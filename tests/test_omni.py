"""Qwen3-Omni second-model pilot on CPU: a tiny random thinker behind the real processor classes.

The processor is Qwen3OmniMoeProcessor with the real chat template (fixture) and a toy
byte-level tokenizer, so the audio-token expansion (26 tokens for 2 s), the first-token
readout and the head check run as on the GPU. The shard fetcher is tested with the real
loader when transformers is 4.57 (the version the pilot pins) and with a stand-in otherwise.

Run: python -m pytest -q tests/test_omni.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import helpers
from erclm import lib_v2 as L
from erclm import omni
from erclm.gitlog import RunsLog
from erclm.host import Host
from erclm.runner import Runner, ensure_pip

FIX = Path(__file__).resolve().parent / "fixtures"
SPECIAL = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|audio_start|>", "<|audio_end|>", "<|audio_pad|>",
           "<|vision_start|>", "<|vision_end|>", "<|image_pad|>", "<|video_pad|>"]
EXTRA = {"image_token": "<|image_pad|>", "audio_token": "<|audio_pad|>", "video_token": "<|video_pad|>",
         "vision_bos_token": "<|vision_start|>", "vision_eos_token": "<|vision_end|>",
         "audio_bos_token": "<|audio_start|>", "audio_eos_token": "<|audio_end|>"}


def make_processor():
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import Qwen2VLImageProcessor, Qwen2VLVideoProcessor, Qwen3OmniMoeProcessor, WhisperFeatureExtractor
    try:  # transformers 4.57 checks the tokenizer class of the processor
        from transformers import Qwen2TokenizerFast as Fast
    except ImportError:
        from transformers import PreTrainedTokenizerFast as Fast
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    corpus = [p["system"] + " " + p["user"] for p in omni.PROMPTS.values()]
    corpus += ["system user assistant A B SPEECH NONSPEECH"]
    trainer = trainers.BpeTrainer(vocab_size=300, special_tokens=SPECIAL,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(corpus * 50, trainer)
    fast = Fast(tokenizer_object=tok, eos_token="<|im_end|>", pad_token="<|endoftext|>",
                                   additional_special_tokens=SPECIAL[1:], extra_special_tokens=EXTRA)
    fe = WhisperFeatureExtractor(feature_size=128, sampling_rate=16000, hop_length=160, chunk_length=300,
                                 n_fft=400, return_attention_mask=True, padding_side="right", dither=0.0)
    return Qwen3OmniMoeProcessor(image_processor=Qwen2VLImageProcessor(), video_processor=Qwen2VLVideoProcessor(),
                                 feature_extractor=fe, tokenizer=fast,
                                 chat_template=(FIX / "qwen3_omni_chat_template.jinja").read_text(encoding="utf-8"))


def make_model(proc, seed=0):
    from transformers import Qwen3OmniMoeConfig, Qwen3OmniMoeForConditionalGeneration
    t = proc.tokenizer
    ids = {k: t.convert_tokens_to_ids(v) for k, v in {
        "audio_token_id": "<|audio_pad|>", "audio_start_token_id": "<|audio_start|>",
        "audio_end_token_id": "<|audio_end|>", "image_token_id": "<|image_pad|>",
        "video_token_id": "<|video_pad|>", "vision_start_token_id": "<|vision_start|>",
        "vision_end_token_id": "<|vision_end|>"}.items()}
    thinker = dict(
        audio_config=dict(d_model=32, encoder_layers=2, encoder_attention_heads=2, encoder_ffn_dim=64,
                          output_dim=64, downsample_hidden_size=16, num_mel_bins=128, n_window=50,
                          n_window_infer=800, conv_chunksize=500),
        vision_config=dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64,
                           deepstack_visual_indexes=[0]),
        text_config=dict(hidden_size=64, intermediate_size=64, moe_intermediate_size=32, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=16, num_experts=4,
                         num_experts_per_tok=2, vocab_size=len(t),
                         rope_scaling={"interleaved": True, "mrope_interleaved": True, "mrope_section": [4, 2, 2],
                                       "rope_type": "default", "type": "default"}),
        user_token_id=t.convert_tokens_to_ids("<|im_start|>"), position_id_per_seconds=13, seconds_per_chunk=2, **ids)
    torch.manual_seed(seed)
    return Qwen3OmniMoeForConditionalGeneration(Qwen3OmniMoeConfig(thinker_config=thinker,
                                                                   enable_audio_output=False)).eval()


PROC = make_processor()


def tiny_omni_loader():
    return make_model(PROC), PROC, {"tiny": True}


def test_prompts_and_labels():
    assert omni.PROMPTS["ab"] is L.PROMPTS["ab"]
    for k, p in omni.PROMPTS.items():
        info = L.label_token_info(PROC.tokenizer, p["labels"])
        assert info["pos_primary"] != info["neg_primary"], k


def test_check_passes_on_the_tiny_model():
    model = make_model(PROC)
    assert not hasattr(model, "talker")
    res = omni.check(model, PROC)
    assert res["failures"] == [], res["failures"]
    assert res["n_audio_in_ids"] == omni.N_AUDIO_TOKENS_2S
    assert res["deterministic"] and res["h_final_dim"] == 64


def test_shard_fetcher_loads_one_shard_at_a_time(tmp_path):
    """Placeholders resolve, each shard exists only while it loads, and the model equals the source."""
    import transformers.modeling_utils as mu
    src = make_model(PROC, seed=3)
    remote = tmp_path / "hub"
    src.save_pretrained(remote, max_shard_size="300KB")
    index = json.loads((remote / "model.safetensors.index.json").read_text())
    local = tmp_path / "local"
    local.mkdir()
    for p in remote.iterdir():
        if p.suffix != ".safetensors":
            (local / p.name).write_bytes(p.read_bytes())
    present = []

    def fetch(name, dest):
        dest.write_bytes((remote / name).read_bytes())

    f = omni.ShardFetcher(local, fetch, log=lambda *_: None)
    shards = f.prepare(omni.thinker_only_index(index))
    assert len(shards) > 1 and all((local / s).stat().st_size == 0 for s in shards)
    if hasattr(mu, "load_shard_file"):  # transformers 4.57, the version the pilot pins
        original = mu.load_shard_file

        def spy(args):
            present.append(sum((local / s).stat().st_size > 0 for s in shards))
            return original(args)

        mu.load_shard_file = f.wrap(spy)
        try:
            from transformers import Qwen3OmniMoeForConditionalGeneration
            model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(local).eval()
        finally:
            mu.load_shard_file = original
        for (k, v), (k2, v2) in zip(src.state_dict().items(), model.state_dict().items()):
            assert k == k2 and torch.equal(v, v2), k
    else:
        def fake_load(args):
            present.append(sum((local / s).stat().st_size > 0 for s in shards))
            assert Path(args[0]).stat().st_size > 0
            return [], None

        wrapped = f.wrap(fake_load)
        for s in shards:
            wrapped((str(local / s),))
    assert present == [1] * len(shards)
    assert sorted(f.loaded) == shards
    assert all((local / s).stat().st_size == 0 for s in shards)


def test_thinker_only_index_drops_the_talker():
    idx = {"metadata": {"total_size": 1}, "weight_map": {"thinker.a": "s1", "talker.b": "s2", "code2wav.c": "s3"}}
    assert omni.thinker_only_index(idx)["weight_map"] == {"thinker.a": "s1"}


def _fake_meta(rng, cond, auc_sep, offset, n=40):
    y = np.r_[np.ones(n // 2, int), np.zeros(n // 2, int)]
    s = rng.standard_normal(n) + auc_sep * y + offset
    return pd.DataFrame({"variant_id": [f"{cond}_{i}" for i in range(n)], "rec_id": [f"r{i}" for i in range(n)],
                         "label": y, "condition": cond, "s_primary": s, "top1_is_label": True})


def test_table_and_registered_reading():
    rng = np.random.default_rng(0)
    # 20 ms: well ordered but shifted to "no speech" (criterion). 40 ms: no evidence (representation).
    ab = pd.concat([_fake_meta(rng, "dur_20ms", 3.0, -4.0), _fake_meta(rng, "dur_40ms", 0.0, 0.0),
                    _fake_meta(rng, "dur_1000ms", 4.0, -2.0)], ignore_index=True)
    ba = ab.assign(s_primary=ab["s_primary"] + rng.normal(0, 0.1, len(ab)))
    hand = ab.assign(s_primary=ab["s_primary"] + rng.normal(0, 0.1, len(ab)))
    table = omni.condition_table({"ab": ab, "ba_speech_second": ba, "hand_csl": hand}, B=200, seed=1)
    assert set(table["prompt"]) == {"ab", "ba_speech_second", "hand_csl", "s_sim"}
    r20 = table.set_index(["prompt", "condition"]).loc[("ab", "dur_20ms")]
    assert r20["ba_L0"] < 0.6 < r20["ba_L1"] and r20["auroc_lo"] < r20["auroc"] < r20["auroc_hi"]
    rule = {"validity_prompt": "hand_csl", "validity_condition": "dur_1000ms", "validity_min_ba": 0.85,
            "full_items": 27160, "max_hours": 25, "read_prompts": ["ab", "s_sim"],
            "read_conditions": ["dur_20ms", "dur_40ms"], "representation_auroc_hi": 0.65,
            "criterion_auroc_lo": 0.6, "criterion_gap": 0.1}
    v = omni.verdict(table, 2.0, rule)
    assert v["reading"]["ab/dur_20ms"]["reading"] == "criterion"
    assert v["reading"]["ab/dur_40ms"]["reading"] in ("representation", "inconclusive")
    assert v["validity"]["ok"] and v["feasibility"]["projected_h"] == pytest.approx(15.1, abs=0.1)
    assert v["full_run"]
    assert not omni.verdict(table, 4.0, rule)["full_run"]


def test_ensure_pip_runs_only_off_the_laptop(monkeypatch):
    calls = []
    monkeypatch.delenv("ERCLM_PIP", raising=False)
    monkeypatch.setattr("subprocess.run", lambda cmd, **kw: calls.append(cmd) or type("R", (), {"returncode": 0})())
    assert not ensure_pip(["x==1"], ["--config", "c.yaml"], "local", execv=lambda *a: calls.append(a))
    assert ensure_pip(["x==1"], ["--config", "c.yaml"], "kaggle", execv=lambda *a: calls.append(a))
    assert calls[0][-1] == "x==1" and calls[1][1][-2:] == ["--config", "c.yaml"]
    assert not ensure_pip(["x==1"], ["--config", "c.yaml"], "kaggle", execv=lambda *a: calls.append(a))
    assert len(calls) == 2


def omni_config(**over):
    rule = {"validity_prompt": "hand_csl", "validity_condition": "dur_1000ms", "validity_min_ba": 0.85,
            "full_items": 27160, "max_hours": 25, "read_prompts": ["ab", "s_sim"],
            "read_conditions": ["dur_20ms"], "representation_auroc_hi": 0.65, "criterion_auroc_lo": 0.6,
            "criterion_gap": 0.1}
    cfg = {"run_name": "omni-test", "backbones": [], "stages": ["data_test", "extract_omni"],
           "pip": ["transformers==4.57.6"],
           "omni": {"prompts": ["ab", "ba_speech_second", "hand_csl"], "conditions": ["dur_20ms", "dur_1000ms"],
                    "rule": rule}}
    return helpers.tiny_config(train=False, **{**cfg, **over})


def test_runner_stage_scores_resumes_and_publishes(tmp_path):
    world = helpers.build_world(tmp_path, train=False)
    cfg = omni_config()
    loads = []

    def loader():
        loads.append(1)
        return tiny_omni_loader()

    def session(n):
        work = world["root"] / f"vm{n}"
        runs = RunsLog(str(world["github"]), None, work / "runs_clone", cfg["run_name"], backoff_s=0)
        return Runner(cfg, Host("local", work), world["remote"], runs, model_loader=helpers.tiny_loader,
                      omni_loader=loader, session_id=f"s{n}")

    r = session(1)
    assert [s.name for s in r.plan()] == ["data_test", "extract_omni"]
    snap = r.run()
    assert snap["stages"]["extract_omni"]["status"] == "done"
    for pk in cfg["omni"]["prompts"]:
        meta, arrays = L.load_store(world["remote"].root / "work" / "omni-test" / "omni" / pk)
        assert len(meta) == 12 * 2 and set(meta["prompt"]) == {pk} and arrays["h_final"].shape == (24, 64)
        assert (meta["n_audio_in_ids"] == omni.N_AUDIO_TOKENS_2S).all()
    files = helpers.ls_runs(world["github"], "omni-test/")
    assert {"omni-test/checks_omni.json", "omni-test/results/omni/condition_table.csv",
            "omni-test/results/omni/verdict.json", "omni-test/extract_omni.json"} <= set(files)
    table = pd.read_csv(world["remote"].root / "work" / "omni-test" / "omni" / "results" / "condition_table.csv")
    assert set(table["prompt"]) == {"ab", "ba_speech_second", "hand_csl", "s_sim"}
    assert len(table) == 4 * 2
    # a second session finds the stage done and loads nothing
    session(2).run()
    assert loads == [1]
    # forced, the stores are rebuilt from scratch
    r3 = Runner(cfg, Host("local", world["root"] / "vm3"), world["remote"],
                RunsLog(str(world["github"]), None, world["root"] / "vm3" / "runs_clone", cfg["run_name"], backoff_s=0),
                omni_loader=loader, session_id="s3", force=["extract_omni"])
    r3.run()
    assert loads == [1, 1]


def test_omni_stage_only_when_asked(tmp_path):
    world = helpers.build_world(tmp_path, train=False)
    cfg = omni_config(stages=None)
    runs = RunsLog(str(world["github"]), None, tmp_path / "rc", cfg["run_name"], backoff_s=0)
    r = Runner(cfg, Host("local", tmp_path / "vm"), world["remote"], runs, omni_loader=tiny_omni_loader)
    assert "extract_omni" in r.universe() and "extract_omni" not in [s.name for s in r.plan()]


def test_config_env_is_set_before_the_runner_starts(tmp_path, monkeypatch):
    """omni-pilot-v1 ran out of memory while quantising (fragmentation). The pilot config sets the
    allocator through env, which main applies before torch is used."""
    import yaml
    from erclm import runner
    monkeypatch.delenv("ERCLM_TEST_ENV", raising=False)
    cfg = yaml.safe_load(Path("configs/omni_pilot.yaml").read_text(encoding="utf-8"))
    assert cfg["env"]["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
    cfg["env"] = {"ERCLM_TEST_ENV": "1"}
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    assert runner.main(["--config", str(path), "--remote-dir", str(tmp_path / "drive"), "--no-runs",
                        "--work", str(tmp_path / "w"), "--plan"]) == 0
    import os
    assert os.environ["ERCLM_TEST_ENV"] == "1"


def test_fetcher_reports_after_each_shard(tmp_path):
    local = tmp_path / "l"
    local.mkdir()
    notes = []
    f = omni.ShardFetcher(local, lambda name, dest: dest.write_bytes(b"x"), log=notes.append,
                          after_each=lambda: ", note")
    f.prepare({"weight_map": {"thinker.a": "s1.safetensors"}})
    f.wrap(lambda args: ([], None))((str(local / "s1.safetensors"),))
    assert notes and notes[0].endswith(", note")


def test_token_spans_put_the_centred_excerpt_on_the_chunk_seam():
    """2 s give two chunks of 1 s with 13 tokens each. The 13th token of a chunk is cut at 1000 ms."""
    sp = omni.token_spans(200, 50)
    assert len(sp) == omni.N_AUDIO_TOKENS_2S
    assert tuple(sp[0]) == (0, 80) and tuple(sp[12]) == (960, 1000) and tuple(sp[13]) == (1000, 1080)
    assert list(omni.region_indices(sp, 20, 2000)) == [11, 12, 13, 14]       # both sides of the seam
    assert list(omni.region_indices(sp, 20, 2000, shift_ms=500)) == [18, 19, 20]  # middle of chunk 2
    assert list(omni.region_indices(sp, 1000, 2000)) == list(range(5, 21))
    assert list(omni.region_indices(sp, 1000, 2000, shift_ms=500)) == list(range(12, 26))
    with pytest.raises(ValueError):
        omni.region_indices(sp, 1000, 2000, shift_ms=600)


def test_tracks_from_prompts_or_explicit():
    o = {"prompts": ["ab", "hand_csl"], "conditions": ["dur_20ms", "dur_1000ms"]}
    assert [t["name"] for t in omni.tracks(o)] == ["ab", "hand_csl"]
    o["tracks"] = ["ab", {"name": "ab_shift500", "prompt": "ab", "shift_ms": 500, "conditions": ["dur_20ms"]}]
    ts = omni.tracks(o)
    assert ts[1] == {"name": "ab_shift500", "prompt": "ab", "shift_ms": 500.0, "conditions": ["dur_20ms"]}
    with pytest.raises(ValueError):
        omni.tracks({**o, "tracks": ["ab", "ab"]})
    with pytest.raises(ValueError):
        omni.tracks({**o, "tracks": [{"prompt": "ab", "conditions": ["dur_40ms"]}]})


def test_forward_keeps_the_probe_states():
    model = make_model(PROC)
    tok = PROC.tokenizer
    labels = L.label_token_info(tok, omni.PROMPTS["ab"]["labels"])
    aid = tok.convert_tokens_to_ids("<|audio_pad|>")
    text = omni.build_text(PROC, omni.PROMPTS["ab"])
    wav = np.random.default_rng(0).standard_normal(32000).astype(np.float32) * 0.01
    scal, a = omni.forward_scores(model, PROC, text, wav, labels, aid, segment_ms=20, layers=(0, 1, 12))
    assert {k: v.shape for k, v in a.items()} == {
        "h_final": (64,), "enc_global_mean": (32,), "h_layer0": (64,), "enc_region_mean": (32,),
        "enc_region_max": (32,), "proj_region_mean": (64,)}   # layer 1 is the last, kept as h_final
    assert (scal["region_k0"], scal["region_k1"]) == (11, 14)
    assert not np.allclose(a["enc_region_mean"], a["enc_global_mean"])
    s2, a2 = omni.forward_scores(model, PROC, text, np.roll(wav, 8000), labels, aid, segment_ms=20, shift_ms=500)
    assert (s2["region_k0"], s2["region_k1"]) == (18, 20) and s2["s_primary"] != scal["s_primary"]
    _, a3 = omni.forward_scores(model, PROC, text, wav, labels, aid)
    assert "enc_region_mean" not in a3 and np.array_equal(a3["enc_global_mean"], a["enc_global_mean"])


def test_runner_stage_with_a_shifted_track(tmp_path):
    world = helpers.build_world(tmp_path, train=False)
    o = omni_config()["omni"]
    cfg = omni_config(omni={**o, "tracks": [
        "ab", "ba_speech_second", {"name": "ab_shift500", "prompt": "ab", "shift_ms": 500,
                                   "conditions": ["dur_20ms"]}]})
    runs = RunsLog(str(world["github"]), None, tmp_path / "rc", cfg["run_name"], backoff_s=0)
    r = Runner(cfg, Host("local", tmp_path / "vm"), world["remote"], runs, omni_loader=tiny_omni_loader)
    snap = r.run()
    assert snap["stages"]["extract_omni"]["status"] == "done"
    root = world["remote"].root / "work" / "omni-test" / "omni"
    m0, a0 = L.load_store(root / "ab")
    m1, a1 = L.load_store(root / "ab_shift500")
    assert len(m0) == 24 and len(m1) == 12 and set(m1["condition"]) == {"dur_20ms"}
    assert (m1["shift_ms"] == 500).all() and (m1["prompt"] == "ab").all() and (m1["track"] == "ab_shift500").all()
    assert set(m0.loc[m0["condition"] == "dur_20ms", "region_k0"]) == {11} and set(m1["region_k0"]) == {18}
    assert {"enc_region_mean", "enc_region_max", "proj_region_mean", "enc_global_mean"} <= set(a1)
    assert not (root / "hand_csl").exists()
    table = pd.read_csv(root / "results" / "condition_table.csv")
    assert set(zip(table["prompt"], table["condition"])) >= {("ab_shift500", "dur_20ms"), ("s_sim", "dur_1000ms")}
    assert len(table[table["prompt"] == "ab_shift500"]) == 1
    res = json.loads(helpers.show(world["github"], "omni-test/extract_omni.json"))
    assert res["n_items"]["ab_shift500"] == 12 and res["n_passes"] == 60
