"""Generation readout for Qwen2-Audio (src/erclm/genread.py) on CPU with the tiny model.

Run: python -m pytest -q tests/test_genread.py
"""
import numpy as np
import pytest

import helpers
from erclm import genread
from erclm import lib_v2 as L
from erclm.gitlog import RunsLog
from erclm.host import Host
from erclm.runner import Runner, StageConfigMismatch


@pytest.mark.parametrize("text,label", [
    ("SPEECH", "SPEECH"), ("NONSPEECH", "NONSPEECH"), ("NONSPEECH.", "NONSPEECH"), ("Non-speech.", "NONSPEECH"),
    ("This is non-speech.", "NONSPEECH"), ("This is speech.", "SPEECH"), ("Speech", "SPEECH"),
    ("No.", "NONSPEECH"), ("Yes, a man is talking.", "SPEECH"), ("SILENCE", "NONSPEECH"),
    ("The sound of music", "NONSPEECH"), ("A woman's voice", "SPEECH"), ("S", None), ("", None),
    ("not speech", "SPEECH")])  # a quirk of the study's parser (the synonym "speech"), kept on purpose
def test_parser_follows_the_study(text, label):
    assert genread.normalize_to_binary(text)[0] == label


def test_parse_levels():
    assert genread.normalize_to_binary("NON-SPEECH")[1] == "L1_NONSPEECH"
    assert genread.normalize_to_binary("No")[1] == "L4_YESNO"
    assert genread.normalize_to_binary("silent room")[1] == "L5_KEYWORDS"


def test_first_step_matches_the_forward_readout():
    model, proc = helpers.tiny_loader()
    labels = L.label_token_info(proc.tokenizer, L.PROMPTS["ab"]["labels"])
    text = L.build_text(proc, L.PROMPTS["ab"])
    wav = np.random.default_rng(0).standard_normal(32000).astype(np.float32) * 0.01
    rec = L.Recorder(model, llm_layers=[0])
    s_fwd, _ = L.forward_collect(model, rec, L.prepare_inputs(proc, text, wav), labels, segment_ms=1000,
                                 n_samples=len(wav))
    rec.remove()
    model.generation_config.repetition_penalty = 1.1  # as in Qwen2-Audio's config, it must not touch the score
    scal, arrays = genread.generate_one(model, proc, text, wav, labels, max_new_tokens=3)
    assert scal["s_primary"] == pytest.approx(s_fwd["s_primary"], abs=1e-4)
    assert scal["top1"] == s_fwd["top1"] and 1 <= scal["n_new_tokens"] <= 3
    assert scal["gen_label"] in ("SPEECH", "NONSPEECH", "UNKNOWN") and isinstance(scal["answer"], str)
    assert arrays["first_top_ids"].shape == (genread.TOPK,) and arrays["first_top_ids"][0] == scal["top1"]
    assert np.all(np.diff(arrays["first_top_logits"]) <= 0)


def test_runner_generates_resumes_and_keeps_old_specs(tmp_path):
    world = helpers.build_world(tmp_path, train=False)
    cfg = helpers.tiny_config(train=False, run_name="gen-test", backbones=["base"],
                              stages=["data_test", "check_base", "extract_base_eval"],
                              generate={"max_new_tokens": 2})

    def session(n, c=cfg):
        work = world["root"] / f"vm{n}"
        runs = RunsLog(str(world["github"]), None, work / "runs_clone", c["run_name"], backoff_s=0)
        return Runner(c, Host("local", work), world["remote"], runs, model_loader=helpers.tiny_loader,
                      session_id=f"s{n}")

    snap = session(1).run()
    assert snap["stages"]["extract_base_eval"]["status"] == "done"
    meta, arrays = L.load_store(world["remote"].root / "work" / "gen-test" / "base" / "eval")
    assert {"answer", "gen_label", "gen_speech", "n_new_tokens", "s_primary"} <= set(meta.columns)
    assert set(arrays) == {"first_top_ids", "first_top_logits"} and len(meta) == 6 * 2 * 7 + 3
    assert (meta["n_new_tokens"] <= 2).all()
    session(2).run()  # resumes, nothing to redo
    # the generation block enters the spec, so the same run without it stops instead of mixing readouts
    plain = {k: v for k, v in cfg.items() if k != "generate"}
    with pytest.raises(StageConfigMismatch):
        session(3, plain).run()
    r = session(4, plain)
    assert "generate" not in r.spec(r.universe()["extract_base_eval"])


@pytest.mark.parametrize("text,label", [
    ("SPEECH", "SPEECH"), ("NONSPEECH.", "NONSPEECH"), ("Non-speech.", "NONSPEECH"), ("NON-SPEECH", "NONSPEECH"),
    ("This is not human speech; it's a sound effect.", "NONSPEECH"),
    ("This is not human speech, it's a sound effect.", "NONSPEECH"), ("This is speech.", "SPEECH"),
    ("SILENCE", "NONSPEECH"), ("A dog barking", "UNKNOWN"), ("Yes", "SPEECH"), ("", "UNKNOWN"),
    ("A) SPEECH", "SPEECH"), ("Speech known", "SPEECH")])
def test_qwen2_parser_follows_the_study(text, label):
    assert genread.parse_qwen2_audio(text) == label


def test_the_two_parsers_differ_on_negations():
    t = "This is not human speech; it's a sound effect."
    assert genread.normalize_to_binary(t)[0] == "SPEECH" and genread.parse_qwen2_audio(t) == "NONSPEECH"
