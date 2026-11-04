"""End-to-end extraction test on a tiny random Qwen2-Audio (CPU, no downloads).

Uses the real Qwen2AudioProcessor and WhisperFeatureExtractor classes with a toy
byte-level BPE tokenizer, so the chat template, audio-token expansion, hooks,
region pooling, feature store and LoRA loader run exactly as in Colab. Only the
vocabulary and the weights are fake.

Run: python tests/test_model_v2.py
"""
import re
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from erclm import lib_v2 as L  # noqa: E402

SPECIAL = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|AUDIO|>", "<|audio_bos|>", "<|audio_eos|>"]


def make_processor():
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast, Qwen2AudioProcessor, WhisperFeatureExtractor
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    corpus = [p["system"] + " " + p["user"] for p in L.PROMPTS.values()]
    corpus += ["system user assistant Audio 1: A B SPEECH NONSPEECH"]
    trainer = trainers.BpeTrainer(vocab_size=300, special_tokens=SPECIAL,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(corpus * 50, trainer)
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="<|endoftext|>", pad_token="<|endoftext|>",
                                   additional_special_tokens=SPECIAL[1:])
    fe = WhisperFeatureExtractor(feature_size=128, sampling_rate=16000, hop_length=160, chunk_length=30, n_fft=400)
    return Qwen2AudioProcessor(feature_extractor=fe, tokenizer=fast)


def make_model(proc, seed=0):
    from transformers import Qwen2AudioConfig, Qwen2AudioForConditionalGeneration
    cfg = Qwen2AudioConfig(
        audio_config=dict(d_model=32, encoder_layers=2, encoder_attention_heads=2, encoder_ffn_dim=64,
                          num_mel_bins=128, max_source_positions=1500),
        text_config=dict(vocab_size=len(proc.tokenizer), hidden_size=32, intermediate_size=64,
                         num_hidden_layers=3, num_attention_heads=2, num_key_value_heads=1,
                         max_position_embeddings=512),
        audio_token_index=proc.tokenizer.convert_tokens_to_ids("<|AUDIO|>"))
    torch.manual_seed(seed)
    return Qwen2AudioForConditionalGeneration(cfg).eval()


def write_bank(d: Path, n_rec_per_class=6, durations=(20, 200, 1000)):
    """Tiny bank named as in the study under review: harmonic 'speech' vs noise bursts in 2 s containers."""
    rng = np.random.default_rng(0)
    n = 32000
    rows = []
    for lab in (1, 0):
        for r in range(n_rec_per_class):
            stem = f"voxconverse_rec{r:02d}a_0001" if lab else f"esc50_1-{1000 + r}-A-{r}_0001"
            for dur in durations:
                x = rng.standard_normal(n) * 1e-4
                k = int(dur * 16)
                t = np.arange(k) / 16000
                seg = (np.sin(2 * np.pi * 150 * t) + 0.5 * np.sin(2 * np.pi * 300 * t)) * (1 + np.sin(2 * np.pi * 4 * t)) \
                    if lab else rng.standard_normal(k)
                x[(n - k) // 2:(n - k) // 2 + k] += 0.3 * seg / np.abs(seg).max()
                name = f"{stem}_1000ms_dur{dur}ms.wav"
                sf.write(str(d / name), x.astype(np.float32), 16000)
                rows.append({"audio_path": f"/somewhere/else/{name}",
                             "ground_truth": "SPEECH" if lab else "NONSPEECH"})
    pd.DataFrame(rows).to_csv(d / "manifest.csv", index=False)
    return d / "manifest.csv"


PROC = make_processor()


def test_prompt_and_tokens():
    for key, prompt in L.PROMPTS.items():
        text = L.build_text(PROC, prompt)
        assert text.count("<|AUDIO|>") == 1 and text.endswith("<|im_start|>assistant\n"), text[-60:]
        info = L.label_token_info(PROC.tokenizer, prompt["labels"])
        assert info["pos_primary"] != info["neg_primary"]
        assert not (set(info["pos_ids"]) & set(info["neg_ids"]))
    wav = np.zeros(32000, np.float32)
    inp = L.prepare_inputs(PROC, L.build_text(PROC, L.PROMPTS["ab"]), wav)
    aid = PROC.tokenizer.convert_tokens_to_ids("<|AUDIO|>")
    assert int((inp["input_ids"][0] == aid).sum()) == 50
    assert tuple(inp["input_features"].shape) == (1, 128, 3000)
    assert int(inp["feature_attention_mask"].sum()) == 200


def test_forward_collect_hooks_and_head():
    model = make_model(PROC)
    rec = L.Recorder(model, llm_layers=(0, -1))
    assert rec.names["final_norm"].endswith("language_model.norm"), rec.names
    info = L.label_token_info(PROC.tokenizer, ("A", "B"))
    wav = (np.random.default_rng(1).standard_normal(32000) * 0.05).astype(np.float32)
    inp = L.prepare_inputs(PROC, L.build_text(PROC, L.PROMPTS["ab"]), wav)
    aid = PROC.tokenizer.convert_tokens_to_ids("<|AUDIO|>")
    scal, arr = L.forward_collect(model, rec, inp, info, segment_ms=200, n_samples=len(wav),
                                  audio_token_id=aid, check_head=True)
    assert scal["n_audio_tokens"] == 50 and scal["n_audio_in_ids"] == 50
    assert scal["head_check_maxdiff"] < 1e-4
    assert (scal["region_k0"], scal["region_k1"]) == (21, 28)
    for k in ("h_final", "enc_global_mean", "enc_region_mean", "enc_region_max", "proj_region_mean",
              "h_layer0", "h_layer2"):
        assert k in arr and arr[k].shape == (32,) and np.isfinite(arr[k]).all(), k
    # the hooked state is the exact input of the unembedding, for the whole vocabulary
    with torch.inference_mode():
        out = model(**inp, use_cache=False)
        full = model.lm_head(rec.data["final_norm"]).float()[0]
    assert torch.allclose(full, out.logits[0, -1].float(), atol=1e-5)
    # determinism at batch size 1
    scal2, arr2 = L.forward_collect(model, rec, inp, info, segment_ms=200, n_samples=len(wav))
    assert scal2["s_primary"] == scal["s_primary"] and np.array_equal(arr2["h_final"], arr["h_final"])
    rec.remove()


def test_run_extraction_resume_and_analysis():
    model = make_model(PROC)
    info = L.label_token_info(PROC.tokenizer, ("A", "B"))
    aid = PROC.tokenizer.convert_tokens_to_ids("<|AUDIO|>")
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        man = L.load_manifest(write_bank(d), audio_root=d)
        cf = L.make_content_free(d / "cf", n_noise_seeds=2)
        rec = L.Recorder(model, llm_layers=(1,))
        store = L.FeatureStore(d / "feats", shard_size=7)
        L.run_extraction(model, PROC, rec, man, store, "ab", info, audio_token_id=aid, max_items=10, log_every=0)
        assert len(store.done_ids()) == 10
        L.run_extraction(model, PROC, rec, pd.concat([man, cf], ignore_index=True), store, "ab", info,
                         audio_token_id=aid, log_every=0)
        meta, arrays = L.load_store(d / "feats")
        assert len(meta) == len(man) + len(cf) and arrays["h_final"].shape == (len(meta), 32)
        assert meta["variant_id"].is_unique and set(meta["role"]) == {"test", "content_free"}
        test = meta[meta.role == "test"].reset_index(drop=True)
        keep = (meta.role == "test").to_numpy()
        arr_t = {k: v[keep] for k, v in arrays.items()}
        s_cf = meta.loc[meta.role == "content_free", "s_primary"].to_numpy()
        R, inf = L.build_readouts(test, arr_t, s_cf=s_cf, n_splits=3, Cs=(1e-2, 1e-1))
        assert {"L0", "L1_bc_cond", "L1_cf", "L2_cv", "P_enc_region_mean_cv", "P_enc_global_mean_cv"} <= set(R)
        tab = L.condition_table(test, R)
        assert len(tab) == 3 * len(R) and tab["auroc"].between(0, 1).all()
        ci, cc = L.cluster_bootstrap(test, R, B=50, contrasts={
            "H2a": [(1, "P_enc_region_mean_cv", "dur_200ms", "auroc"), (-1, "P_enc_region_mean_cv", "dur_20ms", "auroc")]})
        assert len(cc) == 1 and np.isfinite(cc["estimate"]).all()
        thr = L.threshold_table(tab, "ba", 0.9, "duration")
        assert set(thr["status"]) <= {"ok", "ambiguous", "censored_below", "censored_above"}
        rec.remove()


def _randomize_lora(pm, seed=3):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for n, p in pm.named_parameters():
            if "lora_B" in n:
                p.copy_(torch.randn(p.shape, generator=g) * 0.5)


def test_lora_targets_encoder_and_loader_roundtrip():
    from peft import LoraConfig, get_peft_model
    from safetensors.torch import load_file, save_file
    info = L.label_token_info(PROC.tokenizer, ("A", "B"))
    wav = (np.random.default_rng(2).standard_normal(32000) * 0.05).astype(np.float32)
    inp = L.prepare_inputs(PROC, L.build_text(PROC, L.PROMPTS["ab"]), wav)
    # same target list as finetune.py of the study under review
    cfg = LoraConfig(r=4, lora_alpha=16, lora_dropout=0.0, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    pm = get_peft_model(make_model(PROC), cfg)
    rep = L.lora_report(pm)
    print(f"    target list of the study under review on Qwen2-Audio modules -> {rep['by_component']}")
    assert rep["by_component"].get("audio_tower", 0) > 0          # suffix matching reaches the encoder
    assert rep["n_lora_B_zero"] == rep["n_lora_modules"]          # fresh adapter: B is zero
    _randomize_lora(pm)
    pm.eval()
    with torch.inference_mode():
        ref = pm(**inp, use_cache=False).logits[0, -1].float()
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        pm.save_pretrained(str(d / "v5"))
        loaded, rep2 = L.load_lora(make_model(PROC), d / "v5")
        assert rep2["n_lora_B_zero"] == 0 and rep2["remapped"] is None
        with torch.inference_mode():
            got = loaded(**inp, use_cache=False).logits[0, -1].float()
        assert torch.allclose(ref, got, atol=1e-5)
        rec = L.Recorder(loaded)                                   # hooks still resolve inside PEFT
        L.forward_collect(loaded, rec, inp, info, 200, len(wav), check_head=True)
        rec.remove()
        # adapter saved with transformers 4.x module paths
        legacy = d / "v4"
        legacy.mkdir()
        (legacy / "adapter_config.json").write_text((d / "v5" / "adapter_config.json").read_text())
        sd = load_file(str(d / "v5" / "adapter_model.safetensors"))
        sd4 = {re.sub(r"^base_model\.model\.model\.language_model\.", "base_model.model.language_model.model.",
                      re.sub(r"^base_model\.model\.model\.audio_tower\.", "base_model.model.audio_tower.", k)): v
               for k, v in sd.items()}
        assert set(sd4) != set(sd)
        save_file(sd4, str(legacy / "adapter_model.safetensors"))
        loaded4, rep4 = L.load_lora(make_model(PROC), legacy)
        print(f"    legacy-named adapter -> remapped={rep4['remapped']}, zero B={rep4['n_lora_B_zero']}")
        assert rep4["n_lora_B_zero"] == 0
        with torch.inference_mode():
            got4 = loaded4(**inp, use_cache=False).logits[0, -1].float()
        assert torch.allclose(ref, got4, atol=1e-5)


def main():
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception:
            failed += 1
            print(f"FAIL {name}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
