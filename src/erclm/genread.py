"""Generation readout for Qwen2-Audio (2026-11-02). Exploratory, nothing frozen changes.

The readouts of lib_v2 take the first answer token. With the free-answer prompts of the
study under review the model often starts with a word that is not a label ("This", "non",
"Non"), so the first-token score compares two tokens the model does not choose
(docs/exploratory_log.md, "Result of the study prompts"). This module reads the
answer the way the study did. It generates greedily and parses the text with the study's
normalize_to_binary, ported below without changes to its default path. The same pass keeps
the first-step logits, so the two readouts can be compared item by item.

Rule in docs/exploratory_log.md, "Generation readout of the study prompts (2 November)".
"""
from __future__ import annotations

import re
import time
from typing import Optional

import numpy as np
import pandas as pd

from . import lib_v2 as L

TOPK = 20  # first-step logits kept per item, for a first-token score with the tokens the model uses

# ============================================================================ the study's parser

# gbibbo/qwen-vad-lora, src/qsm/utils/normalize.py, normalize_to_binary with mode "auto", no
# letter mapping and no probabilities, which is how the study parsed these prompts.
_YES = ["YES", "SÍ", "SI", "AFFIRMATIVE", "TRUE", "CORRECT", "PRESENT"]
_NO = ["NO", "NEGATIVE", "FALSE", "INCORRECT", "ABSENT", "NOT PRESENT"]
_SPEECH_SYN = ["voice", "voices", "talking", "spoken", "speaking", "speaker", "conversation", "conversational",
               "words", "utterance", "vocal", "human voice", "person talking", "dialogue", "speech", "syllables",
               "phonemes", "formants"]
_NONSPEECH_SYN = ["music", "musical", "song", "melody", "instrumental", "beep", "beeps", "tone", "tones", "pitch",
                  "sine wave", "noise", "noisy", "static", "hiss", "white noise", "silence", "silent", "quiet",
                  "nothing", "empty", "ambient", "environmental", "background", "click", "clicks", "clock", "tick",
                  "ticking"]


def normalize_to_binary(text: str) -> tuple:
    """(label, level), label SPEECH, NONSPEECH or None, level L1_NONSPEECH to L6_UNKNOWN."""
    if not text:
        return None, "L6_UNKNOWN"
    up, low = text.strip().upper(), text.strip().lower()
    if any(k in up for k in ("NONSPEECH", "NON-SPEECH", "NON SPEECH", "NO SPEECH")):
        if "NOT NONSPEECH" not in up and "NOT NON-SPEECH" not in up:
            return "NONSPEECH", "L1_NONSPEECH"
    if "SPEECH" in up and "NOT SPEECH" not in up:
        return "SPEECH", "L2_SPEECH"
    for pat in _YES:
        if re.search(r"\b" + re.escape(pat) + r"\b", up):
            return "SPEECH", "L4_YESNO"
    for pat in _NO:
        if re.search(r"\b" + re.escape(pat) + r"\b", up):
            return "NONSPEECH", "L4_YESNO"
    sp = sum(1 for s in _SPEECH_SYN if s in low)
    ns = sum(1 for s in _NONSPEECH_SYN if s in low)
    if sp > ns:
        return "SPEECH", "L5_KEYWORDS"
    if ns > sp:
        return "NONSPEECH", "L5_KEYWORDS"
    return None, "L6_UNKNOWN"


def parse_qwen2_audio(text: str) -> str:
    """The parser that produced the study's Qwen2-Audio predictions, SPEECH, NONSPEECH or UNKNOWN.

    gbibbo/qwen-vad-lora, src/qsm/models/qwen_audio.py, Qwen2AudioClassifier._parse_response,
    ported without changes. normalize_to_binary above is what the study's Qwen3-Omni classifier
    used. It reads "This is not human speech" as speech, while this one finds the negation before
    SPEECH, as the study's predictions do (147 such answers in 02_base_opro_llm, all NONSPEECH).
    Added on 2 November 2026 as a declared secondary reading (docs/exploratory_log.md).
    """
    response = text.strip().upper()
    norm = re.sub(r"[^A-Z0-9\s]", "", response)
    if "A)" in response or response == "A" or norm == "A":
        return "SPEECH"
    if any(x in response for x in ["B)", "C)", "D)"]) or norm in ["B", "C", "D"]:
        return "NONSPEECH"
    if any(p in norm for p in ["NONSPEECH", "NON SPEECH", "NOSPEECH", "NO SPEECH"]):
        return "NONSPEECH"
    if any(k in norm for k in ["NOISE", "SILENCE", "SILENT", "MUSIC", "BACKGROUND", "AMBIENT", "QUIET"]):
        return "NONSPEECH"
    if "SPEECH" in norm:
        before = norm.split("SPEECH")[0]
        return "NONSPEECH" if any(n in before for n in ["NO", "NOT", "NONE", "WITHOUT", "ISNT", "DOESNT"]) else "SPEECH"
    if any(p in norm for p in ["THERE IS NO", "THERE ISNT", "DOES NOT CONTAIN", "NO AUDIO", "NOT DETECTED"]):
        return "NONSPEECH"
    if any(p in norm for p in ["THERE IS", "CONTAINS", "PRESENT", "YES", "DETECTED", "VOICE", "HUMAN"]):
        return "SPEECH"
    return "UNKNOWN"


# ============================================================================ generation

def generate_one(model, processor, text: str, wav: np.ndarray, label_info: dict, max_new_tokens: int = 16,
                 audio_token_id: Optional[int] = None):
    """Greedy answer and its parse, plus the first-step score of lib_v2 and the top first-step logits."""
    import torch
    inputs = L.prepare_inputs(processor, text, wav).to(L.model_device(model))
    n_in = int(inputs["input_ids"].shape[1])
    tok = processor.tokenizer
    with torch.inference_mode():
        # the model's generation config stays as the study left it (do_sample off, its repetition
        # penalty of 1.1 on). The first-step score uses the raw logits, before that penalty.
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
                             return_dict_in_generate=True, output_logits=True,
                             pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
    new = out.sequences[0, n_in:]
    first = out.logits[0][0].float()
    answer = tok.decode(new, skip_special_tokens=True)
    label, level = normalize_to_binary(answer)
    cfg_eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    eos = {tok.eos_token_id} | set(cfg_eos if isinstance(cfg_eos, (list, tuple)) else [cfg_eos])
    pos, neg = label_info["pos_ids"], label_info["neg_ids"]
    zA, zB = float(first[label_info["pos_primary"]]), float(first[label_info["neg_primary"]])
    top = torch.topk(first, TOPK)
    scal = {"zA": zA, "zB": zB, "s_primary": zA - zB,
            "s_lse": float(torch.logsumexp(first[pos], 0) - torch.logsumexp(first[neg], 0)),
            "top1": int(top.indices[0]), "top1_is_label": bool(int(top.indices[0]) in pos or int(top.indices[0]) in neg),
            "answer": answer, "gen_label": label or "UNKNOWN", "gen_level": level,
            "gen_speech": bool(label == "SPEECH"), "n_new_tokens": int(len(new)),
            "truncated": bool(len(new) >= max_new_tokens and int(new[-1]) not in eos)}
    if audio_token_id is not None:
        scal["n_audio_in_ids"] = int((inputs["input_ids"][0] == audio_token_id).sum())
    arrays = {"first_top_ids": top.indices.cpu().numpy().astype(np.int32),
              "first_top_logits": top.values.cpu().numpy().astype(np.float32)}
    return scal, arrays


def run_generation(model, processor, manifest: pd.DataFrame, store, prompt_key: str, label_info: dict,
                   audio_token_id: Optional[int] = None, max_new_tokens: int = 16, log_every: int = 100,
                   log=print):
    """Generation readout for every manifest row not yet in the store. Resumes like run_extraction."""
    done = store.done_ids()
    todo = manifest[~manifest["variant_id"].astype(str).isin(done)]
    text = L.build_text(processor, L.PROMPTS[prompt_key])
    t0, n = time.time(), len(todo)
    log(f"{len(done)} items already stored, {n} to generate")
    for i, row in enumerate(todo.itertuples(index=False)):
        scal, arrays = generate_one(model, processor, text, L.load_wav(row.path), label_info,
                                    max_new_tokens=max_new_tokens, audio_token_id=audio_token_id)
        meta = {c: getattr(row, c) for c in L.META_COLS}
        meta.update(scal)
        meta["prompt"] = prompt_key
        store.add(meta, arrays)
        if log_every and (i + 1) % log_every == 0:
            el = time.time() - t0
            log(f"  {i + 1}/{n}  {el / (i + 1):.3f} s/item  ETA {(n - i - 1) * el / (i + 1) / 60:.1f} min")
    store.flush()
    log(f"done in {(time.time() - t0) / 60:.1f} min")
