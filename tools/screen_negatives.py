"""Screen of the provenance negatives for speech or singing (W4 of the review of draft v4).

The 114 non-speech windows of the provenance bank were chosen with no annotated speech within
250 ms. This runs an AudioSet tagger (AST, MIT/ast-finetuned-audioset-10-10-0.4593, from the local
Hugging Face cache) on the 1000 ms variant of every clip, speech and non-speech, and ranks the
negatives by their speech and singing probabilities, so a listener can start with the doubtful
ones. A screen, not a label. Exploratory.

    PYTHONPATH=src python tools/screen_negatives.py [--bank _work/closing/prov_audio/prov_bank_v3]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from closing_analyses import WORK, md  # noqa: E402
from erclm import lib_v2 as L2  # noqa: E402

MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"
SPEECH = ("Speech", "Male speech, man speaking", "Female speech, woman speaking", "Child speech, kid speaking",
          "Conversation", "Narration, monologue", "Babbling", "Whispering", "Speech synthesizer", "Shout",
          "Yell", "Chatter", "Hubbub, speech noise, speech babble")
SINGING = ("Singing", "Choir", "Male singing", "Female singing", "Child singing", "Rapping", "Humming", "Yodeling",
           "Chant", "Mantra", "Synthetic singing")


def tag(paths) -> pd.DataFrame:
    import torch
    from transformers import ASTFeatureExtractor, ASTForAudioClassification
    fe = ASTFeatureExtractor.from_pretrained(MODEL, local_files_only=True)
    model = ASTForAudioClassification.from_pretrained(MODEL, local_files_only=True).eval()
    labels = model.config.id2label
    idx = {v: int(k) for k, v in labels.items()}
    missing = [n for n in SPEECH + SINGING + ("Music",) if n not in idx]
    if missing:
        raise RuntimeError(f"AudioSet labels not found in the model: {missing}")
    rows = []
    for p in paths:
        x = L2.load_wav(p)
        with torch.inference_mode():
            prob = torch.sigmoid(model(**fe(x, sampling_rate=16000, return_tensors="pt")).logits[0]).numpy()
        top = np.argsort(prob)[::-1][:3]
        rows.append({"p_speech": float(max(prob[idx[n]] for n in SPEECH)),
                     "p_singing": float(max(prob[idx[n]] for n in SINGING)),
                     "p_music": float(prob[idx["Music"]]),
                     "top3": "; ".join(f"{labels[int(i)]} {prob[i]:.2f}" for i in top)})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bank", default=str(WORK / "prov_audio" / "prov_bank_v3"))
    a = ap.parse_args()
    bank = Path(a.bank)
    meta = pd.read_csv(bank / "metadata.csv")
    clips = meta[meta["duration_ms"] == 1000].reset_index(drop=True)
    out = pd.concat([clips[["base_id", "recording", "ground_truth", "start_s"]],
                     tag([bank / "audio" / p for p in clips["audio_path"]])], axis=1)
    d = WORK / "negatives"
    d.mkdir(parents=True, exist_ok=True)
    out.to_csv(d / "screen.csv", index=False)
    sp, ns = out[out["ground_truth"] == "SPEECH"], out[out["ground_truth"] == "NONSPEECH"]
    print("## Screen of the provenance clips at 1000 ms with AST (AudioSet)\n")
    q = pd.DataFrame({"speech clips": sp[["p_speech", "p_singing", "p_music"]].quantile([0.05, 0.5, 0.95]).stack(),
                      "non-speech clips": ns[["p_speech", "p_singing", "p_music"]].quantile([0.05, 0.5, 0.95]).stack()})
    print(md(q, 3), "\n")
    for thr in (0.1, 0.3, 0.5):
        print(f"- non-speech clips with p_speech >= {thr}: {int((ns['p_speech'] >= thr).sum())} of {len(ns)}, "
              f"with p_singing >= {thr}: {int((ns['p_singing'] >= thr).sum())}")
    print("\n### Non-speech clips to listen to first (p_speech or p_singing >= 0.1), most doubtful first\n")
    doubt = ns[(ns["p_speech"] >= 0.1) | (ns["p_singing"] >= 0.1)].copy()
    doubt["score"] = doubt[["p_speech", "p_singing"]].max(axis=1)
    print(md(doubt.sort_values("score", ascending=False)[["base_id", "start_s", "p_speech", "p_singing", "p_music",
                                                           "top3"]].set_index("base_id"), 2), "\n")


if __name__ == "__main__":
    main()
