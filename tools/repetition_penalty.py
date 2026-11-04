"""The study's repetition penalty on the first generated letter, Qwen2-Audio (2 and 3 November 2026).

Descriptive, decided after reading the generation (docs/exploratory_log.md, "Result of the
generation readout" and "Descriptives of the generation readout (3 November)"). The study generated with the
model's generation_config, whose repetition_penalty of 1.1 divides the positive logit of every
token already in the input (multiplies a negative one). Both OPRO-Template and the hand prompt end
in " NONSPEECH", tokenized " NON", "S", "PE", "ECH", so "S" (50) is in the input and "N" (45) and
"NON" (29563) are not. Here the first step's top-20 logits are read twice, raw and penalized, and
the letter each one picks is compared with the letter the model wrote.

    python tools/repetition_penalty.py <root> [--B 2000] [--out <folder>]

<root> is the one of tools/study_prompts_generation.py, with gen_<p>_<b>/ and q2_base_ba/.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import omni_control as oc  # noqa: E402

S, N, NON = 50, 45, 29563
PROMPTS = ["opro_tmpl", "hand_csl"]   # the two whose answers start with the label
PENALTY = 1.1


def penalize(ids: np.ndarray, logits: np.ndarray, in_input=(S,), penalty: float = PENALTY) -> np.ndarray:
    """HF RepetitionPenaltyLogitsProcessor on the tokens of in_input."""
    hit = np.isin(ids, in_input)
    return np.where(hit, np.where(logits > 0, logits / penalty, logits * penalty), logits)


def letter(tok: np.ndarray) -> np.ndarray:
    return np.where(tok == S, "S", np.where(np.isin(tok, [N, NON]), "N", "?"))


def main(root: Path, B: int = 2000, seed: int = 42, out: Path = None) -> dict:
    root = Path(root)
    out = Path(out or root / "reading_penalty")
    out.mkdir(parents=True, exist_ok=True)
    ref, _ = oc.load(root / "q2_base_ba")
    ref = ref[ref["condition"].isin(oc.DUR + oc.SNR)]
    ids = list(ref.index)
    y = ref["label"].to_numpy() == 1
    res, systems = {}, {}
    for p in PROMPTS:
        for b in ("base", "lora"):
            m, a = oc.aligned(*oc.load(root / f"gen_{p}_{b}", ids=set(ids),
                                       features=("first_top_ids", "first_top_logits")), ids)
            ti, tl = a["first_top_ids"].astype(int), a["first_top_logits"].astype(float)
            raw = letter(ti[:, 0])
            pen = letter(ti[np.arange(len(ti)), penalize(ti, tl).argmax(1)])
            written = m["answer"].fillna("").astype(str).str.strip().str[:1].str.upper().to_numpy()
            both = np.isin(ti, [S]).any(1) & np.isin(ti, [N, NON]).any(1)
            res[f"{b}/{p}"] = {
                "items_with_S_and_N_in_top20": int(both.sum()),
                "penalized_predicts_written": float((pen[both] == written[both]).mean()),
                "raw_S_written_N": int(((raw == "S") & (written == "N")).sum()),
                **{f"{k}_TPR_TNR": [float((v[y] == "S").mean()), float((v[~y] != "S").mean())]
                   for k, v in (("raw", raw), ("penalized", pen))}}
            systems[f"{b}_{p}_raw"] = np.where(raw == "S", 1.0, -1.0)
            systems[f"{b}_{p}_written"] = np.where(written == "S", 1.0, -1.0)
    contrasts = [(f"{b}_{p}_written", f"{b}_{p}_raw") for b in ("base", "lora") for p in PROMPTS]
    contrasts += [(f"{b}_opro_tmpl_{k}", f"{b}_hand_csl_{k}") for b in ("base", "lora") for k in ("raw", "written")]
    t = oc.paired(ref, systems, contrasts, B=B, seed=seed, which=("L0",))
    t.to_csv(out / "penalty.csv", index=False)
    for ax in ("duration", "snr"):
        for w in list(systems) + [f"{x} - {z}" for x, z in contrasts]:
            res[f"{ax}/{w}"] = oc.get(t, ax, "L0", w)
    (out / "reading.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k.split("/")[0] in ("base", "lora")}, indent=1))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out")
    a = ap.parse_args()
    main(Path(a.root), B=a.B, seed=a.seed, out=a.out)
