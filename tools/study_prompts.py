"""The registered reading of the study's prompts on Qwen2-Audio (300 control clips, 30 October 2026).

The rule is in docs/exploratory_log.md, "The study prompts on Qwen2-Audio (30 October)", written
before the runs. For each backbone and each prompt p of the study under review, paired with the
confirmatory A/B prompt on the same 4,200 items and averaged over each axis,

    G = BA_L0(p) - BA_L0(ab)          what choosing the prompt gives with the native rule
    O = oracle(p) - oracle(ab)        the part that comes from ranking (best-threshold BA)
    C = G - O                         the part that comes from the operating point

and whether calibrating without a search reaches it, BA_L1(ab) - BA_L0(p).

    python tools/study_prompts.py <root> [--B 2000] [--out <folder>]

<root> holds the score CSVs downloaded from Drive, as the runner writes them:

    q2_base/, q2_lora/        work/full-v1/<backbone>/eval/                (ab)
    q2_base_ba/, q2_lora_ba/  work/prompt-ba_speech_second/<backbone>/eval/
    <p>_base/, <p>_lora/      work/prompt-<p>/<backbone>/eval/  for p in PROMPTS
    omni/ab/, omni/hand_csl/  work/omni-control-v1/omni/<track>/  (optional, cross-model descriptive)

Intervals are 95% percentiles of a bootstrap over recordings stratified by class, 2,000
replicates, seed 42, one draw shared by every system. L1 is re-estimated in every replicate.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import omni_control as oc  # noqa: E402

PROMPTS = ["hand_csl", "opro_llm_base", "opro_tmpl"]
PRIMARY = {"base": "opro_llm_base", "lora": "opro_tmpl"}  # each backbone's OPRO winner in the study
AXES = {"duration": oc.DUR, "snr": oc.SNR, "short": oc.SHORT}


def derived(r: dict, b: str, p: str, cs: list) -> dict:
    S = lambda k, m: oc.summarise(r[k], m, cs)  # noqa: E731
    ab, q, sim = f"{b}_ab", f"{b}_{p}", f"{b}_sim"
    G = S(q, "L0") - S(ab, "L0")
    O = S(q, "oracle") - S(ab, "oracle")
    return {"G": G, "O": O, "C": G - O, "share_C": (G - O) / G if abs(G) > 1e-9 else np.nan,
            "dAUROC": S(q, "auroc") - S(ab, "auroc"), "L1ab_minus_L0p": S(ab, "L1") - S(q, "L0"),
            "recipe_minus_L0p": S(sim, "L1") - S(q, "L0"), "L1p_minus_L1ab": S(q, "L1") - S(ab, "L1"),
            "BA_L0_p": S(q, "L0"), "BA_L0_ab": S(ab, "L0"), "oracle_p": S(q, "oracle"),
            "oracle_ab": S(ab, "oracle"), "AUROC_p": S(q, "auroc"), "AUROC_ab": S(ab, "auroc")}


def reading(d: dict) -> str:
    """The registered reading of one backbone, prompt and axis."""
    if not d["G"]["ci"][0] > 0.02:
        return "the prompt does not improve the native BA"
    lo, hi = d["share_C"]["ci"]
    if lo > 0.5:
        return "mostly criterion"
    if hi < 0.5:
        return "mostly ranking"
    return "mixed"


def main(root: Path, B: int = 2000, seed: int = 42, out: Path = None):
    root = Path(root)
    out = Path(out or root / "reading_prompts")
    out.mkdir(parents=True, exist_ok=True)
    ref, _ = oc.load(root / "q2_base_ba")
    ref = ref[ref["condition"].isin(oc.DUR + oc.SNR)]
    ids = list(ref.index)
    meta = ref
    systems, tops = {}, {}
    for b in ("base", "lora"):
        sources = {f"{b}_ab": f"q2_{b}", f"{b}_ba": f"q2_{b}_ba", **{f"{b}_{p}": f"{p}_{b}" for p in PROMPTS}}
        for k, folder in sources.items():
            m, _ = oc.aligned(*oc.load(root / folder, ids=set(ids)), ids)
            if not (m["label"].to_numpy() == meta["label"].to_numpy()).all():
                raise SystemExit(f"labels of {folder} differ")
            systems[k] = m["s_primary"].to_numpy(float)
            if "top1_is_label" in m:
                tops[k] = m.groupby("condition")["top1_is_label"].mean().round(3).to_dict()
        systems[f"{b}_sim"] = 0.5 * (systems[f"{b}_ab"] + systems.pop(f"{b}_ba"))
    has_omni = (root / "omni" / "hand_csl").exists()
    if has_omni:
        for tr in ("ab", "hand_csl"):
            m, _ = oc.aligned(*oc.load(root / "omni" / tr, ids=set(ids)), ids)
            systems[f"q3_{tr}"] = m["s_primary"].to_numpy(float)

    def evaluate(w):
        return {k: oc.metrics(meta, s, w) for k, s in systems.items()}

    point = evaluate(None)
    recs = meta["rec_id"].astype(str).to_numpy()
    y = meta["label"].to_numpy().astype(bool)
    reps = [evaluate(w) for w in oc.boot_weights(recs, y, B, seed)]
    res, rows = {"B": B, "seed": seed, "top1_is_label": tops}, []
    for b in ("base", "lora"):
        for p in PROMPTS:
            for ax, cs in AXES.items():
                pt = derived(point, b, p, cs)
                rd = [derived(r, b, p, cs) for r in reps]
                d = {k: {"value": float(v), "ci": [float(np.nanpercentile([x[k] for x in rd], 2.5)),
                                                   float(np.nanpercentile([x[k] for x in rd], 97.5))]}
                     for k, v in pt.items()}
                d["reading"] = reading(d) if ax != "short" else None
                d["calibration_reaches_prompt"] = bool(d["L1ab_minus_L0p"]["ci"][0] > -0.02)
                res[f"{b}/{p}/{ax}"] = d
                rows += [{"backbone": b, "prompt": p, "axis": ax, "quantity": k, **v} for k, v in d.items()
                         if isinstance(v, dict)]
    res["primary"] = {b: {ax: res[f"{b}/{PRIMARY[b]}/{ax}"]["reading"] for ax in ("duration", "snr")}
                      for b in PRIMARY}
    if has_omni:  # does the hand prompt change ordering in the same way in both models
        def hand_gap(r, cs, a, h):
            return oc.summarise(r[h], "auroc", cs) - oc.summarise(r[a], "auroc", cs)
        res["cross_model"] = {}
        for ax, cs in AXES.items():
            for name, a, h in (("qwen3", "q3_ab", "q3_hand_csl"), ("qwen2_base", "base_ab", "base_hand_csl")):
                v = [hand_gap(r, cs, a, h) for r in reps]
                res["cross_model"][f"{name}/{ax}"] = {"value": hand_gap(point, cs, a, h),
                                                      "ci": [float(np.nanpercentile(v, 2.5)),
                                                             float(np.nanpercentile(v, 97.5))]}
    pd.DataFrame([{**{k: r[k] for k in ("backbone", "prompt", "axis", "quantity", "value")},
                   "lo": r["ci"][0], "hi": r["ci"][1]} for r in rows]).to_csv(out / "prompts.csv", index=False)
    (out / "reading.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    print(json.dumps(res["primary"], indent=1))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out")
    a = ap.parse_args()
    main(Path(a.root), B=a.B, seed=a.seed, out=a.out)
