"""Qwen3-Omni against Qwen2-Audio on the same pilot clips, paired by recording (post hoc, 2026-10-29).

Reads the per-item score CSVs of omni-pilot-v2 (work/omni-pilot-v2/omni/<prompt>/) and of
pilot-v1 (work/pilot-v1/<base|lora>/eval/), downloaded from Drive, and reports per condition
the difference in AUROC, best-threshold BA and BA with batch calibration per condition,
with 95% intervals from a bootstrap over recordings stratified by class (2,000 replicates,
seed 42). Descriptive, written after the pilot result. The registered reading is in
results/omni/verdict.json of the run.

    python tools/omni_paired.py <folder with omni_ab/, q2_base/, q2_lora/, ...>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from erclm import lib_v2 as L  # noqa: E402

CONDS = ["dur_20ms", "dur_40ms", "dur_1000ms"]


def load(d: Path) -> pd.DataFrame:
    m = pd.concat([pd.read_csv(p) for p in sorted(d.glob("shard_*.csv"))], ignore_index=True)
    m = m.drop_duplicates("variant_id", keep="last")
    return m[m["condition"].isin(CONDS)].set_index("variant_id")


def metrics(m: pd.DataFrame, s, w=None) -> dict:
    y = m["label"].to_numpy().astype(bool)
    thr = L.apply_rule(m.reset_index(), s, "bc_cond", w=w)
    return {"auroc": L.auc_w(y, s, w), "oracle": L.oracle_ba_w(y, s, w), "L1": L.ba_w(y, s > thr, w),
            "L0": L.ba_w(y, s > 0, w)}


def main(root: Path, B: int = 2000, seed: int = 42):
    omni = {p: load(root / f"omni_{p}") for p in ("ab", "ba_speech_second", "hand_csl")}
    q2 = {b: load(root / f"q2_{b}") for b in ("base", "lora")}
    ids = omni["ab"].index
    for b in q2:
        missing = set(ids) - set(q2[b].index)
        if missing:
            raise SystemExit(f"{len(missing)} omni items are not in pilot-v1 {b}")
    sim = 0.5 * (omni["ab"]["s_primary"] + omni["ba_speech_second"].loc[ids, "s_primary"])
    systems = {"omni_ab": omni["ab"]["s_primary"], "omni_s_sim": sim,
               "omni_hand": omni["hand_csl"].loc[ids, "s_primary"],
               "q2_base_ab": q2["base"].loc[ids, "s_primary"], "q2_lora_ab": q2["lora"].loc[ids, "s_primary"]}
    pairs = [("omni_ab", "q2_base_ab"), ("omni_ab", "q2_lora_ab"), ("omni_hand", "q2_base_ab"),
             ("omni_hand", "omni_ab")]
    rows = []
    meta = omni["ab"]
    for cond in CONDS:
        m = meta[meta["condition"] == cond]
        sel = m.index
        y = m["label"].to_numpy().astype(bool)
        recs = m["rec_id"].astype(str).to_numpy()
        by_class = [np.unique(recs[y == c]) for c in (True, False)]
        rng = np.random.default_rng(seed)
        point = {k: metrics(m, v.loc[sel].to_numpy(float)) for k, v in systems.items()}
        reps = []
        for _ in range(B):
            cnt = {}
            for u in by_class:
                for r in rng.choice(u, size=len(u), replace=True):
                    cnt[r] = cnt.get(r, 0) + 1
            w = np.array([cnt.get(r, 0) for r in recs], dtype=float)
            reps.append({k: metrics(m, v.loc[sel].to_numpy(float), w) for k, v in systems.items()})
        for k in systems:
            rows.append({"condition": cond, "system": k, **{f"{x}": round(point[k][x], 3) for x in point[k]}})
        for a, b in pairs:
            for x in ("auroc", "oracle", "L1"):
                d = np.array([r[a][x] - r[b][x] for r in reps])
                rows.append({"condition": cond, "system": f"{a} - {b}", "metric": x,
                             "diff": round(point[a][x] - point[b][x], 3),
                             "lo": round(float(np.nanpercentile(d, 2.5)), 3),
                             "hi": round(float(np.nanpercentile(d, 97.5)), 3)})
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(out[out["metric"].isna()][["condition", "system", "auroc", "oracle", "L1", "L0"]].to_string(index=False))
    print()
    print(out[out["metric"].notna()][["condition", "system", "metric", "diff", "lo", "hi"]].to_string(index=False))
    return out


if __name__ == "__main__":
    main(Path(sys.argv[1]))
