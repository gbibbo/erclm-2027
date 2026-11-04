"""The registered reading of Qwen3-Omni on the 300 control clips (R1 to R4 and the descriptives).

The rule is in docs/exploratory_log.md, "Qwen3-Omni on the 300 control clips (30 October)", written
on 30 October 2026 before extraction. This script was written the same day, before the run.

    python tools/omni_control.py <root> [--B 2000] [--out <folder>]

<root> holds the stores downloaded from Drive, CSV and npz shards as the runner writes them:

    omni/<track>/       work/omni-control-v1/omni/<track>/  (ab, ba_speech_second, ab_shift500, hand_csl)
    q2_base/, q2_lora/  work/full-v1/<backbone>/eval/       (Qwen2-Audio with ab, all 21,340 items)
    q2_base_ba/, q2_lora_ba/  work/prompt-ba_speech_second/<backbone>/eval/

Every interval is a 95% percentile interval from a bootstrap over recordings stratified by
class, 2,000 replicates, seed 42, with one draw shared by every system so the differences are
paired. Batch calibration per condition (L1) is re-estimated in every replicate. The probes are
cross-fitted once by recording (as L2_cv) and their scores stay fixed in the bootstrap.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from erclm import lib_v2 as L  # noqa: E402
from erclm import omni  # noqa: E402

DUR = ["dur_20ms", "dur_40ms", "dur_60ms", "dur_80ms", "dur_100ms", "dur_200ms", "dur_500ms", "dur_1000ms"]
SNR = ["snr_-10dB", "snr_-5dB", "snr_0dB", "snr_5dB", "snr_10dB", "snr_20dB"]
SHORT = ["dur_20ms", "dur_40ms"]
SETS = {"short": SHORT, "duration": DUR, "snr": SNR}
Q3_PROBES = ["enc_region_mean", "enc_global_mean", "proj_region_mean", "h_layer12", "h_layer24", "h_layer36",
             "h_final"]
Q2_PROBES = ["enc_region_mean", "enc_global_mean", "proj_region_mean", "h_layer8", "h_layer16", "h_layer24",
             "h_final"]


# ============================================================================ loading

def load(d: Path, ids=None, features=()) -> tuple:
    """(meta indexed by variant_id, {feature: array aligned with meta}) from a store folder,
    keeping only `ids` when given and only the requested features."""
    metas, arrays = [], {f: [] for f in features}
    for p in sorted(Path(d).glob("shard_*.csv")):
        m = pd.read_csv(p)
        keep = np.ones(len(m), bool) if ids is None else m["variant_id"].astype(str).isin(ids).to_numpy()
        if features:
            npz = np.load(p.with_suffix(".npz"))
            for f in features:
                if len(npz[f]) != len(m):
                    raise RuntimeError(f"{p}: {f} has {len(npz[f])} rows, meta has {len(m)}")
                arrays[f].append(npz[f][keep].astype(np.float32))
        metas.append(m[keep])
    if not metas:
        raise FileNotFoundError(f"no shards in {d}")
    meta = pd.concat(metas, ignore_index=True)
    arrays = {f: np.concatenate(v) for f, v in arrays.items()}
    last = ~meta["variant_id"].duplicated(keep="last").to_numpy()
    meta, arrays = meta[last].reset_index(drop=True), {f: v[last] for f, v in arrays.items()}
    meta["variant_id"] = meta["variant_id"].astype(str)
    return meta.set_index("variant_id"), arrays


def aligned(meta: pd.DataFrame, arrays: dict, ids) -> tuple:
    pos = pd.Series(np.arange(len(meta)), index=meta.index)
    missing = set(ids) - set(meta.index)
    if missing:
        raise SystemExit(f"{len(missing)} items are missing, e.g. {sorted(missing)[:3]}")
    i = pos.loc[list(ids)].to_numpy()
    return meta.iloc[i], {f: v[i] for f, v in arrays.items()}


# ============================================================================ statistics

def boot_weights(recs: np.ndarray, y: np.ndarray, B: int, seed: int):
    """Replicate weights per item, a draw of recordings with replacement within each class."""
    rng = np.random.default_rng(seed)
    by_class = [np.unique(recs[y == c]) for c in (True, False)]
    codes = {r: i for i, r in enumerate(np.unique(recs))}
    idx = np.array([codes[r] for r in recs])
    for _ in range(B):
        cnt = np.zeros(len(codes))
        for u in by_class:
            np.add.at(cnt, [codes[r] for r in rng.choice(u, size=len(u), replace=True)], 1)
        yield cnt[idx]


def metrics(meta: pd.DataFrame, s: np.ndarray, w=None, which=("auroc", "L0", "L1", "oracle")) -> dict:
    """Per-condition metrics of one score over the rows of meta (one or more conditions)."""
    out = {}
    cond = meta["condition"].to_numpy()
    y_all = meta["label"].to_numpy().astype(bool)
    thr = L.apply_rule(meta.reset_index(), s, "bc_cond", w=w) if "L1" in which else None
    for c in pd.unique(cond):
        k = cond == c
        y, sc, wk = y_all[k], s[k], None if w is None else w[k]
        r = {}
        if "auroc" in which:
            r["auroc"] = L.auc_w(y, sc, wk)
        if "L0" in which:
            r["L0"] = L.ba_w(y, sc > 0, wk)
        if "L1" in which:
            r["L1"] = L.ba_w(y, sc > thr[k], wk)
        if "oracle" in which:
            r["oracle"] = L.oracle_ba_w(y, sc, wk)
        out[c] = r
    return out


def summarise(per_cond: dict, metric: str, conds) -> float:
    vals = [per_cond[c][metric] for c in conds if c in per_cond]
    return float(np.mean(vals)) if len(vals) == len(conds) else float("nan")


def paired(meta: pd.DataFrame, systems: dict, contrasts: list, B: int = 2000, seed: int = 42,
           which=("auroc", "L0", "L1", "oracle"), cross=()) -> pd.DataFrame:
    """Point and 95% interval of each system and each contrast (a - b), per condition and per set.

    systems maps a name to a score aligned with meta. Every system is scored on the same draw.
    cross holds contrasts between readouts, (a, metric of a, b, metric of b), such as the recipe
    with L1 against a native rule (L0).
    """
    def evaluate(w):
        return {k: metrics(meta, s, w, which) for k, s in systems.items()}

    point = evaluate(None)
    recs = meta["rec_id"].astype(str).to_numpy()
    y = meta["label"].to_numpy().astype(bool)
    reps = [evaluate(w) for w in boot_weights(recs, y, B, seed)]
    conds = list(pd.unique(meta["condition"]))
    groups = [(c, [c]) for c in conds] + [(k, v) for k, v in SETS.items() if set(v) <= set(conds)]
    rows = []
    for g, cs in groups:
        for mtr in which:
            for k in systems:
                d = np.array([summarise(r[k], mtr, cs) for r in reps])
                rows.append({"group": g, "metric": mtr, "what": k, "value": summarise(point[k], mtr, cs),
                             "lo": np.nanpercentile(d, 2.5), "hi": np.nanpercentile(d, 97.5)})
            for a, b in contrasts:
                d = np.array([summarise(r[a], mtr, cs) - summarise(r[b], mtr, cs) for r in reps])
                rows.append({"group": g, "metric": mtr, "what": f"{a} - {b}",
                             "value": summarise(point[a], mtr, cs) - summarise(point[b], mtr, cs),
                             "lo": np.nanpercentile(d, 2.5), "hi": np.nanpercentile(d, 97.5)})
        for a, ma, b, mb in cross:
            d = np.array([summarise(r[a], ma, cs) - summarise(r[b], mb, cs) for r in reps])
            rows.append({"group": g, "metric": f"{ma}-{mb}", "what": f"{a} - {b}",
                         "value": summarise(point[a], ma, cs) - summarise(point[b], mb, cs),
                         "lo": np.nanpercentile(d, 2.5), "hi": np.nanpercentile(d, 97.5)})
    return pd.DataFrame(rows)


def get(t: pd.DataFrame, group: str, metric: str, what: str) -> dict:
    r = t[(t["group"] == group) & (t["metric"] == metric) & (t["what"] == what)]
    if len(r) != 1:
        raise KeyError((group, metric, what))
    r = r.iloc[0]
    return {"value": float(r["value"]), "ci": [float(r["lo"]), float(r["hi"])]}


def crossfit(meta: pd.DataFrame, X: np.ndarray, seed: int = 0) -> np.ndarray:
    """Out-of-fold probe scores, folds by recording, C by inner validation (lib_v2, as L2_cv).
    The folds depend only on the labels, the recordings and their order, so two models scored
    on the same aligned items get the same folds."""
    s, _ = L.crossfit_scores(X, meta["label"].to_numpy(int), meta["rec_id"].astype(str).to_numpy(),
                             n_splits=5, seed=seed)
    return s


# ============================================================================ the rule

def r3_reading(delta: dict, gap: float) -> str:
    lo, hi = delta["ci"]
    if not gap > 0:
        return "no gap, only delta is reported"
    half = gap / 2.0
    if lo > 0 and delta["value"] >= half:
        return "the seam explains most of it"
    if lo > 0 and hi < half:
        return "the seam explains a minor part"
    if lo <= 0 and hi < half:
        return "the seam does not explain it"
    return "no conclusion"


def r4_reading(a: dict, b: dict) -> str:
    if a["ci"][1] < 0:
        return "representation, the Qwen3-Omni encoder separates worse"
    if a["ci"][0] > -0.02 and b["ci"][0] > 0.05:
        return "readout, the encoder has the evidence and s does not use it"
    return "no conclusion"


def main(root: Path, B: int = 2000, seed: int = 42, out: Path = None, expect: int = 4200):
    root = Path(root)
    out = Path(out or root / "reading")
    out.mkdir(parents=True, exist_ok=True)
    res = {"B": B, "seed": seed}

    # Qwen3-Omni, the four tracks
    q3, q3a = {}, {}
    for tr in ("ab", "ba_speech_second", "ab_shift500", "hand_csl"):
        feats = Q3_PROBES if tr in ("ab", "ab_shift500") else ()
        q3[tr], q3a[tr] = load(root / "omni" / tr, features=feats)
    ids = list(q3["ab"].index)
    if len(ids) != expect or set(q3["ab"]["condition"]) != set(DUR + SNR):
        raise SystemExit(f"ab has {len(ids)} items over {q3['ab']['condition'].nunique()} conditions")
    meta = q3["ab"]
    for tr in ("ba_speech_second", "hand_csl"):
        q3[tr], _ = aligned(q3[tr], {}, ids)

    # R1, the pilot rule on the runner's table code
    table = omni.condition_table({k: q3[k].reset_index() for k in ("ab", "ba_speech_second")}, B=B, seed=seed)
    rule = {"validity_prompt": "ab", "validity_condition": "dur_1000ms", "validity_min_ba": 0.85,
            "full_items": 27160, "max_hours": 25, "read_prompts": ["ab", "s_sim"],
            "read_conditions": SHORT, "representation_auroc_hi": 0.65, "criterion_auroc_lo": 0.60,
            "criterion_gap": 0.10}
    res["R1"] = omni.verdict(table, None, rule)["reading"]
    table.to_csv(out / "R1_condition_table.csv", index=False)

    # Qwen2-Audio on the same items
    q2 = {}
    q2["base"], q2a = load(root / "q2_base", ids=set(ids), features=("enc_region_mean",) + tuple(Q2_PROBES[1:]))
    q2["base"], q2a = aligned(q2["base"], q2a, ids)
    q2["lora"], _ = aligned(*load(root / "q2_lora", ids=set(ids)), ids)
    for b in ("base", "lora"):
        q2[f"{b}_ba"], _ = aligned(*load(root / f"q2_{b}_ba", ids=set(ids)), ids)
    for k, m in q2.items():
        if not (m["label"].to_numpy() == meta["label"].to_numpy()).all():
            raise SystemExit(f"labels of q2 {k} differ from Qwen3-Omni")

    s = lambda m: m["s_primary"].to_numpy(float)  # noqa: E731
    sys_all = {"q3_ab": s(meta), "q3_ba": s(q3["ba_speech_second"]),
               "q3_sim": 0.5 * (s(meta) + s(q3["ba_speech_second"])), "q3_hand": s(q3["hand_csl"]),
               "q2_base": s(q2["base"]), "q2_lora": s(q2["lora"]),
               "q2_sim": 0.5 * (s(q2["base"]) + s(q2["base_ba"]))}

    # R4, probes with the same folds for both models
    probes = {}
    for f in Q3_PROBES:
        probes[f"q3_P_{f}"] = crossfit(meta, q3a["ab"][f])
    for f in Q2_PROBES:
        probes[f"q2_P_{f}"] = crossfit(meta, q2a[f])
    sys_all.update(probes)
    contrasts = [("q3_ab", "q2_base"), ("q3_ab", "q2_lora"), ("q3_sim", "q2_base"),
                 ("q3_P_enc_region_mean", "q2_P_enc_region_mean"), ("q3_P_enc_region_mean", "q3_ab"),
                 ("q2_P_enc_region_mean", "q2_base"), ("q3_hand", "q3_ab"), ("q3_sim", "q3_ab"),
                 ("q3_sim", "q2_lora"), ("q3_sim", "q2_sim")]
    cross = [("q3_sim", "L1", "q3_ab", "L0"), ("q3_sim", "L1", "q2_lora", "L0"), ("q2_sim", "L1", "q2_lora", "L0")]
    t_all = paired(meta, sys_all, contrasts, B=B, seed=seed, cross=cross)
    t_all.to_csv(out / "paired_all.csv", index=False)

    r2 = get(t_all, "short", "auroc", "q3_ab - q2_base")
    res["R2"] = {"q3_ab - q2_base, 20 and 40 ms": r2,
                 "worse on short crops": bool(r2["ci"][1] < 0),
                 "q3_ab - q2_lora, 20 and 40 ms": get(t_all, "short", "auroc", "q3_ab - q2_lora"),
                 "q3_ab - q2_base, duration": get(t_all, "duration", "auroc", "q3_ab - q2_base"),
                 "q3_ab - q2_base, SNR": get(t_all, "snr", "auroc", "q3_ab - q2_base")}
    a = get(t_all, "short", "auroc", "q3_P_enc_region_mean - q2_P_enc_region_mean")
    b = get(t_all, "short", "auroc", "q3_P_enc_region_mean - q3_ab")
    res["R4"] = {"a": a, "b": b, "reading": r4_reading(a, b),
                 "probes, 20 and 40 ms": {k: get(t_all, "short", "auroc", k) for k in probes}}

    # R3, the seam, on the duration items with the same folds
    dmask = meta["condition"].isin(DUR).to_numpy()
    md = meta[dmask]
    sh, sha = aligned(q3["ab_shift500"], q3a["ab_shift500"], list(md.index))
    sys_d = {"centred": s(md), "shifted": s(sh),
             "centred_P_enc_region_mean": crossfit(md, q3a["ab"]["enc_region_mean"][dmask]),
             "shifted_P_enc_region_mean": crossfit(md, sha["enc_region_mean"])}
    t_d = paired(md, sys_d, [("shifted", "centred"), ("shifted_P_enc_region_mean", "centred_P_enc_region_mean")],
                 B=B, seed=seed)
    t_d.to_csv(out / "R3_seam.csv", index=False)
    delta = get(t_d, "short", "auroc", "shifted - centred")
    gap = -r2["value"]
    res["R3"] = {"delta": delta, "gap": gap, "reading": r3_reading(delta, gap),
                 "delta BA L1": get(t_d, "short", "L1", "shifted - centred"),
                 "delta probe": get(t_d, "short", "auroc", "shifted_P_enc_region_mean - centred_P_enc_region_mean"),
                 "delta 1000 ms": get(t_d, "dur_1000ms", "auroc", "shifted - centred")}

    # descriptives
    res["descriptive"] = {
        "recipe": {ax: {"q3_sim L1 - q3_ab L0": get(t_all, ax, "L1-L0", "q3_sim - q3_ab"),
                        "q3_sim L1 - q2_lora L0": get(t_all, ax, "L1-L0", "q3_sim - q2_lora"),
                        "q2_sim L1 - q2_lora L0": get(t_all, ax, "L1-L0", "q2_sim - q2_lora"),
                        "q3_sim L1 - q2_sim L1": get(t_all, ax, "L1", "q3_sim - q2_sim")}
                   for ax in ("duration", "snr")},
        "prompt": {g: get(t_all, g, "auroc", "q3_hand - q3_ab") for g in ("short", "duration", "snr")},
        "top1_is_label": {tr: q3[tr].groupby("condition")["top1_is_label"].mean().round(3).to_dict()
                          for tr in ("ab", "ba_speech_second", "hand_csl")}}
    (out / "reading.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("R1", "R2", "R3", "R4")}, indent=1, default=float))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out")
    a = ap.parse_args()
    main(Path(a.root), B=a.B, seed=a.seed, out=a.out)
