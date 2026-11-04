"""Exploratory analyses for closing the paper (docs/exploratory_log.md).

Prints Markdown tables for docs/exploratory_log.md. Inputs live outside git, in _work/closing
(step 3 of the closing plan: features and manifests copied from Drive), except the per-item
scores, which are read from the run records in runs/.

    PYTHONPATH=src python tools/closing_analyses.py v1
    PYTHONPATH=src python tools/closing_analyses.py v2
    PYTHONPATH=src python tools/closing_analyses.py tr [--B 2000]
    PYTHONPATH=src python tools/closing_analyses.py anc [--B 2000]
    PYTHONPATH=src python tools/closing_analyses.py cls [--B 2000]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from erclm import lib_v3 as L

WORK = Path("_work/closing")
RUNS = Path(__file__).resolve().parents[1] / "runs"  # the run records shipped with the repository
SCORES = {"base": "full-v1-base/results/base/scores.csv.gz",
          "lora": "full-v1-lora/results/lora/scores.csv.gz"}


def scores_from_runs(backbone: str) -> pd.DataFrame:
    return pd.read_csv(RUNS / SCORES[backbone], compression="gzip")


def test_meta() -> pd.DataFrame:
    return pd.read_csv(WORK / "manifests" / "test_bank_v2.metadata.csv.gz")


def md(df: pd.DataFrame, digits: int = 3) -> str:
    """Markdown table without the optional tabulate dependency."""
    df = df.round(digits)
    idx = [str(n) for n in df.index.names] if any(df.index.names) else [""]
    cols = [" ".join(map(str, c)) if isinstance(c, tuple) else str(c) for c in df.columns]
    lines = ["| " + " | ".join(idx + cols) + " |", "|" + " --- |" * (len(idx) + len(cols))]
    for key, row in df.iterrows():
        keys = list(key) if isinstance(key, tuple) else [key]
        lines.append("| " + " | ".join([str(k) for k in keys] + [str(v) for v in row.tolist()]) + " |")
    return "\n".join(lines)


def v1(floor_db: float) -> None:
    meta = test_meta()
    cov = L.speech_coverage(meta, [WORK / "voxconverse" / "dev", WORK / "voxconverse" / "test"])
    print("## V1. RTTM coverage of the speech crops\n")
    print(md(L.coverage_table(cov)), "\n")

    rel = L.relative_level_db(meta)
    print(f"## V1. Crop level relative to its 1000 ms clip (share below {floor_db:g} dB)\n")
    print(md(L.level_table(rel, floor_db).set_index(["ground_truth", "duration_ms"]), 2), "\n")

    flag = rel.merge(cov[["variant_id", "coverage"]], on="variant_id", how="left")
    flag["suspect"] = (flag["ground_truth"] == "SPEECH") & (
        (flag["rel_db"] < floor_db) | (flag["coverage"].fillna(1.0) < 0.5))
    suspect = set(flag.loc[flag["suspect"], "variant_id"])
    readouts = {"base": ["L0", "L1_bc_cond", "L1_center_cond", "L2"], "lora": ["L0", "L1_bc_cond", "L2"]}
    print("## V1. Sensitivity of BA on the duration axis without suspect speech crops\n")
    for bb, rs in readouts.items():
        sc = scores_from_runs(bb)
        sc = sc[sc["axis"] == "duration"].reset_index(drop=True)
        keep = ~sc["variant_id"].isin(suspect)
        full, filt = L.ba_by_condition(sc, rs), L.ba_by_condition(sc, rs, keep)
        order = sorted(full.index, key=lambda c: int(c.split("_")[1][:-2]))
        tab = pd.concat({"all": full.loc[order], "filtered": filt.loc[order]}, axis=1)
        tab.loc["mean"] = tab.mean()
        print(f"### {bb} (excluded speech crops: {int((~keep).sum())})\n")
        print(md(tab * 100, 1), "\n")
        short = sc[(sc["label"] == 1) & (sc["duration_ms"] <= 40)].copy()
        short["suspect"] = short["variant_id"].isin(suspect)
        hits = {r: short.groupby("suspect").apply(
            lambda g, r=r: (g[f"{r}__score"] > g[f"{r}__thr"]).mean()) for r in rs}
        hits = pd.DataFrame(hits)
        hits.insert(0, "n", short.groupby("suspect").size())
        print(f"Hit rate of speech crops of 20 and 40 ms, {bb}\n")
        print(md(hits, 3), "\n")


def v2() -> None:
    meta = test_meta()
    print("## V2. AUROC of fragment RMS alone, per condition\n")
    print(md(L.level_auroc_by_condition(meta)), "\n")


def train_meta(backbone: str) -> pd.DataFrame:
    """Scalar columns of the LoRA training bank as extracted in the full run (step 3 copy)."""
    parts = [pd.read_csv(p) for p in sorted((WORK / "features" / backbone / "train").glob("shard_*.csv"))]
    return pd.concat(parts, ignore_index=True).drop_duplicates("variant_id", keep="last")


def tr(B: int, seed: int) -> None:
    """Step 8. L1_tr, a threshold fitted on the training bank, frozen and applied to the test bank."""
    from erclm import lib_v2 as L2
    terms = {"tr_minus_bc_cond": {"L1_tr": 1, "L1_bc_cond": -1},
             "tr_minus_cf": {"L1_tr": 1, "L1_cf": -1},
             "tr_minus_oracle": {"L1_tr": 1, "oracle:L0": -1},
             "bc_cond_minus_bc_global": {"L1_bc_cond": 1, "L1_bc_global": -1}}
    for bb in ("base", "lora"):
        mtr = train_meta(bb)
        thr = L.fit_threshold(mtr["s_primary"], mtr["label"])
        sc = scores_from_runs(bb)  # all four axes, so the global BC batch is the paper's
        R = L.threshold_readouts(sc, thr)
        means = {f"mean {k}": {k: 1} for k in R} | {"mean L1*": {"oracle:L0": 1}}
        ci, con = L2.cluster_bootstrap(sc, R, B=B, seed=seed, metrics=("ba", "oracle_ba"),
                                       contrasts=L.axis_mean_contrasts(sc, means | terms))
        ci.to_csv(WORK / f"tr_ci_{bb}.csv", index=False)
        con[["axis", "what"]] = con["contrast"].str.split(":", n=1, expand=True)
        print(f"## L1_tr, {bb}. Threshold {thr:.3f} fitted on {len(mtr)} training variants, B = {B}\n")
        tab = con.set_index(["what", "axis"])[["estimate", "lo", "hi"]] * 100
        print(md(tab, 1), "\n")
        y = sc["label"] == 1
        cls = pd.DataFrame({"s speech": sc[y].groupby("condition")["s_primary"].mean(),
                            "s non-speech": sc[~y].groupby("condition")["s_primary"].mean(),
                            "thr centring": sc.groupby("condition")["L1_center_cond__thr"].first(),
                            "thr BC": sc.groupby("condition")["L1_bc_cond__thr"].first()})
        cls["thr L1_tr"] = thr
        print(f"Mean native score per class and thresholds, {bb}\n")
        print(md(cls, 2), "\n")


def anc(B: int, seed: int) -> None:
    """Step 9. L2_anc on both backbones, against L2 and the LoRA with L0, paired across backbones."""
    import numpy as np
    from erclm import lib_v2 as L2
    cols, heads = {}, {}
    for bb in ("base", "lora"):
        mtr, atr = L2.load_store(WORK / "features" / bb / "train")
        mte, ate = L2.load_store(WORK / "features" / bb / "eval")
        keep = (mte["role"] != "content_free").to_numpy()
        fit_args = (atr["h_final"], mtr["s_primary"].to_numpy(), mtr["label"].to_numpy(), mtr["rec_id"].to_numpy())
        heads[f"{bb} L2_anc"] = L.fit_anchored_head(*fit_args, seed=seed)
        heads[f"{bb} L2_res"] = L.fit_residual_head(*fit_args, seed=seed)
        h_te, s_te = ate["h_final"][keep], mte.loc[keep, "s_primary"].to_numpy()
        cols[bb] = pd.DataFrame({"variant_id": mte.loc[keep, "variant_id"].to_numpy(),
                                 f"{bb}_L2_anc": L.anchored_scores(heads[f"{bb} L2_anc"], h_te, s_te),
                                 f"{bb}_L2_res": L.anchored_scores(heads[f"{bb} L2_res"], h_te, s_te)})
        del atr, ate
    base, lora = scores_from_runs("base"), scores_from_runs("lora")
    m = (base.merge(cols["base"], on="variant_id")
         .merge(lora[["variant_id", "L0__score", "L2__score"]]
                .rename(columns={"L0__score": "lora_L0", "L2__score": "lora_L2"}), on="variant_id")
         .merge(cols["lora"], on="variant_id"))
    if len(m) != len(base):
        raise RuntimeError(f"aligned {len(m)} of {len(base)} items")
    zero = np.zeros(len(m))
    R = {"base_L2": (m["L2__score"].to_numpy(), zero, None),
         "base_L2_anc": (m["base_L2_anc"].to_numpy(), zero, None),
         "lora_L0": (m["lora_L0"].to_numpy(), zero, None),
         "lora_L2": (m["lora_L2"].to_numpy(), zero, None),
         "lora_L2_anc": (m["lora_L2_anc"].to_numpy(), zero, None),
         "base_L2_res": (m["base_L2_res"].to_numpy(), zero, None),
         "lora_L2_res": (m["lora_L2_res"].to_numpy(), zero, None)}
    means = {f"mean {k}": {k: 1} for k in R}
    terms = {"base_anc_minus_lora_L0": {"base_L2_anc": 1, "lora_L0": -1},
             "base_res_minus_lora_L0": {"base_L2_res": 1, "lora_L0": -1},
             "base_anc_minus_base_L2": {"base_L2_anc": 1, "base_L2": -1},
             "base_res_minus_base_L2": {"base_L2_res": 1, "base_L2": -1},
             "lora_anc_minus_lora_L0": {"lora_L2_anc": 1, "lora_L0": -1}}
    con_ba = L.axis_mean_contrasts(m, means | terms, metric="ba")
    con_auc = {k.replace(":", ":auroc ", 1): v for k, v in
               L.axis_mean_contrasts(m, means | terms, metric="auroc").items()}
    ci, con = L2.cluster_bootstrap(m, R, B=B, seed=seed, metrics=("ba", "auroc"), contrasts=con_ba | con_auc)
    ci.to_csv(WORK / "anc_ci.csv", index=False)
    for bb, h in heads.items():
        print(f"{bb}: alpha {h['alpha']:.3f}, C {h['C']:g}, cv AUROC {h['cv_auroc']:.3f}, "
              f"|delta| {np.linalg.norm(h['delta']):.3f}\n")
    con[["axis", "what"]] = con["contrast"].str.split(":", n=1, expand=True)
    print(f"## L2_anc, B = {B}. BA in percent, AUROC times 100\n")
    print(md(con.set_index(["what", "axis"])[["estimate", "lo", "hi"]] * 100, 1), "\n")
    per = ci[(ci["metric"] == "ba") & ci["axis"].isin(["duration", "snr"])]
    per = per.pivot_table(index=["axis", "condition"], columns="readout", values="estimate") * 100
    print("BA per condition\n")
    print(md(per[["base_L2", "base_L2_anc", "base_L2_res", "lora_L0", "lora_L2_anc"]], 1), "\n")


def cls(B: int, seed: int) -> None:
    """Step 10. False alarms per ESC-50 group on the duration and SNR axes, clustered by recording."""
    readouts = {"base": ["L1_bc_cond", "L2"], "lora": ["L0"]}
    rows, per_class = [], []
    for bb, rs in readouts.items():
        sc = scores_from_runs(bb)
        ns = sc[(sc["label"] == 0) & sc["axis"].isin(["duration", "snr"])].copy()
        ns["target"] = ns["base_id"].map(L.esc_class)
        ns["group"] = ns["target"].map(L.esc_group)
        for r in rs:
            fa = (ns[f"{r}__score"] > ns[f"{r}__thr"]).to_numpy()
            for g, idx in ns.groupby("group").indices.items():
                est, lo, hi = L.cluster_rate_ci(fa[idx], ns["rec_id"].to_numpy()[idx], B=B, seed=seed)
                rows.append({"readout": f"{bb} {r}", "group": g, "clips": ns["base_id"].iloc[idx].nunique(),
                             "items": len(idx), "fa": est, "lo": lo, "hi": hi})
            if (bb, r) == ("lora", "L0"):
                vocal = ns["group"] == "human vocal"
                t = pd.DataFrame({"fa": fa[vocal.to_numpy()], "target": ns.loc[vocal, "target"]})
                per_class = t.groupby("target")["fa"].agg(["mean", "size"])
                per_class.index = [L.ESC_HUMAN_VOCAL[i] for i in per_class.index]
    tab = pd.DataFrame(rows).set_index(["readout", "group"])
    tab[["fa", "lo", "hi"]] *= 100
    print(f"## False alarms per ESC-50 group, duration and SNR axes, B = {B}\n")
    print(md(tab, 1), "\n")
    print("LoRA with L0, false-alarm rate per human vocal class\n")
    print(md(per_class, 3), "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("step", choices=["v1", "v2", "tr", "anc", "cls"])
    ap.add_argument("--floor-db", type=float, default=-20.0)
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    if a.step == "v1":
        v1(a.floor_db)
    elif a.step == "v2":
        v2()
    elif a.step == "tr":
        tr(a.B, a.seed)
    elif a.step == "anc":
        anc(a.B, a.seed)
    else:
        cls(a.B, a.seed)


if __name__ == "__main__":
    main()
