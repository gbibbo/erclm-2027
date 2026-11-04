"""Where the speech evidence is linearly accessible, from the encoder to the final state.

docs/exploratory_log.md, "Step 12. Accessibility per layer and heads trained inside the test
bank", and the L2_cv question. For each backbone and stored feature, a
logistic head is fitted two ways and its test scores are cached in _work/closing/acc.
    matched  trained on the LoRA training bank (64 excerpts), as the confirmatory L2
    cv       cross-fitted inside the test bank, folds grouped by source recording, with C
             chosen by inner grouped cross-validation in every fold (lib_v2.crossfit_scores).
             On h_final this is lib_v2's L2_cv.
All of it is exploratory. Features are those the full run stored. Cached scores are keyed by
backbone, feature, mode and a hash of the parameters, so a change of grid or seed refits.
Until 2026-10-21 the cv heads on features other than h_final fixed C to the matched choice,
which left the LoRA depth profile of the main bank uninterpretable (audit of that day).

    PYTHONPATH=src python tools/layer_accessibility.py fit --backbone base --feature h_final --mode cv
    PYTHONPATH=src python tools/layer_accessibility.py report [--B 2000]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from closing_analyses import WORK, md, scores_from_runs  # noqa: E402
from erclm import lib_v2 as L2  # noqa: E402
from erclm import lib_v3 as L  # noqa: E402

FEATURES = ("enc_region_mean", "enc_global_mean", "h_layer8", "h_layer16", "h_layer24", "h_final")
OUT = WORK / "acc"


def _load(bb: str, which: str, feature: str):
    """Like lib_v2.load_store, but reads one feature only, so several fits fit in memory."""
    metas, xs = [], []
    for p in sorted((WORK / "features" / bb / which).glob("shard_*.csv")):
        m = pd.read_csv(p)
        with np.load(p.with_suffix(".npz")) as z:
            x = z[feature]
        if len(x) != len(m):
            raise RuntimeError(f"{p.name}: {feature} has {len(x)} rows, meta has {len(m)}")
        metas.append(m)
        xs.append(x)
    meta, X = pd.concat(metas, ignore_index=True), np.concatenate(xs)
    keep = ~meta["variant_id"].duplicated(keep="last").to_numpy() & (meta["role"] != "content_free").to_numpy()
    return meta[keep].reset_index(drop=True), X[keep].astype(np.float32)


CS = (1e-4, 1e-3, 1e-2, 1e-1)


def _params(mode: str, seed: int) -> dict:
    return {"mode": mode, "Cs": list(CS), "seed": seed, "n_splits": 5, "version": 2}


def fit(bb: str, feature: str, mode: str, seed: int = 42) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    params = _params(mode, seed)
    out = OUT / f"{bb}__{feature}__{mode}.npz"
    meta_json = out.with_suffix(".json")
    if out.exists() and meta_json.exists() and json.loads(meta_json.read_text()).get("params") == params:
        print(f"{out.name} exists with the same parameters, skipping")
        return out
    t0 = time.time()
    mtr, xtr = _load(bb, "train", feature)
    model, info = L2.fit_head(xtr, mtr["label"].to_numpy(int), mtr["rec_id"].to_numpy(), Cs=CS, seed=seed)
    mte, xte = _load(bb, "eval", feature)
    if mode == "matched":
        sc = model.decision_function(xte)
    else:
        sc, infos = L2.crossfit_scores(xte, mte["label"].to_numpy(int), mte["rec_id"].to_numpy(),
                                       n_splits=params["n_splits"], seed=seed, Cs=CS)
        info = {"C_matched": info["C"], "folds": infos}
    tmp = out.with_name(out.stem + ".partial.npz")
    np.savez(tmp, variant_id=mte["variant_id"].to_numpy(str), score=sc)
    tmp.replace(out)
    meta_json.write_text(json.dumps({"params": params, "info": info, "seconds": time.time() - t0}, default=str))
    print(f"{out.name} in {(time.time() - t0) / 60:.1f} min")
    return out


def axis_means(meta: pd.DataFrame, score: np.ndarray) -> dict:
    """Mean BA (score > 0) and mean AUROC over the conditions of the duration and SNR axes."""
    out = {}
    for axis in ("duration", "snr"):
        ba, auc = [], []
        for _, g in meta[meta["axis"] == axis].groupby("condition"):
            y, s = g["label"].to_numpy() == 1, score[g.index.to_numpy()]
            ba.append(L.balanced_accuracy(y, s > 0))
            auc.append(L.auroc(y, s))
        out[f"{axis} BA"], out[f"{axis} AUROC"] = float(np.mean(ba)), float(np.mean(auc))
    return out


def curve(bb: str, feature: str, sizes=(16, 32, 64, 128, 256), reps: int = 3, C: float = 0.01,
          seed: int = 42) -> Path:
    """In-distribution learning curve. Heads cross-fitted inside the test bank as for cv, but each
    training fold is cut to n base excerpts, half per class and at most one per recording, as in
    the LoRA training bank. C is fixed (0.01 by default, the value the matched base head chose;
    the matched LoRA heads chose other values) and written into the CSV, with every repetition,
    so the spread across draws can be reported. 'all' uses the whole training fold."""
    from sklearn.model_selection import StratifiedGroupKFold
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"curve__{bb}__{feature}.csv"
    meta, X = _load(bb, "eval", feature)
    y, rec, bid = meta["label"].to_numpy(int), meta["rec_id"].to_numpy(), meta["base_id"].to_numpy()
    folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed).split(X, y, rec))
    rows = []
    for n in list(sizes) + ["all"]:
        for r in range(1 if n == "all" else reps):
            score = np.full(len(meta), np.nan)
            for k, (tr, te) in enumerate(folds):
                sel = tr
                if n != "all":
                    rng = np.random.default_rng([seed, int(n), r, k])
                    bases = meta.iloc[tr].drop_duplicates("base_id")[["base_id", "rec_id", "label"]]
                    chosen = []
                    for lab in (1, 0):
                        b = bases[bases["label"] == lab]
                        b = b.iloc[rng.permutation(len(b))].drop_duplicates("rec_id")
                        chosen += b["base_id"].iloc[: int(n) // 2].tolist()
                    sel = tr[np.isin(bid[tr], chosen)]
                score[te] = L2._head(C).fit(X[sel], y[sel]).decision_function(X[te])
            rows.append({"backbone": bb, "feature": feature, "n_excerpts": n, "rep": r, "C": C,
                         **axis_means(meta, score)})
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def report(B: int, seed: int) -> None:
    base = scores_from_runs("base")
    lora = scores_from_runs("lora")
    m = base[["variant_id", "base_id", "rec_id", "label", "axis", "condition", "duration_ms"]].merge(
        lora[["variant_id", "L0__score"]].rename(columns={"L0__score": "lora L0"}), on="variant_id")
    R = {"lora L0": m["lora L0"].to_numpy()}
    for p in sorted(q for q in OUT.glob("*__*__*.npz") if ".partial" not in q.name and ".tmp" not in q.name):
        bb, feat, mode = p.stem.split("__")
        z = np.load(p, allow_pickle=False)
        sc = pd.Series(z["score"], index=z["variant_id"])
        R[f"{bb} {feat} {mode}"] = sc.reindex(m["variant_id"]).to_numpy()
    missing = [k for k, v in R.items() if np.isnan(v).any()]
    if missing:
        raise RuntimeError(f"scores missing for {missing}")
    zero = np.zeros(len(m))
    readouts = {k: (v, zero, None) for k, v in R.items()}
    means = {f"mean {k}": {k: 1} for k in R}
    terms = {f"{k} minus lora L0": {k: 1, "lora L0": -1}
             for k in R if k.startswith("base") and k.endswith(("h_final cv", "h_final matched"))}
    con_ba = L.axis_mean_contrasts(m, means | terms, metric="ba")
    con_auc = {k.replace(":", ":auroc ", 1): v for k, v in L.axis_mean_contrasts(m, means, metric="auroc").items()}
    ci, con = L2.cluster_bootstrap(m, readouts, B=B, seed=seed, metrics=("ba", "auroc"),
                                   contrasts=con_ba | con_auc)
    con[["axis", "what"]] = con["contrast"].str.split(":", n=1, expand=True)
    print(f"## Accessibility of the speech evidence, B = {B}\n")
    tab = con[con["axis"].isin(["duration", "snr"])].set_index(["what", "axis"])[["estimate", "lo", "hi"]] * 100
    print(md(tab, 1), "\n")
    per = ci[(ci["metric"] == "auroc") & (ci["axis"] == "duration")]
    per = per.pivot_table(index="condition", columns="readout", values="estimate")
    per = per.loc[sorted(per.index, key=lambda c: int(c.split("_")[1][:-2]))]
    print("AUROC per duration\n")
    print(md(per.T, 3), "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--backbone", choices=["base", "lora"], required=True)
    f.add_argument("--feature", choices=FEATURES, required=True)
    f.add_argument("--mode", choices=["matched", "cv"], required=True)
    r = sub.add_parser("report")
    r.add_argument("--B", type=int, default=2000)
    r.add_argument("--seed", type=int, default=42)
    c = sub.add_parser("curve")
    c.add_argument("--backbone", choices=["base", "lora"], required=True)
    c.add_argument("--feature", choices=FEATURES, required=True)
    a = ap.parse_args()
    if a.cmd == "fit":
        fit(a.backbone, a.feature, a.mode)
    elif a.cmd == "curve":
        curve(a.backbone, a.feature)
    else:
        report(a.B, a.seed)


if __name__ == "__main__":
    main()
