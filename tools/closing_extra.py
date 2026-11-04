"""Numbers cited by the writing plan that were not yet in docs/exploratory_log.md.

    PYTHONPATH=src python tools/closing_extra.py all [--B 2000]

Main bank (cached scores in _work/closing/acc and per-item scores of the run records):
    A  L2_cv against LoRA L0 in BA, AUROC and best-threshold BA, and base L2_cv against LoRA L2_cv
    B  200-to-20 ms drop of the regional encoder probe, matched and cross-fitted
    E  the 20 and 40 ms AUROC lead of L2_cv over LoRA L0 without crops below -30 dBFS
    F  largest share of speech answers of base L0 in any condition, and the worst per-condition
       gap between L1_tr and L1
    G  H4 in AUROC, as a difference in differences of L2 over L0 between backbones
    H  level used two-sidedly, a cross-fitted logistic head on log RMS and its square
    I  the reference of the provenance rule refitted with the supervision of the provenance heads
       (python tools/closing_extra.py matched [--draws 10])
Figure data:
    per-duration paired AUROC differences from LoRA L0, with pointwise 95% intervals, for L2_cv in
    the main bank, L2_cv inside the provenance bank and the main-bank head applied to it
Everything is exploratory and uses the paired cluster bootstrap of lib_v2 with 2,000 replicates.
"""
from __future__ import annotations

import argparse
import os
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from closing_analyses import WORK, md, scores_from_runs, train_meta  # noqa: E402
from erclm import lib_v2 as L2  # noqa: E402
from erclm import lib_v3 as L  # noqa: E402

# provenance run folder, prov-v3 unless ERCLM_PROV_RUN names another (e.g. prov-v3-clean, the
# recordings whose non-speech window a blind listener did not hear as speech)
PROV_DIR = WORK / "runs" / os.environ.get("ERCLM_PROV_RUN", "prov-v3")

DUR = [f"dur_{d}ms" for d in (20, 40, 60, 80, 100, 200, 500, 1000)]


def _cached(name: str, ids: pd.Series) -> np.ndarray:
    z = np.load(WORK / "acc" / f"{name}.npz")
    return pd.Series(z["score"], index=z["variant_id"]).reindex(ids).to_numpy()


def _avg(rd, cs, met, coef=1.0):
    return [(coef / len(cs), rd, c, met) for c in cs]


def main_bank(B: int, seed: int) -> pd.DataFrame:
    base, lora = scores_from_runs("base"), scores_from_runs("lora")
    m = base[base["axis"].isin(["duration", "snr"])].reset_index(drop=True)
    ids = m["variant_id"]
    lo = lora.set_index("variant_id")
    zero = np.zeros(len(m))
    R = {"base L2cv": _cached("base__h_final__cv", ids),
         "lora L2cv": _cached("lora__h_final__cv", ids),
         "lora L0": lo["L0__score"].reindex(ids).to_numpy(),
         "lora L2": lo["L2__score"].reindex(ids).to_numpy(),
         "base L0": m["L0__score"].to_numpy(),
         "base L2": m["L2__score"].to_numpy(),
         "base Preg matched": _cached("base__enc_region_mean__matched", ids),
         "base Preg cv": _cached("base__enc_region_mean__cv", ids)}
    readouts = {k: (v, zero, None) for k, v in R.items()}
    snr = [c for c in m.loc[m["axis"] == "snr", "condition"].unique()]
    con = {}
    for ax, cs in (("duration", DUR), ("snr", snr)):
        for met in ("ba", "auroc", "oracle_ba"):
            con[f"A {ax} {met}: base L2cv - lora L0"] = _avg("base L2cv", cs, met) + _avg("lora L0", cs, met, -1)
            con[f"A {ax} {met}: base L2cv - lora L2cv"] = _avg("base L2cv", cs, met) + _avg("lora L2cv", cs, met, -1)
        # G: (L2 - L0) on base minus (L2 - L0) on LoRA, in AUROC (L1 shares the AUROC of L0)
        con[f"G {ax} auroc: H4 difference in differences"] = (
            _avg("base L2", cs, "auroc") + _avg("base L0", cs, "auroc", -1)
            + _avg("lora L2", cs, "auroc", -1) + _avg("lora L0", cs, "auroc"))
    for nm in ("base Preg matched", "base Preg cv"):
        con[f"B drop 200-20: {nm}"] = [(1.0, nm, "dur_200ms", "auroc"), (-1.0, nm, "dur_20ms", "auroc")]
    for c in DUR:
        con[f"FIG main {c}: base L2cv - lora L0"] = [(1.0, "base L2cv", c, "auroc"), (-1.0, "lora L0", c, "auroc")]
    _, c = L2.cluster_bootstrap(m, readouts, B=B, seed=seed, metrics=("ba", "auroc", "oracle_ba"), contrasts=con)
    c[["estimate", "lo", "hi", "se", "mde80"]] *= 100
    return c


def quiet_free_reference(seed: int) -> pd.DataFrame:
    """E. The 20 and 40 ms AUROC lead of L2_cv over LoRA L0, all crops and crops at or above -30 dBFS."""
    meta = pd.read_csv(WORK / "manifests" / "test_bank_v2.metadata.csv.gz")
    meta["variant_id"] = meta["audio_path"].str[:-4]
    meta["dbfs"] = 20 * np.log10(meta["segment_rms"])
    base, lora = scores_from_runs("base"), scores_from_runs("lora").set_index("variant_id")
    m = base[base["condition"].isin(DUR)].merge(meta[["variant_id", "dbfs"]], on="variant_id")
    m["cv"] = _cached("base__h_final__cv", m["variant_id"])
    m["lora"] = lora["L0__score"].reindex(m["variant_id"]).to_numpy()
    rows = []
    for c in DUR:
        g = m[m["condition"] == c]
        for lab, h in (("all crops", g), ("at or above -30 dBFS", g[g["dbfs"] >= -30])):
            y = h["label"].to_numpy()
            rows.append({"condition": c, "crops": lab, "n": len(h),
                         "lead": 100 * (L.auroc(y, h["cv"]) - L.auroc(y, h["lora"]))})
    t = pd.DataFrame(rows).pivot_table(index="condition", columns="crops", values="lead")
    return t.loc[DUR]


def criterion_extremes() -> dict:
    """F. Largest speech-answer share of base L0 in a condition, and the worst per-condition L1_tr - L1."""
    base = scores_from_runs("base")
    sp = base[base["label"] == 1]
    share = (sp["L0__score"] > sp["L0__thr"]).groupby(sp["condition"]).mean()
    thr = L.fit_threshold(train_meta("base")["s_primary"], train_meta("base")["label"])
    rows = {}
    for cond, g in base.groupby("condition"):
        y = g["label"].to_numpy() == 1
        rows[cond] = 100 * (L.balanced_accuracy(y, g["s_primary"].to_numpy() > thr)
                            - L.balanced_accuracy(y, g["L1_bc_cond__score"].to_numpy() > g["L1_bc_cond__thr"].to_numpy()))
    gap = pd.Series(rows)
    return {"max speech share of base L0 (%)": 100 * share.max(), "condition": share.idxmax(),
            "worst L1_tr - L1 (points)": gap.min(), "worst condition": gap.idxmin(),
            "best L1_tr - L1 (points)": gap.max()}


def level_two_sided(seed: int) -> pd.Series:
    """H. AUROC of a cross-fitted logistic head on log RMS and its square, per duration condition."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold
    meta = pd.read_csv(WORK / "manifests" / "test_bank_v2.metadata.csv.gz")
    meta = meta[meta["condition"].str.startswith("dur")].reset_index(drop=True)
    meta["rec"] = np.where(meta["dataset"] == "voxconverse", meta["recording"].astype(str),
                           meta["source_file"].astype(str).str.rsplit("-", n=2).str[0])
    out = {}
    for cond, g in meta.groupby("condition"):
        lr = np.log10(g["segment_rms"].to_numpy())
        X = np.column_stack([lr, lr ** 2])
        y = (g["ground_truth"] == "SPEECH").to_numpy(int)
        sc = np.zeros(len(g))
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, y, g["rec"]):
            mu, sd = X[tr].mean(0), X[tr].std(0)
            mdl = LogisticRegression(max_iter=1000).fit((X[tr] - mu) / sd, y[tr])
            sc[te] = mdl.decision_function((X[te] - mu) / sd)
        out[cond] = L.auroc(y, sc)
    s = pd.Series(out)
    return s.loc[[f"dur{d}ms" for d in (20, 40, 60, 80, 100, 200, 500, 1000)]]


def figure_provenance(B: int, seed: int) -> pd.DataFrame:
    """Per-duration paired AUROC differences from LoRA L0 inside the provenance bank."""
    p = PROV_DIR
    cv = pd.read_csv(p / "provcv_scores.csv.gz")
    b = pd.read_csv(p / "prov_scores_base.csv.gz").set_index("variant_id").reindex(cv["variant_id"])
    lo = pd.read_csv(p / "prov_scores_lora.csv.gz").set_index("variant_id").reindex(cv["variant_id"])
    m = cv[["variant_id", "base_id", "rec_id", "label", "condition"]].assign(axis="duration")
    zero = np.zeros(len(m))
    R = {"in-bank L2cv": cv["base h_final incv"].to_numpy(),
         "transferred": b["L2 testfit__score"].to_numpy(),
         "lora L0": lo["L0__score"].to_numpy()}
    readouts = {k: (v, zero, None) for k, v in R.items()}
    con = {}
    for c in DUR:
        for k in ("in-bank L2cv", "transferred"):
            con[f"FIG prov {c}: {k} - lora L0"] = [(1.0, k, c, "auroc"), (-1.0, "lora L0", c, "auroc")]
    _, c = L2.cluster_bootstrap(m, readouts, B=B, seed=seed, metrics=("auroc",), contrasts=con)
    c[["estimate", "lo", "hi"]] *= 100
    return c


QUIET = ("no exclusion", "more than 20 dB below its clip", "below -30 dBFS")


def _reference(d: pd.DataFrame, score: np.ndarray) -> dict:
    """Mean AUROC lead over LoRA L0 across the 8 durations and across 20 and 40 ms, in points, for
    each quiet-crop exclusion. d holds the duration-axis items in the order of score."""
    masks = {"no exclusion": np.ones(len(d), bool),
             "more than 20 dB below its clip": ~(d["rel_db"] < -20).to_numpy(),
             "below -30 dBFS": (d["dbfs"] >= -30).to_numpy()}
    out = {}
    for name, mask in masks.items():
        leads = []
        for dms in (20, 40, 60, 80, 100, 200, 500, 1000):
            g = mask & (d["duration_ms"] == dms).to_numpy()
            y = d["label"].to_numpy()[g]
            leads.append(100 * (L.auroc(y, score[g]) - L.auroc(y, d["lora"].to_numpy()[g])))
        out[(name, "all")] = float(np.mean(leads))
        out[(name, "20-40")] = float(np.mean(leads[:2]))
    return out


def matched_reference(seed: int, draws: int, n: int = 182) -> tuple[pd.DataFrame, pd.DataFrame]:
    """I. The provenance rule compares heads cross-fitted inside the provenance bank, whose training
    folds hold about 182 clips (91 recordings with one speech and one non-speech clip), with a
    reference from main-bank heads trained on about 776 excerpts per fold. Here the main-bank L2_cv
    head is refitted with the same outer folds and the same inner choice of C as
    lib_v2.crossfit_scores, but each training fold is cut to n base excerpts, half per class and
    one per recording. 'clips' keeps every condition of the chosen excerpts, 'clips and durations'
    only their 8 duration variants, as in the provenance bank. Each draw is cached with its
    parameters, so a stopped run resumes and a change of parameters refits. Returns the reference
    per draw and the rule's decision with the mean reference."""
    from layer_accessibility import CS, _load
    from closing_gpu import _decide
    from sklearn.model_selection import StratifiedGroupKFold
    out_dir = WORK / "acc" / "matched_ref"
    out_dir.mkdir(parents=True, exist_ok=True)
    meta, X = _load("base", "eval", "h_final")
    y, rec, bid = meta["label"].to_numpy(int), meta["rec_id"].to_numpy(), meta["base_id"].to_numpy()
    dur = (meta["axis"] == "duration").to_numpy()
    variants = {"clips": np.ones(len(meta), bool), "clips and durations": dur}
    folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed).split(X, y, rec))
    params = {"Cs": list(CS), "outer_splits": 5, "inner_splits": 4, "variants": list(variants),
              "sampler": "shuffle excerpts, first per recording, first n/2 per class"}
    for r in range(draws):
        path = out_dir / f"draw{r}_n{n}_seed{seed}.npz"
        params_json = path.with_suffix(".json")
        if path.exists() and params_json.exists() and json.loads(params_json.read_text()) == params:
            continue
        t0 = time.time()
        scores = {v: np.full(len(meta), np.nan) for v in variants}
        chosen_C = {v: [] for v in variants}
        for k, (tr, te) in enumerate(folds):
            rng = np.random.default_rng([seed, n, r, k])
            bases = meta.iloc[tr].drop_duplicates("base_id")[["base_id", "rec_id", "label"]]
            chosen = []
            for lab in (1, 0):
                b = bases[bases["label"] == lab]
                b = b.iloc[rng.permutation(len(b))].drop_duplicates("rec_id")
                chosen += b["base_id"].iloc[: n // 2].tolist()
            picked = tr[np.isin(bid[tr], chosen)]
            for v, keep in variants.items():
                sel = picked[keep[picked]]
                model, info = L2.fit_head(X[sel], y[sel], rec[sel], Cs=CS, n_splits=4, seed=seed + k + 1)
                scores[v][te] = model.decision_function(X[te])
                chosen_C[v].append(info["C"])
        tmp = path.with_name(path.stem + ".partial.npz")
        np.savez(tmp, variant_id=meta["variant_id"].to_numpy(str),
                 **{f"score_{i}": scores[v] for i, v in enumerate(variants)},
                 **{f"C_{i}": np.array(chosen_C[v]) for i, v in enumerate(variants)})
        tmp.replace(path)
        params_json.write_text(json.dumps(params))
        print(f"draw {r} in {(time.time() - t0) / 60:.1f} min, C per fold {chosen_C}", flush=True)

    md_ = pd.read_csv(WORK / "manifests" / "test_bank_v2.metadata.csv.gz")
    md_["variant_id"] = md_["audio_path"].astype(str).str.rsplit(".", n=1).str[0]
    rel = L.relative_level_db(md_).set_index("variant_id")["rel_db"]
    dbfs = 20 * np.log10(md_.set_index("variant_id")["segment_rms"])
    d = meta.loc[dur, ["variant_id", "label", "duration_ms"]].reset_index(drop=True)
    d["lora"] = scores_from_runs("lora").set_index("variant_id")["L0__score"].reindex(d["variant_id"]).to_numpy()
    d["rel_db"] = rel.reindex(d["variant_id"]).to_numpy()
    d["dbfs"] = dbfs.reindex(d["variant_id"]).to_numpy()
    if d[["lora", "dbfs"]].isna().any().any():
        raise RuntimeError("duration items without a LoRA score or a level")
    rows = [{"head": "every excerpt of the fold (cached L2_cv)", "draw": "",
             **_reference(d, _cached("base__h_final__cv", meta["variant_id"])[dur])}]
    for r in range(draws):
        z = np.load(out_dir / f"draw{r}_n{n}_seed{seed}.npz")
        if not (z["variant_id"] == meta["variant_id"].to_numpy(str)).all():
            raise RuntimeError(f"draw {r} was cached with another item order")
        for i, v in enumerate(variants):
            rows.append({"head": f"{n} excerpts, {v}", "draw": r, **_reference(d, z[f"score_{i}"][dur])})
    t = pd.DataFrame(rows)
    t.columns = [c if isinstance(c, str) else f"{c[0]}, {c[1]}" for c in t.columns]

    prov = pd.read_csv(PROV_DIR / "provcv_contrasts.csv").set_index("contrast")
    dec = []
    for head, g in t.groupby("head", sort=False):
        for q in QUIET:
            for span in ("all", "20-40"):
                ref = g[f"{q}, {span}"].mean()
                for a in ("base h_final incv", "lora h_final incv"):
                    p = prov.loc[f"{span}: {a} - lora L0"]
                    dec.append({"reference head": head, "exclusion": q, "durations": span, "contrast": a,
                                "reference": ref, "estimate": p["estimate"], "lo": p["lo"], "hi": p["hi"],
                                "decision": _decide(p["estimate"], p["lo"], p["hi"], ref)})
    return t, pd.DataFrame(dec)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("what", choices=["all", "matched"])
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--draws", type=int, default=10)
    a = ap.parse_args()
    if a.what == "matched":
        t, dec = matched_reference(a.seed, a.draws)
        print("## I. Reference of the provenance rule with matched supervision (AUROC points)\n")
        print(md(t.set_index(["head", "draw"]), 2), "\n")
        print("Mean and range across draws\n")
        summary = t.drop(columns="draw").groupby("head", sort=False).agg(["mean", "min", "max"]).T
        print(md(summary.rename_axis(["reference", "statistic"]), 2), "\n")
        print("### Decision with the mean reference of each head\n")
        print(md(dec.set_index(["reference head", "exclusion", "durations", "contrast"]), 2), "\n")
        dec.to_csv(WORK / "acc" / "matched_ref" / "decisions.csv", index=False)
        t.to_csv(WORK / "acc" / "matched_ref" / "references.csv", index=False)
        return
    c = main_bank(a.B, a.seed)
    print(f"## Main bank, B = {a.B}\n")
    print(md(c[~c["contrast"].str.startswith("FIG")].set_index("contrast")[["estimate", "lo", "hi", "se", "mde80"]], 2), "\n")
    print("## E. AUROC lead of L2_cv over LoRA L0 per duration (points)\n")
    print(md(quiet_free_reference(a.seed), 2), "\n")
    print("## F. Criterion extremes\n")
    for k, v in criterion_extremes().items():
        print(f"- {k}: {v if isinstance(v, str) else round(v, 2)}")
    print("\n## H. Level used two-sidedly, cross-fitted AUROC per duration\n")
    print(md(level_two_sided(a.seed).to_frame("AUROC"), 3), "\n")
    fig = c[c["contrast"].str.startswith("FIG")].copy()
    if (PROV_DIR / "provcv_scores.csv.gz").exists():
        fig = pd.concat([fig, figure_provenance(a.B, a.seed)], ignore_index=True)
    suffix = "" if PROV_DIR.name == "prov-v3" else f"_{PROV_DIR.name}"
    fig.to_csv(WORK / f"figure_auroc_diff{suffix}.csv", index=False)
    print("## Figure data, AUROC difference from LoRA L0 (points)\n")
    print(md(fig.set_index("contrast")[["estimate", "lo", "hi"]], 2), "\n")


if __name__ == "__main__":
    main()
