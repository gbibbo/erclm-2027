"""Analysis of the exploratory GPU runs, decisions D1 and D2 (docs/exploratory_log.md, "Step 11. Provenance control").

    PYTHONPATH=src python tools/closing_gpu.py fetch prov-v3
    PYTHONPATH=src python tools/closing_gpu.py prov [--B 2000]
    PYTHONPATH=src python tools/closing_gpu.py fetch prompt-ab_speech_second   (and the other two)
    PYTHONPATH=src python tools/closing_gpu.py prompt [--B 2000]

Stores go to _work/closing/runs/<run>/<backbone>/eval, downloaded with the project's rclone
remote and checked by size and MD5. Everything here is exploratory.
"""
from __future__ import annotations

import argparse
import os
import json
import sys
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
from erclm.storage import RcloneRemote  # noqa: E402

REMOTE = "erclm:ERC-LM-2027"
PROMPTS = ("ab_speech_second", "ba_speech_first", "ba_speech_second")
HEADS = ("h_final", "enc_region_mean", "enc_global_mean")


def fetch(run: str) -> None:
    remote = RcloneRemote(REMOTE)
    for bb in ("base", "lora"):
        local = WORK / "runs" / run / bb / "eval"
        remote.get_dir(f"work/{run}/{bb}/eval", local)
        n = len(L.load_store_features(local)[0])
        print(f"{run} {bb}: {n} items in {local}")


def _sorted_conditions(conds):
    return sorted(conds, key=lambda c: int(c.split("_")[1][:-2]) if c.startswith("dur_") else 0)


def prov(B: int, seed: int) -> None:
    """Speech against non-speech from the same VoxConverse recordings, with the heads and the
    threshold fitted on the LoRA training bank, next to the duration axis of the main test bank."""
    rows = []
    for bb in ("base", "lora"):
        mtr, xtr = L.load_store_features(WORK / "features" / bb / "train", HEADS)
        mpv, xpv = L.load_store_features(PROV_DIR / bb / "eval", HEADS)
        y_tr, g_tr = mtr["label"].to_numpy(int), mtr["rec_id"].to_numpy()
        s = mpv["s_primary"].to_numpy(float)
        thr = L.fit_threshold(train_meta(bb)["s_primary"], train_meta(bb)["label"])
        zero = np.zeros(len(mpv))
        R = {"L0": (s, zero, None),
             "L1_bc_cond": (s, L2.l1_thresholds(mpv, s, "bc_cond"), "bc_cond"),
             "L1_center_cond": (s, L2.l1_thresholds(mpv, s, "center_cond"), "center_cond"),
             "L1_tr": (s, np.full(len(s), thr), None)}
        mte, xte = L.load_store_features(WORK / "features" / bb / "eval", HEADS)
        for f in HEADS:
            name = "L2" if f == "h_final" else f"P_{f}"
            model, _ = L2.fit_head(xtr[f], y_tr, g_tr, seed=42)
            R[name] = (model.decision_function(xpv[f]), zero, None)
            # the same head fitted on the whole main test bank, VoxConverse speech against ESC-50,
            # which is where a corpus shortcut would be learnt
            model = L2._head(0.01).fit(xte[f], mte["label"].to_numpy(int))
            R[f"{name} testfit"] = (model.decision_function(xpv[f]), zero, None)
        del xte
        per_item = mpv[["variant_id", "base_id", "rec_id", "label", "axis", "condition", "duration_ms"]].copy()
        for k, (sc_, thr_, _) in R.items():
            per_item[f"{k}__score"], per_item[f"{k}__thr"] = sc_, thr_
        per_item.to_csv(PROV_DIR / f"prov_scores_{bb}.csv.gz", index=False)
        ci, _ = L2.cluster_bootstrap(mpv, R, B=B, seed=seed, metrics=("ba", "auroc"))
        ci["bank"], ci["backbone"] = "prov", bb
        rows.append(ci)
        # the same readouts on the duration axis of the main test bank, for comparison
        sc = scores_from_runs(bb)
        sc = sc[sc["axis"] == "duration"].reset_index(drop=True)
        ss = sc["s_primary"].to_numpy(float)
        Rm = {"L0": (ss, np.zeros(len(sc)), None),
              "L1_bc_cond": (ss, L2.l1_thresholds(sc, ss, "bc_cond"), "bc_cond"),
              "L1_center_cond": (ss, L2.l1_thresholds(sc, ss, "center_cond"), "center_cond"),
              "L1_tr": (ss, np.full(len(sc), thr), None),
              "L2": (sc["L2__score"].to_numpy(), np.zeros(len(sc)), None),
              "P_enc_region_mean": (sc["P_enc_region_mean__score"].to_numpy(), np.zeros(len(sc)), None),
              "P_enc_global_mean": (sc["P_enc_global_mean__score"].to_numpy(), np.zeros(len(sc)), None)}
        # on the main bank the in-distribution counterpart of 'testfit' is the cross-fitted head
        for f in HEADS:
            name = "L2" if f == "h_final" else f"P_{f}"
            z = np.load(WORK / "acc" / f"{bb}__{f}__cv.npz")
            cv = pd.Series(z["score"], index=z["variant_id"]).reindex(sc["variant_id"]).to_numpy()
            Rm[f"{name} testfit"] = (cv, np.zeros(len(sc)), None)
        cim, _ = L2.cluster_bootstrap(sc, Rm, B=B, seed=seed, metrics=("ba", "auroc"))
        cim["bank"], cim["backbone"] = "main", bb
        rows.append(cim)
        print(f"{bb}: {mpv['rec_id'].nunique()} recordings, {mpv['base_id'].nunique()} clips, "
              f"{len(mpv)} items, L1_tr threshold {thr:.3f}\n")
    ci = pd.concat(rows, ignore_index=True)
    ci[["estimate", "lo", "hi"]] *= 100
    for met in ("auroc", "ba"):
        t = ci[ci["metric"] == met]
        mean = t.groupby(["backbone", "readout", "bank"])["estimate"].mean().unstack("bank")
        mean["prov - main"] = mean["prov"] - mean["main"]
        print(f"## Provenance control, mean {met.upper()} over the 8 durations (x100)\n")
        print(md(mean, 1), "\n")
    t = ci[(ci["metric"] == "auroc") & ci["readout"].isin(["L0", "L2", "L2 testfit", "P_enc_global_mean",
                                                           "P_enc_global_mean testfit"])]
    per = t.pivot_table(index=["backbone", "readout", "bank"], columns="condition", values="estimate")
    print("## AUROC per duration (x100)\n")
    print(md(per[_sorted_conditions(per.columns)], 1), "\n")
    t = ci[(ci["metric"] == "ba") & (ci["bank"] == "prov")]
    per = t.pivot_table(index=["backbone", "readout"], columns="condition", values="estimate")
    print("## BA per duration on the provenance bank, with 95% intervals in the CSV\n")
    print(md(per[_sorted_conditions(per.columns)], 1), "\n")
    ci.to_csv(PROV_DIR / "prov_ci.csv", index=False)


def prompt(B: int, seed: int) -> None:
    """The confirmatory prompt against the three variants, on the same items, paired."""
    for bb in ("base", "lora"):
        sc = scores_from_runs(bb)
        stores = {v: L.load_store_features(WORK / "runs" / f"prompt-{v}" / bb / "eval")[0] for v in PROMPTS}
        ids = set(sc["variant_id"])
        for v, mv in stores.items():
            ids &= set(mv["variant_id"])
        m = sc[sc["variant_id"].isin(ids)].reset_index(drop=True)
        cols = {"ab": m["s_primary"].to_numpy(float)}
        top1 = {"ab": m["top1_is_label"].mean()}
        for v, mv in stores.items():
            mv = mv.set_index("variant_id").reindex(m["variant_id"])
            cols[v] = mv["s_primary"].to_numpy(float)
            top1[v] = mv["top1_is_label"].astype(bool).mean()
        R = {}
        for v, s in cols.items():
            R[f"{v} L0"] = (s, np.zeros(len(s)), None)
            R[f"{v} BC"] = (s, L2.l1_thresholds(m, s, "bc_cond"), "bc_cond")
        means = {f"mean {k}": {k: 1} for k in R}
        diffs = {f"{v} L0 minus ab L0": {f"{v} L0": 1, "ab L0": -1} for v in PROMPTS}
        con_ba = L.axis_mean_contrasts(m, means | diffs, metric="ba")
        con_auc = {k.replace(":", ":auroc ", 1): v for k, v in
                   L.axis_mean_contrasts(m, {f"mean {v} L0": {f"{v} L0": 1} for v in cols}
                                         | {f"{v} minus ab": {f"{v} L0": 1, "ab L0": -1} for v in PROMPTS},
                                         metric="auroc").items()}
        _, con = L2.cluster_bootstrap(m, R, B=B, seed=seed, metrics=("ba", "auroc"), contrasts=con_ba | con_auc)
        con[["axis", "what"]] = con["contrast"].str.split(":", n=1, expand=True)
        print(f"## Prompt control, {bb}. {m['base_id'].nunique()} clips, {len(m)} items, B = {B}\n")
        print(md(con.set_index(["what", "axis"])[["estimate", "lo", "hi"]] * 100, 1), "\n")
        y = m["label"].to_numpy() == 1
        desc = pd.DataFrame({v: {"mean s speech": s[y].mean(), "mean s non-speech": s[~y].mean(),
                                 "hit L0": (s[y] > 0).mean(), "false alarm L0": (s[~y] > 0).mean(),
                                 "answer is A or B": top1[v]} for v, s in cols.items()}).T
        print(f"Native score and answers per prompt, {bb}\n")
        print(md(desc, 3), "\n")


DEPTH = ("enc_region_mean", "enc_global_mean", "h_layer8", "h_layer16", "h_layer24", "h_final")
CS = (1e-4, 1e-3, 1e-2, 1e-1)


def _decide(est: float, lo: float, hi: float, ref: float) -> str:
    """The rule written in docs/exploratory_log.md, "Decision rule for the contrast inside the provenance
    bank", before the contrast was computed."""
    if lo > 0 and est >= ref / 2:
        return "speech evidence"
    if est < ref / 3 and hi < ref:
        return "shortcut"
    return "inconclusive"


def provcv(B: int, seed: int) -> None:
    """Heads cross-fitted inside the provenance bank (speech and non-speech from the same
    recordings), against the LoRA with L0, with the decision rule fixed beforehand. Also the depth
    profile and the 200-to-20 ms drop of the regional probe inside the bank, and a corpus control
    that separates VoxConverse non-speech from ESC-50 non-speech."""
    base_ps = pd.read_csv(PROV_DIR / "prov_scores_base.csv.gz")
    lora_ps = pd.read_csv(PROV_DIR / "prov_scores_lora.csv.gz")
    R, info = {}, {}
    for bb in ("base", "lora"):
        feats = DEPTH if bb == "base" else ("h_final", "enc_region_mean")
        mpv, xpv = L.load_store_features(PROV_DIR / bb / "eval", feats)
        if not (mpv["variant_id"].to_numpy() == base_ps["variant_id"].to_numpy()).all():
            mpv = mpv.set_index("variant_id").loc[base_ps["variant_id"]].reset_index()
        y, g = mpv["label"].to_numpy(int), mpv["rec_id"].to_numpy()
        for f in feats:
            sc, inf = L2.crossfit_scores(xpv[f], y, g, n_splits=5, seed=seed, Cs=CS)
            R[f"{bb} {f} incv"] = sc
            info[f"{bb} {f}"] = [i["C"] for i in inf]
    per_item = base_ps[["variant_id", "base_id", "rec_id", "label", "condition", "duration_ms"]].copy()
    for k, v in R.items():
        per_item[k] = v
    per_item.to_csv(PROV_DIR / "provcv_scores.csv.gz", index=False)
    R["lora L0"] = lora_ps["L0__score"].to_numpy()
    R["base L0"] = base_ps["L0__score"].to_numpy()
    R["base L2 matched"] = base_ps["L2__score"].to_numpy()
    R["base L2 testfit"] = base_ps["L2 testfit__score"].to_numpy()
    m = base_ps[["variant_id", "base_id", "rec_id", "label", "axis", "condition", "duration_ms"]]
    zero = np.zeros(len(m))
    readouts = {k: (v, zero, None) for k, v in R.items()}
    conds = sorted(m["condition"].unique(), key=lambda c: int(c.split("_")[1][:-2]))
    short = [c for c in conds if c in ("dur_20ms", "dur_40ms")]

    def avg(rd, cs, met="auroc", coef=1.0):
        return [(coef / len(cs), rd, c, met) for c in cs]
    con = {}
    for a in ("base h_final incv", "lora h_final incv", "base L2 testfit", "base L2 matched"):
        con[f"all: {a} - lora L0"] = avg(a, conds) + avg("lora L0", conds, coef=-1.0)
        con[f"20-40: {a} - lora L0"] = avg(a, short) + avg("lora L0", short, coef=-1.0)
    con["all: base h_final incv - lora h_final incv"] = avg("base h_final incv", conds) + avg("lora h_final incv", conds, coef=-1.0)
    for f in DEPTH:
        con[f"depth mean AUROC: base {f} incv"] = avg(f"base {f} incv", conds)
        con[f"depth mean BA: base {f} incv"] = avg(f"base {f} incv", conds, met="ba")
    for f in DEPTH[:-1]:
        con[f"depth AUROC: base {f} incv - base h_final incv"] = avg(f"base {f} incv", conds) + avg("base h_final incv", conds, coef=-1.0)
    for f in DEPTH:
        con[f"depth AUROC: base {f} incv - lora L0"] = avg(f"base {f} incv", conds) + avg("lora L0", conds, coef=-1.0)
    con["BA all: base h_final incv - lora L0"] = avg("base h_final incv", conds, met="ba") + avg("lora L0", conds, met="ba", coef=-1.0)
    con["oracle BA all: base h_final incv - lora L0"] = (avg("base h_final incv", conds, met="oracle_ba")
                                                         + avg("lora L0", conds, met="oracle_ba", coef=-1.0))
    con["oracle BA all: base L0 (L1*)"] = avg("base L0", conds, met="oracle_ba")
    con["oracle BA all: lora L0 (L1*)"] = avg("lora L0", conds, met="oracle_ba")
    con["H2a drop in bank: base enc_region_mean incv 200-20"] = (
        [(1.0, "base enc_region_mean incv", "dur_200ms", "auroc"), (-1.0, "base enc_region_mean incv", "dur_20ms", "auroc")])
    R["base P_enc_region_mean matched"] = base_ps["P_enc_region_mean__score"].to_numpy()
    readouts["base P_enc_region_mean matched"] = (R["base P_enc_region_mean matched"], zero, None)
    con["H2a drop in bank: base enc_region_mean matched 200-20"] = (
        [(1.0, "base P_enc_region_mean matched", "dur_200ms", "auroc"),
         (-1.0, "base P_enc_region_mean matched", "dur_20ms", "auroc")])
    ci, c = L2.cluster_bootstrap(m, readouts, B=B, seed=seed, metrics=("ba", "auroc", "oracle_ba"), contrasts=con)
    c[["estimate", "lo", "hi"]] *= 100
    print(f"## Heads cross-fitted inside the provenance bank, B = {B}\n")
    print("C chosen per fold:", json.dumps(info), "\n")
    rows = []
    for label, ref in (("all", 2.36), ("20-40", 7.65)):
        for a in ("base h_final incv", "lora h_final incv"):
            r = c.set_index("contrast").loc[f"{label}: {a} - lora L0"]
            rows.append({"contrast": f"{label}: {a} - lora L0", "estimate": r["estimate"], "lo": r["lo"],
                         "hi": r["hi"], "reference": ref, "decision": _decide(r["estimate"], r["lo"], r["hi"], ref)})
    print("### Decision rule (AUROC points)\n")
    print(md(pd.DataFrame(rows).set_index("contrast"), 2), "\n")
    print("### All contrasts\n")
    c[["se", "mde80"]] *= 100
    print(md(c.set_index("contrast")[["estimate", "lo", "hi", "se", "mde80"]], 2), "\n")
    per = ci[ci["metric"] == "auroc"].pivot_table(index="readout", columns="condition", values="estimate") * 100
    print("### AUROC per duration inside the provenance bank\n")
    print(md(per[conds], 1), "\n")
    ci.to_csv(PROV_DIR / "provcv_ci.csv", index=False)
    c.to_csv(PROV_DIR / "provcv_contrasts.csv", index=False)

    # corpus control: VoxConverse non-speech (provenance) against ESC-50 non-speech (main bank),
    # duration axis, cross-fitted by recording, per feature of the base backbone
    rows = []
    for f in DEPTH:
        mte, xte = L.load_store_features(WORK / "features" / "base" / "eval", [f])
        keep = ((mte["label"] == 0) & (mte["axis"] == "duration")).to_numpy()
        mpv, xpv = L.load_store_features(PROV_DIR / "base" / "eval", [f])
        kp = (mpv["label"] == 0).to_numpy()
        meta = pd.concat([mte.loc[keep, ["rec_id", "condition"]].assign(corpus=0),
                          mpv.loc[kp, ["rec_id", "condition"]].assign(corpus=1)], ignore_index=True)
        X = np.concatenate([xte[f][keep], xpv[f][kp]])
        del xte
        sc, _ = L2.crossfit_scores(X, meta["corpus"].to_numpy(int), meta["rec_id"].to_numpy(), n_splits=5,
                                   seed=seed, Cs=CS)
        for cond, gi in meta.groupby("condition").indices.items():
            rows.append({"feature": f, "condition": cond, "auroc": L.auroc(meta["corpus"].to_numpy()[gi], sc[gi])})
    t = pd.DataFrame(rows).pivot_table(index="feature", columns="condition", values="auroc") * 100
    print("### Corpus control: VoxConverse non-speech against ESC-50 non-speech, cross-fitted AUROC\n")
    print(md(t[conds], 1), "\n")


def csweep(B: int, seed: int) -> None:
    """Sensitivity of the confirmatory head to its regularisation. The base L2 head is refitted on the
    LoRA training bank with every C of the pre-registered grid and compared with the LoRA with L0, as
    in H3, on the main bank (duration and SNR) and on the provenance bank. Exploratory."""
    mtr, xtr = L.load_store_features(WORK / "features" / "base" / "train", ["h_final"])
    mte, xte = L.load_store_features(WORK / "features" / "base" / "eval", ["h_final"])
    mpv, xpv = L.load_store_features(PROV_DIR / "base" / "eval", ["h_final"])
    y_tr = mtr["label"].to_numpy(int)
    main = scores_from_runs("base")[["variant_id", "base_id", "rec_id", "label", "axis", "condition"]]
    main = main[main["axis"].isin(["duration", "snr"])].reset_index(drop=True)
    lora_main = scores_from_runs("lora").set_index("variant_id")["L0__score"]
    pv = pd.read_csv(PROV_DIR / "prov_scores_lora.csv.gz")
    idx_te = pd.Series(np.arange(len(mte)), index=mte["variant_id"]).reindex(main["variant_id"]).to_numpy()
    idx_pv = pd.Series(np.arange(len(mpv)), index=mpv["variant_id"]).reindex(pv["variant_id"]).to_numpy()
    Rm = {"lora L0": lora_main.reindex(main["variant_id"]).to_numpy()}
    Rp = {"lora L0": pv["L0__score"].to_numpy()}
    for C in CS:
        model = L2._head(C).fit(xtr["h_final"], y_tr)
        Rm[f"L2 C={C:g}"] = model.decision_function(xte["h_final"][idx_te])
        Rp[f"L2 C={C:g}"] = model.decision_function(xpv["h_final"][idx_pv])
    _, cv_info = L2.fit_head(xtr["h_final"], y_tr, mtr["rec_id"].to_numpy(), seed=seed)
    print(f"Grouped CV on the training bank chose C = {cv_info['C']:g} (AUROC {cv_info['cv_auroc']:.4f})\n")
    rows = []
    for bank, meta, R in (("main", main, Rm), ("prov", pv[["variant_id", "base_id", "rec_id", "label", "axis",
                                                             "condition"]], Rp)):
        zero = np.zeros(len(meta))
        readouts = {k: (v, zero, None) for k, v in R.items()}
        terms = {}
        for C in CS:
            k = f"L2 C={C:g}"
            terms[f"BA {k} - lora L0"] = {k: 1, "lora L0": -1}
            terms[f"BA {k}"] = {k: 1}
        con_ba = L.axis_mean_contrasts(meta, terms, metric="ba")
        con_auc = {k.replace(":", ":AUROC ", 1).replace("AUROC BA ", "AUROC "): v
                   for k, v in L.axis_mean_contrasts(meta, terms, metric="auroc").items()}
        _, c = L2.cluster_bootstrap(meta, readouts, B=B, seed=seed, metrics=("ba", "auroc"), contrasts=con_ba | con_auc)
        c["bank"] = bank
        rows.append(c)
    c = pd.concat(rows, ignore_index=True)
    c[["axis", "what"]] = c["contrast"].str.split(":", n=1, expand=True)
    c[["estimate", "lo", "hi"]] *= 100
    print(f"## Regularisation of the matched base head, B = {B}\n")
    print(md(c.set_index(["bank", "axis", "what"])[["estimate", "lo", "hi"]], 2), "\n")
    suffix = "" if PROV_DIR.name == "prov-v3" else f"_{PROV_DIR.name}"
    c.to_csv(WORK / f"csweep_contrasts{suffix}.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("run")
    for name in ("prov", "prompt", "provcv", "csweep"):
        p = sub.add_parser(name)
        p.add_argument("--B", type=int, default=2000)
        p.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    if a.cmd == "fetch":
        fetch(a.run)
    elif a.cmd == "prov":
        prov(a.B, a.seed)
    elif a.cmd == "provcv":
        provcv(a.B, a.seed)
    elif a.cmd == "csweep":
        csweep(a.B, a.seed)
    else:
        prompt(a.B, a.seed)


if __name__ == "__main__":
    main()
