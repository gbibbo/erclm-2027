"""GPU plan of 22 October, exploratory (docs/exploratory_log.md, "GPU plan of 22 October").

I1, channel-cue intervention, on the duration axis.
    A  change of the in-bank lead of base L2_cv over LoRA L0 when every container is band-limited.
    B  change of the transfer gap of the same fold heads, AUROC on their held-out main fold minus
       AUROC on the provenance bank.
    The unfiltered versions come from the stores of full-v1 and prov-v3 with the same code, so both
    versions share items, folds and procedure.
I2, LoRA kept in one part of the model. Share of the LoRA's gain in ranking (AUROC of the native
    score) and in criterion (BA of L0) that each part recovers.
W2 of the review of draft v4. In-bank lead in the main bank minus in-bank lead in the provenance
    bank, with the recordings of both banks resampled.

    PYTHONPATH=src python tools/intervention_analysis.py fetch cue-main [--backbones base,lora]
    PYTHONPATH=src python tools/intervention_analysis.py orig [--B 2000]
    PYTHONPATH=src python tools/intervention_analysis.py cue [--B 2000]
    PYTHONPATH=src python tools/intervention_analysis.py split [--B 2000]
    PYTHONPATH=src python tools/intervention_analysis.py w2 [--B 2000]
    PYTHONPATH=src python tools/intervention_analysis.py negatives [--B 2000]   # W4, doubtful negatives out
    PYTHONPATH=src python tools/intervention_analysis.py quiet [--B 2000]       # quiet crops out of both banks
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
from closing_analyses import WORK, md  # noqa: E402
from erclm import lib_v2 as L2  # noqa: E402
from erclm import lib_v3 as L  # noqa: E402

DUR = (20, 40, 60, 80, 100, 200, 500, 1000)
CS = (1e-4, 1e-3, 1e-2, 1e-1)
OUT = WORK / "interv"
REMOTE = "erclm:ERC-LM-2027"
STORES = {
    "main_o": (WORK / "features" / "base" / "eval", WORK / "features" / "lora" / "eval"),
    "prov_o": (WORK / "runs" / "prov-v3" / "base" / "eval", WORK / "runs" / "prov-v3" / "lora" / "eval"),
    "main_f": (WORK / "runs" / "cue-main" / "base" / "eval", WORK / "runs" / "cue-main" / "lora" / "eval"),
    "prov_f": (WORK / "runs" / "cue-prov" / "base" / "eval", WORK / "runs" / "cue-prov" / "lora" / "eval"),
}
SPLITS = ("split-audio", "split-llm", "split-llm-lower", "split-llm-upper")


def fetch(run: str, backbones=("base", "lora"), splits=("eval",)) -> None:
    from erclm.storage import RcloneRemote
    remote = RcloneRemote(REMOTE)
    for bb in backbones:
        for split in splits:
            local = WORK / "runs" / run / bb / split
            remote.get_dir(f"work/{run}/{bb}/{split}", local)
            print(f"{run} {bb} {split}: {len(L.load_store_features(local)[0])} items in {local}")


# ---------------------------------------------------------------------------- data

def load_bank(key: str, with_x: bool = True):
    """Duration-axis items sorted by variant_id, with the native scores of both backbones as
    s_base and s_lora, and the final state of the base backbone."""
    db, dl = STORES[key]
    mb, xb = L.load_store_features(db, ["h_final"] if with_x else [])
    keep = (mb["axis"] == "duration").to_numpy()
    order = np.argsort(mb.loc[keep, "variant_id"].astype(str).to_numpy(), kind="mergesort")
    meta = mb[keep].iloc[order].reset_index(drop=True)
    x = xb["h_final"][keep][order] if with_x else None
    ml = L.load_store_features(dl)[0].set_index("variant_id")
    meta = meta.rename(columns={"s_primary": "s_base"})
    meta["s_lora"] = ml["s_primary"].reindex(meta["variant_id"]).to_numpy()
    if meta["s_lora"].isna().any():
        raise RuntimeError(f"{key}: {int(meta['s_lora'].isna().sum())} items without a LoRA score")
    return meta, x


def fold_heads(meta: pd.DataFrame, x: np.ndarray, xp: np.ndarray, seed: int = 42) -> dict:
    """The folds, inner choice of C and seeds of lib_v2.crossfit_scores, keeping every fold head
    to score the provenance bank as well."""
    from sklearn.model_selection import StratifiedGroupKFold
    y, g = meta["label"].to_numpy(int), meta["rec_id"].to_numpy()
    cv, fold = np.full(len(y), np.nan), np.full(len(y), -1)
    prov, chosen = np.empty((5, len(xp))), []
    for f, (tr, te) in enumerate(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed).split(x, y, g)):
        if set(g[tr]) & set(g[te]):
            raise RuntimeError("recording leakage between folds")
        model, info = L2.fit_head(x[tr], y[tr], g[tr], Cs=CS, n_splits=4, seed=seed + f + 1)
        cv[te], fold[te] = model.decision_function(x[te]), f
        prov[f] = model.decision_function(xp)
        chosen.append(info["C"])
    return {"cv": cv, "fold": fold, "prov": prov, "C": np.array(chosen)}


def heads(version: str, seed: int = 42):
    """Main and provenance banks of one version ('o' unfiltered, 'f' filtered) with their fold
    heads, cached with their parameters."""
    OUT.mkdir(parents=True, exist_ok=True)
    main, xm = load_bank(f"main_{version}")
    prov, xp = load_bank(f"prov_{version}")
    path, pj = OUT / f"heads_{version}.npz", OUT / f"heads_{version}.json"
    params = {"Cs": list(CS), "seed": seed, "outer": 5, "inner": 4, "n_main": len(main), "n_prov": len(prov)}
    if path.exists() and pj.exists() and json.loads(pj.read_text()) == params:
        z = np.load(path, allow_pickle=False)
        if (z["main_ids"] == main["variant_id"].astype(str).to_numpy()).all() \
                and (z["prov_ids"] == prov["variant_id"].astype(str).to_numpy()).all():
            return main, prov, {k: z[k] for k in ("cv", "fold", "prov", "C")}
    t0 = time.time()
    h = fold_heads(main, xm, xp, seed)
    tmp = OUT / f"heads_{version}.partial.npz"
    np.savez(tmp, main_ids=main["variant_id"].astype(str).to_numpy().astype("U"),
             prov_ids=prov["variant_id"].astype(str).to_numpy().astype("U"), **h)
    tmp.replace(path)
    pj.write_text(json.dumps(params))
    print(f"fold heads {version} in {(time.time() - t0) / 60:.1f} min, C per fold {h['C'].tolist()}", flush=True)
    return main, prov, h


# ---------------------------------------------------------------------------- bootstrap

class Boot:
    """Recording weights as in lib_v2.cluster_bootstrap, resampled within each class stratum."""

    def __init__(self, meta: pd.DataFrame):
        rec = meta["rec_id"].astype(str).to_numpy()
        self.uniq, self.inv = np.unique(rec, return_inverse=True)
        strat = pd.Series(meta["label"].to_numpy(), index=rec).groupby(level=0).first().reindex(self.uniq).to_numpy()
        self.strata = [np.flatnonzero(strat == s) for s in np.unique(strat)]

    def draw(self, rng) -> np.ndarray:
        counts = np.zeros(len(self.uniq))
        for ids in self.strata:
            np.add.at(counts, rng.choice(ids, size=len(ids), replace=True), 1.0)
        return counts[self.inv]


def mean_auc(meta, s, w=None, durs=DUR, mask=None) -> float:
    y, d = meta["label"].to_numpy(), meta["duration_ms"].to_numpy()
    out = []
    for dur in durs:
        i = d == dur if mask is None else (d == dur) & mask
        out.append(L2.auc_w(y[i], s[i], None if w is None else w[i]))
    return float(np.mean(out))


def summarise(point: dict, reps: dict) -> pd.DataFrame:
    rows = []
    for k, v in point.items():
        r = np.asarray(reps[k])
        lo, hi = np.nanpercentile(r, [2.5, 97.5])
        rows.append({"quantity": k, "estimate": 100 * v, "lo": 100 * lo, "hi": 100 * hi,
                     "se": 100 * float(np.nanstd(r, ddof=1))})
    return pd.DataFrame(rows).set_index("quantity")


def boot_two_banks(main, prov, fn, B: int, seed: int) -> pd.DataFrame:
    """fn(wm, wp) -> {name: value}, on unit weights for the estimate and on independent recording
    resamples of the two banks for the 95% percentile interval."""
    point = fn(None, None)
    bm, bp, rng = Boot(main), Boot(prov), np.random.default_rng(seed)
    reps = {k: [] for k in point}
    for _ in range(B):
        r = fn(bm.draw(rng), bp.draw(rng))
        for k in point:
            reps[k].append(r[k])
    return summarise(point, reps)


def read_rule(est: float, lo: float, hi: float, original: float) -> str:
    """Reading rule of I1 (docs/exploratory_log.md, "I1. Intervention on channel cues"), in the units
    of the arguments."""
    if hi < 0 and est <= -original / 2:
        return "the filter explains it"
    if lo > -original / 4:
        return "the filter does not explain it"
    return "inconclusive"


# ---------------------------------------------------------------------------- I1 and W2

def gap_terms(main, prov, h, wm, wp) -> tuple:
    """In-bank lead over LoRA L0 and transfer gap of the same fold heads, for one version."""
    lead = mean_auc(main, h["cv"], wm) - mean_auc(main, main["s_lora"].to_numpy(), wm)
    gaps = [mean_auc(main, h["cv"], wm, mask=h["fold"] == k) - mean_auc(prov, h["prov"][k], wp) for k in range(5)]
    return lead, float(np.mean(gaps))


def orig(B: int, seed: int) -> pd.DataFrame:
    main, prov, h = heads("o", seed)

    def fn(wm, wp):
        lead, gap = gap_terms(main, prov, h, wm, wp)
        return {"A lead, unfiltered": lead, "B gap, unfiltered": gap,
                "held-out main AUROC of the fold heads": float(np.mean(
                    [mean_auc(main, h["cv"], wm, mask=h["fold"] == k) for k in range(5)])),
                "provenance AUROC of the fold heads": float(np.mean([mean_auc(prov, h["prov"][k], wp)
                                                                     for k in range(5)]))}
    return boot_two_banks(main, prov, fn, B, seed)


def cue(B: int, seed: int) -> tuple:
    mo, po, ho = heads("o", seed)
    mf, pf, hf = heads("f", seed)
    for a, b in ((mo, mf), (po, pf)):
        if not (a["variant_id"].astype(str).to_numpy() == b["variant_id"].astype(str).to_numpy()).all():
            raise RuntimeError("the filtered and unfiltered banks do not hold the same items")

    # sensitivity added after the blind listening of 22 Oct, outside the registered rule: B without
    # the recordings whose non-speech window was heard as clear or possible speech
    clean = np.ones(len(po), bool)
    lab_path = WORK / "negatives" / "listening.csv"
    if lab_path.exists():
        lab = pd.read_csv(lab_path)
        rec_of = po.drop_duplicates("base_id").set_index("base_id")["rec_id"]
        bad = set(rec_of.reindex(lab.loc[lab["label"].isin(["voz", "voz_posible"]), "base_id"]).dropna())
        clean = ~po["rec_id"].isin(bad).to_numpy()
    poc, pfc = po[clean].reset_index(drop=True), pf[clean].reset_index(drop=True)
    hoc, hfc = {**ho, "prov": ho["prov"][:, clean]}, {**hf, "prov": hf["prov"][:, clean]}

    def fn(wm, wp):
        lo_, go = gap_terms(mo, po, ho, wm, wp)
        lf, gf = gap_terms(mf, pf, hf, wm, wp)
        out = {"A lead, unfiltered": lo_, "A lead, filtered": lf, "A change": lf - lo_,
               "B gap, unfiltered": go, "B gap, filtered": gf, "B change": gf - go}
        wpc = None if wp is None else wp[clean]
        gco, gcf = gap_terms(mo, poc, hoc, wm, wpc)[1], gap_terms(mf, pfc, hfc, wm, wpc)[1]
        out.update({"B gap, unfiltered, heard-clean negatives": gco, "B gap, filtered, heard-clean negatives": gcf,
                    "B change, heard-clean negatives": gcf - gco})
        for bank, a, b, w in (("main", mo, mf, wm), ("provenance", po, pf, wp)):
            for s in ("s_lora", "s_base"):
                out[f"control {bank} {s}: filtered - unfiltered"] = (
                    mean_auc(b, b[s].to_numpy(), w) - mean_auc(a, a[s].to_numpy(), w))
        return out
    tab = boot_two_banks(mo, po, fn, B, seed)
    rules = []
    for est, orig_ in (("A change", "A lead, unfiltered"), ("B change", "B gap, unfiltered")):
        r = tab.loc[est]
        rules.append({"estimand": est, "original": tab.loc[orig_, "estimate"], "estimate": r["estimate"],
                      "lo": r["lo"], "hi": r["hi"],
                      "reading": read_rule(r["estimate"], r["lo"], r["hi"], tab.loc[orig_, "estimate"])})
    return tab, pd.DataFrame(rules).set_index("estimand"), corpus_control(mo, mf, po, pf, seed)


def corpus_control(mo, mf, po, pf, seed: int) -> pd.DataFrame:
    """Cross-fitted AUROC of a head separating provenance non-speech (VoxConverse) from main-bank
    non-speech (ESC-50), per duration, without and with the filter. Descriptive."""
    rows = {}
    for tag, mk, pk in (("unfiltered", "main_o", "prov_o"), ("filtered", "main_f", "prov_f")):
        m, xm = load_bank(mk)
        p, xp = load_bank(pk)
        km, kp = (m["label"] == 0).to_numpy(), (p["label"] == 0).to_numpy()
        meta = pd.concat([m.loc[km, ["rec_id", "duration_ms"]].assign(corpus=0),
                          p.loc[kp, ["rec_id", "duration_ms"]].assign(corpus=1)], ignore_index=True)
        sc, _ = L2.crossfit_scores(np.concatenate([xm[km], xp[kp]]), meta["corpus"].to_numpy(int),
                                   meta["rec_id"].astype(str).to_numpy(), n_splits=5, seed=seed, Cs=CS)
        rows[tag] = {d: 100 * L2.auc_w(meta["corpus"].to_numpy()[meta["duration_ms"].to_numpy() == d],
                                       sc[meta["duration_ms"].to_numpy() == d]) for d in DUR}
    return pd.DataFrame(rows).T.rename_axis("corpus AUROC by duration (ms)")


def w2(B: int, seed: int) -> pd.DataFrame:
    """In-bank lead, main bank minus provenance bank, all crops. Main-bank heads with full
    supervision (the cached L2_cv on every axis, and the duration-only fold heads) and with the
    supervision of the provenance heads (the 10 draws of step I, averaged inside each replicate)."""
    main, prov, h = heads("o", seed)
    ids = main["variant_id"].astype(str)
    z = np.load(WORK / "acc" / "base__h_final__cv.npz")
    full_all = pd.Series(z["score"], index=z["variant_id"]).reindex(ids).to_numpy()
    draws = []
    for r in range(10):
        d = np.load(WORK / "acc" / "matched_ref" / f"draw{r}_n182_seed42.npz")
        draws.append(pd.Series(d["score_1"], index=d["variant_id"]).reindex(ids).to_numpy())
    pc = pd.read_csv(WORK / "runs" / "prov-v3" / "provcv_scores.csv.gz").set_index("variant_id")
    incv = pc["base h_final incv"].reindex(prov["variant_id"].astype(str)).to_numpy()
    if np.isnan(full_all).any() or any(np.isnan(d).any() for d in draws) or np.isnan(incv).any():
        raise RuntimeError("scores missing for some items")
    s_lm, s_lp = main["s_lora"].to_numpy(), prov["s_lora"].to_numpy()

    def fn(wm, wp):
        lora_m = mean_auc(main, s_lm, wm)
        lead_p = mean_auc(prov, incv, wp) - mean_auc(prov, s_lp, wp)
        lead_all = mean_auc(main, full_all, wm) - lora_m
        lead_dur = mean_auc(main, h["cv"], wm) - lora_m
        lead_182 = float(np.mean([mean_auc(main, d, wm) for d in draws])) - lora_m
        return {"provenance in-bank lead": lead_p,
                "main lead, every axis": lead_all, "main lead, duration only": lead_dur,
                "main lead, 182 excerpts": lead_182,
                "main minus provenance, every axis": lead_all - lead_p,
                "main minus provenance, duration only": lead_dur - lead_p,
                "main minus provenance, 182 excerpts": lead_182 - lead_p}
    return boot_two_banks(main, prov, fn, B, seed)


# ---------------------------------------------------------------------------- provenance sensitivities

def _prov_scores():
    """Provenance items (duration axis, sorted) with LoRA L0, the in-bank base head and the
    unfiltered fold heads, plus the main bank with its fold heads."""
    main, prov, h = heads("o")
    pc = pd.read_csv(WORK / "runs" / "prov-v3" / "provcv_scores.csv.gz").set_index("variant_id")
    prov = prov.copy()
    prov["incv"] = pc["base h_final incv"].reindex(prov["variant_id"].astype(str)).to_numpy()
    return main, prov, h


def _prov_table(main, prov, h, keep: np.ndarray, B: int, seed: int) -> pd.DataFrame:
    p = prov[keep].reset_index(drop=True)
    hp = {**h, "prov": h["prov"][:, keep]}

    def fn(wm, wp):
        lora = mean_auc(p, p["s_lora"].to_numpy(), wp)
        return {"LoRA L0, provenance": lora,
                "LoRA L0 at 1000 ms, provenance": mean_auc(p, p["s_lora"].to_numpy(), wp, durs=(1000,)),
                "in-bank lead, provenance": mean_auc(p, p["incv"].to_numpy(), wp) - lora,
                "B gap of the same heads": gap_terms(main, p, hp, wm, wp)[1]}
    return boot_two_banks(main, p, fn, B, seed)


def negatives(B: int, seed: int, thresholds=(0.5, 0.3)) -> pd.DataFrame:
    """Provenance results without the recordings whose non-speech window an AudioSet tagger marks
    as speech or singing (tools/screen_negatives.py), dropping the speech window of the same
    recording too. Thresholds fixed before looking at the effect. Heads are not refitted."""
    main, prov, h = _prov_scores()
    scr = pd.read_csv(WORK / "negatives" / "screen.csv")
    ns = scr[scr["ground_truth"] == "NONSPEECH"]
    rec_of = prov.drop_duplicates("base_id").set_index("base_id")["rec_id"]
    out = []
    for thr in (None,) + tuple(thresholds):
        flagged = set() if thr is None else set(
            rec_of.reindex(ns.loc[(ns["p_speech"] >= thr) | (ns["p_singing"] >= thr), "base_id"]).dropna())
        keep = ~prov["rec_id"].isin(flagged).to_numpy()
        t = _prov_table(main, prov, h, keep, B, seed)
        t["negatives dropped"] = "none" if thr is None else f"p >= {thr} ({len(flagged)} recordings)"
        out.append(t.reset_index())
    return pd.concat(out).set_index(["negatives dropped", "quantity"])


def negatives_listened(B: int, seed: int) -> pd.DataFrame:
    """The same results with the labels of a blind listener (tools/blind_listening.py), dropping each
    recording whose non-speech window was heard as speech, with its speech window too. Non-verbal
    vocal sounds stay, as ESC-50 vocal classes are non-speech in the main bank."""
    main, prov, h = _prov_scores()
    lab = pd.read_csv(WORK / "negatives" / "listening.csv")
    rec_of = prov.drop_duplicates("base_id").set_index("base_id")["rec_id"]
    out = []
    for name, labels in (("none", ()), ("heard as clear speech", ("voz",)),
                         ("heard as clear or possible speech", ("voz", "voz_posible")),
                         ("all but those heard as no voice", ("voz", "voz_posible", "vocal_no_verbal"))):
        flagged = set(rec_of.reindex(lab.loc[lab["label"].isin(labels), "base_id"]).dropna())
        t = _prov_table(main, prov, h, ~prov["rec_id"].isin(flagged).to_numpy(), B, seed)
        t["recordings dropped"] = f"{name} ({len(flagged)})"
        out.append(t.reset_index())
    return pd.concat(out).set_index(["recordings dropped", "quantity"])


def quiet_symmetric(B: int, seed: int) -> pd.DataFrame:
    """The provenance in-bank contrast with the quiet-crop exclusions applied to the provenance
    crops too (levels from the bank metadata), next to the matched references of step I."""
    main, prov, h = _prov_scores()
    meta = pd.read_csv(WORK / "manifests" / "prov_bank_v3.metadata.csv.gz")
    meta["variant_id"] = meta["audio_path"].astype(str).str.rsplit(".", n=1).str[0]
    rel = L.relative_level_db(meta).set_index("variant_id")["rel_db"].reindex(prov["variant_id"].astype(str))
    dbfs = (20 * np.log10(meta.set_index("variant_id")["segment_rms"])).reindex(prov["variant_id"].astype(str))
    if rel.isna().any() or dbfs.isna().any():
        raise RuntimeError("provenance crops without a level")
    out = []
    for name, keep in (("no exclusion", np.ones(len(prov), bool)),
                       ("more than 20 dB below its clip", ~(rel < -20).to_numpy()),
                       ("below -30 dBFS", (dbfs >= -30).to_numpy())):
        t = _prov_table(main, prov, h, keep, B, seed)
        t["exclusion"] = f"{name} ({int((~keep).sum())} crops out)"
        out.append(t.reset_index())
    return pd.concat(out).set_index(["exclusion", "quantity"])


# ---------------------------------------------------------------------------- review of draft v5 (23 Oct)

def _draws(n: int, ids) -> list:
    return [pd.Series(d["score_1"], index=d["variant_id"]).reindex(ids).to_numpy()
            for d in (np.load(WORK / "acc" / "matched_ref" / f"draw{r}_n{n}_seed42.npz") for r in range(10))]


def _clean_prov():
    """Clean provenance items with the heads refitted inside them, the transferred head and LoRA L0."""
    clean = WORK / "runs" / "prov-v3-clean"
    pc = pd.read_csv(clean / "provcv_scores.csv.gz")
    pb = pd.read_csv(clean / "prov_scores_base.csv.gz").set_index("variant_id").reindex(pc["variant_id"])
    pl = pd.read_csv(clean / "prov_scores_lora.csv.gz").set_index("variant_id").reindex(pc["variant_id"])
    prov = pc[["variant_id", "base_id", "rec_id", "label", "duration_ms"]].copy()
    return prov, {"incv": pc["base h_final incv"].to_numpy(), "transferred": pb["L2 testfit__score"].to_numpy(),
                  "lora": pl["L0__score"].to_numpy()}


def clean_joint(B: int, seed: int) -> pd.DataFrame:
    """N3 of the review of draft v5. In-bank lead of main-bank heads with 122 excerpts per fold (mean of
    10 draws inside each replicate) minus that of heads refitted inside the clean provenance bank,
    over all durations and at 20 and 40 ms, with both banks resampled. Absolute AUROCs beside it,
    and the spread of the main lead across the fixed training draws."""
    main, _, _ = heads("o", seed)
    ids = main["variant_id"].astype(str)
    draws = _draws(122, ids)
    prov, S = _clean_prov()
    s_lm = main["s_lora"].to_numpy()
    out = {}
    for span, durs in (("all durations", DUR), ("20 and 40 ms", (20, 40))):
        def fn(wm, wp, durs=durs):
            lm, lp = mean_auc(main, s_lm, wm, durs=durs), mean_auc(prov, S["lora"], wp, durs=durs)
            hm = float(np.mean([mean_auc(main, x, wm, durs=durs) for x in draws]))
            hp = mean_auc(prov, S["incv"], wp, durs=durs)
            return {"main head, 122": hm, "main LoRA L0": lm, "main lead": hm - lm,
                    "clean provenance head": hp, "clean provenance LoRA L0": lp, "clean provenance lead": hp - lp,
                    "main lead minus provenance lead": (hm - lm) - (hp - lp)}
        t = boot_two_banks(main, prov, fn, B, seed)
        per_draw = [100 * (mean_auc(main, x, durs=durs) - mean_auc(main, s_lm, durs=durs)) for x in draws]
        t["draws min"], t["draws max"] = np.nan, np.nan
        t.loc["main lead", ["draws min", "draws max"]] = min(per_draw), max(per_draw)
        out[span] = t
    return pd.concat(out, names=["span"])


def split_extra(B: int, seed: int) -> pd.DataFrame:
    """N1 of the review of draft v5. For the base, the LoRA and each restricted LoRA, mean native BA, AUROC
    and best-threshold BA (diagnostic, not a readout) per axis, and the encoder's effect given the
    base or the adapted language model (encoder only minus base, full LoRA minus language model only)."""
    metas = {run: L.load_store_features(WORK / "runs" / run / "lora" / "eval")[0].set_index("variant_id")["s_primary"]
             for run in SPLITS}
    ids = sorted(set.intersection(*[set(s.index) for s in metas.values()]))
    base = L.load_store_features(STORES["main_o"][0])[0].set_index("variant_id")
    lora = L.load_store_features(STORES["main_o"][1])[0].set_index("variant_id")
    meta = base.loc[ids, ["base_id", "rec_id", "label", "axis", "condition"]].reset_index()
    S = {"base": base.loc[ids, "s_primary"].to_numpy(), "lora": lora.loc[ids, "s_primary"].to_numpy(),
         **{run: metas[run].loc[ids].to_numpy() for run in SPLITS}}
    y = meta["label"].to_numpy()
    groups = {ax: [np.flatnonzero((meta["axis"] == ax).to_numpy() & (meta["condition"] == c).to_numpy())
                   for c in sorted(meta.loc[meta["axis"] == ax, "condition"].unique())] for ax in ("duration", "snr")}

    def fn(w):
        out = {}
        for ax, idxs in groups.items():
            for name, s in S.items():
                ww = [None if w is None else w[i] for i in idxs]
                out[f"{ax} {name} BA"] = np.mean([L2.ba_w(y[i], s[i] > 0, wi) for i, wi in zip(idxs, ww)])
                out[f"{ax} {name} AUROC"] = np.mean([L2.auc_w(y[i], s[i], wi) for i, wi in zip(idxs, ww)])
                out[f"{ax} {name} best-threshold BA"] = np.mean([L2.oracle_ba_w(y[i], s[i], wi)
                                                                 for i, wi in zip(idxs, ww)])
            for met in ("AUROC", "BA", "best-threshold BA"):
                out[f"{ax} encoder effect given base LM, {met}"] = out[f"{ax} split-audio {met}"] - out[f"{ax} base {met}"]
                out[f"{ax} encoder effect given adapted LM, {met}"] = out[f"{ax} lora {met}"] - out[f"{ax} split-llm {met}"]
        return out
    point = fn(None)
    bt, rng = Boot(meta), np.random.default_rng(seed)
    reps = {k: [] for k in point}
    for _ in range(B):
        r = fn(bt.draw(rng))
        for k in point:
            reps[k].append(r[k])
    return summarise(point, reps)


I3_RUNS = ("split-audio-llm-lower", "split-audio-llm-upper")


def i3(B: int, seed: int) -> tuple:
    """I3 (docs/exploratory_log.md, "Reading rule for I3, written on 23 October before extraction").
    Operating-point loss P(v), best-threshold BA minus native BA, for the eight combinations of
    encoder, lower and upper language-model updates.
    D_lower = P(encoder and upper) - P(LoRA), D_upper = P(encoder and lower) - P(LoRA), against
    G = P(base) - P(LoRA). A half is necessary if the lower limit of its D exceeds G/2 and not
    necessary if the upper limit is below G/4."""
    runs = SPLITS + I3_RUNS
    metas = {run: L.load_store_features(WORK / "runs" / run / "lora" / "eval")[0].set_index("variant_id")["s_primary"]
             for run in runs}
    ids = sorted(set.intersection(*[set(s.index) for s in metas.values()]))
    if any(len(s) != len(ids) for s in metas.values()):
        raise RuntimeError("the restricted runs do not hold the same items")
    base = L.load_store_features(STORES["main_o"][0])[0].set_index("variant_id")
    lora = L.load_store_features(STORES["main_o"][1])[0].set_index("variant_id")
    meta = base.loc[ids, ["base_id", "rec_id", "label", "axis", "condition"]].reset_index()
    S = {"base": base.loc[ids, "s_primary"].to_numpy(), "lora": lora.loc[ids, "s_primary"].to_numpy(),
         **{run: metas[run].loc[ids].to_numpy() for run in runs}}
    y = meta["label"].to_numpy()
    groups = {ax: [np.flatnonzero((meta["axis"] == ax).to_numpy() & (meta["condition"] == c).to_numpy())
                   for c in sorted(meta.loc[meta["axis"] == ax, "condition"].unique())] for ax in ("duration", "snr")}
    if sum(len(i) for idx in groups.values() for i in idx) != len(meta):
        raise RuntimeError("condition groups do not cover the items")

    def fn(w):
        out = {}
        for ax, idxs in groups.items():
            ww = [None if w is None else w[i] for i in idxs]
            for name, s in S.items():
                ba = np.mean([L2.ba_w(y[i], s[i] > 0, wi) for i, wi in zip(idxs, ww)])
                best = np.mean([L2.oracle_ba_w(y[i], s[i], wi) for i, wi in zip(idxs, ww)])
                out[f"{ax} {name} AUROC"] = np.mean([L2.auc_w(y[i], s[i], wi) for i, wi in zip(idxs, ww)])
                out[f"{ax} {name} BA"], out[f"{ax} {name} best-threshold BA"] = ba, best
                out[f"{ax} {name} P"] = best - ba
            out[f"{ax} G"] = out[f"{ax} base P"] - out[f"{ax} lora P"]
            out[f"{ax} D_lower"] = out[f"{ax} split-audio-llm-upper P"] - out[f"{ax} lora P"]
            out[f"{ax} D_upper"] = out[f"{ax} split-audio-llm-lower P"] - out[f"{ax} lora P"]
            out[f"{ax} D_lower - G/2"] = out[f"{ax} D_lower"] - out[f"{ax} G"] / 2
            out[f"{ax} D_upper - G/4"] = out[f"{ax} D_upper"] - out[f"{ax} G"] / 4
            for half, keep in (("lower", "split-audio-llm-upper"), ("upper", "split-audio-llm-lower")):
                for met in ("AUROC", "BA"):
                    out[f"{ax} removing {half}, {met} change"] = out[f"{ax} {keep} {met}"] - out[f"{ax} lora {met}"]
        return out
    point = fn(None)
    bt, rng = Boot(meta), np.random.default_rng(seed)
    reps = {k: [] for k in point}
    for _ in range(B):
        r = fn(bt.draw(rng))
        for k in point:
            reps[k].append(r[k])
    tab = summarise(point, reps)
    reading = []
    for ax in groups:
        g = tab.loc[f"{ax} G", "estimate"]
        for half in ("lower", "upper"):
            r = tab.loc[f"{ax} D_{half}"]
            verdict = ("necessary" if r["lo"] > g / 2 else "not necessary" if r["hi"] < g / 4 else "inconclusive")
            reading.append({"axis": ax, "half removed": half, "D": r["estimate"], "lo": r["lo"], "hi": r["hi"],
                            "G": g, "reading": verdict})
    return tab, pd.DataFrame(reading).set_index(["axis", "half removed"])


def cross_filter(seed: int, boot_B: int = 0):
    """N4 of the review of draft v5. In-bank lead over LoRA L0 on the main bank for heads fitted on one
    version of the audio (unfiltered or band-passed) and scored on either version, with the same
    folds and the inner choice of C of lib_v2.crossfit_scores. LoRA L0 comes from the test version."""
    from sklearn.model_selection import StratifiedGroupKFold
    mo, xo = load_bank("main_o")
    mf, xf = load_bank("main_f")
    if not (mo["variant_id"].astype(str).to_numpy() == mf["variant_id"].astype(str).to_numpy()).all():
        raise RuntimeError("the two versions do not hold the same items")
    y, g = mo["label"].to_numpy(int), mo["rec_id"].to_numpy()
    X = {"unfiltered": xo, "filtered": xf}
    lora = {"unfiltered": mo["s_lora"].to_numpy(), "filtered": mf["s_lora"].to_numpy()}
    scores = {(tr_v, te_v): np.full(len(y), np.nan) for tr_v in X for te_v in X}
    for f, (tr, te) in enumerate(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed).split(xo, y, g)):
        for tr_v in X:
            model, _ = L2.fit_head(X[tr_v][tr], y[tr], g[tr], Cs=CS, n_splits=4, seed=seed + f + 1)
            for te_v in X:
                scores[(tr_v, te_v)][te] = model.decision_function(X[te_v][te])
    rows = []
    for (tr_v, te_v), s in scores.items():
        rows.append({"head fitted on": tr_v, "scored on": te_v, "head AUROC": 100 * mean_auc(mo, s),
                     "LoRA L0 AUROC": 100 * mean_auc(mo, lora[te_v]),
                     "lead": 100 * (mean_auc(mo, s) - mean_auc(mo, lora[te_v]))})
    table = pd.DataFrame(rows).set_index(["head fitted on", "scored on"])
    if not boot_B:
        return table

    # follow-up of the review of draft v5 (23 Oct): paired intervals for the fixed-head comparison
    def fn(w):
        u, f = scores[("unfiltered", "unfiltered")], scores[("unfiltered", "filtered")]
        head_loss = mean_auc(mo, f, w) - mean_auc(mo, u, w)
        lora_loss = mean_auc(mo, lora["filtered"], w) - mean_auc(mo, lora["unfiltered"], w)
        refit = (mean_auc(mo, scores[("filtered", "filtered")], w) - mean_auc(mo, lora["filtered"], w)
                 - mean_auc(mo, u, w) + mean_auc(mo, lora["unfiltered"], w))
        return {"fixed heads, head AUROC change": head_loss, "LoRA L0 AUROC change": lora_loss,
                "fixed heads, change in lead": head_loss - lora_loss, "refitted heads, change in lead": refit}
    point = fn(None)
    bt, rng = Boot(mo), np.random.default_rng(seed)
    reps = {k: [] for k in point}
    for _ in range(boot_B):
        r = fn(bt.draw(rng))
        for k in point:
            reps[k].append(r[k])
    return table, summarise(point, reps)


# ---------------------------------------------------------------------------- figure data, draft v5

def figure_supervision(B: int, seed: int) -> pd.DataFrame:
    """Per duration, AUROC of heads minus LoRA L0 of the same bank, with pointwise 95% bands.
    Main bank, L2_cv with every excerpt of the fold and with 122 (mean of 10 draws of step I),
    the supervision of the clean provenance heads.
    Provenance bank without the recordings heard as speech, L2_cv refitted inside it and the head
    fitted on the whole main bank. Writes _work/closing/figure_v5_supervision.csv."""
    main, _, _ = heads("o", seed)
    ids = main["variant_id"].astype(str)
    z = np.load(WORK / "acc" / "base__h_final__cv.npz")
    full = pd.Series(z["score"], index=z["variant_id"]).reindex(ids).to_numpy()
    draws = _draws(122, ids)
    clean = WORK / "runs" / "prov-v3-clean"
    pc = pd.read_csv(clean / "provcv_scores.csv.gz")
    pb = pd.read_csv(clean / "prov_scores_base.csv.gz").set_index("variant_id").reindex(pc["variant_id"])
    pl = pd.read_csv(clean / "prov_scores_lora.csv.gz").set_index("variant_id").reindex(pc["variant_id"])
    prov = pc[["variant_id", "base_id", "rec_id", "label", "duration_ms"]].copy()
    S_p = {"incv": pc["base h_final incv"].to_numpy(), "transferred": pb["L2 testfit__score"].to_numpy(),
           "lora": pl["L0__score"].to_numpy()}
    if np.isnan(full).any() or any(np.isnan(v).any() for v in S_p.values()):
        raise RuntimeError("scores missing for some items")
    s_lm = main["s_lora"].to_numpy()

    def fn(wm, wp):
        out = {}
        for d in DUR:
            lm = mean_auc(main, s_lm, wm, durs=(d,))
            lp = mean_auc(prov, S_p["lora"], wp, durs=(d,))
            out[("main, every excerpt", d)] = mean_auc(main, full, wm, durs=(d,)) - lm
            out[("main, 122 excerpts", d)] = float(np.mean([mean_auc(main, x, wm, durs=(d,)) for x in draws])) - lm
            out[("provenance, refitted inside", d)] = mean_auc(prov, S_p["incv"], wp, durs=(d,)) - lp
            out[("provenance, transferred", d)] = mean_auc(prov, S_p["transferred"], wp, durs=(d,)) - lp
        return out
    tab = boot_two_banks(main, prov, fn, B, seed).reset_index()
    tab[["series", "dur"]] = pd.DataFrame(tab["quantity"].tolist(), index=tab.index)
    tab = tab.drop(columns="quantity")[["series", "dur", "estimate", "lo", "hi", "se"]]
    tab.to_csv(WORK / "figure_v5_supervision.csv", index=False)
    return tab


# ---------------------------------------------------------------------------- I2

def split(B: int, seed: int) -> tuple:
    """Shares of the LoRA's gain recovered by each restricted LoRA, on the items of the split runs."""
    metas = {}
    for run in SPLITS:
        m = L.load_store_features(WORK / "runs" / run / "lora" / "eval")[0]
        metas[run] = m.set_index("variant_id")["s_primary"]
    ids = sorted(set.intersection(*[set(s.index) for s in metas.values()]))
    if any(len(s) != len(ids) for s in metas.values()):
        raise RuntimeError("the split runs do not hold the same items")
    base = L.load_store_features(STORES["main_o"][0])[0].set_index("variant_id")
    lora = L.load_store_features(STORES["main_o"][1])[0].set_index("variant_id")
    meta = base.loc[ids, ["base_id", "rec_id", "label", "axis", "condition"]].reset_index()
    S = {"base": base.loc[ids, "s_primary"].to_numpy(), "lora": lora.loc[ids, "s_primary"].to_numpy(),
         **{run: metas[run].loc[ids].to_numpy() for run in SPLITS}}
    y = meta["label"].to_numpy()
    # positions in the whole meta, not in an axis subset
    groups = {ax: [np.flatnonzero((meta["axis"] == ax).to_numpy() & (meta["condition"] == c).to_numpy())
                   for c in sorted(meta.loc[meta["axis"] == ax, "condition"].unique())]
              for ax in ("duration", "snr")}
    if sum(len(i) for idx in groups.values() for i in idx) != len(meta):
        raise RuntimeError("condition groups do not cover the items")

    def axis_means(w):
        out = {}
        for ax, idxs in groups.items():
            for name, s in S.items():
                ww = [None if w is None else w[i] for i in idxs]
                out[(ax, name, "auroc")] = np.mean([L2.auc_w(y[i], s[i], wi) for i, wi in zip(idxs, ww)])
                out[(ax, name, "ba")] = np.mean([L2.ba_w(y[i], s[i] > 0, wi) for i, wi in zip(idxs, ww)])
        return out

    def shares(w):
        a = axis_means(w)
        out = {}
        for ax in groups:
            for met, tag in (("auroc", "R"), ("ba", "C")):
                den = a[(ax, "lora", met)] - a[(ax, "base", met)]
                for run in SPLITS:
                    out[f"{ax} {tag} {run}"] = (a[(ax, run, met)] - a[(ax, "base", met)]) / den
                out[f"{ax} {met} LoRA minus base"] = den
        return out, a
    point, levels = shares(None)
    bt, rng = Boot(meta), np.random.default_rng(seed)
    reps = {k: [] for k in point}
    for _ in range(B):
        r, _ = shares(bt.draw(rng))
        for k in point:
            reps[k].append(r[k])
    rows = []
    for k, v in point.items():
        lo, hi = np.nanpercentile(reps[k], [2.5, 97.5])
        rows.append({"quantity": k, "estimate": v, "lo": lo, "hi": hi})
    tab = pd.DataFrame(rows).set_index("quantity")
    lev = pd.Series({f"{ax} {name} {met}": 100 * v for (ax, name, met), v in levels.items()}).to_frame("mean over the axis")
    reading = []
    for ax in groups:
        for tag in ("R", "C"):
            for a_, b_ in (("split-audio", "split-llm"), ("split-llm-lower", "split-llm-upper")):
                ra, rb = tab.loc[f"{ax} {tag} {a_}"], tab.loc[f"{ax} {tag} {b_}"]
                if ra["lo"] > 0.5 and rb["hi"] < 0.5:
                    r = f"{a_} carries it"
                elif rb["lo"] > 0.5 and ra["hi"] < 0.5:
                    r = f"{b_} carries it"
                elif ra["hi"] < 0.5 and rb["hi"] < 0.5:
                    r = "both parts are needed"
                elif ra["lo"] > 0.5 and rb["lo"] > 0.5:
                    r = "either part suffices"
                else:
                    r = "mixed"
                reading.append({"axis": ax, "share": tag, "pair": f"{a_} / {b_}", "reading": r})
    return tab, lev, pd.DataFrame(reading)


# ---------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("what", choices=["fetch", "orig", "cue", "split", "w2", "negatives", "listened", "quiet", "figure",
                                        "cleanjoint", "splitextra", "crossfilter", "i3"])
    ap.add_argument("run", nargs="?")
    ap.add_argument("--backbones", default="base,lora")
    ap.add_argument("--splits", default="eval", help="fetch only, e.g. eval,train")
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    if a.what == "fetch":
        fetch(a.run, a.backbones.split(","), a.splits.split(","))
    elif a.what == "orig":
        print(f"## I1 unfiltered, AUROC points, B = {a.B}\n")
        print(md(orig(a.B, a.seed), 2), "\n")
    elif a.what == "negatives":
        print(f"## Provenance results without doubtful negatives, AUROC points, B = {a.B}\n")
        print(md(negatives(a.B, a.seed), 2), "\n")
    elif a.what == "cleanjoint":
        print(f"## Clean provenance against main-bank heads with 122 excerpts, AUROC points, B = {a.B}\n")
        print(md(clean_joint(a.B, a.seed), 2))
    elif a.what == "splitextra":
        print(f"## Restricted LoRA, native and best-threshold BA, B = {a.B}\n")
        print(md(split_extra(a.B, a.seed), 2))
    elif a.what == "i3":
        tab, reading = i3(a.B, a.seed)
        print(f"## I3, encoder with one half of the language model, BA points, B = {a.B}\n")
        print(md(tab, 2), "\n")
        print("### Reading rule\n")
        print(md(reading, 2), "\n")
    elif a.what == "crossfilter":
        print("## Heads fitted and scored on unfiltered or band-passed audio, main bank, AUROC points\n")
        table, ci = cross_filter(a.seed, a.B)
        print(md(table, 2), "\n")
        print(md(ci, 2))
    elif a.what == "figure":
        print(md(figure_supervision(a.B, a.seed).set_index(["series", "dur"]), 2))
    elif a.what == "listened":
        print(f"## Provenance results without the negatives a blind listener heard as speech, B = {a.B}\n")
        print(md(negatives_listened(a.B, a.seed), 2), "\n")
    elif a.what == "quiet":
        print(f"## Provenance results with the quiet-crop exclusions on both banks, AUROC points, B = {a.B}\n")
        print(md(quiet_symmetric(a.B, a.seed), 2), "\n")
    elif a.what == "w2":
        print(f"## W2, in-bank lead in the main bank minus the provenance bank, AUROC points, B = {a.B}\n")
        print(md(w2(a.B, a.seed), 2), "\n")
    elif a.what == "cue":
        tab, rules, corpus = cue(a.B, a.seed)
        print(f"## I1, channel-cue intervention, AUROC points, B = {a.B}\n")
        print(md(tab, 2), "\n")
        print("### Reading rule\n")
        print(md(rules, 2), "\n")
        print("### Corpus control, provenance non-speech against main-bank non-speech\n")
        print(md(corpus, 1), "\n")
    else:
        tab, lev, reading = split(a.B, a.seed)
        print(f"## I2, LoRA kept in one part of the model, B = {a.B}\n")
        print(md(lev, 1), "\n")
        print(md(tab, 3), "\n")
        print("### Reading rule\n")
        print(md(reading.set_index(["axis", "share", "pair"]), 3), "\n")


if __name__ == "__main__":
    main()
