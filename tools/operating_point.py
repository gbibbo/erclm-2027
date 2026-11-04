"""Is the dominant failure the operating point? Rules S, R and N of 23 October (afternoon), written
before computing in docs/exploratory_log.md, "Is the dominant failure the operating point?
(23 October, afternoon)". Exploratory.

S  symmetric prompting. The native score averaged over "A) speech, B) non-speech" and
   "A) non-speech, B) speech" (and, secondarily, over the four formats of the prompt control),
   read with the native threshold 0, against batch calibration on the original format.
R  why H3 fails. C chosen by validation that leaves out one SNR level, and the head's BA with its
   native threshold, with batch calibration and with a frozen threshold from the training bank,
   against LoRA L0 under H3's margin.
N  the residual at 20 and 40 ms without the vocal classes of ESC-50 in the main bank.
G1 the training-free recipe on held-out recordings. g1t checks whether its calibration transfers.

    PYTHONPATH=src python tools/operating_point.py s|r|n|g1|g1t [--B 2000]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from closing_analyses import WORK, md  # noqa: E402
import intervention_analysis as I  # noqa: E402
from erclm import lib_v2 as L2  # noqa: E402
from erclm import lib_v3 as L  # noqa: E402

CS = (1e-4, 1e-3, 1e-2, 1e-1)
AXES = ("duration", "snr")
# ESC-50 targets: human non-verbal vocal sounds and animal vocalisations (rule N)
VOCAL = set(L.ESC_HUMAN_VOCAL) | {0, 1, 2, 3, 4, 5, 6, 8, 9, 14}


def _groups(meta: pd.DataFrame) -> dict:
    g = {ax: [np.flatnonzero((meta["axis"] == ax).to_numpy() & (meta["condition"] == c).to_numpy())
              for c in sorted(meta.loc[meta["axis"] == ax, "condition"].unique())] for ax in AXES}
    if sum(len(i) for idx in g.values() for i in idx) != len(meta):
        raise RuntimeError("condition groups do not cover the items")
    return g


def _axis_stats(y, s, idxs, w, thr=None) -> tuple:
    """Mean over the conditions of an axis of AUROC, BA at thr (0 if None) and best-threshold BA."""
    au, ba, best = [], [], []
    for i in idxs:
        wi = None if w is None else w[i]
        t = 0.0 if thr is None else thr[i]
        au.append(L2.auc_w(y[i], s[i], wi))
        ba.append(L2.ba_w(y[i], s[i] > t, wi))
        best.append(L2.oracle_ba_w(y[i], s[i], wi))
    return float(np.mean(au)), float(np.mean(ba)), float(np.mean(best))


def _boot(meta, fn, B, seed) -> pd.DataFrame:
    point = fn(None)
    bt, rng = I.Boot(meta), np.random.default_rng(seed)
    reps = {k: [] for k in point}
    for _ in range(B):
        r = fn(bt.draw(rng))
        for k in point:
            reps[k].append(r[k])
    return I.summarise(point, reps)


# ---------------------------------------------------------------------------- S

def symmetric(B: int, seed: int) -> pd.DataFrame:
    runs = {"ab_speech_second": "prompt-ab_speech_second", "ba_speech_first": "prompt-ba_speech_first",
            "ba_speech_second": "prompt-ba_speech_second"}
    out = {}
    for bb in ("base", "lora"):
        S = {k: L.load_store_features(WORK / "runs" / r / bb / "eval")[0].set_index("variant_id")["s_primary"]
             for k, r in runs.items()}
        ids = sorted(set.intersection(*[set(v.index) for v in S.values()]))
        full = L.load_store_features(I.STORES["main_o"][0 if bb == "base" else 1])[0].set_index("variant_id")
        S["ab"] = full["s_primary"]
        meta = full.loc[ids, ["base_id", "rec_id", "label", "axis", "condition"]].reset_index()
        sc = {k: v.reindex(ids).to_numpy(float) for k, v in S.items()}
        if any(np.isnan(v).any() for v in sc.values()):
            raise RuntimeError("missing scores")
        sc["sym2"] = (sc["ab"] + sc["ba_speech_second"]) / 2
        sc["sym4"] = (sc["ab"] + sc["ab_speech_second"] + sc["ba_speech_first"] + sc["ba_speech_second"]) / 4
        lora_ab = L.load_store_features(I.STORES["main_o"][1])[0].set_index("variant_id")["s_primary"].reindex(ids).to_numpy()
        y, G = meta["label"].to_numpy(), _groups(meta)

        def fn(w):
            r = {}
            for ax, idxs in G.items():
                bc = L2.apply_rule(meta, sc["ab"], "bc_cond", w)
                a_ab, b_ab, best_ab = _axis_stats(y, sc["ab"], idxs, w)
                _, b_bc, _ = _axis_stats(y, sc["ab"], idxs, w, thr=bc)
                a2, b2, best2 = _axis_stats(y, sc["sym2"], idxs, w)
                a4, b4, best4 = _axis_stats(y, sc["sym4"], idxs, w)
                _, b_lora, _ = _axis_stats(y, lora_ab, idxs, w)
                r.update({f"{ax} AUROC original": a_ab, f"{ax} AUROC sym2": a2, f"{ax} AUROC sym4": a4,
                          f"{ax} BA original L0": b_ab, f"{ax} BA original L1": b_bc,
                          f"{ax} BA sym2": b2, f"{ax} BA sym4": b4, f"{ax} best BA sym2": best2,
                          f"{ax} operating-point loss sym2": best2 - b2,
                          f"{ax} BA sym2 - original L1": b2 - b_bc, f"{ax} BA sym2 - original L0": b2 - b_ab,
                          f"{ax} BA sym2 - LoRA L0 (original format)": b2 - b_lora,
                          f"{ax} AUROC sym2 - original": a2 - a_ab, f"{ax} BA sym4 - original L1": b4 - b_bc})
            return r
        out[bb] = _boot(meta, fn, B, seed)
    return pd.concat(out, names=["backbone"])


# ---------------------------------------------------------------------------- R

def _cv_select(xtr, mtr, scheme: str, seed: int) -> tuple:
    """C chosen by validation that leaves out one SNR level (loso) or validates on +20 dB after
    training up to +10 dB (toward_clean), with validation recordings apart from training ones."""
    from sklearn.model_selection import GroupKFold
    y, g, snr = mtr["label"].to_numpy(int), mtr["rec_id"].to_numpy(), mtr["snr_db"].to_numpy(float)
    rec_folds = list(GroupKFold(n_splits=4).split(xtr, y, g))
    levels = sorted(np.unique(snr)) if scheme == "loso" else [20.0]
    scores = {}
    for C in CS:
        au = []
        for k in levels:
            for tr, va in rec_folds:
                trm = tr[(snr[tr] != k) if scheme == "loso" else (snr[tr] <= 10)]
                vam = va[snr[va] == k]
                model = L2._head(C).fit(xtr[trm], y[trm])
                au.append(L2.auc_w(y[vam], model.decision_function(xtr[vam])))
        scores[C] = float(np.mean(au))
    return max(scores, key=scores.get), scores


def why_h3(B: int, seed: int) -> tuple:
    mtr, xtr = L.load_store_features(WORK / "features" / "base" / "train", ["h_final"])
    xtr, ytr, gtr = xtr["h_final"], mtr["label"].to_numpy(int), mtr["rec_id"].to_numpy()
    sel = {}
    for scheme in ("loso", "toward_clean"):
        sel[scheme] = _cv_select(xtr, mtr, scheme, seed)
    mte, xte = L.load_store_features(WORK / "features" / "base" / "eval", ["h_final"])
    keep = mte["axis"].isin(AXES).to_numpy()
    mte, xte = mte[keep].reset_index(drop=True), xte["h_final"][keep]
    lora = L.load_store_features(I.STORES["main_o"][1])[0].set_index("variant_id")["s_primary"]
    s_lora = lora.reindex(mte["variant_id"]).to_numpy(float)
    heads = {"C=0.01 (registered)": 1e-2, f"C={sel['loso'][0]} (leave one SNR out)": sel["loso"][0]}
    if sel["toward_clean"][0] != sel["loso"][0]:
        heads[f"C={sel['toward_clean'][0]} (toward clean)"] = sel["toward_clean"][0]
    S, FROZEN = {}, {}
    from sklearn.model_selection import StratifiedGroupKFold
    for name, C in heads.items():
        S[name] = L2._head(C).fit(xtr, ytr).decision_function(xte)
        cv = np.full(len(ytr), np.nan)       # cross-fitted training scores for the frozen threshold
        for tr, va in StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed).split(xtr, ytr, gtr):
            cv[va] = L2._head(C).fit(xtr[tr], ytr[tr]).decision_function(xtr[va])
        FROZEN[name] = L.fit_threshold(cv, ytr)
    y, G = mte["label"].to_numpy(), _groups(mte)

    def fn(w):
        r = {}
        for ax, idxs in G.items():
            _, b_lora, best_lora = _axis_stats(y, s_lora, idxs, w)
            r[f"{ax} LoRA L0 BA"] = b_lora
            for name, s in S.items():
                a, b0, best = _axis_stats(y, s, idxs, w)
                _, bbc, _ = _axis_stats(y, s, idxs, w, thr=L2.apply_rule(mte, s, "bc_cond", w))
                _, bfz, _ = _axis_stats(y, s, idxs, w, thr=np.full(len(s), FROZEN[name]))
                r.update({f"{ax} {name} AUROC": a, f"{ax} {name} best BA": best,
                          f"{ax} {name} BA native - LoRA": b0 - b_lora,
                          f"{ax} {name} BA batch-calibrated - LoRA": bbc - b_lora,
                          f"{ax} {name} BA frozen threshold - LoRA": bfz - b_lora})
        return r
    return _boot(mte, fn, B, seed), sel, FROZEN


# ---------------------------------------------------------------------------- N

def residual(B: int, seed: int) -> pd.DataFrame:
    main, _, _ = I.heads("o", seed)
    vocal = main["base_id"].astype(str).map(lambda b: b.startswith("esc50") and L.esc_class(b) in VOCAL)
    nonvocal = ~((main["label"] == 0) & vocal).to_numpy()
    prov, Sp = I._clean_prov()
    lab = pd.read_csv(WORK / "negatives" / "listening.csv")
    rec_of = prov.drop_duplicates("base_id").set_index("base_id")["rec_id"]
    vocal_recs = set(rec_of.reindex(lab.loc[lab["label"] == "vocal_no_verbal", "base_id"]).dropna())
    out = {}
    for tag, mkeep, pkeep in (("all negatives", np.ones(len(main), bool), np.ones(len(prov), bool)),
                              ("main without ESC-50 vocal classes", nonvocal, np.ones(len(prov), bool)),
                              ("both without vocal negatives", nonvocal, ~prov["rec_id"].isin(vocal_recs).to_numpy())):
        m = main[mkeep].reset_index(drop=True)
        p = prov[pkeep].reset_index(drop=True)
        ids = m["variant_id"].astype(str)
        draws = I._draws(122, ids)
        s_lm = m["s_lora"].to_numpy()
        sp = {k: v[pkeep] for k, v in Sp.items()}

        def fn(wm, wp):
            r = {}
            for span, durs in (("20 and 40 ms", (20, 40)), ("all durations", I.DUR)):
                lm = I.mean_auc(m, s_lm, wm, durs=durs)
                hm = float(np.mean([I.mean_auc(m, x, wm, durs=durs) for x in draws]))
                lp = I.mean_auc(p, sp["lora"], wp, durs=durs)
                hp = I.mean_auc(p, sp["incv"], wp, durs=durs)
                r.update({f"{span} main LoRA L0": lm, f"{span} main lead": hm - lm,
                          f"{span} provenance LoRA L0": lp, f"{span} provenance lead": hp - lp,
                          f"{span} residual": (hm - lm) - (hp - lp)})
            return r
        t = I.boot_two_banks(m, p, fn, B, seed)
        t.index = [f"{tag} | {q}" for q in t.index]
        out[tag] = t
        print(f"{tag}: main {len(m)} items, provenance {p['rec_id'].nunique()} recordings", flush=True)
    return pd.concat(out.values())


# ---------------------------------------------------------------------------- G1

def g1(B: int, seed: int) -> tuple:
    """Rule G1 (docs/exploratory_log.md, "Rule for G1, written on 23 October before extraction"). The
    symmetric score on clips from recordings outside the prompt control, with batch calibration per
    condition and with a frozen threshold fitted on the training bank, against LoRA L0. The
    registered batch is the evaluated held-out items (29% speech). A declared sensitivity
    calibrates on the whole condition batch of the bank (970 clips, balanced by design), still
    label-free, with the threshold fixed across replicates."""
    g1dir = WORK / "runs" / "prompt-ba_speech_second-full" / "base"
    ab = L.load_store_features(I.STORES["main_o"][0])[0]
    ab = ab[ab["axis"].isin(AXES)].set_index("variant_id")
    ba2 = L.load_store_features(g1dir / "eval")[0].set_index("variant_id")["s_primary"]
    lora = L.load_store_features(I.STORES["main_o"][1])[0].set_index("variant_id")["s_primary"]
    ids = sorted(set(ab.index) & set(ba2.index))
    meta_all = ab.loc[ids, ["base_id", "rec_id", "label", "axis", "condition", "duration_ms"]].reset_index()
    s_ab, s_ba2 = ab.loc[ids, "s_primary"].to_numpy(float), ba2.reindex(ids).to_numpy(float)
    sym_all = (s_ab + s_ba2) / 2
    pilot = L.load_store_features(WORK / "runs" / "prompt-ba_speech_second" / "base" / "eval")[0]
    held = ~meta_all["rec_id"].isin(set(pilot["rec_id"])).to_numpy()
    # label-free thresholds from the whole balanced condition batch (sensitivity)
    thr_full = L2.apply_rule(meta_all, sym_all, "bc_cond")
    thr_full_ab = L2.apply_rule(meta_all, s_ab, "bc_cond")
    # frozen thresholds on the training bank
    tr_ab = L.load_store_features(WORK / "features" / "base" / "train")[0].set_index("variant_id")
    tr_ba2 = L.load_store_features(g1dir / "train")[0].set_index("variant_id")["s_primary"]
    tid = sorted(set(tr_ab.index) & set(tr_ba2.index))
    ytr = tr_ab.loc[tid, "label"].to_numpy()
    sym_tr = (tr_ab.loc[tid, "s_primary"].to_numpy(float) + tr_ba2.reindex(tid).to_numpy(float)) / 2
    frozen_sym = L.fit_threshold(sym_tr, ytr)
    frozen_ab = L.fit_threshold(tr_ab.loc[tid, "s_primary"].to_numpy(float), ytr)
    meta = meta_all[held].reset_index(drop=True)
    sym, sab, slo = sym_all[held], s_ab[held], lora.reindex(meta["variant_id"]).to_numpy(float)
    tf_sym, tf_ab = thr_full[held], thr_full_ab[held]
    if np.isnan(slo).any() or np.isnan(sym).any():
        raise RuntimeError("missing scores")
    y, G = meta["label"].to_numpy(), _groups(meta)

    def fn(w):
        r = {}
        for ax, idxs in G.items():
            a_sym, b0_sym, best_sym = _axis_stats(y, sym, idxs, w)
            a_ab, b0_ab, best_ab = _axis_stats(y, sab, idxs, w)
            _, b_lo, _ = _axis_stats(y, slo, idxs, w)
            _, bbc_sym, _ = _axis_stats(y, sym, idxs, w, thr=L2.apply_rule(meta, sym, "bc_cond", w))
            _, bbc_ab, _ = _axis_stats(y, sab, idxs, w, thr=L2.apply_rule(meta, sab, "bc_cond", w))
            _, bfz_sym, _ = _axis_stats(y, sym, idxs, w, thr=np.full(len(sym), frozen_sym))
            _, bfz_ab, _ = _axis_stats(y, sab, idxs, w, thr=np.full(len(sab), frozen_ab))
            _, bfull_sym, _ = _axis_stats(y, sym, idxs, w, thr=tf_sym)
            _, bfull_ab, _ = _axis_stats(y, sab, idxs, w, thr=tf_ab)
            r.update({f"{ax} LoRA L0 BA": b_lo, f"{ax} AUROC original": a_ab, f"{ax} AUROC sym": a_sym,
                      f"{ax} best BA sym": best_sym, f"{ax} BA sym L1": bbc_sym, f"{ax} BA original L1": bbc_ab,
                      f"{ax} BA sym frozen": bfz_sym, f"{ax} BA original frozen": bfz_ab,
                      f"{ax} BA sym L1 - LoRA (primary on duration)": bbc_sym - b_lo,
                      f"{ax} BA sym frozen - LoRA": bfz_sym - b_lo,
                      f"{ax} BA sym L1 - original L1": bbc_sym - bbc_ab,
                      f"{ax} BA sym frozen - original frozen": bfz_sym - bfz_ab,
                      f"{ax} AUROC sym - original": a_sym - a_ab,
                      f"{ax} BA sym L1 on the full batch - LoRA (sensitivity)": bfull_sym - b_lo,
                      f"{ax} BA original L1 on the full batch - LoRA (sensitivity)": bfull_ab - b_lo})
        return r
    tab = _boot(meta, fn, B, seed)
    info = {"held-out clips": int(meta["base_id"].nunique()),
            "speech share": float((meta.drop_duplicates("base_id")["label"] == 1).mean()),
            "frozen threshold sym": frozen_sym, "frozen threshold original": frozen_ab}
    return tab, info


def _bc_value(s) -> float:
    p = 1 / (1 + np.exp(-np.asarray(s, float)))
    return float(np.log(p.mean() / (1 - p.mean())))


def g1_transfer(B: int, seed: int, n_cal: int = 150, reps: int = 200) -> tuple:
    """Post hoc, 25 October. Does the G1 recipe need the test batch? Per-condition thresholds are
    computed once on separate unlabeled clips from the discovery recordings and applied clip by clip
    to the 470 held-out clips. Sources are the 300 clips of the prompt control (50% speech), all 500
    clips of those recordings (70% speech) and subsamples of n_cal of the 500 with a set speech share.
    Labels only build the subsamples. The thresholds never see them."""
    g1dir = WORK / "runs" / "prompt-ba_speech_second-full" / "base"
    ab = L.load_store_features(I.STORES["main_o"][0])[0]
    ab = ab[ab["axis"].isin(AXES)].set_index("variant_id")
    ba2 = L.load_store_features(g1dir / "eval")[0].set_index("variant_id")["s_primary"]
    lora = L.load_store_features(I.STORES["main_o"][1])[0].set_index("variant_id")["s_primary"]
    ids = sorted(set(ab.index) & set(ba2.index))
    meta_all = ab.loc[ids, ["base_id", "rec_id", "label", "axis", "condition"]].reset_index()
    sym_all = (ab.loc[ids, "s_primary"].to_numpy(float) + ba2.reindex(ids).to_numpy(float)) / 2
    pilot = L.load_store_features(WORK / "runs" / "prompt-ba_speech_second" / "base" / "eval")[0]
    held = ~meta_all["rec_id"].isin(set(pilot["rec_id"])).to_numpy()
    in_pilot = meta_all["base_id"].isin(set(pilot["base_id"])).to_numpy()
    cond_all = meta_all["condition"].to_numpy()
    conds = sorted(set(cond_all))

    def thresholds(mask) -> dict:
        return {c: _bc_value(sym_all[mask & (cond_all == c)]) for c in conds}

    meta = meta_all[held].reset_index(drop=True)
    sym, slo = sym_all[held], lora.reindex(meta["variant_id"]).to_numpy(float)
    y, G = meta["label"].to_numpy(), _groups(meta)
    sources = {"prompt-control clips (300, 50% speech)": in_pilot & ~held,
               "all clips of the discovery recordings (500, 70% speech)": ~held}
    thr = {k: meta["condition"].map(thresholds(m)).to_numpy(float) for k, m in sources.items()}
    if any(np.isnan(t).any() for t in thr.values()):
        raise RuntimeError("a condition has no calibration clips")

    def fn(w):
        r = {}
        for ax, idxs in G.items():
            _, b_lo, _ = _axis_stats(y, slo, idxs, w)
            _, b_own, _ = _axis_stats(y, sym, idxs, w, thr=L2.apply_rule(meta, sym, "bc_cond", w))
            r[f"{ax} recipe on its own batch - LoRA L0"] = b_own - b_lo
            for k, t in thr.items():
                r[f"{ax} recipe, thresholds from {k} - LoRA L0"] = _axis_stats(y, sym, idxs, w, thr=t)[1] - b_lo
        return r
    tab = _boot(meta, fn, B, seed)

    # speech share of the calibration batch, n_cal clips drawn from the 500 by base clip
    base = meta_all[~held].drop_duplicates("base_id")
    sp, ns = base.loc[base["label"] == 1, "base_id"].to_numpy(), base.loc[base["label"] == 0, "base_id"].to_numpy()
    rng, rows = np.random.default_rng(seed), []
    ref = {ax: _axis_stats(y, slo, idxs, None)[1] for ax, idxs in G.items()}
    for share in (0.1, 0.2, 0.3, 0.5, 0.7, 0.9):
        k = int(round(share * n_cal))
        res = {ax: [] for ax in AXES}
        for _ in range(reps):
            pick = np.concatenate([rng.choice(sp, k, replace=False), rng.choice(ns, n_cal - k, replace=False)])
            t = meta["condition"].map(thresholds(meta_all["base_id"].isin(pick).to_numpy())).to_numpy(float)
            for ax, idxs in G.items():
                res[ax].append(_axis_stats(y, sym, idxs, None, thr=t)[1])
        for ax in AXES:
            v = 100 * np.asarray(res[ax])
            rows.append({"axis": ax, "speech share of the calibration batch": share, "recipe BA": v.mean(),
                         "2.5%": np.percentile(v, 2.5), "97.5%": np.percentile(v, 97.5),
                         "recipe - LoRA L0": v.mean() - 100 * ref[ax]})
    return tab, pd.DataFrame(rows).set_index("axis")


# ---------------------------------------------------------------------------- P and G2

FORMAT_RUNS = {"ba_speech_second": "prompt-ba_speech_second-full", "ab_speech_second": "prompt-ab_speech_second-full"}


def _formats(names, split: str = "eval") -> pd.DataFrame:
    """Base native scores of the duration and SNR axes (eval) or of the training bank (train), one
    column per prompt format. "ab" is the original format of the main extraction."""
    cols = {}
    for k in names:
        d = (WORK / "features" / "base" / split) if k == "ab" else (WORK / "runs" / FORMAT_RUNS[k] / "base" / split)
        m = L.load_store_features(d)[0]
        if split == "eval":
            m = m[m["axis"].isin(AXES)]
        cols[k] = m.set_index("variant_id")
    ids = sorted(set.intersection(*[set(v.index) for v in cols.values()]))
    keep = ["base_id", "rec_id", "label"] + (["axis", "condition"] if split == "eval" else [])
    out = cols[names[0]].loc[ids, keep].reset_index()
    for k, v in cols.items():
        out[k] = v["s_primary"].reindex(ids).to_numpy(float)
    return out


def _held_out(meta: pd.DataFrame) -> tuple:
    """Clips of recordings outside the prompt control (the G1 test set) and the 300 control clips."""
    pilot = L.load_store_features(WORK / "runs" / "prompt-ba_speech_second" / "base" / "eval")[0]
    held = ~meta["rec_id"].isin(set(pilot["rec_id"])).to_numpy()
    return held, meta["base_id"].isin(set(pilot["base_id"])).to_numpy() & ~held


def _bc_w(s, w) -> float:
    return float(L2.logit(np.sum(w * L2.sigmoid(s)) / np.sum(w)))


def _mixture(s, w, t0, tied: bool, iters: int = 500, tol: float = 1e-9):
    """Two-Gaussian mixture with free weights fitted by weighted EM from the split of the batch at
    t0 (rule P). Returns the speech weight, the means and the variances, or None when a weight
    falls below 0.02."""
    r = (s > t0).astype(float)
    W, prev = w.sum(), -np.inf
    for _ in range(iters):
        n1 = float(np.sum(w * r))
        pi = n1 / W
        if not 0.02 <= pi <= 0.98:
            return None
        mu1, mu0 = np.sum(w * r * s) / n1, np.sum(w * (1 - r) * s) / (W - n1)
        d1, d0 = w * r * (s - mu1) ** 2, w * (1 - r) * (s - mu0) ** 2
        if tied:
            v1 = v0 = max(float((d1.sum() + d0.sum()) / W), 1e-6)
        else:
            v1, v0 = max(float(d1.sum() / n1), 1e-6), max(float(d0.sum() / (W - n1)), 1e-6)
        l1 = np.log(pi) - 0.5 * np.log(2 * np.pi * v1) - (s - mu1) ** 2 / (2 * v1)
        l0 = np.log(1 - pi) - 0.5 * np.log(2 * np.pi * v0) - (s - mu0) ** 2 / (2 * v0)
        m = np.logaddexp(l0, l1)
        r = np.exp(l1 - m)
        ll = float(np.sum(w * m) / W)
        if abs(ll - prev) < tol:
            break
        prev = ll
    if mu1 < mu0:
        pi, mu0, mu1, v0, v1 = 1 - pi, mu1, mu0, v1, v0
    return pi, mu0, mu1, v0, v1


def _crossing(mu0, mu1, v0, v1) -> float:
    """Where two Gaussians with equal weights cross between their means, else the midpoint."""
    a, b = 1 / v0 - 1 / v1, 2 * (mu1 / v1 - mu0 / v0)
    c = mu0 ** 2 / v0 - mu1 ** 2 / v1 + np.log(v0 / v1)
    roots = [-c / b] if abs(a) < 1e-12 else np.roots([a, b, c])
    inside = [float(np.real(x)) for x in roots if abs(np.imag(x)) < 1e-12 and mu0 <= np.real(x) <= mu1]
    return inside[0] if inside else (mu0 + mu1) / 2


def _saerens(p0, w, pi_tr: float, iters: int = 500, tol: float = 1e-9) -> tuple:
    """EM of Saerens et al. (2002). The speech share of the batch and the posteriors adjusted to it,
    from posteriors calibrated under the training share pi_tr."""
    pi = pi_tr
    for _ in range(iters):
        a, b = pi / pi_tr * p0, (1 - pi) / (1 - pi_tr) * (1 - p0)
        p = a / (a + b)
        new = float(np.clip(np.sum(w * p) / np.sum(w), 1e-6, 1 - 1e-6))
        if abs(new - pi) < tol:
            pi = new
            break
        pi = new
    return pi, p


class PRule:
    """Label-free per-condition thresholds of rule P on one score, from any batch. platt maps the
    score to posteriors calibrated on the training bank, whose speech share is pi_tr."""

    def __init__(self, platt: tuple, pi_tr: float):
        self.platt, self.pi_tr = platt, pi_tr

    def batch(self, s, w) -> dict:
        keep = w > 0
        s, w = s[keep], w[keep]
        t1 = _bc_w(s, w)
        tied, free = _mixture(s, w, t1, True), _mixture(s, w, t1, False)
        p0 = L2.sigmoid(self.platt[0] * s + self.platt[1])
        pi2, p = _saerens(p0, w, self.pi_tr)
        omega = p / (2 * pi2) + (1 - p) / (2 * (1 - pi2))
        return {"L1": t1, "P1": t1 if tied is None else (tied[1] + tied[2]) / 2,
                "P1 own variances": t1 if free is None else _crossing(*free[1:]),
                "P2": _bc_w(s, w * omega), "P1 failed": tied is None,
                "P1 speech share": np.nan if tied is None else tied[0], "P2 speech share": pi2}

    def thresholds(self, meta, s, w=None, mask=None) -> tuple:
        """Per-item thresholds of every method for the items of meta, estimated on the items in mask
        of each condition (all of them by default), plus the per-condition fits."""
        w = np.ones(len(s)) if w is None else np.asarray(w, float)
        mask = np.ones(len(s), bool) if mask is None else mask
        cond = meta["condition"].to_numpy()
        fits = {c: self.batch(s[mask & (cond == c)], w[mask & (cond == c)]) for c in np.unique(cond)}
        out = {k: np.array([fits[c][k] for c in cond], float) for k in ("L1", "P1", "P1 own variances", "P2")}
        return out, fits


def _platt(s, y) -> tuple:
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression(C=1e6, max_iter=1000).fit(np.asarray(s, float).reshape(-1, 1), y)
    return float(m.coef_[0, 0]), float(m.intercept_[0])


def _score_setup(pair: tuple) -> tuple:
    """The average of two formats on the axes and on the training bank, with rule P fitted."""
    ev = _formats(("ab",) + tuple(k for k in FORMAT_RUNS if k in pair), "eval")
    s_all = ev[list(pair)].mean(axis=1).to_numpy()
    tr = _formats(pair, "train")
    s_tr, y_tr = tr[list(pair)].mean(axis=1).to_numpy(), tr["label"].to_numpy(int)
    return ev, s_all, PRule(_platt(s_tr, y_tr), float(y_tr.mean()))


def prevalence(B: int, seed: int, pair=("ab", "ba_speech_second"), n_cal: int = 150, reps: int = 200) -> tuple:
    """Rule P (docs/exploratory_log.md, "Rule for P, a calibration that does not depend on the share
    of speech, written on 25 October before computing"). Calibration that should not depend on the
    speech share of the batch. P1, a two-Gaussian mixture with a common variance, threshold at the
    midpoint of its means. P2, the EM of Saerens et al. on posteriors from a Platt map fitted on the training bank,
    then L1 on the batch reweighted to 50%. Sweep over the g1t calibration batches (same draws) and
    a paired bootstrap on the 470 held-out clips."""
    meta_all, s_all, rule = _score_setup(pair)
    held, in_pilot = _held_out(meta_all)
    lora = L.load_store_features(I.STORES["main_o"][1])[0].set_index("variant_id")["s_primary"]
    meta = meta_all[held].reset_index(drop=True)
    s, slo = s_all[held], lora.reindex(meta["variant_id"]).to_numpy(float)
    if np.isnan(slo).any() or np.isnan(s).any():
        raise RuntimeError("missing scores")
    y, G = meta["label"].to_numpy(), _groups(meta)
    methods = ("L1", "P1", "P1 own variances", "P2")

    def from_batch(mask) -> tuple:
        """Thresholds fitted once on the clips in mask of meta_all, mapped to the held-out items."""
        _, fits = rule.thresholds(meta_all, s_all, mask=mask)
        cond = meta["condition"].to_numpy()
        return {k: np.array([fits[c][k] for c in cond], float) for k in methods}, fits

    fixed = {"300 prompt-control clips, 50% speech": from_batch(in_pilot),
             "500 discovery clips, 70% speech": from_batch(~held)}

    def fn(w):
        r = {}
        own, _ = rule.thresholds(meta, s, w)
        for ax, idxs in G.items():
            _, b_lo, _ = _axis_stats(y, slo, idxs, w)
            for src, thr in [("test batch", own)] + [(k, v[0]) for k, v in fixed.items()]:
                ba = {k: _axis_stats(y, s, idxs, w, thr=thr[k])[1] for k in methods}
                for k in ("P1", "P2"):
                    r[f"{ax} | {src} | {k} - LoRA L0"] = ba[k] - b_lo
                    r[f"{ax} | {src} | {k} - L1"] = ba[k] - ba["L1"]
                r[f"{ax} | {src} | L1 - LoRA L0"] = ba["L1"] - b_lo
        return r
    tab = _boot(meta, fn, B, seed) if B else None    # B = 0 gives the sweep alone

    base = meta_all[~held].drop_duplicates("base_id")
    sp, ns = base.loc[base["label"] == 1, "base_id"].to_numpy(), base.loc[base["label"] == 0, "base_id"].to_numpy()
    rng, rows, fails = np.random.default_rng(seed), [], 0
    ref = {ax: _axis_stats(y, slo, idxs, None)[1] for ax, idxs in G.items()}
    for share in (0.1, 0.2, 0.3, 0.5, 0.7, 0.9):
        k = int(round(share * n_cal))
        res = {(ax, m): [] for ax in AXES for m in methods}
        est = {"P1": [], "P2": []}
        for _ in range(reps):       # the same draws, in the same order, as g1_transfer
            pick = np.concatenate([rng.choice(sp, k, replace=False), rng.choice(ns, n_cal - k, replace=False)])
            thr, fits = from_batch(meta_all["base_id"].isin(pick).to_numpy())
            fails += sum(f["P1 failed"] for f in fits.values())
            est["P1"].append(np.nanmean([f["P1 speech share"] for f in fits.values()]))
            est["P2"].append(np.mean([f["P2 speech share"] for f in fits.values()]))
            for ax, idxs in G.items():
                for m in methods:
                    res[(ax, m)].append(_axis_stats(y, s, idxs, None, thr=thr[m])[1])
        for (ax, m), v in res.items():
            v = 100 * np.asarray(v)
            rows.append({"axis": ax, "method": m, "speech share": share, "BA": v.mean(),
                         "2.5%": np.percentile(v, 2.5), "97.5%": np.percentile(v, 97.5),
                         "BA - LoRA L0": v.mean() - 100 * ref[ax],
                         "estimated share": np.mean(est[m]) if m in est else np.nan})
    sweep = pd.DataFrame(rows)
    reading = {}
    for ax in AXES:
        l1_50 = sweep.query("axis == @ax and method == 'L1' and `speech share` == 0.5")["BA"].iloc[0]
        for m in methods:
            v = sweep.query("axis == @ax and method == @m")["BA"]
            reading[(ax, m)] = {"range": v.max() - v.min(), "lowest": v.min(), "L1 at 50%": l1_50,
                                "flat (range <= 2)": v.max() - v.min() <= 2.0,
                                "no cost (lowest >= L1 at 50% - 2)": v.min() >= l1_50 - 2.0}
    info = {"held-out clips": int(meta["base_id"].nunique()), "P1 failed fits in the sweep": int(fails),
            "Platt map (slope, intercept)": rule.platt, "training speech share": rule.pi_tr,
            "fixed-batch speech share estimates": {k: {m: float(np.nanmean([f[f"{m} speech share"] for f in v[1].values()]))
                                                       for m in ("P1", "P2")} for k, v in fixed.items()}}
    return tab, sweep, pd.DataFrame(reading).T, info


def methods(B: int, seed: int, pair=("ab", "ba_speech_second")) -> tuple:
    """Main table of draft v6. On the 470 held-out clips, BA, AUROC and best-threshold BA (a
    diagnostic) per axis for the original format and the two-format average with L0, L1, P1 (test
    batch, re-estimated in each replicate) and the frozen threshold, and for LoRA L0 and L1, with the
    contrasts the text quotes."""
    meta_all, s_all, rule = _score_setup(pair)
    held, _ = _held_out(meta_all)
    lora = L.load_store_features(I.STORES["main_o"][1])[0].set_index("variant_id")["s_primary"]
    meta = meta_all[held].reset_index(drop=True)
    sc = {"original": meta_all["ab"].to_numpy()[held], "recipe": s_all[held],
          "LoRA": lora.reindex(meta["variant_id"]).to_numpy(float)}
    if any(np.isnan(v).any() for v in sc.values()):
        raise RuntimeError("missing scores")
    tr = _formats(tuple(dict.fromkeys(("ab",) + tuple(pair))), "train")
    ytr = tr["label"].to_numpy(int)
    frozen = {"original": L.fit_threshold(tr["ab"].to_numpy(), ytr),
              "recipe": L.fit_threshold(tr[list(pair)].mean(axis=1).to_numpy(), ytr)}
    y, G = meta["label"].to_numpy(), _groups(meta)

    def fn(w):
        r, ba = {}, {}
        p1 = rule.thresholds(meta, sc["recipe"], w)[0]["P1"]
        for ax, idxs in G.items():
            for k, s in sc.items():
                a, b0, best = _axis_stats(y, s, idxs, w)
                ba[(ax, k, "L0")] = b0
                ba[(ax, k, "L1")] = _axis_stats(y, s, idxs, w, thr=L2.apply_rule(meta, s, "bc_cond", w))[1]
                r.update({f"{ax} | {k} | AUROC": a, f"{ax} | {k} | best BA": best,
                          f"{ax} | {k} | BA L0": b0, f"{ax} | {k} | BA L1": ba[(ax, k, "L1")]})
                if k != "LoRA":
                    ba[(ax, k, "frozen")] = _axis_stats(y, s, idxs, w, thr=np.full(len(s), frozen[k]))[1]
                    r[f"{ax} | {k} | BA frozen"] = ba[(ax, k, "frozen")]
            ba[(ax, "recipe", "P1")] = _axis_stats(y, sc["recipe"], idxs, w, thr=p1)[1]
            r[f"{ax} | recipe | BA P1"] = ba[(ax, "recipe", "P1")]
            lo0, lo1 = ba[(ax, "LoRA", "L0")], ba[(ax, "LoRA", "L1")]
            r.update({f"{ax} | recipe L1 - LoRA L0": ba[(ax, "recipe", "L1")] - lo0,
                      f"{ax} | recipe P1 - LoRA L0": ba[(ax, "recipe", "P1")] - lo0,
                      f"{ax} | recipe frozen - LoRA L0": ba[(ax, "recipe", "frozen")] - lo0,
                      f"{ax} | recipe L1 - LoRA L1": ba[(ax, "recipe", "L1")] - lo1,
                      f"{ax} | recipe P1 - LoRA L1": ba[(ax, "recipe", "P1")] - lo1,
                      f"{ax} | recipe L1 - original L1": ba[(ax, "recipe", "L1")] - ba[(ax, "original", "L1")],
                      f"{ax} | recipe L0 - original L0": ba[(ax, "recipe", "L0")] - ba[(ax, "original", "L0")],
                      f"{ax} | original L1 - original L0": ba[(ax, "original", "L1")] - ba[(ax, "original", "L0")],
                      f"{ax} | AUROC recipe - original": r[f"{ax} | recipe | AUROC"] - r[f"{ax} | original | AUROC"],
                      f"{ax} | AUROC recipe - LoRA": r[f"{ax} | recipe | AUROC"] - r[f"{ax} | LoRA | AUROC"],
                      f"{ax} | best BA recipe - LoRA L0": r[f"{ax} | recipe | best BA"] - lo0,
                      f"{ax} | best BA recipe - best BA LoRA": r[f"{ax} | recipe | best BA"] - r[f"{ax} | LoRA | best BA"]})
        return r
    return _boot(meta, fn, B, seed), {"frozen thresholds": frozen, "held-out clips": int(meta["base_id"].nunique())}


def per_duration(B: int, seed: int, pair=("ab", "ba_speech_second")) -> pd.DataFrame:
    """Question 3 of the calibrated review of draft v6. Per duration on the 470 held-out clips, the recipe
    with L1 (test batch, re-estimated in each replicate) minus LoRA L0 and minus LoRA with L1, the
    recipe with per-condition thresholds from the 300 control clips minus LoRA L0, and the recipe with
    P1 minus LoRA L0. Descriptive, the registered endpoint is the axis mean."""
    meta_all, s_all, rule = _score_setup(pair)
    held, in_pilot = _held_out(meta_all)
    lora = L.load_store_features(I.STORES["main_o"][1])[0].set_index("variant_id")["s_primary"]
    meta = meta_all[held].reset_index(drop=True)
    s, slo = s_all[held], lora.reindex(meta["variant_id"]).to_numpy(float)
    cond_all = meta_all["condition"].to_numpy()
    ctl = meta["condition"].map({c: _bc_value(s_all[in_pilot & (cond_all == c)]) for c in set(cond_all)}).to_numpy(float)
    y, cond = meta["label"].to_numpy(), meta["condition"].to_numpy()
    durs = [f"dur_{d}ms" for d in I.DUR]

    def fn(w):
        ww = np.ones(len(y)) if w is None else w
        t1 = L2.apply_rule(meta, s, "bc_cond", w)
        tl = L2.apply_rule(meta, slo, "bc_cond", w)
        tp = rule.thresholds(meta, s, w)[0]["P1"]
        r = {}
        for c in durs:
            i = cond == c
            ba = lambda sc, t: L2.ba_w(y[i], sc[i] > t[i], ww[i])
            lo0 = ba(slo, np.zeros(len(y)))
            r[f"{c[4:]} | recipe L1 - LoRA L0"] = ba(s, t1) - lo0
            r[f"{c[4:]} | recipe L1 - LoRA L1"] = ba(s, t1) - ba(slo, tl)
            r[f"{c[4:]} | recipe, control thresholds - LoRA L0"] = ba(s, ctl) - lo0
            r[f"{c[4:]} | recipe P1 - LoRA L0"] = ba(s, tp) - lo0
        return r
    return _boot(meta, fn, B, seed)


def calibrated_lora(B: int, seed: int) -> dict:
    """Post hoc, 26 October. Batch calibration averages probabilities, which the LoRA's large logits
    saturate. The median of each condition batch, the prior-matching variant of L1 (lib_v2
    median_cond), assumes a balanced batch and works in logit space. On the balanced banks only, the
    whole duration axis of the main bank and the 76 clean provenance recordings, with thresholds on
    each whole condition batch re-estimated in every replicate."""
    out = {}
    ev = _formats(("ab", "ba_speech_second"), "eval")
    ev = ev[ev["axis"] == "duration"].reset_index(drop=True)
    lora = L.load_store_features(I.STORES["main_o"][1])[0].set_index("variant_id")["s_primary"]
    W = WORK / "runs"
    pb = L.load_store_features(W / "prov-v3" / "base" / "eval")[0].set_index("variant_id")
    pa = L.load_store_features(W / "prompt-ba_speech_second-prov" / "base" / "eval")[0].set_index("variant_id")["s_primary"]
    pl = L.load_store_features(W / "prov-v3" / "lora" / "eval")[0].set_index("variant_id")["s_primary"]
    clean = set(pd.read_csv(W / "prov-v3-clean" / "provcv_scores.csv.gz")["rec_id"])
    pm = pb[pb["rec_id"].isin(clean)].sort_index()
    banks = {"main bank, 970 clips, duration axis": (ev, ev[["ab", "ba_speech_second"]].mean(axis=1).to_numpy(),
                                                     lora.reindex(ev["variant_id"]).to_numpy(float), True),
             "clean provenance bank, 76 recordings": (pm.reset_index(), (pm["s_primary"].to_numpy() + pa.reindex(pm.index).to_numpy()) / 2,
                                                      pl.reindex(pm.index).to_numpy(), False)}
    for tag, (m, s_r, s_l, strat) in banks.items():
        y, cond = m["label"].to_numpy(), m["condition"].to_numpy()
        conds = [f"dur_{d}ms" for d in I.DUR]

        def thr(s, w, rule):
            t = np.zeros(len(s))
            for c in conds:
                i = (cond == c) & (w > 0)
                t[cond == c] = _bc_w(s[i], w[i]) if rule == "L1" else L2.weighted_median(s[i], w[i])
            return t

        def fn(w):
            w = np.ones(len(y)) if w is None else w
            ba = lambda s, t, cs=conds: float(np.mean([L2.ba_w(y[cond == c], s[cond == c] > t[cond == c], w[cond == c]) for c in cs]))
            best = float(np.mean([L2.oracle_ba_w(y[cond == c], s_l[cond == c], w[cond == c]) for c in conds]))
            tl1, tlm, tr1, trm = thr(s_l, w, "L1"), thr(s_l, w, "median"), thr(s_r, w, "L1"), thr(s_r, w, "median")
            l0, l1, lm = ba(s_l, np.zeros(len(y))), ba(s_l, tl1), ba(s_l, tlm)
            r1, rm = ba(s_r, tr1), ba(s_r, trm)
            return {"LoRA L0": l0, "LoRA L1": l1, "LoRA median": lm, "LoRA best (diagnostic)": best,
                    "recipe L1": r1, "recipe median": rm,
                    "LoRA median - LoRA L0": lm - l0, "LoRA best - LoRA median": best - lm,
                    "recipe L1 - LoRA median": r1 - lm, "recipe median - LoRA median": rm - lm,
                    "20 ms | recipe L1 - LoRA median": ba(s_r, tr1, ["dur_20ms"]) - ba(s_l, tlm, ["dur_20ms"]),
                    "20 ms | LoRA median - LoRA L0": ba(s_l, tlm, ["dur_20ms"]) - ba(s_l, np.zeros(len(y)), ["dur_20ms"])}

        point = fn(None)
        reps = {k: [] for k in point}
        rng = np.random.default_rng(seed)
        if strat:
            bt = I.Boot(m)
            draw = lambda: bt.draw(rng)
        else:
            uniq, inv = np.unique(m["rec_id"].astype(str).to_numpy(), return_inverse=True)
            draw = lambda: np.bincount(rng.choice(len(uniq), len(uniq)), minlength=len(uniq)).astype(float)[inv]
        for _ in range(B):
            for k, v in fn(draw()).items():
                reps[k].append(v)
        out[tag] = I.summarise(point, reps)
    return out


def crossing(B: int, seed: int, csv: str = "") -> pd.DataFrame:
    """Post hoc, 26 October, for Fig. 1 of draft v6. On the 470 held-out clips, per condition
    BA of the recipe (L1 on the test batch, as registered), of LoRA L0 and of the LoRA with the
    batch median of its logits. The median needs a balanced batch, so its per-condition thresholds
    come from the whole condition batch of the main bank (970 clips, balanced, unlabelled) and stay
    fixed across replicates, as in the full-batch sensitivity of G1."""
    meta_all, s_all, _ = _score_setup(("ab", "ba_speech_second"))
    held, _ = _held_out(meta_all)
    lora = L.load_store_features(I.STORES["main_o"][1])[0].set_index("variant_id")["s_primary"]
    l_all = lora.reindex(meta_all["variant_id"]).to_numpy(float)
    cond_all = meta_all["condition"].to_numpy()
    med = {c: float(np.median(l_all[cond_all == c])) for c in np.unique(cond_all)}
    meta = meta_all[held].reset_index(drop=True)
    s, sl = s_all[held], l_all[held]
    tm = meta["condition"].map(med).to_numpy(float)
    y, cond, G = meta["label"].to_numpy(), meta["condition"].to_numpy(), _groups(meta)
    conds = [f"dur_{d}ms" for d in I.DUR] + [f"snr_{k}dB" for k in (-10, -5, 0, 5, 10, 20)]

    def fn(w):
        ww = np.ones(len(y)) if w is None else w
        t1 = L2.apply_rule(meta, s, "bc_cond", w)
        r = {}
        for c in conds:
            i = cond == c
            r[f"{c} | recipe"] = L2.ba_w(y[i], s[i] > t1[i], ww[i])
            r[f"{c} | LoRA L0"] = L2.ba_w(y[i], sl[i] > 0, ww[i])
            r[f"{c} | LoRA median"] = L2.ba_w(y[i], sl[i] > tm[i], ww[i])
            r[f"{c} | LoRA best"] = L2.oracle_ba_w(y[i], sl[i], ww[i])
        for ax, idxs in G.items():
            rec = _axis_stats(y, s, idxs, w, thr=t1)[1]
            lm = _axis_stats(y, sl, idxs, w, thr=tm)[1]
            l0 = _axis_stats(y, sl, idxs, w)[1]
            best = _axis_stats(y, sl, idxs, w)[2]
            r.update({f"{ax} | recipe - LoRA median": rec - lm, f"{ax} | LoRA median - LoRA L0": lm - l0,
                      f"{ax} | LoRA best - LoRA median": best - lm})
        return r
    tab = _boot(meta, fn, B, seed)
    if csv:
        rows = []
        for q in tab.index:
            if " | " in q and q.split(" | ")[0] in conds:
                c, series = q.split(" | ")
                rows.append({"condition": c, "series": series, "estimate": tab.loc[q, "estimate"],
                             "lo": tab.loc[q, "lo"], "hi": tab.loc[q, "hi"]})
        pd.DataFrame(rows).to_csv(csv, index=False)
    return tab


def _p1(s, w) -> float:
    """Threshold of P1 on one weighted batch, falling back to L1 as rule P does."""
    keep = w > 0
    s, w = s[keep], w[keep]
    t1 = _bc_w(s, w)
    fit = _mixture(s, w, t1, True)
    return t1 if fit is None else (fit[1] + fit[2]) / 2


def provenance_check(B: int, seed: int) -> dict:
    """Rule V (docs/exploratory_log.md, "Rule for V, the fixed recipe on the clean provenance bank,
    written on 26 October before extraction"). The locked recipe on the provenance bank, where
    speech and non-speech share recordings. L1 and P1 are cross-calibrated, each half of the
    recordings taking per-condition thresholds from the other half, re-estimated in every replicate. The bootstrap
    resamples recordings without strata, since every recording holds both classes."""
    W = WORK / "runs"
    base = L.load_store_features(W / "prov-v3" / "base" / "eval")[0].set_index("variant_id")
    ba2 = L.load_store_features(W / "prompt-ba_speech_second-prov" / "base" / "eval")[0].set_index("variant_id")["s_primary"]
    lora = L.load_store_features(W / "prov-v3" / "lora" / "eval")[0].set_index("variant_id")["s_primary"]
    clean = set(pd.read_csv(W / "prov-v3-clean" / "provcv_scores.csv.gz")["rec_id"])
    out = {}
    for tag, recs in (("76 clean recordings (primary)", clean), ("all 114 recordings", set(base["rec_id"]))):
        m = base[base["rec_id"].isin(recs)].sort_index()
        sc = {"original": m["s_primary"].to_numpy(float), "LoRA": lora.reindex(m.index).to_numpy(float)}
        sc["recipe"] = (sc["original"] + ba2.reindex(m.index).to_numpy(float)) / 2
        if any(np.isnan(v).any() for v in sc.values()):
            raise RuntimeError(f"{tag}: missing scores")
        m = m.reset_index()
        y, cond = m["label"].to_numpy(), m["condition"].to_numpy()
        conds = [f"dur_{d}ms" for d in I.DUR]
        uniq, inv = np.unique(m["rec_id"].astype(str).to_numpy(), return_inverse=True)
        half = np.zeros(len(uniq), int)
        half[np.random.default_rng(seed).permutation(len(uniq))[: len(uniq) // 2]] = 1
        half = half[inv]

        def thr(s, w, rule, cross=True):
            t = np.zeros(len(s))
            for c in conds:
                for h in (0, 1):
                    dst = (cond == c) & (half == h)
                    src = (cond == c) & ((half != h) if cross else np.ones(len(s), bool))
                    t[dst] = _bc_w(s[src], w[src]) if rule == "L1" else _p1(s[src], w[src])
            return t

        def ba(s, t, w, c=None):
            cs = conds if c is None else [c]
            return float(np.mean([L2.ba_w(y[cond == k], s[cond == k] > t[cond == k], w[cond == k]) for k in cs]))

        def auc(s, w):
            return float(np.mean([L2.auc_w(y[cond == k], s[cond == k], w[cond == k]) for k in conds]))

        def fn(w):
            w = np.ones(len(y)) if w is None else w
            t = {(k, r): thr(sc[k], w, r) for k in ("original", "recipe", "LoRA") for r in ("L1", "P1")}
            lo0, lo1 = ba(sc["LoRA"], np.zeros(len(y)), w), ba(sc["LoRA"], t[("LoRA", "L1")], w)
            rec1, recp = ba(sc["recipe"], t[("recipe", "L1")], w), ba(sc["recipe"], t[("recipe", "P1")], w)
            r = {"LoRA L0 BA": lo0, "LoRA L1 cross BA": lo1, "recipe L1 cross BA": rec1, "recipe P1 cross BA": recp,
                 "original L1 cross BA": ba(sc["original"], t[("original", "L1")], w),
                 "AUROC recipe": auc(sc["recipe"], w), "AUROC original": auc(sc["original"], w),
                 "AUROC LoRA": auc(sc["LoRA"], w),
                 "recipe L1 cross - LoRA L0 (primary)": rec1 - lo0,
                 "recipe L1 cross - LoRA L1 cross": rec1 - lo1,
                 "recipe P1 cross - LoRA L0": recp - lo0, "recipe P1 cross - LoRA L1 cross": recp - lo1,
                 "recipe L1 cross - original L1 cross": rec1 - ba(sc["original"], t[("original", "L1")], w),
                 "AUROC recipe - LoRA": auc(sc["recipe"], w) - auc(sc["LoRA"], w),
                 "AUROC recipe - original": auc(sc["recipe"], w) - auc(sc["original"], w),
                 "recipe L1 on the whole batch - LoRA L0": ba(sc["recipe"], thr(sc["recipe"], w, "L1", cross=False), w) - lo0}
            for c in conds:
                d = c[4:]
                r[f"{d} | recipe L1 cross - LoRA L0"] = ba(sc["recipe"], t[("recipe", "L1")], w, c) - ba(sc["LoRA"], np.zeros(len(y)), w, c)
                r[f"{d} | recipe L1 cross - LoRA L1 cross"] = ba(sc["recipe"], t[("recipe", "L1")], w, c) - ba(sc["LoRA"], t[("LoRA", "L1")], w, c)
            return r

        point = fn(None)
        rng, reps = np.random.default_rng(seed), {k: [] for k in point}
        for _ in range(B):
            cnt = np.bincount(rng.choice(len(uniq), len(uniq)), minlength=len(uniq)).astype(float)
            for k, v in fn(cnt[inv]).items():
                reps[k].append(v)
        out[tag] = I.summarise(point, reps)
        print(f"{tag}: {len(uniq)} recordings, {len(m)} items", flush=True)
    return out


def g2(B: int, seed: int) -> tuple:
    """Rule G2 (docs/exploratory_log.md, "Rule for G2, written on 25 October before extraction").
    s_seg, the average of the two speech-second formats, against s_sim of G1 on the same 470
    held-out clips, both with L1 per condition on the test batch. Sensitivity with per-condition
    thresholds from the 300 balanced control clips."""
    ev = _formats(("ab", "ba_speech_second", "ab_speech_second"), "eval")
    lora = L.load_store_features(I.STORES["main_o"][1])[0].set_index("variant_id")["s_primary"]
    sc_all = {"sim": ev[["ab", "ba_speech_second"]].mean(axis=1).to_numpy(),
              "seg": ev[["ab_speech_second", "ba_speech_second"]].mean(axis=1).to_numpy()}
    held, in_pilot = _held_out(ev)
    cond_all = ev["condition"].to_numpy()
    meta = ev[held].reset_index(drop=True)
    sc = {k: v[held] for k, v in sc_all.items()}
    slo = lora.reindex(meta["variant_id"]).to_numpy(float)
    if np.isnan(slo).any() or any(np.isnan(v).any() for v in sc.values()):
        raise RuntimeError("missing scores")
    ctl = {k: meta["condition"].map({c: _bc_value(v[in_pilot & (cond_all == c)]) for c in set(cond_all)}).to_numpy(float)
           for k, v in sc_all.items()}
    y, G = meta["label"].to_numpy(), _groups(meta)

    def fn(w):
        r = {}
        for ax, idxs in G.items():
            a_lo, b_lo, _ = _axis_stats(y, slo, idxs, w)
            st = {}
            for k, s in sc.items():
                a, _, best = _axis_stats(y, s, idxs, w)
                _, b1, _ = _axis_stats(y, s, idxs, w, thr=L2.apply_rule(meta, s, "bc_cond", w))
                _, bc, _ = _axis_stats(y, s, idxs, w, thr=ctl[k])
                st[k] = (a, b1, bc, best)
                r.update({f"{ax} AUROC s_{k}": a, f"{ax} BA s_{k} L1": b1, f"{ax} best BA s_{k}": best})
            r.update({f"{ax} LoRA L0 BA": b_lo, f"{ax} LoRA L0 AUROC": a_lo,
                      f"{ax} BA s_seg L1 - s_sim L1 (primary on SNR)": st["seg"][1] - st["sim"][1],
                      f"{ax} BA s_seg L1 - LoRA L0": st["seg"][1] - b_lo,
                      f"{ax} AUROC s_seg - s_sim": st["seg"][0] - st["sim"][0],
                      f"{ax} AUROC s_seg - LoRA L0": st["seg"][0] - a_lo,
                      f"{ax} BA s_seg, control thresholds - LoRA L0 (sensitivity)": st["seg"][2] - b_lo,
                      f"{ax} BA s_seg, control thresholds - s_sim, control thresholds": st["seg"][2] - st["sim"][2]})
        return r
    tab = _boot(meta, fn, B, seed)
    info = {"held-out clips": int(meta["base_id"].nunique()),
            "speech share": float((meta.drop_duplicates("base_id")["label"] == 1).mean())}
    return tab, info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("what", choices=["s", "r", "n", "g1", "g1t", "p", "p2", "g2", "methods", "methods2", "v", "perdur", "lora_cal", "crossing"])
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sweep-csv", default="", help="p and p2, also write the sweep for the figure of draft v6")
    a = ap.parse_args()
    if a.what == "s":
        print(f"## S, symmetric prompting, points, B = {a.B}\n")
        print(md(symmetric(a.B, a.seed), 2))
    elif a.what == "r":
        tab, sel, frozen = why_h3(a.B, a.seed)
        print(f"## R, why H3 fails, BA and AUROC points, B = {a.B}\n")
        for k, (c, sc) in sel.items():
            print(f"- C chosen by {k}: {c}, validation AUROC per C {sc}")
        print(f"- frozen thresholds on cross-fitted training scores: {frozen}\n")
        print(md(tab, 2))
    elif a.what == "g1":
        tab, info = g1(a.B, a.seed)
        print(f"## G1, training-free recipe on held-out recordings, points, B = {a.B}")
        print(info)
        print(md(tab, 2))
    elif a.what == "g1t":
        tab, sweep = g1_transfer(a.B, a.seed)
        print(f"## G1 transfer, thresholds from separate unlabeled clips, BA points, B = {a.B}\n")
        print(md(tab, 2))
        print("\nSpeech share of a calibration batch of 150 clips from the discovery recordings, 200 draws\n")
        print(md(sweep, 2))
    elif a.what in ("p", "p2"):
        pair = ("ab", "ba_speech_second") if a.what == "p" else ("ab_speech_second", "ba_speech_second")
        tab, sweep, reading, info = prevalence(a.B, a.seed, pair)
        print(f"## P, calibration that should not depend on the speech share, on the average of {pair}, "
              f"BA points, B = {a.B}\n")
        print(info, "\n")
        print("### Reading of the sweep\n")
        print(md(reading, 2))
        print("\n### Sweep, 150 calibration clips from the discovery recordings, the 200 draws of g1t\n")
        print(md(sweep.set_index("axis"), 2))
        if tab is not None:
            print("\n### Paired bootstrap on the 470 held-out clips\n")
            print(md(tab, 2))
        if a.sweep_csv:
            sweep.to_csv(a.sweep_csv, index=False)
    elif a.what in ("methods", "methods2"):
        pair = ("ab", "ba_speech_second") if a.what == "methods" else ("ab_speech_second", "ba_speech_second")
        tab, info = methods(a.B, a.seed, pair)
        print(f"## Methods on the 470 held-out clips, recipe = average of {pair}, points, B = {a.B}\n")
        print(info, "\n")
        print(md(tab, 2))
    elif a.what == "perdur":
        print(f"## Per duration on the 470 held-out clips, BA points, B = {a.B}\n")
        print(md(per_duration(a.B, a.seed), 2))
    elif a.what == "crossing":
        print(f"## Per condition on the 470 held-out clips, recipe against LoRA L0 and the median-calibrated "
              f"LoRA, BA points, B = {a.B}\n")
        print(md(crossing(a.B, a.seed, a.sweep_csv), 2))
    elif a.what == "lora_cal":
        print(f"## Calibrating the LoRA in logit space, balanced banks, BA points, B = {a.B}")
        for tag, tab in calibrated_lora(a.B, a.seed).items():
            print(f"\n### {tag}\n")
            print(md(tab, 2))
    elif a.what == "v":
        print(f"## V, the locked recipe on the provenance bank, BA and AUROC points, B = {a.B}\n")
        for tag, tab in provenance_check(a.B, a.seed).items():
            print(f"\n### {tag}\n")
            print(md(tab, 2))
    elif a.what == "g2":
        tab, info = g2(a.B, a.seed)
        print(f"## G2, the two speech-second formats against s_sim, points, B = {a.B}")
        print(info)
        print(md(tab, 2))
    else:
        print(f"## N, residual at 20 and 40 ms without vocal negatives, AUROC points, B = {a.B}\n")
        print(md(residual(a.B, a.seed), 2))


if __name__ == "__main__":
    main()
