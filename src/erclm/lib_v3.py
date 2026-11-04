"""ERC-LM analysis library v3 (2026-10-21).

Exploratory analyses for closing the paper, as laid out in docs/exploratory_log.md.
lib_v2 stays frozen. Nothing here changes a confirmatory readout, metric or hypothesis,
and every new readout has its own name.

V1  speech coverage of the centred crops, from the VoxConverse RTTM annotations, and
    their energy relative to the 1000 ms clip they come from.
V2  how much the level metadata (fragment RMS, applied gain, noise sigma) alone
    separates speech from non-speech in each condition.
P   prompt control (decision D1 of the plan). The A/B prompt with the letter of the
    speech option and its position crossed. Exploratory only. The confirmatory prompt
    stays lib_v2.PROMPTS["ab"].
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

SR = 16000
BASE_MS = 1000


def n_samples(ms: float) -> int:
    """Same rounding as bankgen.n_samples."""
    return int(round(ms * SR / 1000.0))


# ============================================================================ V1 coverage

def read_rttm_intervals(path) -> list:
    """Speech intervals (start_s, end_s) of an RTTM file, merged over speakers."""
    segs = []
    for line in Path(path).read_text().splitlines():
        p = line.split()
        if len(p) >= 5 and p[0] == "SPEAKER":
            a = float(p[3])
            segs.append((a, a + float(p[4])))
    return merge_intervals(segs)


def merge_intervals(segs: Iterable) -> list:
    out = []
    for a, b in sorted(segs):
        if b <= a:
            continue
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def interval_coverage(intervals: Sequence, a: float, b: float) -> float:
    """Fraction of [a, b) covered by the merged intervals."""
    if b <= a:
        return 0.0
    covered = 0.0
    for s, e in intervals:
        if e <= a:
            continue
        if s >= b:
            break
        covered += min(e, b) - max(s, a)
    return covered / (b - a)


def crop_bounds(start_s: float, duration_ms: float) -> tuple:
    """Source-time bounds of the centred crop that bankgen.center_cut takes from a base clip."""
    n_base, n = n_samples(BASE_MS), n_samples(duration_ms)
    off = (n_base - n) // 2
    a = start_s + off / SR
    return a, a + n / SR


def find_rttm(rttm_dirs: Sequence, recording: str) -> Optional[Path]:
    for d in rttm_dirs:
        p = Path(d) / f"{recording}.rttm"
        if p.exists():
            return p
    return None


def speech_coverage(meta: pd.DataFrame, rttm_dirs: Sequence) -> pd.DataFrame:
    """RTTM speech coverage of the crop of every VoxConverse variant of the duration axis.

    meta is a bank metadata table (bankgen columns). Returns one row per variant with
    variant_id, base_id, duration_ms and coverage in [0, 1].
    """
    rows, cache = [], {}
    sel = meta[(meta["dataset"] == "voxconverse") & meta["condition"].astype(str).str.startswith("dur")]
    for r in sel.itertuples(index=False):
        if r.recording not in cache:
            p = find_rttm(rttm_dirs, r.recording)
            if p is None:
                raise FileNotFoundError(f"no RTTM for recording {r.recording}")
            cache[r.recording] = read_rttm_intervals(p)
        a, b = crop_bounds(float(r.start_s), float(r.duration_ms))
        rows.append({"variant_id": str(r.audio_path).rsplit(".", 1)[0], "base_id": r.base_id,
                     "duration_ms": int(r.duration_ms),
                     "coverage": interval_coverage(cache[r.recording], a, b)})
    return pd.DataFrame(rows)


def coverage_table(cov: pd.DataFrame) -> pd.DataFrame:
    """Share of speech crops per duration that are fully, mostly, partly and not covered."""
    out = {}
    for d, g in cov.groupby("duration_ms"):
        c = g["coverage"].to_numpy()
        out[d] = {"n": len(c),
                  "full": float(np.mean(c >= 1.0 - 1e-9)),
                  "at_least_half": float(np.mean(c >= 0.5 - 1e-9)),
                  "below_half": float(np.mean(c < 0.5 - 1e-9)),
                  "none": float(np.mean(c <= 1e-9))}
    return pd.DataFrame(out).T.rename_axis("duration_ms")


def relative_level_db(meta: pd.DataFrame) -> pd.DataFrame:
    """RMS of each duration-axis crop in dB relative to the 1000 ms variant of the same clip."""
    d = meta[meta["condition"].astype(str).str.startswith("dur")].copy()
    ref = d[d["duration_ms"] == BASE_MS].set_index("base_id")["segment_rms"]
    d["rel_db"] = 20.0 * np.log10(d["segment_rms"] / d["base_id"].map(ref))
    d["variant_id"] = d["audio_path"].astype(str).str.rsplit(".", n=1).str[0]
    return d[["variant_id", "base_id", "ground_truth", "duration_ms", "rel_db"]].reset_index(drop=True)


def level_table(rel: pd.DataFrame, floor_db: float = -20.0) -> pd.DataFrame:
    """Per class and duration, the median and 10th percentile of rel_db and the share below floor_db."""
    rows = []
    for (lab, d), g in rel.groupby(["ground_truth", "duration_ms"]):
        x = g["rel_db"]
        rows.append({"ground_truth": lab, "duration_ms": d, "p10_db": x.quantile(0.1),
                     "median_db": x.median(), "share_below": float((x < floor_db).mean())})
    return pd.DataFrame(rows)


# ============================================================================ V2 level cues

def auroc(y, s) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(np.asarray(y).astype(int), np.asarray(s, dtype=float)))


def level_auroc_by_condition(meta: pd.DataFrame) -> pd.DataFrame:
    """AUROC of log fragment RMS for speech against non-speech, per condition.

    Reported as max(AUROC, 1 - AUROC), the best a level threshold can do in either
    direction, with the direction kept apart. Within an SNR condition the noise sigma
    is the fragment RMS times a constant, so it carries no further information.
    """
    rows = []
    for cond, g in meta.groupby("condition"):
        y = (g["ground_truth"] == "SPEECH").to_numpy()
        a = auroc(y, np.log10(g["segment_rms"].to_numpy()))
        rows.append({"condition": cond, "auroc": max(a, 1.0 - a),
                     "speech_louder": bool(a >= 0.5)})
    return pd.DataFrame(rows).set_index("condition")


# ============================================================================ P prompt control

SYSTEM_PROMPT = "You classify audio content."  # the same as lib_v2.SYSTEM_PROMPT
SPEECH_OPTION = "SPEECH (human voice)"
NONSPEECH_OPTION = "NONSPEECH (music/noise/silence/animals)"


def ab_prompt(speech_letter: str = "A", speech_first: bool = True) -> dict:
    """The forced-choice prompt with the speech option under speech_letter, listed first or second.

    labels are (speech letter, non-speech letter), so the native score s stays
    z(speech) - z(non-speech) whatever the mapping. ab_prompt("A", True) is lib_v2's "ab".
    """
    if speech_letter not in ("A", "B"):
        raise ValueError("speech_letter must be A or B")
    other = "B" if speech_letter == "A" else "A"
    lines = [f"{speech_letter}) {SPEECH_OPTION}", f"{other}) {NONSPEECH_OPTION}"]
    if not speech_first:
        lines.reverse()
    return {"system": SYSTEM_PROMPT,
            "user": "Choose one:\n" + "\n".join(lines) + "\n\nAnswer with A or B ONLY.",
            "labels": (speech_letter, other)}


PROMPTS = {
    "ab_speech_second": ab_prompt("A", speech_first=False),
    "ba_speech_first": ab_prompt("B", speech_first=True),
    "ba_speech_second": ab_prompt("B", speech_first=False),
    # Prompts of the study under review (gbibbo/qwen-vad-lora, results/CONSOLIDATED_MATRIX_RESULTS.md,
    # cells 1A to 3A and 1B). hand_csl is the hand prompt of every model there, the same as
    # omni.HAND_CSL. opro_llm_base is the prompt OPRO-LLM chose for the base (82.6 % BA there).
    # The LoRA's OPRO-Template prompt is lib_v2's opro_tmpl. Exploratory, read at the first token.
    "hand_csl": {"system": SYSTEM_PROMPT,
                 "user": "Does this audio contain human speech? Answer SPEECH or NONSPEECH.",
                 "labels": ("SPEECH", "NONSPEECH")},
    "opro_llm_base": {"system": SYSTEM_PROMPT,
                      "user": "Is this audio human speech? Answer: SPEECH or NON-SPEECH.",
                      "labels": ("SPEECH", "NON-SPEECH")},
}


def register_prompts(table: dict) -> dict:
    """Add the v3 prompts to a prompt table (lib_v2.PROMPTS at run time) without touching its entries."""
    for k, v in PROMPTS.items():
        if k in table and table[k] != v:
            raise ValueError(f"prompt {k} already defined differently")
        table.setdefault(k, v)
    return table


# ============================================================================ L1_tr threshold transfer

def fit_threshold(s, y) -> float:
    """Threshold on s that maximises balanced accuracy on labelled data (the LoRA training bank).

    Candidates are midpoints between consecutive distinct scores. When several reach the
    maximum, the median of them is returned, so the choice does not hinge on one item.
    """
    s, y = np.asarray(s, dtype=float), np.asarray(y).astype(bool)
    u = np.unique(s)
    if len(u) < 2:
        return float(u[0]) if len(u) else 0.0
    cand = np.concatenate([[u[0] - 1.0], (u[:-1] + u[1:]) / 2.0, [u[-1] + 1.0]])
    pos, neg = np.sort(s[y]), np.sort(s[~y])
    tpr = 1.0 - np.searchsorted(pos, cand, side="right") / len(pos)
    tnr = np.searchsorted(neg, cand, side="right") / len(neg)
    ba = 0.5 * (tpr + tnr)
    best = cand[np.isclose(ba, ba.max())]
    return float(np.median(best))


def threshold_readouts(scores: pd.DataFrame, thr_tr: float, s_key: str = "s_primary") -> dict:
    """{name: (score, threshold, rule)} for the threshold readouts of one backbone, in the
    format of lib_v2.cluster_bootstrap. Batch rules are re-estimated in every replicate.
    L1_cf keeps the stored content-free threshold and L1_tr the frozen one from training."""
    from . import lib_v2 as L2
    s = scores[s_key].to_numpy(float)
    R = {"L0": (s, np.zeros(len(s)), None)}
    for m in L2.BATCH_RULES:
        R[f"L1_{m}"] = (s, L2.l1_thresholds(scores, s, m), m)
    if "L1_cf__thr" in scores:
        R["L1_cf"] = (s, scores["L1_cf__thr"].to_numpy(float), None)
    R["L1_tr"] = (s, np.full(len(s), float(thr_tr)), None)
    return R


def axis_mean_contrasts(scores: pd.DataFrame, terms: dict, metric: str = "ba") -> dict:
    """Contrasts for cluster_bootstrap that average over the conditions of each axis.

    terms: {label: {readout: coefficient}}, e.g. {"tr_minus_bc": {"L1_tr": 1, "L1_bc_cond": -1}}.
    A readout key of the form "oracle:<readout>" uses that readout's oracle BA.
    """
    out = {}
    for axis, g in scores.groupby("axis"):
        conds = list(dict.fromkeys(g["condition"]))
        for label, coefs in terms.items():
            rows = []
            for rd, c in coefs.items():
                met, name = ("oracle_ba", rd.split(":", 1)[1]) if rd.startswith("oracle:") else (metric, rd)
                rows += [(c / len(conds), name, cond, met) for cond in conds]
            out[f"{axis}:{label}"] = rows
    return out


# ============================================================================ L2_anc anchored head

def fit_anchored_head(h, s, y, groups, Cs=(1e-4, 1e-3, 1e-2, 1e-1), anchor_scale: float = 100.0,
                      n_splits: int = 4, seed: int = 0) -> dict:
    """L2_anc, a logistic head on the frozen state that starts from the native direction.

    The score is alpha * s + delta' z + b, with z the state standardised on the training
    data. The l2 penalty falls on delta. The native score enters multiplied by anchor_scale,
    so its own coefficient is penalised by a negligible amount and alpha stays free. C is
    chosen by grouped cross-validation on AUROC, as for L2 in lib_v2.fit_head.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold
    h = np.asarray(h, dtype=np.float32)
    s = np.asarray(s, dtype=np.float64)
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups)
    mu, sd = h.mean(0), h.std(0)
    sd[sd == 0] = 1.0

    def design(hh, ss, m, d):
        return np.column_stack([ss * anchor_scale, (hh - m) / d])

    def fit(hh, ss, yy, C):
        m, d = hh.mean(0), hh.std(0)
        d[d == 0] = 1.0
        return LogisticRegression(C=C, max_iter=5000).fit(design(hh, ss, m, d), yy), m, d

    best = None
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for C in Cs:
        aucs = []
        for tr, va in cv.split(h, y, groups):
            model, m, d = fit(h[tr], s[tr], y[tr], C)
            aucs.append(auroc(y[va], model.decision_function(design(h[va], s[va], m, d))))
        score = float(np.mean(aucs))
        if best is None or score > best[1] + 1e-12:
            best = (C, score)
    model = LogisticRegression(C=best[0], max_iter=5000).fit(design(h, s, mu, sd), y)
    coef = model.coef_[0]
    return {"C": best[0], "cv_auroc": best[1], "alpha": float(coef[0] * anchor_scale),
            "delta": coef[1:].astype(np.float64), "b": float(model.intercept_[0]),
            "mean": mu, "std": sd}


def _fit_offset_logistic(Z, s, y, C: float) -> tuple:
    """Logistic regression with offset s and coefficient 1 on it: min sum log(1 + exp(-y'(s + Z d + b))) + |d|^2 / (2C)."""
    from scipy.optimize import minimize
    yy = 2.0 * np.asarray(y, dtype=float) - 1.0
    Z = np.asarray(Z, dtype=np.float64)

    def f(w):
        d, b = w[:-1], w[-1]
        m = yy * (s + Z @ d + b)
        loss = np.logaddexp(0.0, -m).sum() + 0.5 * (d @ d) / C
        g = -yy / (1.0 + np.exp(m))
        return loss, np.append(Z.T @ g + d / C, g.sum())

    res = minimize(f, np.zeros(Z.shape[1] + 1), jac=True, method="L-BFGS-B", options={"maxiter": 2000})
    return res.x[:-1], float(res.x[-1])


def fit_residual_head(h, s, y, groups, Cs=(1e-4, 1e-3, 1e-2, 1e-1, 1.0), n_splits: int = 4, seed: int = 0) -> dict:
    """L2_res, a head that keeps the native score with coefficient 1 and learns a penalised correction.

    The score is s + delta' z + b. A strong penalty leaves the native direction with a fitted
    threshold, a weak one approaches a free head. This is the head regularised towards
    u_A - u_B instead of towards zero. C is chosen by grouped cross-validation on AUROC.
    """
    from sklearn.model_selection import StratifiedGroupKFold
    h = np.asarray(h, dtype=np.float32)
    s = np.asarray(s, dtype=np.float64)
    y = np.asarray(y, dtype=int)

    def stdz(hh, m=None, d=None):
        if m is None:
            m, d = hh.mean(0), hh.std(0)
            d[d == 0] = 1.0
        return (hh - m) / d, m, d

    best = None
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for C in Cs:
        aucs = []
        for tr, va in cv.split(h, y, np.asarray(groups)):
            ztr, m, d = stdz(h[tr])
            delta, b = _fit_offset_logistic(ztr, s[tr], y[tr], C)
            aucs.append(auroc(y[va], s[va] + stdz(h[va], m, d)[0] @ delta + b))
        score = float(np.mean(aucs))
        if best is None or score > best[1] + 1e-12:
            best = (C, score)
    z, mu, sd = stdz(h)
    delta, b = _fit_offset_logistic(z, s, y, best[0])
    return {"C": best[0], "cv_auroc": best[1], "alpha": 1.0, "delta": delta, "b": b, "mean": mu, "std": sd}


def anchored_scores(head: dict, h, s) -> np.ndarray:
    z = (np.asarray(h, dtype=np.float32) - head["mean"]) / head["std"]
    return head["alpha"] * np.asarray(s, dtype=np.float64) + z @ head["delta"] + head["b"]


# ============================================================================ per-class errors

ESC_HUMAN_VOCAL = {20: "crying_baby", 21: "sneezing", 23: "breathing", 24: "coughing", 26: "laughing",
                   28: "snoring"}
ESC_HUMAN_OTHER = {22: "clapping", 25: "footsteps", 27: "brushing_teeth", 29: "drinking_sipping"}


def esc_class(base_id: str) -> int:
    """ESC-50 target from a base id such as esc50_1-100032-A-0_0007."""
    return int(str(base_id).rsplit("_", 1)[0].rsplit("-", 1)[1])


def esc_group(target: int) -> str:
    if target in ESC_HUMAN_VOCAL:
        return "human vocal"
    if target in ESC_HUMAN_OTHER:
        return "human other"
    return "other"


def cluster_rate_ci(hit, clusters, B: int = 2000, seed: int = 0, alpha: float = 0.05) -> tuple:
    """Mean of a 0/1 outcome with a percentile interval from resampling clusters."""
    hit = np.asarray(hit, dtype=float)
    uniq, inv = np.unique(np.asarray(clusters).astype(str), return_inverse=True)
    sums, counts = np.bincount(inv, weights=hit), np.bincount(inv).astype(float)
    rng = np.random.default_rng(seed)
    reps = np.empty(B)
    for b in range(B):
        w = np.bincount(rng.integers(len(uniq), size=len(uniq)), minlength=len(uniq))
        reps[b] = (w @ sums) / (w @ counts)
    lo, hi = np.percentile(reps, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(hit.mean()), float(lo), float(hi)


# ============================================================================ stores

def load_store_features(d, features: Sequence[str] = (), drop_content_free: bool = True) -> tuple:
    """Meta and selected feature arrays of a lib_v2 feature store folder, reading only those keys.

    Duplicated variant ids keep their last copy, as in lib_v2.load_store. Arrays come back
    as float32. Content-free containers are dropped unless asked for.
    """
    metas, xs = [], {f: [] for f in features}
    for p in sorted(Path(d).glob("shard_*.csv")):
        m = pd.read_csv(p)
        if features:
            with np.load(p.with_suffix(".npz")) as z:
                for f in features:
                    if len(z[f]) != len(m):
                        raise RuntimeError(f"{p.name}: {f} has {len(z[f])} rows, meta has {len(m)}")
                    xs[f].append(z[f])
        metas.append(m)
    if not metas:
        raise FileNotFoundError(f"no shards in {d}")
    meta = pd.concat(metas, ignore_index=True)
    keep = ~meta["variant_id"].duplicated(keep="last").to_numpy()
    if drop_content_free:
        keep &= (meta["role"] != "content_free").to_numpy()
    arrays = {f: np.concatenate(v)[keep].astype(np.float32) for f, v in xs.items()}
    return meta[keep].reset_index(drop=True), arrays


# ============================================================================ BA helpers

def balanced_accuracy(y: np.ndarray, pred: np.ndarray) -> float:
    y, pred = np.asarray(y).astype(bool), np.asarray(pred).astype(bool)
    if y.all() or (~y).all():
        return float("nan")
    return 0.5 * (pred[y].mean() + (~pred[~y]).mean())


def ba_by_condition(scores: pd.DataFrame, readouts: Sequence, keep: Optional[pd.Series] = None) -> pd.DataFrame:
    """BA per condition and readout from a scores table with <readout>__score and __thr columns.

    keep, if given, is a boolean mask over the rows. Thresholds are the stored ones.
    """
    df = scores if keep is None else scores[keep.to_numpy()]
    out = {}
    for cond, g in df.groupby("condition"):
        y = g["label"].to_numpy() == 1
        out[cond] = {r: balanced_accuracy(y, g[f"{r}__score"].to_numpy() > g[f"{r}__thr"].to_numpy())
                     for r in readouts}
    return pd.DataFrame(out).T.rename_axis("condition")


# ============================================================================ GPU interventions (22 Oct)
# Two exploratory experiments added after the review of draft v4 (docs/exploratory_log.md,
# "GPU plan of 22 October"). I1 band-limits every container the model hears, to remove the
# channel cues the audit found (VoxConverse has no energy above 7.4 kHz, and the corpora differ
# below 40 Hz). I2 keeps the trained LoRA in one part of the model only. Neither changes a
# confirmatory readout, and both are switched on by their own configuration keys.

def bandpass(x, sr: int = SR, lo: float = 60.0, hi: float = 6000.0, order: int = 8) -> np.ndarray:
    """Zero-phase Butterworth band-pass over the whole container (scipy sosfiltfilt)."""
    from scipy.signal import butter, sosfiltfilt
    sos = butter(order, [lo, hi], btype="bandpass", fs=sr, output="sos")
    return sosfiltfilt(sos, np.asarray(x, dtype=np.float64)).astype(np.float32)


AUDIO_TRANSFORMS = {"bandpass": bandpass}


def transform_audio(manifest: pd.DataFrame, out_dir, spec: dict) -> pd.DataFrame:
    """Passes the audio of every manifest row through a transform and returns the manifest with
    paths to the new files. spec is {"kind": name in AUDIO_TRANSFORMS, **parameters}. Files are
    written as 32-bit float WAV under a temporary name and renamed, and files already present are
    reused, so a resumed session does not redo them."""
    import soundfile as sf
    from .lib_v2 import load_wav
    params = {k: v for k, v in spec.items() if k != "kind"}
    fn = AUDIO_TRANSFORMS[spec["kind"]]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for vid, p in zip(manifest["variant_id"].astype(str), manifest["path"]):
        q = out_dir / f"{vid}.wav"
        if not q.exists():
            tmp = out_dir / f"{vid}.partial.wav"
            sf.write(str(tmp), fn(load_wav(p), SR, **params), SR, subtype="FLOAT")
            tmp.replace(q)
        paths.append(str(q))
    out = manifest.copy()
    out["path"] = paths
    return out


def parse_lora_keep(keep: str) -> tuple:
    """'audio_tower', 'language_model' or 'language_model:16-31' as (component, (first, last) or None)."""
    comp, _, rng = keep.partition(":")
    if comp not in ("audio_tower", "language_model"):
        raise ValueError(f"lora_keep {keep!r}: the component must be audio_tower or language_model")
    if not rng:
        return comp, None
    a, _, b = rng.partition("-")
    first, last = int(a), int(b or a)
    if first > last:
        raise ValueError(f"lora_keep {keep!r}: empty layer range")
    return comp, (first, last)


def restrict_lora(model, keep: str) -> dict:
    """Switches off every LoRA module outside `keep` by setting its scaling to 0. The trained
    weights stay as they are. Returns how many modules stay on and off per component. `keep` can
    join several parts with '+', e.g. 'audio_tower+language_model:0-15' (added on 23 Oct for I3; a
    single part behaves as before)."""
    import re
    parts = [parse_lora_keep(p) for p in keep.split("+")]
    kept, off = {}, {}
    for name, mod in model.named_modules():
        lb = getattr(mod, "lora_B", None)
        if lb is None or not hasattr(lb, "values") or len(lb) == 0:
            continue
        c = "audio_tower" if "audio_tower" in name else "language_model" if "language_model" in name else "other"
        m = re.search(r"\.layers\.(\d+)\.", name)
        layer = int(m.group(1)) if m else None
        on = any(c == comp and (rng is None or (layer is not None and rng[0] <= layer <= rng[1]))
                 for comp, rng in parts)
        if not on:
            for adapter in list(mod.scaling):
                mod.scaling[adapter] = 0.0
        d = kept if on else off
        d[c] = d.get(c, 0) + 1
    if not kept:
        raise ValueError(f"no LoRA module matches lora_keep {keep!r}")
    return {"keep": keep, "kept": kept, "disabled": off}
