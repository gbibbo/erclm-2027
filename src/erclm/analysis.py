"""Per-backbone and joint analyses, ported from the v2 notebook (sections 6 and 7).

per_backbone() works on extracted features. joint() works on the score tables
that per_backbone() exports, so H3 and H4 can be rerun from the run records in
runs/ without GPU or Drive:

    from erclm import analysis, export
    out = analysis.joint({"base": export.load_scores(".../base/scores.csv.gz"),
                          "lora": export.load_scores(".../lora/scores.csv.gz")}, B=2000, seed=42)

The contrasts, metrics and thresholds are the notebook's, unchanged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import lib_v2 as L


def split_eval(meta: pd.DataFrame, arrays: dict):
    """Test items, their arrays, and the scores of the content-free containers."""
    is_test = (meta["role"] == "test").to_numpy()
    mt = meta[is_test].reset_index(drop=True)
    at = {k: v[is_test] for k, v in arrays.items()}
    s_cf = meta.loc[meta["role"] == "content_free", "s_primary"].to_numpy()
    return mt, at, s_cf


def _conds(tab: pd.DataFrame, axis: str) -> list:
    return tab.loc[tab.axis == axis].sort_values("level")["condition"].unique().tolist()


def backbone_contrasts(tab: pd.DataFrame, L2N: str, PREG: str) -> dict:
    dur, snr = _conds(tab, "duration"), _conds(tab, "snr")
    contrasts = {}
    if snr:
        contrasts["H1_snr_mean_BC_minus_L0_ba"] = L.mean_contrast(snr, "L1_bc_cond", "L0", "ba")
    if dur:
        contrasts["dur_mean_oracle_minus_L0_ba"] = [t for c in dur for t in
                                                    ((1 / len(dur), "L0", c, "oracle_ba"),
                                                     (-1 / len(dur), "L0", c, "ba"))]
        contrasts[f"dur_mean_{L2N}_minus_BC_ba"] = L.mean_contrast(dur, L2N, "L1_bc_cond", "ba")
    if {"dur_20ms", "dur_200ms"} <= set(dur):
        contrasts["H2a_S1_auroc_200_minus_20"] = [(1, PREG, "dur_200ms", "auroc"), (-1, PREG, "dur_20ms", "auroc")]
        contrasts[f"H2b_({L2N}-BC)_20_minus_200"] = [(1, L2N, "dur_20ms", "ba"), (-1, "L1_bc_cond", "dur_20ms", "ba"),
                                                     (-1, L2N, "dur_200ms", "ba"), (1, "L1_bc_cond", "dur_200ms", "ba")]
    return contrasts


def per_backbone(meta: pd.DataFrame, arrays: dict, train=None, B: int = 2000, seed: int = 42,
                 prevalence_reps: int = 20) -> dict:
    """Notebook section 6. Returns {'tables', 'meta', 'readouts', 'fit_info', 'summary'}."""
    mt, at, s_cf = split_eval(meta, arrays)
    R, fit_info = L.build_readouts(mt, at, s_cf=s_cf, train=train)
    L2N, PREG = L.head_name(R, "L2"), L.head_name(R, "P_enc_region_mean")
    tab = L.condition_table(mt, R)
    eff = L.efficiency_table(tab, "L0", PREG)
    contrasts = backbone_contrasts(tab, L2N, PREG)
    ci, cc = L.cluster_bootstrap(mt, R, B=B, seed=seed, contrasts=contrasts,
                                 metrics=("auroc", "ba", "oracle_ba"))
    thr = pd.concat([L.threshold_table(tab, "ba", 0.90, "duration"), L.threshold_table(tab, "ba", 0.75, "snr"),
                     L.threshold_table(tab, "auroc", 0.90, "duration")], ignore_index=True)
    prev = L.prevalence_check(mt, mt["s_primary"].to_numpy(), reps=prevalence_reps, seed=seed)
    fit = pd.DataFrame([{"readout": k, **v} if isinstance(v, dict) else {"readout": k, "folds": str(v)}
                        for k, v in fit_info.items()])
    summary = {"n_items": int(len(mt)), "n_base_ids": int(mt["base_id"].nunique()),
               "n_recordings": int(mt["rec_id"].nunique()), "n_content_free": int(len(s_cf)),
               "s_cf_mean": float(np.mean(s_cf)) if len(s_cf) else None,
               "top1_is_label_share": float(mt["top1_is_label"].astype(bool).mean()),
               "confirmatory": train is not None, "head": L2N, "probe": PREG, "bootstrap_B": B, "seed": seed}
    return {"tables": {"condition_table": tab, "efficiency": eff, "bootstrap_ci": ci, "contrasts": cc,
                       "thresholds": thr, "prevalence": prev, "fit_info": fit},
            "meta": mt, "readouts": R, "fit_info": fit_info, "summary": summary}


def joint(runs: dict, B: int = 2000, seed: int = 42, h3_margin: float = 0.02) -> dict:
    """Notebook section 7 on {'base': (meta, readouts), 'lora': (meta, readouts)}."""
    if set(runs) != {"base", "lora"}:
        raise ValueError(f"joint analysis needs base and lora, got {sorted(runs)}")
    jm, JR = L.combine_runs(runs)
    confirm = "base:L2" in JR and "lora:L2" in JR
    L2b, L2l = ("base:L2", "lora:L2") if confirm else ("base:L2_cv", "lora:L2_cv")

    def jconds(axis):
        sub = jm[jm.axis == axis]
        lv = sub.duration_ms if axis == "duration" else sub.snr_db
        return sub.assign(lv=lv).sort_values("lv")["condition"].unique().tolist()

    jc = {}
    for axis in ("duration", "snr"):
        cs = jconds(axis)
        if not cs:
            continue
        jc[f"H3_{axis}_base{L2b.split(':')[1]}_minus_loraL0_ba"] = L.mean_contrast(cs, L2b, "lora:L0", "ba")
        jc[f"H4_{axis}_did_L2_minus_BC_ba"] = L.did_contrast(cs, L2b, "base:L1_bc_cond", L2l, "lora:L1_bc_cond")
        jc[f"{axis}_base_BC_minus_L0_ba"] = L.mean_contrast(cs, "base:L1_bc_cond", "base:L0", "ba")
        jc[f"{axis}_lora_BC_minus_L0_ba"] = L.mean_contrast(cs, "lora:L1_bc_cond", "lora:L0", "ba")
    ci_j, cc_j = L.cluster_bootstrap(jm, JR, B=B, seed=seed, contrasts=jc, metrics=("auroc", "ba"))
    cc_j["confirmatory"] = confirm
    h3 = cc_j.contrast.str.startswith("H3_")
    cc_j.loc[h3, "noninferior"] = cc_j.loc[h3, "lo"] > -h3_margin
    h3_joint = bool(cc_j.loc[h3, "noninferior"].all()) if h3.any() else None
    summary = {"n_items": int(len(jm)), "n_recordings": int(jm["rec_id"].nunique()), "confirmatory": confirm,
               "h3_noninferior_all_axes": h3_joint, "h3_margin": h3_margin, "bootstrap_B": B, "seed": seed}
    return {"tables": {"joint_contrasts": cc_j, "joint_bootstrap_ci": ci_j}, "summary": summary}
