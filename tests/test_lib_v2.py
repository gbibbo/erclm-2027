"""Tests for erclm_lib_v2 that need no GPU or model weights.

Real-data fixtures (passed by environment variables):
  ERCLM_PRED_CSV  a predictions.csv from audits/round2/b2_normalization (repository of the study under review)
  ERCLM_SDT_CSV   erc_lm_2027_SDT_duracion_SNR.csv (independently computed SDT table)

Run: python tests/test_lib_v2.py
"""
import os
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from erclm import lib_v2 as L  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures"
PRED_CSV = os.environ.get("ERCLM_PRED_CSV", str(FIXTURES / "predictions_csl_06_lora_opro_template.csv"))
SDT_CSV = os.environ.get("ERCLM_SDT_CSV", str(FIXTURES / "erc_lm_2027_SDT_duracion_SNR.csv"))


# ----------------------------------------------------------------- A. manifest

def test_parse_real_names_match_csl_labels():
    if not PRED_CSV:
        print("  (skipped: ERCLM_PRED_CSV not set)")
        return
    raw = pd.read_csv(PRED_CSV)
    parsed = pd.DataFrame([L.parse_variant(p) for p in raw["audio_path"]])
    assert (parsed["axis"].to_numpy() == raw["variant_type"].astype(str).to_numpy()).all()
    assert (parsed["condition"].to_numpy() == raw["condition"].astype(str).to_numpy()).all()
    assert parsed.loc[parsed.axis == "snr", "duration_ms"].eq(1000).all()
    assert set(parsed.loc[parsed.axis == "snr", "snr_db"]) == {-10, -5, 0, 5, 10, 20}
    assert set(parsed.loc[parsed.axis == "reverb", "t60_s"]) == {0.0, 0.3, 1.0, 2.5}


def test_parse_factorial_training_name():
    f = L.parse_variant("voxconverse_hkzpa_036_1000ms_dur100ms_snr-10dB.wav")
    assert f["axis"] == "factorial" and f["duration_ms"] == 100 and f["snr_db"] == -10
    assert f["condition"] == "dur_100ms|snr_-10dB"
    g = L.parse_variant("x/esc50_1-45645-A-31_0152_1000ms_snr+0dB.wav")
    assert g["axis"] == "snr" and g["condition"] == "snr_0dB" and g["snr_db"] == 0


def test_ids_on_real_names():
    names = ["voxconverse_wewoz_0454_1000ms_dur500ms.wav",
             "esc50_1-45645-A-31_0152_1000ms_dur20ms.wav",
             "esc50_1-45645-B-31_0001_1000ms_filterbandpass.wav"]
    assert L.base_id(names[0]) == "voxconverse_wewoz_0454"
    assert L.base_id(names[1]) == "esc50_1-45645-A-31_0152"
    assert L.recording_id(names[0]) == "vox:wewoz"
    assert L.recording_id(names[1]) == L.recording_id(names[2]) == "esc:1-45645"
    if PRED_CSV:
        raw = pd.read_csv(PRED_CSV)
        b = raw["audio_path"].map(L.base_id)
        assert b.nunique() == 970 and b.value_counts().eq(22).all()
        r = raw["audio_path"].map(L.recording_id)
        assert r.str.startswith("other:").sum() == 0
        assert r.nunique() == 650


def test_load_manifest_and_pilot():
    if not PRED_CSV:
        print("  (skipped: ERCLM_PRED_CSV not set)")
        return
    m = L.load_manifest(PRED_CSV, audio_root="/content/audio")
    assert len(m) == 21340 and set(m["label"]) == {0, 1}
    assert Path(m["path"].iloc[0]).as_posix().startswith("/content/audio/") and m["path"].iloc[0].endswith(".wav")
    assert all(c in m.columns for c in L.META_COLS)
    p = L.select_pilot(m, n_per_class=50, seed=42)
    bases = p.drop_duplicates("base_id")
    assert len(p) == 2200 and len(bases) == 100
    assert bases["label"].value_counts().to_dict() == {1: 50, 0: 50}
    assert bases["rec_id"].is_unique
    p2 = L.select_pilot(m, n_per_class=50, seed=42)
    assert p2["variant_id"].tolist() == p["variant_id"].tolist()


# ----------------------------------------------------------------- B. geometry

def test_token_geometry():
    assert L.n_audio_tokens(32000) == 50
    assert L.n_audio_tokens(480000) == 750
    for d in (20, 40, 60, 80, 100, 200, 500, 1000):
        idx = L.region_token_indices(d, 50)
        assert idx.min() >= 0 and idx.max() <= 49
        assert 24 in idx and 25 in idx
        span_ms = (idx.max() - idx.min() + 1) * 40
        assert span_ms >= d
    assert L.region_token_indices(20, 50).tolist() == [23, 24, 25, 26]
    assert L.region_token_indices(1000, 50).tolist() == list(range(11, 39))
    assert L.region_token_indices(1000, 50, margin=0).tolist() == list(range(12, 38))
    assert L.region_token_indices(0, 50).tolist() == [24, 25]


# ----------------------------------------------------------------- E. metrics

def test_auc_ba_against_sklearn():
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    rng = np.random.default_rng(0)
    for trial in range(20):
        n = rng.integers(20, 300)
        y = rng.integers(0, 2, n)
        if y.min() == y.max():
            y[0] = 1 - y[0]
        s = np.round(rng.normal(size=n) + y * rng.uniform(0, 2), 1)   # rounding creates ties
        w = rng.integers(0, 4, n).astype(float)
        if w[y == 1].sum() == 0 or w[y == 0].sum() == 0:
            w[:] = 1
        assert abs(L.auc_w(y, s) - roc_auc_score(y, s)) < 1e-10
        assert abs(L.auc_w(y, s, w) - roc_auc_score(y, s, sample_weight=w)) < 1e-10
        pred = s > 0.3
        assert abs(L.ba_w(y, pred) - balanced_accuracy_score(y, pred)) < 1e-10
        assert abs(L.ba_w(y, pred, w) - balanced_accuracy_score(y, pred, sample_weight=w)) < 1e-10


def test_oracle_ba_is_max_over_thresholds():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 200)
    s = np.round(rng.normal(size=200) + y, 1)
    brute = max(L.ba_w(y, s > t) for t in np.r_[np.unique(s) - 1e-9, np.inf])
    assert abs(L.oracle_ba_w(y, s) - max(brute, 0.5)) < 1e-12
    w = rng.integers(1, 3, 200).astype(float)
    brute_w = max(L.ba_w(y, s > t, w) for t in np.r_[np.unique(s) - 1e-9, np.inf])
    assert abs(L.oracle_ba_w(y, s, w) - max(brute_w, 0.5)) < 1e-12


def test_sdt_reproduces_reference_table():
    if not SDT_CSV:
        print("  (skipped: ERCLM_SDT_CSV not set)")
        return
    t = pd.read_csv(SDT_CSV)
    worst = 0.0
    for r in t.itertuples():
        out = L.sdt_counts(r.tp, r.fn, r.fp, r.tn)
        worst = max(worst, abs(out["dprime"] - r.dprime_corrected), abs(out["c"] - r.c_corrected))
    assert worst < 1e-4, worst


def test_d_a_and_efficiency():
    assert abs(L.d_a_from_auc(0.5)) < 1e-12
    assert abs(L.d_a_from_auc(0.76025) - 1.0) < 1e-3          # Phi(1/sqrt2) = 0.76025
    e = L.rel_efficiency([1.0, 2.0, 1.0], [2.0, 1.0, 0.0])
    assert abs(e[0] - 0.25) < 1e-12 and abs(e[1] - 4.0) < 1e-12 and np.isnan(e[2])


# ----------------------------------------------------------------- D. readouts

def _synthetic(n_rec_per_class=40, items_per_rec=2, conds=("dur_20ms", "dur_200ms", "dur_1000ms"),
               dim=16, seed=0):
    rng = np.random.default_rng(seed)
    rows, H = [], []
    strength = {"dur_20ms": 0.3, "dur_200ms": 1.0, "dur_1000ms": 2.0}
    bias = {"dur_20ms": -2.0, "dur_200ms": -1.0, "dur_1000ms": 0.0}   # conservative native criterion
    direction = np.random.default_rng(12345).normal(size=dim)   # same signal axis for every seed
    for lab in (1, 0):
        for r in range(n_rec_per_class):
            rec = f"{'vox' if lab else 'esc'}:{r}"
            for it in range(items_per_rec):
                base = f"{rec}_{it}"
                for c in conds:
                    z = lab * strength[c]
                    h = rng.normal(size=dim) + z * direction
                    s = 1.2 * z + bias[c] + rng.normal()
                    rows.append({"variant_id": f"{base}_{c}", "base_id": base, "rec_id": rec, "label": lab,
                                 "role": "test", "axis": "duration", "condition": c,
                                 "duration_ms": float(c.split("_")[1][:-2]), "snr_db": np.nan, "t60_s": 0.0,
                                 "reverb": "none", "filter": "none", "s_primary": s})
                    H.append(h)
    meta = pd.DataFrame(rows)
    arrays = {"h_final": np.asarray(H, dtype=np.float16),
              "enc_region_mean": (np.asarray(H) + rng.normal(size=(len(H), dim))).astype(np.float16)}
    return meta, arrays


def test_l1_thresholds_properties():
    meta, _ = _synthetic()
    s = meta["s_primary"].to_numpy()
    y = meta["label"].to_numpy()
    for m in ("bc_global", "bc_cond", "center_cond", "median_cond"):
        thr = L.l1_thresholds(meta, s, m)
        assert thr.shape == s.shape and np.isfinite(thr).all()
    thr = L.l1_thresholds(meta, s, "bc_cond")
    for c, g in meta.groupby("condition").indices.items():
        assert abs(thr[g][0] - L.logit(np.mean(L.sigmoid(s[g])))) < 1e-12
    med = L.l1_thresholds(meta, s, "median_cond")
    for c, g in meta.groupby("condition").indices.items():
        assert abs(np.mean(s[g] > med[g]) - 0.5) < 0.02
    # calibration can only move BA towards the oracle, never above it
    for c, g in meta.groupby("condition").indices.items():
        o = L.oracle_ba_w(y[g], s[g])
        for m in ("none", "bc_cond", "center_cond", "median_cond"):
            assert L.ba_w(y[g], s[g] > L.l1_thresholds(meta, s, m)[g]) <= o + 1e-12
    cf = L.l1_thresholds(meta, s, "cf", s_cf=[0.5, 1.5])
    assert np.allclose(cf, 1.0)


def test_crossfit_and_readouts():
    meta, arrays = _synthetic()
    y = meta["label"].to_numpy()
    R, info = L.build_readouts(meta, arrays, Cs=(1e-2, 1e-1), n_splits=4)
    assert {"L0", "L1_bc_cond", "L2_cv", "P_enc_region_mean_cv"} <= set(R)
    assert "L2" not in R                                            # cross-fitted heads are exploratory
    for name, r in R.items():
        s, thr, rule = L.readout_parts(r)
        assert np.isfinite(s).all() and s.shape == (len(meta),) and np.isfinite(thr).all()
        assert (rule is not None) == (name in {f"L1_{m}" for m in L.BATCH_RULES})
    tab = L.condition_table(meta, R)
    l0 = tab[(tab.readout == "L0") & (tab.condition == "dur_1000ms")].iloc[0]
    bc = tab[(tab.readout == "L1_bc_cond") & (tab.condition == "dur_1000ms")].iloc[0]
    assert abs(l0.auroc - bc.auroc) < 1e-12                        # L1 never changes AUROC
    l2 = tab[(tab.readout == "L2_cv") & (tab.condition == "dur_1000ms")].iloc[0]
    assert l2.auroc > 0.9
    assert bc.ba > l0.ba + 0.05                                     # biased native criterion is corrected
    assert bc.ba <= bc.oracle_ba + 1e-12
    eff = L.efficiency_table(tab, "L0", L.head_name(R, "P_enc_region_mean"))
    assert len(eff) == 3 and "rel_efficiency" in eff
    pv = L.pivot(tab, "ba", "duration")
    assert pv.shape == (3, len(R)) and list(pv.index.get_level_values(0)) == [20.0, 200.0, 1000.0]
    # matched supervision path: train on one synthetic set, evaluate on another
    meta_tr, arrays_tr = _synthetic(seed=5)
    R2, info2 = L.build_readouts(meta, arrays, train=(meta_tr, arrays_tr), Cs=(1e-2, 1e-1))
    assert L.auc_w(y, R2["L2"][0]) > 0.75 and "L2_cv" not in R2


def test_crossfit_never_splits_recordings():
    meta, arrays = _synthetic()
    from sklearn.model_selection import StratifiedGroupKFold
    g = meta["rec_id"].to_numpy()
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(arrays["h_final"], meta.label, g):
        assert not (set(g[tr]) & set(g[te]))


# ----------------------------------------------------------------- F. bootstrap

def test_cluster_bootstrap():
    meta, arrays = _synthetic()
    R = {"L0": (meta["s_primary"].to_numpy(), np.zeros(len(meta))),
         "L1_bc_cond": (meta["s_primary"].to_numpy(),
                        L.l1_thresholds(meta, meta["s_primary"].to_numpy(), "bc_cond"))}
    contrasts = {"bc_minus_l0_20ms": [(1, "L1_bc_cond", "dur_20ms", "ba"), (-1, "L0", "dur_20ms", "ba")],
                 "auc_1000_minus_20": [(1, "L0", "dur_1000ms", "auroc"), (-1, "L0", "dur_20ms", "auroc")]}
    contrasts["mean_bc_minus_l0"] = L.mean_contrast(["dur_20ms", "dur_200ms", "dur_1000ms"], "L1_bc_cond", "L0", "ba")
    contrasts["headroom_1000"] = [(1, "L0", "dur_1000ms", "oracle_ba"), (-1, "L0", "dur_1000ms", "ba")]
    ci, cc = L.cluster_bootstrap(meta, R, B=200, seed=1, contrasts=contrasts, metrics=("auroc", "ba", "oracle_ba"))
    assert ((ci["lo"] <= ci["estimate"] + 1e-12) & (ci["estimate"] <= ci["hi"] + 1e-12)).all()
    assert cc.set_index("contrast").loc["headroom_1000", "lo"] > 0     # biased L0 leaves criterion headroom
    assert len(L.mean_contrast(["a", "b"], "x", "y", "ba")) == 4
    row = cc.set_index("contrast").loc["auc_1000_minus_20"]
    assert row["lo"] > 0                                            # strong synthetic effect is detected
    tab = L.condition_table(meta, R)
    est = ci[(ci.readout == "L0") & (ci.condition == "dur_20ms") & (ci.metric == "auroc")]["estimate"].iloc[0]
    ref = tab[(tab.readout == "L0") & (tab.condition == "dur_20ms")]["auroc"].iloc[0]
    assert abs(est - ref) < 1e-12


# ----------------------------------------------------------------- v2 additions

def test_weighted_median():
    rng = np.random.default_rng(3)
    for n in (1, 2, 5, 10, 11):
        x = rng.normal(size=n)
        assert abs(L.weighted_median(x) - np.median(x)) < 1e-12
        w = rng.integers(0, 4, n)
        if w.sum() == 0:
            w[0] = 1
        assert abs(L.weighted_median(x, w) - np.median(np.repeat(x, w))) < 1e-12


def test_rules_with_integer_weights_equal_expanded_sample():
    meta, _ = _synthetic(n_rec_per_class=10)
    s = meta["s_primary"].to_numpy()
    rng = np.random.default_rng(4)
    w = rng.integers(0, 3, len(s)).astype(float)
    for g, idx in meta.groupby("condition").indices.items():     # every condition keeps some weight
        if w[idx].sum() == 0:
            w[idx[0]] = 1
    rep = np.repeat(np.arange(len(s)), w.astype(int))
    meta_exp = meta.iloc[rep].reset_index(drop=True)
    for rule in L.BATCH_RULES:
        a = L.apply_rule(meta, s, rule, w)
        b = L.apply_rule(meta_exp, s[rep], rule)
        for cond in meta["condition"].unique():
            va = a[(meta["condition"] == cond).to_numpy()][0]
            vb = b[(meta_exp["condition"] == cond).to_numpy()][0]
            assert abs(va - vb) < 1e-9, (rule, cond, va, vb)
        assert np.allclose(L.apply_rule(meta, s, rule), L.l1_thresholds(meta, s, rule))


def test_bootstrap_recalibrates_batch_rules():
    meta, _ = _synthetic()
    s = meta["s_primary"].to_numpy()
    thr = L.l1_thresholds(meta, s, "bc_cond")
    R = {"frozen": (s, thr, None), "recal": (s, thr, "bc_cond")}
    ci, _ = L.cluster_bootstrap(meta, R, B=300, seed=2, metrics=("ba",))
    tab = L.condition_table(meta, R)
    for cond in meta["condition"].unique():
        f = ci[(ci.readout == "frozen") & (ci.condition == cond)].iloc[0]
        r = ci[(ci.readout == "recal") & (ci.condition == cond)].iloc[0]
        assert abs(f.estimate - r.estimate) < 1e-12                 # same point estimate
        assert abs(r.estimate - tab[(tab.readout == "recal") & (tab.condition == cond)].ba.iloc[0]) < 1e-12
    widths = ci.assign(w=ci.hi - ci.lo).groupby("readout")["w"].mean()
    assert widths["recal"] != widths["frozen"]                       # offset uncertainty now enters


def test_combine_runs_paired_did_is_exact():
    meta, arrays = _synthetic(n_rec_per_class=12)
    R, _ = L.build_readouts(meta, arrays, Cs=(1e-2,), n_splits=3, probe_features=())
    shuffled = meta.sample(frac=1.0, random_state=0).reset_index()
    perm = shuffled["index"].to_numpy()
    R_sh = {k: tuple(np.asarray(v)[perm] if isinstance(v, np.ndarray) else v for v in r) for k, r in R.items()}
    jm, JR = L.combine_runs({"base": (meta, R), "lora": (shuffled.drop(columns="index"), R_sh)})
    assert len(jm) == len(meta) and set(JR) == {f"{t}:{k}" for t in ("base", "lora") for k in R}
    assert np.allclose(JR["base:L0"][0], JR["lora:L0"][0])          # realigned despite shuffling
    conds = list(meta["condition"].unique())
    contrasts = {"H4_did": L.did_contrast(conds, "base:L2_cv", "base:L1_bc_cond", "lora:L2_cv", "lora:L1_bc_cond"),
                 "H3_like": L.mean_contrast(conds, "base:L2_cv", "lora:L0", "ba")}
    ci, cc = L.cluster_bootstrap(jm, JR, B=100, seed=0, metrics=("ba",), contrasts=contrasts)
    did = cc.set_index("contrast").loc["H4_did"]
    assert abs(did.estimate) < 1e-12 and abs(did.lo) < 1e-12 and abs(did.hi) < 1e-12 and did.se < 1e-12
    assert {"se", "mde80"} <= set(cc.columns)
    bad = shuffled.drop(columns="index").copy()
    bad.loc[0, "label"] = 1 - bad.loc[0, "label"]
    try:
        L.combine_runs({"base": (meta, R), "lora": (bad, R_sh)})
        raise AssertionError("label mismatch not detected")
    except ValueError:
        pass


def test_prevalence_check():
    meta, _ = _synthetic(n_rec_per_class=100)
    s = meta["s_primary"].to_numpy()
    pc = L.prevalence_check(meta, s, prevalences=(0.2, 0.5, 0.8), reps=30, seed=0)
    assert set(pc["prevalence"]) == {0.2, 0.5, 0.8} and {"L0", "L1_bc_cond", "oracle"} <= set(pc.columns)
    l0 = pc.pivot_table(index="condition", columns="prevalence", values="L0")
    assert (l0.max(axis=1) - l0.min(axis=1)).max() < 0.03          # fixed threshold: BA is prevalence-free
    bc = pc.pivot_table(index="condition", columns="prevalence", values="L1_bc_cond")
    assert (bc.max(axis=1) - bc.min(axis=1)).max() > 0.01           # batch rule moves with prevalence


# ----------------------------------------------------------------- G. thresholds

def test_crossing_policy():
    lv = [20, 40, 80, 160]
    r = L.crossing(lv, [0.6, 0.8, 0.95, 0.97], 0.9, "log2")
    assert r["status"] == "ok" and 40 < r["threshold"] < 80
    assert abs(r["threshold"] - 40 * 2 ** (2 / 3)) < 1e-9          # (0.9-0.8)/(0.95-0.8) of an octave
    assert L.crossing(lv, [0.95] * 4, 0.9)["status"] == "censored_below"
    assert L.crossing(lv, [0.5] * 4, 0.9)["status"] == "censored_above"
    amb = L.crossing(lv, [0.95, 0.8, 0.95, 0.97], 0.9)
    assert amb["status"] == "ambiguous" and 40 < amb["threshold"] < 80
    assert L.crossing(lv, [0.95, 0.97, 0.95, 0.8], 0.9)["status"] == "ambiguous"
    s = L.crossing([-10, -5, 0, 5], [0.6, 0.7, 0.8, 0.9], 0.75, "linear")
    assert s["status"] == "ok" and abs(s["threshold"] - (-2.5)) < 1e-9


# ----------------------------------------------------------------- C. store

def test_feature_store_roundtrip_and_resume():
    with tempfile.TemporaryDirectory() as d:
        st = L.FeatureStore(d, shard_size=3)
        for i in range(7):
            st.add({"variant_id": f"v{i}", "x": i}, {"h": np.full(4, i, np.float16)})
        st.flush()
        assert st.done_ids() == {f"v{i}" for i in range(7)}
        meta, arr = L.load_store(d)
        assert len(meta) == 7 and arr["h"].shape == (7, 4)
        assert (arr["h"][:, 0] == meta["x"].to_numpy()).all()
        # an orphan npz (crash before its CSV) is ignored, and new shards get fresh indices
        Path(d, "shard_00099.npz").write_bytes(b"garbage")
        st2 = L.FeatureStore(d, shard_size=10)
        st2.add({"variant_id": "v7", "x": 7}, {"h": np.full(4, 7, np.float16)})
        st2.flush()
        meta, arr = L.load_store(d)
        assert len(meta) == 8 and (arr["h"][:, 0] == meta["x"].to_numpy()).all()


def main():
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception:
            failed += 1
            print(f"FAIL {name}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
