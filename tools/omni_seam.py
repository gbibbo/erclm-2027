"""The registered reading of the excerpt-position sweep in Qwen3-Omni (R5, 2 November 2026).

The rule is in docs/exploratory_log.md, "Sweep of the crop position in Qwen3-Omni (2 November)",
written before omni-seam-v1. On the 300 control clips at 20, 40 and 60 ms, paired by item, the
AUROC of s with the excerpt at each position against the excerpt on the seam (position 0).

    python tools/omni_seam.py <root> [--B 2000] [--out <folder>]

<root> holds, as the runner writes them,

    omni/ab/, omni/ab_shift500/, omni/hand_csl/   work/omni-control-v1/omni/<track>/
    seam/<track>/                                 work/omni-seam-v1/omni/<track>/
    q2_base/                                      work/full-v1/base/eval/   (the gap to Qwen2-Audio)

Intervals are 95% percentiles of a bootstrap over recordings stratified by class, 2,000
replicates, seed 42, one draw for every position. The encoder probes are cross-fitted once
by recording with the same folds for every position and stay fixed in the bootstrap.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import omni_control as oc  # noqa: E402

CONDS = ["dur_20ms", "dur_40ms", "dur_60ms"]
SHIFTS = {-500: ("seam", "ab_shift-500"), -250: ("seam", "ab_shift-250"), 0: ("omni", "ab"),
          40: ("seam", "ab_shift40"), 250: ("seam", "ab_shift250"), 500: ("omni", "ab_shift500"),
          750: ("seam", "ab_shift750")}
OFF_SEAM = [-500, -250, 250, 500, 750]


def name(sh: int) -> str:
    return f"pos{sh:+d}"


def reading(t, group: str = "short3") -> dict:
    g = lambda what: oc.get(t, group, "auroc", what)  # noqa: E731
    d = {sh: g(f"{name(sh)} - {name(0)}") for sh in OFF_SEAM + [40]}
    sym = g(f"{name(-500)} - {name(500)}")
    edge = g(f"{name(500)} - {name(40)}")
    out = {"delta": {str(k): v for k, v in d.items()}, "symmetry": sym, "centre_minus_edge": edge}
    hurt = [sh for sh in OFF_SEAM if d[sh]["ci"][0] > 0]
    out["R5a"] = ("the seam hurts at every position tested" if len(hurt) == len(OFF_SEAM)
                  else f"the rotation improves only at {hurt}")
    out["R5b"] = ("the position within the block matters, not the position in the container" if sym["ci"][0] <= 0 <= sym["ci"][1]
                  else "the position in the container also matters")
    split = d[40]["ci"][0] > 0 and d[40]["value"] >= d[500]["value"] / 2
    out["R5c"] = {"splitting the crop across blocks explains most of it": bool(split),
                  "being at the edge of a block also hurts": bool(edge["ci"][0] > 0)}
    return out


def main(root: Path, B: int = 2000, seed: int = 42, out: Path = None, with_probes: bool = True):
    root = Path(root)
    out = Path(out or root / "reading_seam")
    out.mkdir(parents=True, exist_ok=True)
    feats = ("enc_region_mean",) if with_probes else ()
    m0, a0 = oc.load(root / "omni" / "ab", features=feats)
    m0 = m0[m0["condition"].isin(CONDS)]
    ids = list(m0.index)
    meta, systems, probes = m0, {}, {}
    for sh, (where, track) in SHIFTS.items():
        m, a = oc.aligned(*oc.load(root / where / track, ids=set(ids), features=feats), ids)
        if not (m["label"].to_numpy() == meta["label"].to_numpy()).all():
            raise SystemExit(f"labels of {track} differ")
        systems[name(sh)] = m["s_primary"].to_numpy(float)
        if with_probes:
            probes[f"P_{name(sh)}"] = oc.crossfit(meta, a["enc_region_mean"])
    q2, _ = oc.aligned(*oc.load(root / "q2_base", ids=set(ids)), ids)
    systems["q2_base"] = q2["s_primary"].to_numpy(float)
    contrasts = [(name(sh), name(0)) for sh in OFF_SEAM + [40]]
    contrasts += [(name(-500), name(500)), (name(500), name(40)), (name(-250), name(750)),
                  ("q2_base", name(0)), ("q2_base", name(500))]
    contrasts += [(f"P_{name(sh)}", f"P_{name(0)}") for sh in OFF_SEAM + [40] if with_probes]
    oc.SETS["short3"] = CONDS
    t = oc.paired(meta, {**systems, **probes}, contrasts, B=B, seed=seed, which=("auroc", "L0", "L1"))
    t.to_csv(out / "seam.csv", index=False)
    res = {"B": B, "seed": seed, "R5": reading(t)}
    # hand prompt with the excerpt off the seam, at 20 and 40 ms (descriptive)
    short = meta[meta["condition"].isin(oc.SHORT)]
    sids = list(short.index)
    hs = {}
    for k, (where, track) in {"hand_0": ("omni", "hand_csl"), "hand_500": ("seam", "hand_csl_shift500"),
                              "ab_0": ("omni", "ab"), "ab_500": ("omni", "ab_shift500"),
                              "q2_base": ("q2", None)}.items():
        src = root / "q2_base" if where == "q2" else root / where / track
        m, _ = oc.aligned(*oc.load(src, ids=set(sids)), sids)
        hs[k] = m["s_primary"].to_numpy(float)
    th = oc.paired(short, hs, [("hand_500", "ab_500"), ("hand_500", "hand_0"), ("q2_base", "hand_500"),
                               ("q2_base", "ab_500")], B=B, seed=seed, which=("auroc",))
    th.to_csv(out / "hand.csv", index=False)
    res["hand"] = {w: oc.get(th, "short", "auroc", w) for w in ("hand_500 - ab_500", "hand_500 - hand_0",
                                                                "q2_base - hand_500", "q2_base - ab_500")}
    (out / "reading.json").write_text(json.dumps(res, indent=2, default=float, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: res["R5"][k] for k in ("R5a", "R5b", "R5c")}, indent=1, ensure_ascii=False))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out")
    a = ap.parse_args()
    main(Path(a.root), B=a.B, seed=a.seed, out=a.out)
