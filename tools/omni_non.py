"""The study prompt in Qwen3-Omni scored with "NON" among the non-speech tokens (2 November 2026).

Descriptive, decided after reading the excerpt-position sweep (R5). With the study prompt the most
likely first token is "NON" (29563), and s_primary compares "S" (50) with "N" (45) only. The logits
of the three come from h_final, stored after the final norm, times those rows of thinker.lm_head at
the run's revision, fetched by byte range from the Hugging Face shard. The recomputed "S" and "N"
logits must match the stored zA and zB.

    python tools/omni_non.py <root> [--rows <rows.npz>] [--B 2000]

<root> is the one of tools/omni_seam.py. Without --rows the three rows are fetched (12 KB).
"""
import argparse
import json
import struct
import sys
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import omni_control as oc  # noqa: E402

REPO, REV = "Qwen/Qwen3-Omni-30B-A3B-Instruct", "26291f793822fb6be9555850f06dfe95f2d7e695"
TOKENS = {"S": 50, "N": 45, "NON": 29563}


def _get(url: str, a: int, b: int) -> bytes:
    req = urllib.request.Request(url, headers={"Range": f"bytes={a}-{b}"})
    with urllib.request.urlopen(req) as r:
        return r.read()


def fetch_rows() -> dict:
    base = f"https://huggingface.co/{REPO}/resolve/{REV}/"
    with urllib.request.urlopen(base + "model.safetensors.index.json") as r:
        shard = json.load(r)["weight_map"]["thinker.lm_head.weight"]
    url = base + shard
    n = struct.unpack("<Q", _get(url, 0, 7))[0]
    t = json.loads(_get(url, 8, 8 + n - 1))["thinker.lm_head.weight"]
    if t["dtype"] != "BF16":
        raise SystemExit(f"lm_head is {t['dtype']}")
    d, rows = t["shape"][1], {}
    for k, tok in TOKENS.items():
        a = 8 + n + t["data_offsets"][0] + tok * d * 2
        raw = np.frombuffer(_get(url, a, a + 2 * d - 1), dtype=np.uint16).astype(np.uint32) << 16
        rows[k] = raw.view(np.float32).astype(np.float64)
    return rows


def main(root: Path, rows: dict, B: int = 2000, seed: int = 42) -> dict:
    root = Path(root)
    m0, _ = oc.load(root / "omni" / "ab")
    short = m0[m0["condition"].isin(oc.SHORT)]
    ids, S, res = list(short.index), {}, {"first_token": {}}
    for k, src in {"seam": root / "omni" / "hand_csl", "centre": root / "seam" / "hand_csl_shift500"}.items():
        m, a = oc.aligned(*oc.load(src, ids=set(ids), features=("h_final",)), ids)
        z = {t: a["h_final"].astype(np.float64) @ w for t, w in rows.items()}
        err = max(np.abs(z["S"] - m["zA"]).max(), np.abs(z["N"] - m["zB"]).max())
        if err > 0.05:
            raise SystemExit(f"{k}: recomputed logits differ from zA, zB by {err:.3f}")
        res[f"max_err_{k}"] = float(err)
        res["first_token"][k] = {str(t): int(c) for t, c in m["top1"].value_counts().items()}
        S[f"{k}_registered"] = m["s_primary"].to_numpy(float)
        S[f"{k}_with_non"] = z["S"] - np.logaddexp(z["N"], z["NON"])
    t = oc.paired(short, S, [("centre_registered", "seam_registered"), ("centre_with_non", "seam_with_non")],
                  B=B, seed=seed, which=("auroc", "L0", "L1"))
    for metric in ("auroc", "L0", "L1"):
        for w in ("seam_with_non", "centre_with_non", "centre_with_non - seam_with_non",
                  "centre_registered - seam_registered"):
            res[f"{metric}/{w}"] = oc.get(t, "short", metric, w)
    print(json.dumps(res, indent=1, default=float))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--rows")
    ap.add_argument("--B", type=int, default=2000)
    a = ap.parse_args()
    if a.rows:
        z = np.load(a.rows)
        rows = {k: z[str(tok)].astype(np.float64) for k, tok in TOKENS.items()}
    else:
        rows = fetch_rows()
    main(Path(a.root), rows, B=a.B)
