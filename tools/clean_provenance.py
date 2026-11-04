"""Provenance stores without the recordings whose non-speech window a blind listener heard as
speech (labels voz and voz_posible in _work/closing/negatives/listening.csv, from
tools/blind_listening.py). Each dropped recording loses its speech window too. Content-free
containers stay. Writes _work/closing/runs/prov-v3-clean/<backbone>/eval as one shard, so the
provenance analyses rerun on it with ERCLM_PROV_RUN=prov-v3-clean and the original folder stays
untouched.

    PYTHONPATH=src python tools/clean_provenance.py
    ERCLM_PROV_RUN=prov-v3-clean PYTHONPATH=src python tools/closing_gpu.py prov
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from closing_analyses import WORK  # noqa: E402

SRC, DST = WORK / "runs" / "prov-v3", WORK / "runs" / "prov-v3-clean"
DROP = ("voz", "voz_posible")


def main() -> None:
    lab = pd.read_csv(WORK / "negatives" / "listening.csv")
    bad_bases = set(lab.loc[lab["label"].isin(DROP), "base_id"])
    for bb in ("base", "lora"):
        metas, arrays = [], {}
        for p in sorted((SRC / bb / "eval").glob("shard_*.csv")):
            m = pd.read_csv(p)
            with np.load(p.with_suffix(".npz")) as z:
                for k in z.files:
                    if len(z[k]) != len(m):
                        raise RuntimeError(f"{p.name}: {k} has {len(z[k])} rows, meta has {len(m)}")
                    arrays.setdefault(k, []).append(z[k])
            metas.append(m)
        meta = pd.concat(metas, ignore_index=True)
        arrays = {k: np.concatenate(v) for k, v in arrays.items()}
        content = meta["role"] != "content_free"
        bad_recs = set(meta.loc[content & meta["base_id"].isin(bad_bases), "rec_id"])
        keep = (~content | ~meta["rec_id"].isin(bad_recs)).to_numpy()
        out = DST / bb / "eval"
        out.mkdir(parents=True, exist_ok=True)
        for old in out.glob("shard_*"):
            old.unlink()
        meta[keep].reset_index(drop=True).to_csv(out / "shard_000.csv", index=False)
        np.savez(out / "shard_000.npz", **{k: v[keep] for k, v in arrays.items()})
        kept = meta[keep & content.to_numpy()]
        print(f"{bb}: dropped {len(bad_recs)} recordings, kept {kept['rec_id'].nunique()} recordings and "
              f"{len(kept)} items, {int((~content).sum())} content-free containers")


if __name__ == "__main__":
    main()
