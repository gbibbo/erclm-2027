"""Packs an audio bank as <name>.tar plus .sha256, validates it and uploads it to Drive.

Run it on the machine that holds the WAVs (a local machine, an external disk, a cluster).

    python tools/pack_bank.py --name test_bank_v1 \\
        --csv path/to/test_metadata.csv --audio path/to/audio/test --out /tmp/banks \\
        --expect rows=21340 conditions=22 base_ids=970 rec_ids=650 speech_share=0.5 \\
        --names tests/fixtures/predictions_csl_06_lora_opro_template.csv \\
        --upload erclm:ERC-LM-2027

--csv takes any CSV with an audio-path column and a label column, for example
a predictions.csv of the repository of the study under review, whose paths point to another machine. Only
the file names are used. The bank is validated before packing and nothing is
uploaded if a check fails (unless --allow-failures).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from erclm import banks  # noqa: E402
from erclm import lib_v2 as L  # noqa: E402
from erclm.storage import RcloneRemote  # noqa: E402


def _expect(items):
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        out[k] = float(v) if "." in v else int(v)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--name", required=True, help="e.g. test_bank_v1 or train_bank_v1")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--audio", required=True, help="folder with the WAV files")
    ap.add_argument("--out", required=True, help="where the tar is written")
    ap.add_argument("--role", default=None, help="test or train (default from the name)")
    ap.add_argument("--expect", nargs="*", help="key=value checks: rows, conditions, base_ids, rec_ids, speech_share")
    ap.add_argument("--names", help="CSV whose audio_path file names must equal the bank's")
    ap.add_argument("--upload", help="rclone base, e.g. erclm:ERC-LM-2027. The tar goes to <base>/data/")
    ap.add_argument("--rclone-config", default=None)
    ap.add_argument("--allow-failures", action="store_true")
    args = ap.parse_args(argv)

    role = args.role or ("train" if "train" in args.name else "test")
    man = L.load_manifest(args.csv, audio_root=args.audio, role=role)
    print(f"{len(man)} rows, checking every WAV")
    rep = banks.validate(man, _expect(args.expect), check_audio=True, names_csv=args.names)
    print(json.dumps({k: v for k, v in rep.items() if k != "audio_bad"}, indent=2, default=str))
    if not rep["ok"] and not args.allow_failures:
        sys.exit(f"validation failed: {rep['failures']}")
    tar, sha = banks.make_tar(args.name, args.csv, args.audio, args.out)
    print(f"{tar} ({tar.stat().st_size / 2**30:.2f} GB) sha256 {sha}")
    if args.upload:
        remote = RcloneRemote(args.upload, config=args.rclone_config)
        for p in (tar, tar.with_name(tar.name + ".sha256")):
            info = remote.put(p, f"data/{p.name}")
            print(f"uploaded {p.name} -> {args.upload}/data/ {info}")
        (Path(args.out) / f"{args.name}.report.json").write_text(json.dumps(rep, indent=2, default=str))
        remote.put(Path(args.out) / f"{args.name}.report.json", f"data/manifests/{args.name}.report.json")


if __name__ == "__main__":
    main()
