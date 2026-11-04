"""Runs the launcher on Kaggle from a local machine, with the Kaggle CLI.

    python tools/kaggle_push.py --config smoke --extra "--force check_base"
    kaggle kernels status <user>/erclm-launcher

Builds the kernel folder in _work/kaggle/ from tools/build_launcher.py with
CONFIG, CODE_REF and EXTRA_ARGS filled in, writes kernel-metadata.json (private,
internet on, two T4) and runs `kaggle kernels push`, which starts a run. The
credentials come from the private dataset <user>/erclm-credentials,
because Kaggle does not hand its secrets to runs pushed with the CLI (Kaggle CLI
issue 582). Results reach the runs branch as with Colab, unless --extra holds --no-runs.

--cpu runs without GPU, which costs no GPU quota. --build-only writes the
folder and stops.

    python tools/kaggle_push.py --make-credentials <file with the GitHub token>

creates or updates that dataset, always private, from the local rclone.conf
(base64) and a GitHub token reserved for Kaggle. Nothing is printed.
"""
import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SLUG = "erclm-launcher"
CREDENTIALS = "erclm-credentials"  # private dataset with erclm_credentials.json, see make_credentials()


def _builder():
    spec = importlib.util.spec_from_file_location("build_launcher", ROOT / "tools" / "build_launcher.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_kernel(out: Path, config: str, ref: str = "main", extra: str = "", user: str = "gabrielbibb",
                 gpu: bool = True) -> Path:
    B = _builder()
    cells = B.parametrized_cells(config, ref, extra)
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    B.build(out / "launcher.ipynb", cells=cells)
    meta = {"id": f"{user}/{SLUG}", "title": SLUG, "code_file": "launcher.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": gpu, "enable_internet": True,
            "machine_shape": "NvidiaTeslaT4" if gpu else "", "dataset_sources": [f"{user}/{CREDENTIALS}"],
            "competition_sources": [],
            "kernel_sources": [], "model_sources": []}
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    return out


def _kaggle() -> str:
    kaggle = shutil.which("kaggle")
    if not kaggle:
        sys.exit("kaggle CLI not found. Install it with: uv tool install kaggle")
    return kaggle


def make_credentials(token_file, user: str = "gabrielbibb") -> int:
    """Create or update the private credentials dataset. Prints no secret."""
    import base64
    token =Path(token_file).read_text(encoding="utf-8-sig").strip()
    if not token.startswith(("github_pat_", "ghp_")):
        sys.exit(f"{token_file} does not hold a GitHub token")
    conf = subprocess.run(["rclone", "config", "file"], capture_output=True, text=True, check=True)
    conf_path = Path(conf.stdout.strip().splitlines()[-1])
    creds = {"GH_TOKEN": token, "RCLONE_CONF": base64.b64encode(conf_path.read_bytes()).decode()}
    return publish_credentials(creds, user)


def publish_credentials(creds: dict, user: str = "gabrielbibb") -> int:
    """Create the private dataset with erclm_credentials.json, or add a version. Prints no secret."""
    import tempfile
    kaggle, ref = _kaggle(), f"{user}/{CREDENTIALS}"
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "erclm_credentials.json").write_text(json.dumps(creds))
        (d / "dataset-metadata.json").write_text(json.dumps(
            {"title": CREDENTIALS, "id": ref, "licenses": [{"name": "other"}]}))
        exists = subprocess.run([kaggle, "datasets", "status", ref], capture_output=True, text=True)
        if exists.returncode == 0 and "ready" in exists.stdout.lower():
            cmd = [kaggle, "datasets", "version", "-p", str(d), "-m", "credentials updated"]
        else:
            cmd = [kaggle, "datasets", "create", "-p", str(d)]  # private unless --public is passed
        res = subprocess.run(cmd, capture_output=True, text=True)
    for secret in creds.values():
        res.stdout, res.stderr = res.stdout.replace(secret, "***"), res.stderr.replace(secret, "***")
    print(res.stdout.strip(), res.stderr.strip(), sep="\n")
    return res.returncode


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--make-credentials", metavar="TOKEN_FILE",
                    help="create or update the private credentials dataset, then exit")
    ap.add_argument("--config", help="smoke, pilot or full")
    ap.add_argument("--ref", default="main", help="branch or SHA to run (default main)")
    ap.add_argument("--extra", default="", help='extra runner arguments, e.g. "--backbones base"')
    ap.add_argument("--user", default="gabrielbibb", help="Kaggle username")
    ap.add_argument("--cpu", action="store_true", help="no GPU (no quota used)")
    ap.add_argument("--build-only", action="store_true")
    args = ap.parse_args(argv)
    if args.make_credentials:
        return make_credentials(args.make_credentials, args.user)
    if not args.config:
        ap.error("--config is required")
    out = build_kernel(ROOT / "_work" / "kaggle", args.config, args.ref, args.extra, args.user, gpu=not args.cpu)
    print("kernel folder", out)
    if args.build_only:
        return 0
    res = subprocess.run([_kaggle(), "kernels", "push", "-p", str(out)], capture_output=True, text=True)
    print(res.stdout.strip(), res.stderr.strip(), sep="\n")
    return res.returncode


if __name__ == "__main__":
    sys.exit(main())
