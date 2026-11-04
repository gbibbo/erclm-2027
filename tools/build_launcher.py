"""Builds notebooks/launcher.ipynb, the three-cell launcher for Colab and Kaggle.

The launcher holds no experiment logic. Cell 1 reads the secrets and installs
rclone, cell 2 checks out CODE_REF and installs requirements.txt, cell 3 runs
the runner. tests/test_launcher.py checks that the committed notebook matches
this builder.

Run: python tools/build_launcher.py
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "launcher.ipynb"

INTRO = """# ERC-LM 2027 · Launcher

Choose `CONFIG` in cell 1 and press "Run all". If the session is cut off, do the same again and the runner resumes where it stopped.

| CONFIG | What it does |
| --- | --- |
| `smoke` | Smoke test. rclone uploads and downloads a file, `env.json` is pushed to `runs`, checks of the base model |
| `pilot` | Pilot, feasibility only. 50 excerpts per class, base model and LoRA, full analysis |
| `full` | Confirmatory run. Full main bank, training bank and joint analysis |

Secrets: `RCLONE_CONF`, and `GH_TOKEN` to publish to the `runs` branch (`HF_TOKEN` is optional). With `--no-runs` in `EXTRA_ARGS` nothing is published and `GH_TOKEN` is not needed. In Colab the secrets go in the key panel, and this notebook must be given access to them. In Kaggle they go in Add-ons › Secrets, with GPU and internet switched on. Keep the tab open during the run. Without `--no-runs`, results and errors reach the `runs` branch on their own."""

CELL1 = '''# 1 · Platform, secrets and rclone
CONFIG = "pilot"      # smoke | pilot | full
CODE_REF = "main"     # branch or SHA, main by default
EXTRA_ARGS = ""       # optional, e.g. "--no-runs", "--backbones base" or "--force analyze_base"
REPO = "gbibbo/erclm-2027"

import os, shutil, subprocess, sys, urllib.request, zipfile

# Kaggle first: its image derives from Colab's and also sets COLAB_RELEASE_TAG
HOST = ("kaggle" if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or os.path.isdir("/kaggle/working") else
        "colab" if os.environ.get("COLAB_RELEASE_TAG") else "local")
print("platform", HOST)
try:
    urllib.request.urlopen("https://github.com", timeout=15)
    print("internet ok")
except Exception as _exc:  # Kaggle without a verified phone runs offline, and its secrets need internet
    print(f"internet: no access ({type(_exc).__name__}). On Kaggle, verify the phone number of the account")


def _kaggle_credentials():
    """Kaggle does not hand its secrets to runs pushed with the CLI, so those runs read
    them from a private dataset named erclm-credentials (docs/running.md)."""
    import glob, json
    for _p in glob.glob("/kaggle/input/**/erclm_credentials.json", recursive=True):
        with open(_p) as _fh:
            print("credentials from the private dataset")
            return json.load(_fh)
    return {}


_FILE_SECRETS = _kaggle_credentials() if HOST == "kaggle" else {}


def _secret(name):
    if os.environ.get(name):
        return os.environ[name]
    if _FILE_SECRETS.get(name):
        return _FILE_SECRETS[name]
    if os.environ.get("ERCLM_NO_PLATFORM_SECRETS"):  # Colab CLI runs: no secrets UI to ask
        return None
    try:
        if HOST == "colab":
            from google.colab import userdata
            return userdata.get(name)
        if HOST == "kaggle":
            from kaggle_secrets import UserSecretsClient
            return UserSecretsClient().get_secret(name)
    except Exception as exc:  # the reason only, never a value
        print(f"  {name} not available ({type(exc).__name__}: {str(exc)[:160]})")
        return None
    return None


for _name in ("GH_TOKEN", "RCLONE_CONF", "HF_TOKEN"):
    _value = _secret(_name)
    if _value:
        os.environ[_name] = _value  # the runner is a subprocess and reads them from here
    print(f"{_name:12s} {'ok' if _value else 'missing'}")
_required = ("RCLONE_CONF",) if "--no-runs" in EXTRA_ARGS else ("GH_TOKEN", "RCLONE_CONF")  # GH_TOKEN only publishes
_missing = [n for n in _required if not os.environ.get(n)]
if _missing:
    raise SystemExit(f"Missing secrets {_missing}. See docs/running.md")

if shutil.which("rclone") is None:
    # Pinned release from GitHub. downloads.rclone.org answers 403 to Python's default user agent
    _zip = "/tmp/rclone.zip"
    for _url in ("https://github.com/rclone/rclone/releases/download/v1.75.2/rclone-v1.75.2-linux-amd64.zip",
                 "https://downloads.rclone.org/v1.75.2/rclone-v1.75.2-linux-amd64.zip"):
        try:
            _req = urllib.request.Request(_url, headers={"User-Agent": "curl/8.5.0"})
            with urllib.request.urlopen(_req, timeout=120) as _r, open(_zip, "wb") as _f:
                shutil.copyfileobj(_r, _f)
            break
        except Exception as _exc:
            print("could not download rclone from", _url, "->", _exc)
    else:
        raise SystemExit("Could not install rclone")
    _bin = "/usr/local/bin" if os.access("/usr/local/bin", os.W_OK) else os.path.expanduser("~/.local/bin")
    os.makedirs(_bin, exist_ok=True)
    with zipfile.ZipFile(_zip) as z:
        _member = next(m for m in z.namelist() if m.endswith("/rclone"))
        with z.open(_member) as src, open(f"{_bin}/rclone", "wb") as dst:
            shutil.copyfileobj(src, dst)
    os.chmod(f"{_bin}/rclone", 0o755)
    os.environ["PATH"] = _bin + os.pathsep + os.environ["PATH"]
print(subprocess.run(["rclone", "version"], capture_output=True, text=True).stdout.splitlines()[0])'''

CELL2 = '''# 2 · Code at CODE_REF and dependencies
import base64

CODE_DIR = {"colab": "/content/erclm-2027", "kaggle": "/kaggle/working/erclm-2027"}.get(
    HOST, os.path.abspath("erclm-2027"))
_git_env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
if os.environ.get("GH_TOKEN"):  # the public repository needs no token to check out
    _basic = base64.b64encode(f"x-access-token:{os.environ['GH_TOKEN']}".encode()).decode()
    _git_env.update({"GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": "http.https://github.com/.extraheader",
                     "GIT_CONFIG_VALUE_0": f"AUTHORIZATION: basic {_basic}"})


def _git(*args):
    r = subprocess.run(["git", *args], cwd=CODE_DIR, env=_git_env, capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"git {args[0]} failed: {r.stderr.strip()}")
    return r.stdout.strip()


shutil.rmtree(CODE_DIR, ignore_errors=True)
os.makedirs(CODE_DIR)
_git("init", "-q")
_git("remote", "add", "origin", f"https://github.com/{REPO}.git")
_git("fetch", "-q", "--depth", "1", "origin", CODE_REF)
_git("checkout", "-q", "FETCH_HEAD")
SHA = _git("rev-parse", "HEAD")
os.environ.update(ERCLM_CODE_SHA=SHA, ERCLM_CODE_REF=CODE_REF)
print("code", CODE_REF, SHA)

_pip = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", f"{CODE_DIR}/requirements.txt"],
                      capture_output=True, text=True)
if _pip.returncode:
    print(_pip.stdout[-3000:], _pip.stderr[-3000:])
    raise SystemExit("pip install failed")
print("dependencies installed")'''

CELL3 = '''# 3 · Runner. If the session is cut off, run everything again and it resumes on its own
import shlex

_cmd = [sys.executable, "-u", "-m", "erclm.runner", "--config", f"configs/{CONFIG}.yaml", *shlex.split(EXTRA_ARGS)]
_proc = subprocess.Popen(_cmd, cwd=CODE_DIR, env={**os.environ, "PYTHONPATH": f"{CODE_DIR}/src"},
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
for _line in _proc.stdout:
    print(_line, end="")
if _proc.wait():
    raise SystemExit(f"The runner exited with code {_proc.returncode}. The details are in runs/<run_id>/error.md")
print("done")'''

CELLS = [("intro", "markdown", INTRO), ("platform", "code", CELL1), ("code", "code", CELL2),
         ("run", "code", CELL3)]


def parametrized_cells(config: str, ref: str = "main", extra: str = "") -> list:
    """CELLS with CONFIG, CODE_REF and EXTRA_ARGS filled in, for Kaggle and the Colab CLI."""
    cell1 = CELL1
    for old, new in (('CONFIG = "pilot"', f"CONFIG = {config!r}"), ('CODE_REF = "main"', f"CODE_REF = {ref!r}"),
                     ('EXTRA_ARGS = ""', f"EXTRA_ARGS = {extra!r}")):
        if cell1.count(old) != 1:
            raise RuntimeError(f"launcher cell 1 no longer contains {old!r}")
        cell1 = cell1.replace(old, new)
    return [(cid, kind, cell1 if cid == "platform" else src) for cid, kind, src in CELLS]


def build(path: Path = OUT, cells=None) -> Path:
    import nbformat
    nb = nbformat.v4.new_notebook()
    nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                   "language_info": {"name": "python"},
                   "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"}}
    for cid, kind, src in cells or CELLS:
        cell = nbformat.v4.new_markdown_cell(src) if kind == "markdown" else nbformat.v4.new_code_cell(src)
        cell.id = cid
        nb.cells.append(cell)
    nbformat.validate(nb)
    path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, str(path))
    return path


if __name__ == "__main__":
    print("wrote", build())
