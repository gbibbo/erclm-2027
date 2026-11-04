"""Remote storage for heavy artefacts.

RcloneRemote talks to Google Drive through rclone. LocalRemote stores files in
a local folder and stands in for Drive in the tests. Both share the write
protocol of put(): upload to <name>.part, rename to <name>, then check size and
MD5 of the remote file against the local one. A file counts as stored only
after put() returns. Paths are relative to the remote root and use '/'.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional

PART = ".part"


class StorageError(RuntimeError):
    pass


def md5_file(path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def sha256_file(path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _join(*parts) -> str:
    return "/".join(p.strip("/") for p in parts if p and p.strip("/"))


class Remote:
    """Write protocol and helpers on top of seven backend primitives."""

    retries = 3
    backoff_s = 2.0

    # ------------------------------------------------------------ primitives
    def _upload(self, local: Path, rel: str): raise NotImplementedError
    def _download(self, rel: str, local: Path): raise NotImplementedError
    def _move(self, src: str, dst: str): raise NotImplementedError
    def _stat(self, rel: str) -> Optional[dict]: raise NotImplementedError
    def _list(self, rel: str) -> dict: raise NotImplementedError
    def _delete(self, rel: str): raise NotImplementedError
    def _purge(self, rel: str): raise NotImplementedError

    def describe(self) -> str:
        return type(self).__name__

    # ------------------------------------------------------------ protocol
    def _retry(self, what: str, fn):
        last = None
        for attempt in range(self.retries):
            try:
                return fn()
            except StorageError as exc:
                last = exc
                time.sleep(self.backoff_s * (2 ** attempt))
        raise StorageError(f"{what} failed after {self.retries} attempts: {last}")

    def put(self, local, rel: str) -> dict:
        """Atomic, verified upload. Returns {'size', 'md5'} of the stored file."""
        local = Path(local)
        want = {"size": local.stat().st_size, "md5": md5_file(local)}

        def attempt():
            part = rel + PART
            self._upload(local, part)
            self._move(part, rel)  # replaces an existing file (rclone moveto deletes it first)
            got = self._stat(rel)
            if got is None:
                raise StorageError(f"{rel} missing after rename")
            if got["size"] != want["size"] or (got.get("md5") and got["md5"] != want["md5"]):
                raise StorageError(f"{rel} verification failed: remote {got}, local {want}")
            return want

        return self._retry(f"put {rel}", attempt)

    def put_bytes(self, data: bytes, rel: str) -> dict:
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / Path(rel).name
            p.write_bytes(data)
            return self.put(p, rel)

    def put_json(self, obj, rel: str) -> dict:
        return self.put_bytes(json.dumps(obj, indent=2, sort_keys=True, default=str).encode(), rel)

    def get(self, rel: str, local, expect: Optional[dict] = None) -> Path:
        """Download and verify against the remote listing (or `expect`)."""
        local = Path(local)
        local.parent.mkdir(parents=True, exist_ok=True)

        def attempt():
            info = expect or self._stat(rel)
            if info is None:
                raise FileNotFoundError(rel)
            tmp = local.with_name(local.name + PART)
            self._download(rel, tmp)
            size = tmp.stat().st_size
            if size != info["size"] or (info.get("md5") and md5_file(tmp) != info["md5"]):
                tmp.unlink(missing_ok=True)
                raise StorageError(f"{rel} download does not match the remote listing")
            tmp.replace(local)
            return local

        return self._retry(f"get {rel}", attempt)

    def get_bytes(self, rel: str) -> Optional[bytes]:
        if self._stat(rel) is None:
            return None
        with tempfile.TemporaryDirectory() as d:
            return self.get(rel, Path(d) / "x").read_bytes()

    def get_json(self, rel: str):
        data = self.get_bytes(rel)
        return None if data is None else json.loads(data.decode())

    def get_dir(self, rel: str, local) -> Path:
        """Flat download of the files in a remote folder."""
        local = Path(local)
        listing = self.ls(rel)
        if not listing:
            raise FileNotFoundError(f"remote folder {rel} is empty or missing")
        for name, info in listing.items():
            if name.endswith(PART):
                continue
            target = local / name
            if target.exists() and target.stat().st_size == info["size"] and \
                    (not info.get("md5") or md5_file(target) == info["md5"]):
                continue
            self.get(_join(rel, name), target, expect=info)
        return local

    def _download_many(self, rel: str, names: list, local_dir: Path):
        for n in names:
            self._download(_join(rel, n), local_dir / n)

    def get_many(self, rel: str, names, local_dir, listing: Optional[dict] = None) -> list:
        """Download several files of one folder, skipping local copies that already match.

        Backends may fetch them in one call. Every file is verified against the
        listing afterwards and fetched again on its own if it does not match.
        """
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        listing = self.ls(rel) if listing is None else listing
        names = list(names)

        def fresh(n):
            p, info = local_dir / n, listing[n]
            return p.is_file() and p.stat().st_size == info["size"] and \
                (not info.get("md5") or md5_file(p) == info["md5"])

        need = [n for n in names if not fresh(n)]
        if need:
            try:
                self._download_many(rel, need, local_dir)
            except StorageError:
                pass  # retried one by one below
            for n in need:
                if not fresh(n):
                    self.get(_join(rel, n), local_dir / n, expect=listing[n])
        return [local_dir / n for n in names]

    def ls(self, rel: str) -> dict:
        """{name: {'size', 'md5'}} for the files directly under rel. {} if missing."""
        return self._list(rel)

    def stat(self, rel: str) -> Optional[dict]:
        return self._stat(rel)

    def exists(self, rel: str) -> bool:
        return self._stat(rel) is not None

    def delete(self, rel: str):
        if self._stat(rel) is not None:
            self._delete(rel)

    def purge(self, rel: str):
        self._purge(rel)

    def dedupe(self, rel: str):
        """Google Drive allows two files with one name. No-op elsewhere."""


class LocalRemote(Remote):
    """A folder that behaves like the Drive root. Used by tests and local runs."""

    backoff_s = 0.0

    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def describe(self) -> str:
        return f"local:{self.root}"

    def _p(self, rel: str) -> Path:
        return self.root / rel

    def _upload(self, local, rel):
        dst = self._p(rel)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local, dst)

    def _download(self, rel, local):
        shutil.copyfile(self._p(rel), local)

    def _move(self, src, dst):
        d = self._p(dst)
        d.parent.mkdir(parents=True, exist_ok=True)
        self._p(src).replace(d)

    def _stat(self, rel):
        p = self._p(rel)
        if not p.is_file():
            return None
        return {"size": p.stat().st_size, "md5": md5_file(p)}

    def _list(self, rel):
        d = self._p(rel)
        if not d.is_dir():
            return {}
        return {p.name: {"size": p.stat().st_size, "md5": md5_file(p)}
                for p in sorted(d.iterdir()) if p.is_file()}

    def _delete(self, rel):
        self._p(rel).unlink(missing_ok=True)

    def _purge(self, rel):
        p = self._p(rel)
        if p.is_dir():
            shutil.rmtree(p)
        elif p.exists():
            p.unlink()


class RcloneRemote(Remote):
    """Google Drive (or any rclone remote) through the rclone CLI.

    base is 'remote:folder', e.g. 'erclm:ERC-LM-2027'. A plain local path also
    works, which is how the tests exercise this class when rclone is installed.
    """

    def __init__(self, base: str, config: Optional[str] = None, rclone: str = "rclone",
                 flags=("--retries", "3", "--low-level-retries", "10")):
        self.base = base.rstrip("/")
        self.config = config
        self.rclone = rclone
        self.flags = list(flags)

    def describe(self) -> str:
        return f"rclone:{self.base}"

    def _target(self, rel: str) -> str:
        return f"{self.base}/{rel}" if rel else self.base

    def _run(self, *args, check: bool = True) -> subprocess.CompletedProcess:
        cmd = [self.rclone, *args, *self.flags]
        if self.config:
            cmd += ["--config", str(self.config)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if check and res.returncode != 0:
            raise StorageError(f"rclone {args[0]} exited {res.returncode}: {res.stderr.strip()[-800:]}")
        return res

    def _upload(self, local, rel):
        self._run("copyto", str(local), self._target(rel))

    def _download(self, rel, local):
        self._run("copyto", self._target(rel), str(local))

    def _move(self, src, dst):
        self._run("moveto", self._target(src), self._target(dst))

    def _download_many(self, rel, names, local_dir):
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
            fh.write("\n".join(names) + "\n")
        try:
            self._run("copy", self._target(rel), str(local_dir), "--files-from-raw", fh.name)
        finally:
            Path(fh.name).unlink(missing_ok=True)

    @staticmethod
    def _entry(e: dict) -> dict:
        hashes = {k.lower(): v for k, v in (e.get("Hashes") or {}).items()}
        return {"size": int(e["Size"]), "md5": hashes.get("md5") or None}

    def _stat(self, rel):
        res = self._run("lsjson", "--stat", "--hash", self._target(rel), check=False)
        if res.returncode != 0:
            # rclone exits 3 (directory not found) or 4 (file not found)
            if res.returncode in (3, 4) or "not found" in res.stderr.lower():
                return None
            raise StorageError(f"rclone lsjson --stat exited {res.returncode}: {res.stderr.strip()[-800:]}")
        e = json.loads(res.stdout)
        if e.get("IsDir"):
            return None
        return self._entry(e)

    def _list(self, rel):
        res = self._run("lsjson", "--files-only", "--hash", self._target(rel), check=False)
        if res.returncode != 0:
            if res.returncode in (3, 4) or "not found" in res.stderr.lower():
                return {}
            raise StorageError(f"rclone lsjson exited {res.returncode}: {res.stderr.strip()[-800:]}")
        out = {}
        for e in json.loads(res.stdout or "[]"):
            if e["Name"] in out:
                raise StorageError(f"two remote files named {rel}/{e['Name']}; run dedupe")
            out[e["Name"]] = self._entry(e)
        return out

    def _delete(self, rel):
        self._run("deletefile", self._target(rel))

    def _purge(self, rel):
        res = self._run("purge", self._target(rel), check=False)
        if res.returncode not in (0, 3, 4) and "not found" not in res.stderr.lower():
            raise StorageError(f"rclone purge exited {res.returncode}: {res.stderr.strip()[-800:]}")

    def dedupe(self, rel: str):
        if ":" in self.base.split("/")[0]:
            self._run("dedupe", "--dedupe-mode", "newest", self._target(rel), check=False)
