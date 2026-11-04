"""Storage protocol on the fake Drive, and RcloneRemote on a local path when rclone is installed."""
import json
import shutil

import pytest

import helpers  # noqa: F401  (puts src on the path)
from erclm.storage import LocalRemote, RcloneRemote, StorageError, md5_file

RCLONE = shutil.which("rclone")


@pytest.fixture(params=["local", "rclone"])
def remote(request, tmp_path):
    if request.param == "rclone":
        if not RCLONE:
            pytest.skip("rclone not installed")
        (tmp_path / "rr").mkdir()
        (tmp_path / "empty.conf").write_text("")
        r = RcloneRemote(str(tmp_path / "rr"), config=str(tmp_path / "empty.conf"))
    else:
        r = LocalRemote(tmp_path / "lr")
    r.backoff_s = 0.0
    return r


def test_put_get_roundtrip_is_atomic_and_verified(remote, tmp_path):
    src = tmp_path / "a.bin"
    src.write_bytes(b"x" * 100_000)
    info = remote.put(src, "work/run/a.bin")
    assert info == {"size": 100_000, "md5": md5_file(src)}
    listing = remote.ls("work/run")
    assert set(listing) == {"a.bin"} and listing["a.bin"]["size"] == 100_000
    assert listing["a.bin"]["md5"] in (None, info["md5"])
    out = remote.get("work/run/a.bin", tmp_path / "back" / "a.bin")
    assert out.read_bytes() == src.read_bytes()
    assert not list((tmp_path / "back").glob("*.part"))


def test_put_overwrites_and_missing_paths(remote, tmp_path):
    src = tmp_path / "a.txt"
    src.write_text("one")
    remote.put(src, "x/a.txt")
    src.write_text("second version")
    remote.put(src, "x/a.txt")
    assert remote.get_bytes("x/a.txt") == b"second version"
    assert set(remote.ls("x")) == {"a.txt"}
    assert remote.stat("x/none.txt") is None and remote.get_bytes("x/none.txt") is None
    assert remote.ls("no/such/dir") == {}
    remote.delete("x/a.txt")
    remote.delete("x/a.txt")
    assert remote.ls("x") == {}


def test_json_many_dir_and_purge(remote, tmp_path):
    remote.put_json({"a": 1}, "state/r/s.done.json")
    assert remote.get_json("state/r/s.done.json") == {"a": 1}
    for i in range(3):
        remote.put_bytes(f"file {i}".encode(), f"d/f{i}.txt")
    local = tmp_path / "many"
    paths = remote.get_many("d", ["f0.txt", "f2.txt"], local)
    assert [p.read_text() for p in paths] == ["file 0", "file 2"]
    (local / "f0.txt").write_text("stale")
    remote.get_many("d", ["f0.txt"], local)
    assert (local / "f0.txt").read_text() == "file 0"
    remote.get_dir("d", tmp_path / "dir")
    assert sorted(p.name for p in (tmp_path / "dir").iterdir()) == ["f0.txt", "f1.txt", "f2.txt"]
    remote.purge("d")
    assert remote.ls("d") == {}


def test_put_detects_a_bad_remote_copy(tmp_path):
    class Truncating(LocalRemote):
        def _move(self, src, dst):
            super()._move(src, dst)
            p = self._p(dst)
            p.write_bytes(p.read_bytes()[:-10])

    r = Truncating(tmp_path / "r")
    r.backoff_s = 0.0
    src = tmp_path / "a.bin"
    src.write_bytes(b"y" * 1000)
    with pytest.raises(StorageError, match="verification failed"):
        r.put(src, "a.bin")


def test_get_rejects_a_copy_that_differs_from_the_listing(tmp_path):
    r = LocalRemote(tmp_path / "r")
    r.backoff_s = 0.0
    r.put_bytes(b"z" * 500, "a.bin")
    expect = r.stat("a.bin")
    (tmp_path / "r" / "a.bin").write_bytes(b"z" * 499)
    with pytest.raises(StorageError):
        r.get("a.bin", tmp_path / "out.bin", expect=expect)
    assert not (tmp_path / "out.bin").exists()


def test_rclone_entry_parsing():
    e = {"Name": "a", "Size": 12, "Hashes": {"MD5": "abc"}}
    assert RcloneRemote._entry(e) == {"size": 12, "md5": "abc"}
    assert RcloneRemote._entry({"Name": "b", "Size": 3}) == {"size": 3, "md5": None}
    assert json.loads("[]") == []
