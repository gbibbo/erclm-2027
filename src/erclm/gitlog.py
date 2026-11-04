"""Publishes small run outputs to the orphan `runs` branch.

Only refs/heads/runs is ever pushed. Every publish fetches the tip of runs,
resets a private shallow clone to it, writes the files under <run_id>/, commits
as erclm-bot and pushes. A rejected push is retried from the fetch with growing
waits. A run writes only inside its own folder, so fetch, reset and reapply has
the effect of pull --rebase and can never conflict.

The token travels in an HTTP header set through GIT_CONFIG_* variables. It
never appears in a command line, a remote URL or .git/config.
"""
from __future__ import annotations

import base64
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Iterable, Optional

BRANCH = "runs"
BOT_NAME = "erclm-bot"
BOT_EMAIL = "erclm-bot@users.noreply.github.com"


class GitLogError(RuntimeError):
    pass


def repo_url(repo: str) -> str:
    """'owner/name' -> https URL. Full URLs and local paths pass through."""
    if "://" in repo or repo.startswith("/") or Path(repo).exists():
        return repo
    return f"https://github.com/{repo}.git"


def auth_env(token: Optional[str], url: str) -> dict:
    env = {"GIT_TERMINAL_PROMPT": "0"}
    if token and url.startswith("https://"):
        basic = base64.b64encode(f"x-access-token:{token}".encode()).decode()
        host = "/".join(url.split("/")[:3]) + "/"
        env.update({"GIT_CONFIG_COUNT": "1",
                    "GIT_CONFIG_KEY_0": f"http.{host}.extraheader",
                    "GIT_CONFIG_VALUE_0": f"AUTHORIZATION: basic {basic}"})
    return env


class RunsLog:
    def __init__(self, repo: str, token: Optional[str], clone_dir, run_id: str,
                 retries: int = 6, backoff_s: float = 2.0, redact: Iterable[str] = ()):
        self.url = repo_url(repo)
        self.token = token
        self.dir = Path(clone_dir)
        self.run_id = run_id.strip("/")
        self.retries = retries
        self.backoff_s = backoff_s
        self.secrets = [s for s in (token, *redact) if s]
        self._lock = threading.Lock()
        self._env = {**os.environ, **auth_env(token, self.url)}

    def _clean(self, text: str) -> str:
        for s in self.secrets:
            text = text.replace(s, "***")
        return text

    def _git(self, *args, check: bool = True, cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
        res = subprocess.run(["git", *args], cwd=str(cwd or self.dir), env=self._env,
                             capture_output=True, text=True)
        if check and res.returncode != 0:
            raise GitLogError(self._clean(f"git {args[0]} exited {res.returncode}: {res.stderr.strip()[-800:]}"))
        return res

    def _ensure_clone(self):
        if (self.dir / ".git").is_dir():
            self._git("remote", "set-url", "origin", self.url)   # follows a renamed repository
            return
        self.dir.mkdir(parents=True, exist_ok=True)
        self._git("init", "-q")
        self._git("remote", "add", "origin", self.url)

    def _sync(self):
        self._git("fetch", "-q", "--depth", "1", "origin", f"+refs/heads/{BRANCH}:refs/remotes/origin/{BRANCH}")
        self._git("checkout", "-q", "-B", BRANCH, f"origin/{BRANCH}")
        self._git("reset", "-q", "--hard", f"origin/{BRANCH}")

    def publish(self, files: dict, delete: Iterable[str] = (), message: str = "") -> bool:
        """Write files {relpath: bytes | str | Path} under <run_id>/ and push.

        Returns False when there was nothing new to commit.
        """
        delete = list(delete)
        with self._lock:
            self._ensure_clone()
            last = None
            for attempt in range(self.retries):
                try:
                    self._sync()
                    root = self.dir / self.run_id
                    for rel, content in files.items():
                        p = root / rel
                        p.parent.mkdir(parents=True, exist_ok=True)
                        if isinstance(content, Path):
                            p.write_bytes(content.read_bytes())
                        elif isinstance(content, bytes):
                            p.write_bytes(content)
                        else:
                            p.write_text(str(content), encoding="utf-8")
                    for rel in delete:
                        (root / rel).unlink(missing_ok=True)
                    self._git("add", "-A", "--", self.run_id)
                    if self._git("diff", "--cached", "--quiet", check=False).returncode == 0:
                        return False
                    msg = message or f"{self.run_id}: update"
                    self._git("-c", f"user.name={BOT_NAME}", "-c", f"user.email={BOT_EMAIL}",
                              "commit", "-q", "-m", msg)
                    self._git("push", "-q", "origin", f"HEAD:refs/heads/{BRANCH}")
                    return True
                except GitLogError as exc:
                    last = exc
                    time.sleep(self.backoff_s * (2 ** attempt))
            raise GitLogError(f"publish to {BRANCH} failed after {self.retries} attempts: {last}")


class NullRunsLog:
    """Stand-in when publishing is switched off (--no-runs)."""

    run_id = "local"

    def publish(self, files: dict, delete: Iterable[str] = (), message: str = "") -> bool:
        return False
