"""Publishing to the runs branch of a bare repo that plays GitHub."""
import base64

import helpers
from erclm.gitlog import RunsLog, auth_env


def test_publish_writes_only_the_run_folder_of_runs(tmp_path):
    bare = helpers.make_github(tmp_path)
    main_before = helpers.git("--git-dir", str(bare), "rev-parse", "main")
    log = RunsLog(str(bare), None, tmp_path / "clone", "pilot-v1", backoff_s=0)
    assert log.publish({"env.json": '{"a": 1}', "results/t.csv": b"x,y\n1,2\n"}, message="start")
    assert helpers.show(bare, "pilot-v1/env.json") == b'{"a": 1}'
    assert helpers.show(bare, "pilot-v1/results/t.csv") == b"x,y\n1,2\n"
    assert helpers.show(bare, "README.md") == b"runs\n"
    assert not log.publish({"env.json": '{"a": 1}'})          # nothing new
    assert log.publish({"progress.json": "{}"}, delete=["env.json"])
    assert helpers.show(bare, "pilot-v1/env.json") is None
    assert helpers.git("--git-dir", str(bare), "rev-parse", "main") == main_before
    author = helpers.git("--git-dir", str(bare), "log", "-1", "--format=%an", "runs").strip()
    assert author == "erclm-bot"


def test_rejected_push_is_retried_on_top_of_the_new_tip(tmp_path):
    bare = helpers.make_github(tmp_path)
    a = RunsLog(str(bare), None, tmp_path / "a", "run-a", backoff_s=0)
    b = RunsLog(str(bare), None, tmp_path / "b", "run-b", backoff_s=0)
    sync = a._sync
    raced = []

    def racing_sync():
        sync()
        if not raced:                       # another session pushes between our fetch and our push
            raced.append(1)
            b.publish({"x.json": "{}"})

    a._sync = racing_sync
    assert a.publish({"y.json": "{}"})
    assert helpers.show(bare, "run-a/y.json") == b"{}" and helpers.show(bare, "run-b/x.json") == b"{}"


def test_token_stays_out_of_urls_and_config(tmp_path):
    token = "github_pat_SECRET123456789"
    env = auth_env(token, "https://github.com/gbibbo/erclm-2027.git")
    assert token not in " ".join(env.values())
    assert base64.b64encode(f"x-access-token:{token}".encode()).decode() in env["GIT_CONFIG_VALUE_0"]
    assert env["GIT_CONFIG_KEY_0"] == "http.https://github.com/.extraheader"
    bare = helpers.make_github(tmp_path)
    log = RunsLog(str(bare), token, tmp_path / "clone", "r", backoff_s=0)
    log.publish({"a.txt": "1"})
    assert token not in (tmp_path / "clone" / ".git" / "config").read_text()
    assert log._clean(f"fatal: {token} rejected") == "fatal: *** rejected"


def test_an_existing_clone_follows_a_renamed_repository(tmp_path):
    old = helpers.make_github(tmp_path)
    RunsLog(str(old), None, tmp_path / "clone", "r", backoff_s=0).publish({"a.txt": "1"})
    new = tmp_path / "renamed.git"
    old.rename(new)
    assert RunsLog(str(new), None, tmp_path / "clone", "r", backoff_s=0).publish({"b.txt": "2"})
    assert helpers.show(new, "r/b.txt") == b"2"
