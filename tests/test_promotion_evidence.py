"""B-061 — `hearth adapters promote --report` must not promote on a report a human could type.

Every test here is an attack that PROMOTED on the code before the fix (the reviewer's
reproducer promoted a nonexistent adapter from a hand-written report, with a task mismatch,
on a prereg committed seconds earlier in a throwaway repo). Each must now be refused, and
the adapter must still be a candidate afterwards.

The legitimate offline path — `hearth eval --report-json` then `adapters promote --report
--prereg` — is exercised first, so every refusal below is a refusal of the *attack*, not of
the setup. Reports are produced by the real CLI; where a test needs to isolate one binding
check it re-signs an edited report with the install's key (a forger who has read the key),
which is exactly the case the binding checks — not the signature — must catch.
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from hearth.cli import app
from hearth.providers.base import GenResult
from hearth.registry import AdapterStore
from hearth.training import attest
from hearth.training.eval import as_golden_set

runner = CliRunner()

BASE = "mlx-community/Qwen2.5-3B-Instruct-4bit"
ROWS = [{"prompt": f"p{i}", "expected": "A" if i % 2 == 0 else f"B{i}"} for i in range(40)]
ANSWERS = {r["prompt"]: r["expected"] for r in ROWS}


class _Provider:
    """Base parrots "A"; ONLY the adapter directory named `winner` answers correctly."""

    name = "fake"

    def __init__(self, winner: str = "extract-1") -> None:
        self.winner = winner

    def generate(self, req):
        prompt = req.messages[-1].content
        good = bool(req.adapter) and Path(req.adapter).name == self.winner
        return GenResult(text=ANSWERS[prompt] if good else "A", model=req.model, backend="fake")


class _TestBackends(frozenset):
    """The real promotable-backend allowlist, plus fake providers defined in test modules.

    Promotion refuses scores from anything but the MLX pool (B-084); these offline tests
    stand a fake in for it. Only classes whose module is a ``test_*`` module are admitted,
    so the echo stub and any plugin stay refused even here.
    """

    def __contains__(self, identity: object) -> bool:
        return super().__contains__(identity) or str(identity).startswith("test_")


def allow_test_backends(monkeypatch) -> None:
    from hearth.training import promotion

    monkeypatch.setattr(promotion, "PROMOTABLE_BACKENDS",
                        _TestBackends(promotion.PROMOTABLE_BACKENDS))


@pytest.fixture(autouse=True)
def fake_provider(monkeypatch):
    monkeypatch.setattr("hearth.cli.select_provider", lambda settings: _Provider())
    allow_test_backends(monkeypatch)


def _git(cwd: Path, *args: str, env: dict | None = None) -> str:
    full_env = {**os.environ, **(env or {})}
    return subprocess.run(
        ["git", "-c", "user.email=t@example.com", "-c", "user.name=test", *args],
        cwd=str(cwd), check=True, capture_output=True, text=True, env=full_env,
    ).stdout.strip()


class World:
    """An isolated HEARTH_HOME, adapters with weights, and a repo holding golden + prereg."""

    def __init__(self, tmp_path: Path) -> None:
        self.tmp = tmp_path
        self.home = tmp_path / ".hearth"
        self.repo = tmp_path / "repo"
        self.repo.mkdir()
        _git(self.repo, "init", "-q")
        self.golden = self.repo / "golden.jsonl"
        self.golden.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in ROWS))
        self.prereg = self.repo / "prereg.yaml"
        self.report = tmp_path / "report.json"
        self.store = AdapterStore(path=self.home / "adapters.json")
        # The repo is this install's evals repository (B-081), set before any measurement.
        from hearth.training.prereg import pin_anchor

        pin_anchor(self.home, self.repo)

    @property
    def env(self) -> dict[str, str]:
        return {"COLUMNS": "300", "HEARTH_HOME": str(self.home), "HEARTH_BACKEND": "echo"}

    def adapter(self, adapter_id: str = "extract-1", task: str = "extract", base: str = BASE):
        path = self.tmp / "weights" / adapter_id
        path.mkdir(parents=True, exist_ok=True)
        (path / "adapters.safetensors").write_bytes(f"weights of {adapter_id}".encode())
        self.store.register(adapter_id, base_model=base, task=task, train_run_id="r",
                            adapter_path=str(path))
        return path

    def write_prereg(self, task: str = "extract") -> None:
        sha = as_golden_set(task, [(r["prompt"], r["expected"]) for r in ROWS]).sha
        body = {
            "task": task, "hypothesis": "the adapter learns the labels", "golden_sha": sha,
            "metric": "exact",
            "generation": {"temperature": 0.0, "max_tokens": 24, "seed": None,
                           "system_hash": ""},
            "bar": {"test": "auto", "alpha": 0.05, "min_effect": 0.0, "min_n": 30},
            "stopping_rule": "one run", "kill_condition": "no lift",
        }
        self.prereg.write_text(yaml.safe_dump(body, sort_keys=False), encoding="utf-8")

    def commit(self, *names: str, when: datetime | None = None) -> str:
        _git(self.repo, "add", *names)
        env = {}
        if when is not None:
            env = {"GIT_COMMITTER_DATE": when.isoformat(), "GIT_AUTHOR_DATE": when.isoformat()}
        _git(self.repo, "commit", "-qm", "commit " + " ".join(names), env=env)
        return _git(self.repo, "rev-parse", "HEAD")

    def registered(self) -> str:
        """Golden set + prereg committed together (the legitimate setup)."""
        self.write_prereg()
        return self.commit("golden.jsonl", "prereg.yaml")

    def eval(self, adapter_id: str = "extract-1", *extra: str):
        return runner.invoke(
            app,
            ["eval", adapter_id, "--golden", str(self.golden), "--metric", "exact",
             "--max-tokens", "24", *extra],
            env=self.env,
        )

    def eval_report(self, adapter_id: str = "extract-1", *extra: str) -> dict:
        result = self.eval(adapter_id, "--report-json", str(self.report), *extra)
        assert result.exit_code == 0, result.output
        return json.loads(self.report.read_text())

    def promote(self, adapter_id: str = "extract-1", report: Path | None = None,
                prereg: Path | None = None):
        return runner.invoke(
            app,
            ["adapters", "promote", adapter_id, "--report", str(report or self.report),
             "--prereg", str(prereg or self.prereg)],
            env=self.env,
        )

    def write_report(self, payload: dict, *, resign: bool) -> None:
        if resign:
            payload = attest.sign(payload, attest.load_key(self.home))
        self.report.write_text(json.dumps(payload, indent=2, sort_keys=True))

    def status(self, adapter_id: str = "extract-1") -> str:
        return self.store.get(adapter_id).status


@pytest.fixture
def world(tmp_path) -> World:
    return World(tmp_path)


def _refused(result, world: World, adapter_id: str = "extract-1", text: str = "") -> None:
    flat = " ".join(result.output.split())
    assert result.exit_code == 1, flat
    assert "Promoted" not in flat
    assert world.status(adapter_id) == "candidate", flat
    if text:
        assert text in flat, flat


# -- the legitimate path still works ----------------------------------------------------


def test_the_legitimate_offline_path_promotes_and_records_its_evidence(world):
    world.adapter()
    prereg_commit = world.registered()
    _git(world.repo, "commit", "--allow-empty", "-qm", "unrelated later commit")
    payload = world.eval_report()
    assert payload["signature"]["alg"] == "hmac-sha256"
    assert payload["candidate_weights_sha"]
    result = world.promote()
    assert result.exit_code == 0, result.output
    entry = world.store.get("extract-1")
    assert entry.status == "promoted"
    proof = entry.promotion_proof
    assert proof["gate"] == "verified" and proof["evidence"] == "signed-report"
    # The commit that wrote the bar — not HEAD, which an unrelated commit has moved.
    assert proof["prereg_commit"] == prereg_commit
    assert proof["prereg_introduced_commit"] == prereg_commit
    assert proof["prereg_commit"] != _git(world.repo, "rev-parse", "HEAD")
    assert proof["candidate_weights_sha"] == payload["candidate_weights_sha"]
    assert len(proof["report_sha"]) == 64
    key = attest.key_path(world.home)
    assert oct(key.stat().st_mode & 0o777) == "0o600"


def test_the_measuring_path_records_the_introducing_commit_not_head(world):
    world.adapter()
    prereg_commit = world.registered()
    _git(world.repo, "commit", "--allow-empty", "-qm", "unrelated later commit")
    result = world.eval("extract-1", "--prereg", str(world.prereg), "--promote")
    assert result.exit_code == 0, result.output
    proof = world.store.get("extract-1").promotion_proof
    assert proof["evidence"] == "measured"
    assert proof["prereg_commit"] == prereg_commit != _git(world.repo, "rev-parse", "HEAD")


# -- a report a human could type ----------------------------------------------------------


def test_the_reviewer_forgery_is_refused_with_no_key_on_the_install(world):
    """The B-061 reproducer: hand-written report, nonexistent weights, task mismatch."""
    world.store.register("bogus-ad", base_model=BASE, task="extract", train_run_id="none",
                         adapter_path="/nonexistent")
    world.write_prereg(task="classify")
    world.commit("prereg.yaml")
    world.report.write_text(json.dumps({
        "candidate": {"task": "classify", "metric": "exact_match", "score": 1.0,
                      "per_example": [1.0] * 30, "n": 30, "golden_sha": "ab" * 32},
        "incumbent": {"task": "classify", "metric": "exact_match", "score": 0.0,
                      "per_example": [0.0] * 30, "n": 30, "golden_sha": "ab" * 32},
    }))
    _refused(world.promote("bogus-ad"), world, "bogus-ad", "Unusable eval report")


def test_a_hand_written_report_is_refused_when_a_key_exists(world):
    world.adapter()
    world.registered()
    payload = world.eval_report()
    del payload["signature"]
    world.write_report(payload, resign=False)
    _refused(world.promote(), world, text="not signed")


def test_an_edited_report_is_refused(world):
    """Flip the incumbent to all-wrong to manufacture significance; the MAC no longer fits."""
    world.adapter()
    world.registered()
    payload = world.eval_report()
    payload["incumbent"]["per_example"] = [0.0] * 40
    payload["incumbent"]["score"] = 0.0
    world.write_report(payload, resign=False)
    _refused(world.promote(), world, text="edited")


def test_a_report_signed_by_another_install_is_refused(world, tmp_path):
    world.adapter()
    world.registered()
    payload = world.eval_report()
    other_key = attest.load_key(tmp_path / "elsewhere", create=True)
    world.report.write_text(json.dumps(attest.sign(payload, other_key)))
    _refused(world.promote(), world, text="different install")


def test_a_world_readable_key_authenticates_nothing(world):
    world.adapter()
    world.registered()
    world.eval_report()
    attest.key_path(world.home).chmod(0o644)
    _refused(world.promote(), world, text="group/other")


# -- a signed report that is evidence for something else ----------------------------------


def test_a_report_for_another_adapter_cannot_promote_this_one(world):
    world.adapter("extract-1")
    world.adapter("extract-2")
    world.registered()
    world.eval_report("extract-1")
    _refused(world.promote("extract-2"), world, "extract-2", "measured adapter 'extract-1'")


@pytest.mark.parametrize(
    ("edit", "text"),
    [
        (lambda p: p.update(schema="hearth.eval-report/1"), "schema"),
        (lambda p: p.update(candidate_id="someone-else"), "measured adapter"),
        (lambda p: p.update(task="classify"), "report task 'classify'"),
        (lambda p: p["candidate"].update(task="classify"), "(candidate 'classify')"),
        (lambda p: p.update(base_model="mlx-community/Qwen2.5-14B-Instruct-4bit"), "base model"),
        (lambda p: p["candidate"].update(model_id="other+extract-1"), "model_id"),
        (lambda p: p.update(adapter_path="/elsewhere"), "adapter_path"),
        (lambda p: p.update(candidate_weights_sha="0" * 64), "weights changed"),
        (lambda p: p.update(measured_at="2020-01-01T00:00:00+00:00"), "measured_at"),
        (lambda p: p.update(incumbent_role="incumbent"), "incumbent is the base model"),
        (lambda p: p.update(incumbent_id="some-other-base"), "incumbent is the base model"),
        (lambda p: p["incumbent"].update(model_id="x"), "incumbent report model_id"),
        (lambda p: p["incumbent"].update(task="classify"), "incumbent task"),
    ],
)
def test_each_binding_between_report_and_adapter_is_checked(world, edit, text):
    """A forger holding the key re-signs an edited report: the binding checks must catch it."""
    world.adapter()
    world.registered()
    payload = world.eval_report()
    del payload["signature"]
    edit(payload)
    world.write_report(payload, resign=True)
    _refused(world.promote(), world, text=text)


def test_weights_changed_after_the_measurement_are_a_different_candidate(world):
    weights = world.adapter()
    world.registered()
    world.eval_report()
    (weights / "adapters.safetensors").write_bytes(b"retrained after the eval")
    _refused(world.promote(), world, text="weights changed")


def test_weights_deleted_after_the_measurement_cannot_be_promoted(world):
    weights = world.adapter()
    world.registered()
    world.eval_report()
    for f in weights.iterdir():
        f.unlink()
    _refused(world.promote(), world, text="weights not found")


def test_weights_swapped_during_eval_promote_are_not_the_bytes_measured(world, monkeypatch):
    """The measuring path re-hashes before promoting: the bytes promoted are the bytes scored."""
    weights = world.adapter()
    world.registered()

    class _Swapping(_Provider):
        def generate(self, req):
            (weights / "adapters.safetensors").write_bytes(b"swapped mid-eval")
            return super().generate(req)

    monkeypatch.setattr("hearth.cli.select_provider", lambda settings: _Swapping())
    result = world.eval("extract-1", "--prereg", str(world.prereg), "--promote")
    _refused(result, world, text="weights changed during the eval")


def test_eval_refuses_to_measure_an_adapter_with_no_weights(world):
    world.store.register("ghost", base_model=BASE, task="extract", train_run_id="r",
                         adapter_path=str(world.tmp / "nope"))
    world.registered()
    result = world.eval("ghost", "--report-json", str(world.report))
    assert result.exit_code == 1
    assert "weights not found" in " ".join(result.output.split())
    assert not world.report.exists()


def test_a_report_that_beat_the_base_cannot_displace_a_since_promoted_adapter(world):
    world.adapter("extract-1")
    world.adapter("extract-0")
    world.registered()
    world.eval_report("extract-1")  # incumbent: the base model
    world.store.promote("extract-0", gate_passed=True)
    _refused(world.promote("extract-1"), world, text="'extract-0' is the promoted adapter")
    assert world.status("extract-0") == "promoted"


def test_an_incumbent_whose_weights_changed_since_is_not_the_one_beaten(world):
    world.adapter("extract-1")
    old = world.adapter("extract-0")
    world.store.promote("extract-0", gate_passed=True)
    world.registered()
    payload = world.eval_report("extract-1")
    assert payload["incumbent_role"] == "incumbent" and payload["incumbent_id"] == "extract-0"
    (old / "adapters.safetensors").write_bytes(b"incumbent retrained")
    _refused(world.promote("extract-1"), world, text="incumbent 'extract-0' weights changed")


def test_measured_on_another_base_is_not_promotable(world):
    world.adapter()
    world.registered()
    other = "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"
    world.eval_report("extract-1", "--base", other)
    _refused(world.promote(), world, text="base model")
    result = world.eval("extract-1", "--base", other, "--prereg", str(world.prereg), "--promote")
    _refused(result, world, text="measured on base")


# -- prereg provenance --------------------------------------------------------------------


def test_a_prereg_committed_after_the_measurement_is_refused(world):
    world.adapter()
    world.commit("golden.jsonl")
    world.eval_report()  # measured with no bar registered yet
    world.write_prereg()
    world.commit("prereg.yaml", when=datetime.now(tz=UTC) + timedelta(hours=1))
    _refused(world.promote(), world, text="AFTER the measurement")


def test_eval_promote_refuses_a_prereg_committed_after_the_run_started(world):
    world.adapter()
    world.write_prereg()
    world.commit("golden.jsonl", "prereg.yaml", when=datetime.now(tz=UTC) + timedelta(hours=1))
    result = world.eval("extract-1", "--prereg", str(world.prereg), "--promote")
    _refused(result, world, text="AFTER the measurement")


def test_a_bar_moved_in_a_later_commit_is_refused(world):
    """Commit a bar, measure, then commit a different (still valid) bar: not the one in force."""
    world.adapter()
    world.registered()
    world.eval_report()
    world.prereg.write_text(world.prereg.read_text().replace("min_n: 30", "min_n: 35"))
    world.commit("prereg.yaml", when=datetime.now(tz=UTC) + timedelta(hours=1))
    _refused(world.promote(), world, text="AFTER the measurement")


def test_a_prereg_in_a_throwaway_repo_is_refused(world, tmp_path):
    """The golden set is versioned in `repo`; a bar committed in another repo does not count."""
    world.adapter()
    world.commit("golden.jsonl")
    other = tmp_path / "throwaway"
    other.mkdir()
    _git(other, "init", "-q")
    world.write_prereg()
    (other / "prereg.yaml").write_text(world.prereg.read_text())
    world.prereg.unlink()
    _git(other, "add", "prereg.yaml")
    _git(other, "commit", "-qm", "prereg")
    world.eval_report()
    _refused(world.promote(prereg=other / "prereg.yaml"), world, text="versioned in")
    result = world.eval("extract-1", "--prereg", str(other / "prereg.yaml"), "--promote")
    _refused(result, world, text="versioned in")


def test_an_uncommitted_golden_set_is_refused(world):
    world.adapter()
    world.write_prereg()
    world.commit("prereg.yaml")  # golden.jsonl left untracked
    world.eval_report()
    _refused(world.promote(), world, text="golden set was not committed")
    result = world.eval("extract-1", "--prereg", str(world.prereg), "--promote")
    _refused(result, world, text="golden set was not committed")
