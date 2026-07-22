from pathlib import Path

from bilab.smoke import run_smoke

ROOT = Path(__file__).parents[1]


def test_smoke_runs_end_to_end(tmp_path: Path) -> None:
    document = run_smoke(ROOT, ROOT / "experiments/smoke/manifest.json", tmp_path)
    decisions = document["decision_demonstrations"]
    assert decisions["succeeded"]["status"] == "succeeded"
    assert decisions["succeeded"]["budget_violations"] == []
    assert decisions["failed"]["status"] == "failed"
    assert decisions["rejected_incomparable"]["status"] == "rejected_incomparable"
    assert len(document["trials"]) == 6
    assert (tmp_path / "results.json").is_file()
    assert (tmp_path / "manifest.completed.json").is_file()
    assert "does not measure intelligence" in (tmp_path / "report.md").read_text()
