"""The CI workflows: Node comes from the newest release, not the runner's tool cache."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
WORKFLOWS = sorted((ROOT / ".github" / "workflows").glob("*.yml"))


@pytest.mark.parametrize("workflow", WORKFLOWS, ids=lambda path: path.name)
def test_setup_node_checks_latest(workflow):
    text = workflow.read_text()
    assert text.count("actions/setup-node") == text.count("check-latest: true")
