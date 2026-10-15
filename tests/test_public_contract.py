import json
from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_sample_records_are_jsonl_and_typed():
    rows = [json.loads(line) for line in (ROOT / "examples/train.jsonl").read_text().splitlines()]
    assert {row["input"]["answer_space"]["kind"] for row in rows} == {"choice", "scalar", "point"}
    assert all("target" in row and "id" in row for row in rows)


def test_public_entrypoints_exist():
    for name in ("predict.py", "train.py", "evaluate.py"):
        assert (ROOT / "scripts" / name).is_file()


def test_release_readme_does_not_reference_private_workspace_paths():
    for name in ("README.md", "README-zh.md"):
        text = (ROOT / name).read_text()
        assert "/mnt/workspace" not in text
        assert "/root/" not in text
