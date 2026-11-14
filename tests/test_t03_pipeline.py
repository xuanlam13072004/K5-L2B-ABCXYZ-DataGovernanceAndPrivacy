"""T03 wiring tests (Lead). They must stay green after every merge.

Member 6 adds algorithm tests (glare on object, erase detection, ...) in tests/test_t03.py.
"""
import json

import pandas as pd

from src import main as runner

CMD = ["--config", "configs/lab.yaml", "--track", "t03", "--offline", "--seed", "42", "--run-name", "poc"]


def _run(out):
    return runner.main(CMD + ["--output", str(out)])


def test_t03_end_to_end_and_reproducible(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    assert _run(a) == 0
    for f in ["qc.csv", "transform_manifest.csv", "review_queue.csv", "contact_sheet.html", "decision.json", "run_info.json"]:
        assert (a / f).exists(), f
    qc = pd.read_csv(a / "qc.csv")
    holdout = pd.read_csv("fixtures/t03_synthetic/holdout_transformed.csv")
    assert set(qc.image_id) == set(holdout.image_id) and not qc.image_id.duplicated().any()
    assert set(qc.pred_unusable.unique()) <= {0, 1}
    decision = json.loads((a / "decision.json").read_text())
    assert decision["reason_code"] and decision["owner"]
    assert _run(b) == 0                                   # same seed -> identical predictions
    assert (a / "qc.csv").read_text() == (b / "qc.csv").read_text()


def _reviewed(out, reviewer):
    q = pd.read_csv(out / "review_queue.csv", dtype=str).fillna("")
    q.loc[0, ["reviewer", "reviewer_decision", "override_value", "review_note"]] = [reviewer, "edit", "1", "test override"]
    q.to_csv(out / "review_queue_reviewed.csv", index=False)
    return q.loc[0, "item_id"]


def test_t03_review_override_is_applied_and_audited(tmp_path):
    assert _run(tmp_path) == 0
    item = _reviewed(tmp_path, "Tester")
    assert _run(tmp_path) == 0
    qc = pd.read_csv(tmp_path / "qc.csv").set_index("image_id")
    assert qc.loc[item, "pred_unusable"] == 1
    events = [json.loads(line) for line in (tmp_path / "audit_log.jsonl").read_text().splitlines()]
    assert any(e["item_id"] == item and e["actor"] == "Tester" and e["action"] == "override:edit" for e in events)


def test_t03_review_without_reviewer_fails_clearly(tmp_path):
    assert _run(tmp_path) == 0
    _reviewed(tmp_path, "")                               # decision with no reviewer name
    assert _run(tmp_path) == 2
