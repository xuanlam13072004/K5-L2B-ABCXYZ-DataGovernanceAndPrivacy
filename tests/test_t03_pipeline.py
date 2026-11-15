"""T03 wiring tests (Lead). They must stay green after every merge.

The wiring tests run on the small instructor sample (fixtures/t03_synthetic, 320x180) to stay fast;
test_t03_default_config_runs_on_team_images checks the real default: the team's 30 images (data/t03_real).
Member 6 adds algorithm tests (glare on object, erase detection, ...) in tests/test_t03.py.
"""
import json

import pandas as pd

from src import main as runner

CMD = ["--config", "configs/lab.yaml", "--track", "t03", "--offline", "--seed", "42", "--run-name", "poc",
       "--input", "fixtures/t03_synthetic"]


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


def test_t03_default_config_runs_on_team_images(tmp_path):
    out = tmp_path / "real"
    assert runner.main(["--config", "configs/lab.yaml", "--offline", "--seed", "42", "--run-name", "poc",
                        "--output", str(out)]) == 0
    holdout = pd.read_csv("data/t03_real/holdout_transformed.csv")
    qc = pd.read_csv(out / "qc.csv")
    assert set(qc.image_id) == set(holdout.image_id)
    manifest = pd.read_csv(out / "transform_manifest.csv")
    required = ["image_id", "parent_image", "transform", "severity", "seed"]
    assert len(manifest) > 0 and manifest[required].notna().all().all()      # provenance completeness = 100%
    assert set(manifest.parent_image) <= set(pd.read_csv("data/t03_real/originals.csv").query("split == 'dev'").image_id)
    gold = pd.read_csv(out / "dev_gold.csv", keep_default_na=False)
    assert set(gold.image_id) == set(manifest.image_id) and gold.unusable.isin([0, 1]).all()


def _review_all(out, decision, reviewer="Tester"):
    q = pd.read_csv(out / "review_queue.csv", dtype=str).fillna("")
    q["reviewer"], q["reviewer_decision"], q["review_note"] = reviewer, decision, "test"
    q.to_csv(out / "review_queue_reviewed.csv", index=False)
    return q


def test_t03_every_review_decision_is_audited(tmp_path):
    assert _run(tmp_path) == 0
    q = _review_all(tmp_path, "approve")
    assert _run(tmp_path) == 0
    events = [json.loads(line) for line in (tmp_path / "audit_log.jsonl").read_text().splitlines()]
    approved = {e["item_id"] for e in events if e["action"] == "review:approve" and e["actor"] == "Tester"}
    assert approved == set(q.item_id)
    assert json.loads((tmp_path / "decision.json").read_text())["metrics"]["n_review_approved"] == len(q)


def test_t03_escalated_review_holds_the_release(tmp_path):
    assert _run(tmp_path) == 0
    _review_all(tmp_path, "escalate")
    assert _run(tmp_path) == 0
    d = json.loads((tmp_path / "decision.json").read_text())
    assert d["status"] == "HOLD" and d["reason_code"] == "REVIEW_ESCALATED"
