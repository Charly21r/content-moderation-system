import json
from pathlib import Path

MAX_FPR_DELTA = 0.05  # 5 percentage points allowed


def test_bias_constraints():
    root = Path(__file__).resolve().parents[1]
    report_path = root / "models" / "text_toxicity" / "artifacts" / "bias_report.json"

    assert report_path.exists(), (
        f"bias_report.json not found at {report_path}. "
        "Run scripts/run_bias_eval.py before running tests."
    )

    with report_path.open("r") as f:
        metrics = json.load(f)

    # Check all FPR deltas for all labels / groups
    for key, value in metrics.items():
        if key.endswith("_FPR_delta") and value is not None:
            assert abs(value) <= MAX_FPR_DELTA, (
                f"Bias constraint violated: {key}={value:.4f} "
                f"exceeds allowed {MAX_FPR_DELTA:.4f}"
            )
