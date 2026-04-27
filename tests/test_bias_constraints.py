import json
from pathlib import Path

# Policy: we allow maximum +5 percentage points extra false positives on
# lexical-identity mentions vs non-lexical (validation data)
MAX_FPR_DELTA = 0.05


def _load_bias_report() -> dict:
    root = Path(__file__).resolve().parents[1]
    report_path = root / "models" / "text_toxicity" / "artifacts" / "bias_report.json"

    assert report_path.exists(), (
        f"bias_report.json not found at {report_path}. Run scripts/run_bias_eval.py before running tests."
    )

    try:
        return json.loads(report_path.read_text())
    except json.JSONDecodeError as e:
        raise AssertionError(f"bias_report.json is not valid JSON: {e}") from e


def test_bias_constraints():
    """
    CI fairness gate (real validation only):
    - Enforces that lex-vs-nonlex FPR deltas do not exceed MAX_FPR_DELTA.

    Notes:
    - We do NOT gate on templated_identity_sensitivity; that's a diagnostic stress test.
    """
    report = _load_bias_report()

    assert "val" in report, "bias_report.json missing top-level key: 'val'"

    metrics = report["val"]
    assert isinstance(metrics, dict), "'val' section must be a JSON object"

    # We expect at least one FPR delta metric to exist
    fpr_delta_keys = [k for k in metrics if k.endswith("_FPR_delta")]
    assert fpr_delta_keys, (
        "No '*_FPR_delta' keys found under report['val']. "
        "Make sure compute_metrics() writes FPR deltas for the validation slice eval."
    )

    violations = []
    skipped = []

    for key in sorted(fpr_delta_keys):
        value = metrics.get(key)

        if value is None:
            skipped.append(key)
            continue

        try:
            v = float(value)
        except (TypeError, ValueError):
            violations.append(f"{key} has non-numeric value: {value!r}")
            continue

        # we constraint only for over-flagging on these words mentions
        if v > MAX_FPR_DELTA:
            violations.append(f"{key}={v:.6f} > {MAX_FPR_DELTA:.6f}")

    if violations:
        extra = ""
        if skipped:
            extra = f"\nSkipped (not computable): {', '.join(skipped)}"

        raise AssertionError(
            "Fairness gate failed (FPR delta exceeded allowed threshold):\n"
            + "\n".join(f"- {msg}" for msg in violations)
            + extra
        )
