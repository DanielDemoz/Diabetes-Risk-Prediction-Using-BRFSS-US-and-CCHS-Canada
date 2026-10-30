import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from predict import MODEL, predict  # noqa: E402


def test_probability_in_range():
    r = predict("50-54", "Female", 27, "Good", ["HighBP"])
    assert 0 < r["probability"] < 1


def test_beats_simple_rule():
    m = MODEL["metrics"]
    assert m["demo_logistic"]["auc"] > m["baseline_age_bmi"]["auc"]
    assert m["demo_logistic"]["brier"] < m["baseline_age_bmi"]["brier"]


def test_calibrated_on_average():
    m = MODEL["metrics"]["demo_logistic"]
    assert abs(m["mean_predicted"] - m["actual_rate"]) < 0.01


def test_income_and_education_are_not_inputs():
    assert not any(k.startswith(("income=", "education=", "Income", "Education")) for k in MODEL["coef"])


def test_every_fairness_group_reported():
    groups = {s["group"] for s in MODEL["segments"]}
    assert groups == {"Sex", "Age group", "Household income", "Education"}


def test_higher_risk_profile_scores_higher():
    low = predict("30-34", "Female", 22, "Excellent", ["PhysActivity", "Fruits", "Veggies"])["probability"]
    high = predict("65-69", "Male", 36, "Fair", ["HighBP", "HighChol", "DiffWalk"])["probability"]
    assert high > low


def test_unknown_value_rejected():
    try:
        predict("50-54", "Other", 27, "Good")
    except ValueError:
        return
    raise AssertionError("unknown value accepted")
