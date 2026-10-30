"""Score one survey profile with the exported model (docs/model.json).

Educational use only. This is not a diagnosis.

Example:
    python predict.py --age 50-54 --sex Female --bmi 31 --gen-health Good --yes HighBP PhysActivity
"""
import argparse
import json
import math
from pathlib import Path

MODEL = json.loads((Path(__file__).parent / "docs" / "model.json").read_text())


def bmi_band(bmi):
    cuts = MODEL["bmi_bins"]
    i = 0
    while i < len(cuts) and bmi >= cuts[i]:
        i += 1
    return MODEL["options"]["bmi_band"][i]


def predict(age, sex, bmi, gen_health, yes=()):
    """Return the estimated share with diagnosed diabetes for this profile."""
    choices = {"age": age, "sex": sex, "bmi_band": bmi_band(float(bmi)), "gen_health": gen_health}
    z = MODEL["intercept"]
    for k, v in choices.items():
        if v not in MODEL["options"][k]:
            raise ValueError(f"Unknown value for {k}: {v!r}")
        z += MODEL["coef"].get(f"{k}={v}", 0.0)
    for k in yes:
        if k not in MODEL["binary"]:
            raise ValueError(f"Unknown yes/no question: {k!r}")
        z += MODEL["coef"].get(f"{k}=1", 0.0)
    p = 1 / (1 + math.exp(-z))
    return {"probability": p, "above_review_line": p >= MODEL["threshold"], "base_rate": MODEL["data"]["base_rate_train"]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--age", required=True, choices=MODEL["options"]["age"])
    ap.add_argument("--sex", required=True, choices=MODEL["options"]["sex"])
    ap.add_argument("--bmi", required=True, type=float)
    ap.add_argument("--gen-health", default="Good", choices=MODEL["options"]["gen_health"])
    ap.add_argument("--yes", nargs="*", default=[], choices=list(MODEL["binary"]))
    a = ap.parse_args()
    r = predict(a.age, a.sex, a.bmi, a.gen_health, a.yes)
    print(f"Share with diagnosed diabetes among similar answers: {r['probability']:.1%} (average {r['base_rate']:.1%})")
    print("Educational use only. Not a diagnosis. For screening in Canada, see https://diabetes.ca/static/risk-assessment.html")
