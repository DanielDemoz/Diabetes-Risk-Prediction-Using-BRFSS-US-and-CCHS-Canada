"""Train and audit the diabetes model on BRFSS 2015, then export docs/model.json.

Run from the repo root:  python model/train.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
THRESHOLD = 0.20

raw = pd.read_csv(ROOT / "data" / "diabetes_012_health_indicators_BRFSS2015.csv")

# Label: told they have diabetes (2) vs no diabetes (0). Prediabetes (1) is
# dropped, because it is a different condition and the Canadian survey has no
# matching answer.
n_prediabetes = int((raw.Diabetes_012 == 1).sum())
d = raw[raw.Diabetes_012 != 1].copy()
d["y"] = (d.Diabetes_012 == 2).astype(int)

AGE_LABELS = {1: "18-24", 2: "25-29", 3: "30-34", 4: "35-39", 5: "40-44", 6: "45-49", 7: "50-54",
              8: "55-59", 9: "60-64", 10: "65-69", 11: "70-74", 12: "75-79", 13: "80+"}
BMI_BINS = [0, 18.5, 25, 30, 35, 40, 200]
BMI_LABELS = ["Under 18.5", "18.5-24.9", "25-29.9", "30-34.9", "35-39.9", "40+"]
GEN_LABELS = {1: "Excellent", 2: "Very good", 3: "Good", 4: "Fair", 5: "Poor"}

d["age"] = d.Age.astype(int).map(AGE_LABELS)
d["bmi_band"] = pd.cut(d.BMI, BMI_BINS, right=False, labels=BMI_LABELS).astype(str)
d["gen_health"] = d.GenHlth.astype(int).map(GEN_LABELS)
d["sex"] = d.Sex.map({0: "Female", 1: "Male"})

BINARY = {
    "HighBP": "High blood pressure",
    "HighChol": "High cholesterol",
    "Smoker": "Smoked 100+ cigarettes in life",
    "PhysActivity": "Physical activity in past 30 days",
    "Fruits": "Fruit at least once a day",
    "Veggies": "Vegetables at least once a day",
    "HvyAlcoholConsump": "Heavy drinking",
    "HeartDiseaseorAttack": "Heart disease or heart attack",
    "Stroke": "Ever had a stroke",
    "DiffWalk": "Serious difficulty walking or climbing stairs",
}
CATEGORICAL = ["age", "sex", "bmi_band", "gen_health"]
# Income and education predict diabetes, but they are left out as inputs on
# purpose. They are used only to check whether the model works equally well
# across groups.
HARMONIZED = ["HighBP", "HighChol", "Smoker", "PhysActivity", "Fruits", "Veggies"]
ALL_RAW = [c for c in raw.columns if c != "Diabetes_012"]


def design(frame, cats, bins):
    x = pd.get_dummies(frame[cats].astype(str), prefix_sep="=", dtype=float)
    for b in bins:
        x[f"{b}=1"] = frame[b].astype(float)
    return x


tr_idx, te_idx = train_test_split(d.index, test_size=0.3, random_state=SEED, stratify=d.y)
tr, te = d.loc[tr_idx], d.loc[te_idx]


def fit_lr(cats, bins, C=1.0):
    Xtr = design(tr, cats, bins)
    Xte = design(te, cats, bins).reindex(columns=Xtr.columns, fill_value=0.0)
    m = LogisticRegression(C=C, max_iter=3000).fit(Xtr, tr.y)
    return m, Xtr, Xte, m.predict_proba(Xte)[:, 1]


def scores(y, p):
    alert = p >= THRESHOLD
    return {
        "auc": round(float(roc_auc_score(y, p)), 3),
        "pr_auc": round(float(average_precision_score(y, p)), 3),
        "brier": round(float(brier_score_loss(y, p)), 3),
        "mean_predicted": round(float(p.mean()), 3),
        "actual_rate": round(float(np.mean(y)), 3),
        "recall": round(float(alert[y == 1].mean()), 3),
        "precision": round(float(np.asarray(y)[alert].mean()), 3) if alert.any() else None,
        "flagged_share": round(float(alert.mean()), 3),
    }


metrics = {}

# 1. Demo model (used in the web page).
demo, Xtr, Xte, p_demo = fit_lr(CATEGORICAL, list(BINARY), C=1.0)
metrics["demo_logistic"] = scores(te.y.values, p_demo)

# 2. The variables both surveys share, the same set the notebook used for
#    Canada. BMI is banded because the Canadian file only has BMI classes.
_, _, _, p_h2 = fit_lr(["bmi_band"], HARMONIZED)
metrics["harmonized_6_plus_bmi"] = scores(te.y.values, p_h2)

# 3. Simple baseline: age band and BMI band only.
_, _, _, p_base = fit_lr(["age", "bmi_band"], [])
metrics["baseline_age_bmi"] = scores(te.y.values, p_base)

# 4. Same model plus income and education, to measure what leaving them out costs.
d["income"] = d.Income.astype(int).astype(str)
d["education"] = d.Education.astype(int).astype(str)
tr, te = d.loc[tr_idx], d.loc[te_idx]
_, _, _, p_ie = fit_lr(CATEGORICAL + ["income", "education"], list(BINARY), C=1.0)
metrics["demo_plus_income_education"] = scores(te.y.values, p_ie)

# 5. Gradient boosting on every raw column, including income and education.
hgb = HistGradientBoostingClassifier(random_state=SEED, max_iter=300, learning_rate=0.06)
hgb.fit(tr[ALL_RAW], tr.y)
p_hgb = hgb.predict_proba(te[ALL_RAW])[:, 1]
metrics["gb_all_columns"] = scores(te.y.values, p_hgb)

# 6. The original notebook set-up: every column, prediabetes counted as diabetes,
#    SMOTE oversampling. Re-run to show its effect on calibration.
try:
    from imblearn.over_sampling import SMOTE

    o = raw.copy()
    o["y"] = (o.Diabetes_012 > 0).astype(int)
    otr, ote = train_test_split(o, test_size=0.3, random_state=SEED, stratify=o.y)
    Xs, ys = SMOTE(random_state=SEED).fit_resample(otr[ALL_RAW], otr.y)
    om = LogisticRegression(max_iter=10000).fit(Xs, ys)
    metrics["original_notebook_smote"] = scores(ote.y.values, om.predict_proba(ote[ALL_RAW])[:, 1])
except ImportError:
    pass

# Calibration of the demo model, in ten equal-size groups.
cal = []
order = np.argsort(p_demo)
for chunk in np.array_split(order, 10):
    cal.append({"pred": round(float(p_demo[chunk].mean()), 3), "actual": round(float(te.y.values[chunk].mean()), 3), "n": int(len(chunk))})

# Results by group. Income and education are audit-only.
INCOME = {1: "Under $25k", 2: "Under $25k", 3: "Under $25k", 4: "Under $25k", 5: "$25k-50k", 6: "$25k-50k",
          7: "$50k-75k", 8: "$75k+"}
EDU = {1: "No high school diploma", 2: "No high school diploma", 3: "No high school diploma",
       4: "High school graduate", 5: "Some college", 6: "College graduate"}
AGEG = lambda a: "18-44" if a <= 5 else ("45-64" if a <= 9 else "65+")
groups = {
    "Sex": te.sex,
    "Age group": te.Age.astype(int).map(AGEG),
    "Household income": te.Income.astype(int).map(INCOME),
    "Education": te.Education.astype(int).map(EDU),
}
segments = []
for gname, col in groups.items():
    for val in sorted(col.unique()):
        m = (col == val).values
        y, p = te.y.values[m], p_demo[m]
        alert = p >= THRESHOLD
        segments.append({
            "group": gname, "value": val, "n": int(m.sum()),
            "actual": round(float(y.mean()), 3), "predicted": round(float(p.mean()), 3),
            "auc": round(float(roc_auc_score(y, p)), 3),
            "recall": round(float(alert[y == 1].mean()), 3),
            "flagged": round(float(alert.mean()), 3),
        })

# Support: how many training rows share each input value.
support = {c: int(v) for c, v in Xtr.sum().items()}

coef = {c: round(float(w), 5) for c, w in zip(Xtr.columns, demo.coef_[0])}
out = {
    "intercept": round(float(demo.intercept_[0]), 5),
    "coef": coef,
    "support": support,
    "options": {
        "age": list(AGE_LABELS.values()),
        "sex": ["Female", "Male"],
        "bmi_band": BMI_LABELS,
        "gen_health": list(GEN_LABELS.values()),
    },
    "bmi_bins": BMI_BINS[1:-1],
    "binary": BINARY,
    "threshold": THRESHOLD,
    "data": {
        "rows_total": int(len(raw)),
        "duplicates": int(raw.duplicated().sum()),
        "prediabetes_dropped": n_prediabetes,
        "rows_used": int(len(d)),
        "train": int(len(tr)), "test": int(len(te)),
        "base_rate_train": round(float(tr.y.mean()), 3),
    },
    "metrics": metrics,
    "calibration": cal,
    "segments": segments,
    "notebook_results": {
        "note": "Reported by the original 2025 notebook. Not re-run here: the CCHS file is not in this repository.",
        "brfss": {"logistic_regression": 0.8149, "random_forest": 0.7913, "xgboost": 0.8206},
        "cchs": {"logistic_regression": 0.7729, "random_forest": 0.7055, "xgboost": 0.7315},
    },
}
(ROOT / "docs").mkdir(exist_ok=True)
(ROOT / "docs" / "model.json").write_text(json.dumps(out, indent=1))
print(json.dumps(metrics, indent=1))
for s in segments:
    print(s)
