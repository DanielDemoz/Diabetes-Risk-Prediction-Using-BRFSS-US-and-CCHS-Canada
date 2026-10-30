# Diabetes Risk Model: US and Canadian survey data

How well do health survey answers predict diabetes, and for whom does that work less well?

**Live demo:** https://danieldemoz.github.io/Diabetes-Risk-Prediction-Using-BRFSS-US-and-CCHS-Canada/
**Status:** working model on real public survey data. A learning project, not a medical tool.

## What it does

You enter a survey profile: age, sex, BMI, general health and ten yes/no questions. The model estimates how common diagnosed diabetes is among US adults who answered the same way. It runs in the browser, and each result shows:

- the average rate for everyone in the data,
- the factors that moved the estimate most,
- warnings for groups where the model is weaker.

The page links to [CANRISK from Diabetes Canada](https://diabetes.ca/static/risk-assessment.html), a validated screening tool, for anyone with a real concern.

## Results on held-back data

Trained on 70% of BRFSS 2015 answers (174,334) and tested on the other 30% (74,715).

| Model | AUC | Brier | Avg predicted | Actual |
|---|---|---|---|---|
| Simple rule: age and BMI only | 0.747 | 0.109 | 14.1% | 14.2% |
| Inputs both surveys share | 0.777 | 0.106 | 14.1% | 14.2% |
| **Logistic regression (used)** | **0.826** | **0.099** | **14.2%** | **14.2%** |
| Same model plus income and education | 0.827 | 0.098 | 14.2% | 14.2% |
| Gradient boosting, all 21 columns | 0.834 | 0.097 | 14.2% | 14.2% |
| Original notebook set-up (SMOTE) | 0.815 | 0.178 | 38.8% | 15.8% |

**Honest findings:**

1. **SMOTE broke the percentages.** The original notebook oversampled the minority class. Ranking stayed good, but the average prediction was 38.8% against an actual 15.8%, and two thirds of people were flagged. The rebuilt model does not oversample, and its percentages match reality (see the calibration chart in the demo).
2. **The US vs Canada gap was mostly about inputs.** The notebook's US models used all 21 columns, while the Canadian models used only blood pressure, cholesterol, BMI class, smoking, activity, fruit and vegetables. Trained on the same kinds of inputs, the US data scores 0.777, close to Canada's 0.773. The US label also counted prediabetes; the Canadian one did not.
3. **Income and education are not needed as inputs.** Adding them improves AUC by 0.001, so they are left out and used only to audit fairness.

## Fairness check (held-back data, 20% review line)

| Group | People | Actual | Predicted | AUC | Caught |
|---|---|---|---|---|---|
| Female | 41,661 | 13.2% | 13.1% | 0.84 | 65% |
| Male | 33,054 | 15.4% | 15.5% | 0.81 | 67% |
| Age 18-44 | 16,230 | 4.3% | 4.1% | 0.84 | **29%** |
| Age 65+ | 25,927 | 21.0% | 21.0% | **0.76** | 71% |
| Income under $25k | 16,860 | 23.8% | **21.4%** | 0.78 | 75% |
| Income $75k+ | 26,863 | 8.1% | 9.1% | 0.84 | 52% |
| No high school diploma | 3,942 | 26.7% | **23.9%** | 0.76 | 77% |

- Lower-income and less-educated groups are under-estimated by 2 to 3 points.
- Younger adults with diabetes are mostly missed at a single 20% line.
- Ranking is weaker for adults 65 and over.

The full table is in the demo and in [docs/model.json](docs/model.json).

## Responsible AI, following the Reliable AI Adoption Framework

1. **Business problem:** understand which survey answers relate to diabetes, and whether a model from one country holds in another. Success means honest percentages and similar performance across groups.
2. **Data readiness:** BRFSS 2015 health indicators. Self-reported, one year, no survey weights. Prediabetes answers (4,631) were removed so the label matches the Canadian survey. See [data/README.md](data/README.md).
3. **Governance and risk:** health is high stakes, so the tool is educational and advisory only. Income and education are excluded as inputs and used for the fairness audit.
4. **Technology:** logistic regression with one-hot features, exported to JSON and run in the browser. Chosen over gradient boosting because it scores almost the same and every factor can be shown.
5. **People and adoption:** each result explains its main factors in plain words and points to CANRISK and a health professional.
6. **Outcomes and monitoring:** recheck calibration and the fairness table on any new data. Validate on Canadian data before any use in Canada.

See the [model card](MODEL_CARD.md) for details.

## How to run

```bash
pip install -r requirements.txt
python model/train.py      # rebuilds docs/model.json from data/
python predict.py --age 50-54 --sex Female --bmi 31 --yes HighBP PhysActivity
pytest -q
```

Open the demo locally with `python -m http.server -d docs`.

The original exploration, including the Canadian survey (CCHS) analysis, is in [notebooks/us_canada_comparison_2025.ipynb](notebooks/us_canada_comparison_2025.ipynb). The CCHS file is not included in this repository.

## Project history

The 2025 version trained models with SMOTE and showed a web dashboard with randomly generated patients. A summary report from that time listed scores that did not match the notebook, so it was removed. The project was rebuilt in October 2026 with a calibrated model, a fairness audit, and a re-check of the cross-country comparison.

## Tech stack

Python, pandas, scikit-learn, vanilla JavaScript, GitHub Pages.

## License

Code: MIT. Data: see [data/README.md](data/README.md).

*For educational and research purposes only. Not for medical decisions.*
