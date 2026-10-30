# Model card: diabetes risk from survey answers

## Model details
- **Type:** logistic regression (scikit-learn, C = 1.0) on one-hot encoded features.
- **Output:** the estimated share of adults with diagnosed diabetes among people who gave the same survey answers.
- **Version:** October 2026.
- **Owner:** Daniel S. Demoz (portfolio project).

## Intended use
- **For:** learning, and helping a public health analyst think about which groups to reach with screening.
- **Not for:** diagnosis, treatment, insurance, employment, or any decision about an individual. People with a concern should use a validated tool such as [CANRISK](https://diabetes.ca/static/risk-assessment.html) and see a health professional.

## Inputs
Age band (13 bands), sex, BMI band (6 bands), self-rated general health (5 levels), and ten yes/no answers: high blood pressure, high cholesterol, smoked 100+ cigarettes, physical activity in the past 30 days, fruit daily, vegetables daily, heavy drinking, heart disease or heart attack, stroke, and serious difficulty walking.

Income and education are **not** inputs. They are used only for the fairness audit.

## Data
- **Source:** BRFSS 2015 (CDC), in the cleaned "Diabetes Health Indicators" version. See [data/README.md](data/README.md).
- **Rows:** 253,680. Prediabetes answers (4,631) removed, leaving 249,049.
- **Label:** ever told by a health professional that they have diabetes (1) vs no diabetes (0). 14.2% positive.
- **Split:** random, stratified, 70% train (174,334) and 30% test (74,715). Seed 42.
- **Not included:** survey weights, lab results, family history, ethnicity.

## Performance (test set)
| Model | AUC | PR-AUC | Brier |
|---|---|---|---|
| Age and BMI only | 0.747 | 0.315 | 0.109 |
| **This model** | **0.826** | **0.433** | **0.099** |
| Gradient boosting, all 21 columns | 0.834 | 0.448 | 0.097 |

At a 20% review line the model flags 25.4% of people and catches 65.5% of people with diabetes, with 36.6% precision.

Calibration: average prediction 14.2%, actual 14.2%. Each tenth of the test set is within 2 points of its actual rate.

## Results by group (test set)
| Group | People | Actual | Predicted | AUC | Caught at 20% |
|---|---|---|---|---|---|
| Female | 41,661 | 13.2% | 13.1% | 0.836 | 64.5% |
| Male | 33,054 | 15.4% | 15.5% | 0.812 | 66.6% |
| Age 18-44 | 16,230 | 4.3% | 4.1% | 0.844 | 29.2% |
| Age 45-64 | 32,558 | 13.7% | 13.7% | 0.828 | 64.9% |
| Age 65+ | 25,927 | 21.0% | 21.0% | 0.758 | 70.7% |
| Income under $25k | 16,860 | 23.8% | 21.4% | 0.784 | 75.3% |
| Income $25k-50k | 18,347 | 15.6% | 15.9% | 0.798 | 65.7% |
| Income $50k-75k | 12,645 | 12.4% | 12.7% | 0.814 | 59.3% |
| Income $75k+ | 26,863 | 8.1% | 9.1% | 0.840 | 51.7% |
| No high school diploma | 3,942 | 26.7% | 23.9% | 0.755 | 76.5% |
| High school graduate | 18,436 | 18.3% | 17.8% | 0.798 | 70.7% |
| Some college | 20,524 | 15.0% | 14.8% | 0.818 | 66.0% |
| College graduate | 31,813 | 9.8% | 10.4% | 0.840 | 55.8% |

## Known limitations
- The data is cross-sectional. The model describes who has diagnosed diabetes, not who will develop it.
- Answers are self-reported. Undiagnosed diabetes is counted as "no".
- Coefficients are associations, not causes. Heavy drinking lowers the estimate, which most likely reflects who drinks and habit changes after diagnosis.
- No survey weights, so results describe respondents, not the US population.
- 23,899 rows exactly repeat another row. With mostly yes/no answers this is expected from different people, so they were kept.
- Trained on US data from 2015. It has not been validated on Canadian data.

## Ethical considerations
The data is anonymous survey data. The model is weaker for lower-income and less-educated groups and for younger adults, which are groups where missed cases matter. Any real use would need a group-aware threshold, clinician input, and validation on the target population.

## Monitoring
- Recheck calibration and the group table on any new data.
- Validate on Canadian survey data before any use in Canada.
- Review the threshold with clinicians, weighing missed cases against extra follow-ups.
