# Data

`diabetes_012_health_indicators_BRFSS2015.csv` is the cleaned "Diabetes Health Indicators Dataset" published on Kaggle by Alex Teboul. It is derived from the 2015 Behavioral Risk Factor Surveillance System (BRFSS), a telephone health survey run by the US Centers for Disease Control and Prevention: https://www.cdc.gov/brfss/

- **Rows:** 253,680 survey answers. 21 health indicators and one label.
- **Label `Diabetes_012`:** 0 = no diabetes, 1 = prediabetes, 2 = diabetes. The model drops prediabetes rows.
- **Not included:** survey weights, so the rows describe the people who answered, not the US population.
- **Personal information:** none. The file has no names, dates or locations.

Check the Kaggle dataset page for the redistribution terms of the cleaned file.

The Canadian Community Health Survey (CCHS) public-use file used in the notebook comes from Statistics Canada and is not stored here.
