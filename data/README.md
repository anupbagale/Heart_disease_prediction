# Data

`raw/` holds the four processed cohorts of the **UCI Heart Disease** dataset,
one CSV per hospital, with a header row and empty cells for missing values.
Values are the original UCI numeric codes, unchanged.

| File | Centre | Patients | Without disease |
|---|---|---|---|
| `cleveland.csv` | Cleveland Clinic Foundation, USA | 303 | 164 |
| `hungarian.csv` | Hungarian Institute of Cardiology, Budapest | 294 | 188 |
| `switzerland.csv` | University Hospitals Zurich and Basel | 123 | 8 |
| `va.csv` | VA Medical Center, Long Beach, USA | 200 | 51 |

**Source.** Janosi A, Steinbrunn W, Pfisterer M, Detrano R. *Heart Disease* [Dataset].
UCI Machine Learning Repository, 1989. https://doi.org/10.24432/C52P4X.
Licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Principal investigators: Andras Janosi (Budapest), William Steinbrunn (Zurich),
Matthias Pfisterer (Basel), Robert Detrano (VA Long Beach and Cleveland).

**Provenance.** The files were created from UCI's `processed.<site>.data`
files. To regenerate them from UCI directly, or to check the committed copies
against UCI, run:

```bash
python scripts/download_data.py --check
```

## Columns

| Column | Meaning | Codes |
|---|---|---|
| `age` | Age in years | |
| `sex` | Sex | 1 = male, 0 = female |
| `cp` | Chest pain type | 1 typical angina, 2 atypical angina, 3 non-anginal pain, 4 asymptomatic |
| `trestbps` | Resting blood pressure (mmHg) | |
| `chol` | Serum cholesterol (mg/dl) | 0 = not measured (see below) |
| `fbs` | Fasting blood sugar > 120 mg/dl | 1 = true |
| `restecg` | Resting ECG | 0 normal, 1 ST-T wave abnormality, 2 left ventricular hypertrophy |
| `thalach` | Maximum heart rate achieved | |
| `exang` | Exercise-induced angina | 1 = yes |
| `oldpeak` | ST depression induced by exercise relative to rest (mm) | |
| `slope` | Slope of the peak exercise ST segment | 1 upsloping, 2 flat, 3 downsloping |
| `ca` | Major vessels (0-3) coloured by fluoroscopy | |
| `thal` | Thallium stress test result | 3 normal, 6 fixed defect, 7 reversible defect |
| `num` | Angiographic diagnosis | 0 = < 50 % narrowing; 1-4 = > 50 % narrowing |

## Cleaning applied in `src/heartdisease/data.py`

* **Outcome.** `target = 1` if `num > 0`. The 1-4 levels are not a reliable
  severity scale and Hungary only records 0/1, so the binary outcome used in
  the original studies is the appropriate one.
* **Impossible zeros.** Cholesterol of 0 mg/dl (all 123 Swiss patients and 49 at
  VA) and one VA resting blood pressure of 0 mmHg mean "not measured" and are
  set to missing.
* **Structural missingness.** `ca`, `thal` and `slope` are mostly missing
  outside Cleveland (see `results/figures/missingness.png`). This is why the
  analyses also use a 10-feature set without them.

Nothing else is changed. Imputation happens inside the model pipelines so it
learns only from training data.
