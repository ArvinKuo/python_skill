# python_skill — Telco Customer Churn Prediction

Predicts which telecom customers are likely to leave (churn), using the
[Telco Customer Churn](data/raw/Telco-Customer-Churn.csv) dataset
(7,043 customers, 21 columns).

## Setting up a new environment

Copying the project does **not** install anything automatically.
`requirements.txt` is only a list of packages; you have to install it
yourself once in every new environment.

**Requirements:** Python 3.10 or newer (developed and tested on Python 3.13)
and internet access for the first install.

**Windows (PowerShell)**

```powershell
cd <path-to>\python_skill               # always work from the project root
python -m venv .venv                    # recommended: isolated environment for this project
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest tests                  # check that everything works
python main.py
```

**Linux / macOS**

```bash
cd <path-to>/python_skill
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest tests
python main.py
```

If PowerShell refuses to run `Activate.ps1` ("running scripts is disabled"),
run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or skip
activation and call `.\.venv\Scripts\python.exe` directly instead of `python`.

### Things to watch out for

- **Use `python -m pip`, not `pip`, and use `-r`.** `python -m pip` installs
  into the same Python that runs `main.py`; a bare `pip` may belong to a
  different Python on machines with several installs. Without `-r`, pip
  looks for a package literally called `requirements.txt` and fails.
- **Activate the virtual environment in every new terminal.** Packages
  installed in `.venv` are only visible while it is active. A
  `ModuleNotFoundError` (e.g. `No module named 'sklearn'`) usually means the
  environment is not active or the install step was skipped.
- **Run everything from the project root.** The default paths
  (`data/raw/...`, `output/...`), the `from src...` imports and the tests are
  all relative to the current folder. Running `python some\other\path\main.py`
  from a different folder will not find the CSV.
- **Bring the raw data with you.** `data/raw/Telco-Customer-Churn.csv` is
  required. When moving the project with git, make sure the CSV is committed
  and pushed. It is not excluded by `.gitignore`, but it is only included if
  it has been committed.
- **Package versions are not pinned.** `requirements.txt` has no version
  numbers, so a new install gets the latest releases. The project was
  developed with pandas 2.3.0, numpy 2.5.1, scikit-learn 1.9.1,
  matplotlib 3.10.3 and pytest 8.3.3. If results must be exactly
  reproducible, pin these versions (e.g. `pandas==2.3.0`).
- **Generated files do not need to be copied.** Everything in `output/` and
  `data/processed/` is recreated by `python main.py`. Do not commit `.venv/`
  (already excluded by `.gitignore`); create it again on the new machine.
- **No screen is needed.** The chart is saved directly to a PNG file and no
  window is opened, so the pipeline also runs on servers and CI runners.
- **Close the raw CSV in Excel before running, if possible.** Reading works
  while Excel has the file open, but Windows locks the file, so tools that
  check or copy it (e.g. `Get-FileHash`) can fail.

## How to run

```powershell
python main.py                # validate -> clean -> explore -> define X/y -> split -> train -> evaluate -> predictions.csv
python -m pytest tests        # run the unit tests
```

| Option         | Default                               | What it is                          |
|----------------|---------------------------------------|-------------------------------------|
| `--input`      | `data/raw/Telco-Customer-Churn.csv`   | Raw data (read only, never changed) |
| `--report`     | `output/validation_report.txt`        | Data validation report              |
| `--processed`  | `data/processed/telco_churn_clean.csv`| Cleaned and encoded data            |
| `--eda-report` | `output/eda_report.txt`               | Exploratory analysis report         |
| `--chart`      | `output/churn_by_contract.png`        | Churn rate by contract type chart   |
| `--model-report` | `output/model_report.txt`           | Split settings, pipeline and model coefficients |
| `--evaluation-report` | `output/evaluation_report.txt` | Test-set metrics and confusion matrix |
| `--output`     | `output/predictions.csv`              | One prediction per test-set customer |

## Project structure

```
main.py                 Command-line entry point
src/validation.py       Task 3: checks the raw data and writes a plain-language report
src/cleaning.py         Task 4: converts the raw data into a numeric feature matrix
src/eda.py              Task 5: churn rates by customer group, report and chart
src/features.py         Task 6: splits cleaned data into features (X), target (y) and customer IDs
src/model.py            Task 7-8: train/test split and Logistic Regression pipeline
src/evaluation.py       Task 9-10: test-set metrics and predictions.csv
tests/                  Unit tests for validation, cleaning, analysis, features, model and evaluation
data/raw/               Original CSV (never modified)
data/processed/         Output of the cleaning step
output/                 Reports, charts and predictions
```

## Task 3 — Data validation

`src/validation.py` answers these questions and saves the answers to
`output/validation_report.txt`: row/column counts, duplicate customer IDs,
missing values (blank text counts as missing), data type of every column,
whether `TotalCharges` can be read as a number, whether `tenure` is between
0 and 120 months, whether charges are non-negative, whether `Churn` contains
only `Yes`/`No`, and whether any category column contains an unknown value.
The status is **PASS** when nothing is found, otherwise **REVIEW** with a
numbered list of the items to look at.

Result on the raw data: **REVIEW**. The only finding is that `TotalCharges`
is stored as text and contains 11 blank values. Everything else passed:
no duplicate IDs, no negative values, tenure 0–72 months, `Churn` only
`Yes`/`No`, no unexpected category values.

## Task 4 — Data cleaning

Cleaning is done by `clean()` in [src/cleaning.py](src/cleaning.py). Running
`python main.py` prints a summary of every change and saves the result to
`data/processed/telco_churn_clean.csv` (customerID, 41 feature columns,
Churn). Below is each change, the evidence for it, and why it was made.

### 1. The raw CSV is never modified

- **What:** The raw file is only read. Cleaning works on a copy in memory and
  the result is written to a different file in `data/processed/`. `main.py`
  refuses to run if `--processed` points at the input file.
- **Why:** Keeping the original lets anyone re-run the whole pipeline from the
  source, compare before/after, and change a cleaning decision later without
  losing information. A unit test confirms `clean()` does not change the
  DataFrame it receives.

### 2. `TotalCharges` converted from text to a number

- **What we found:** pandas reads `TotalCharges` as text because 11 rows
  contain a single space (`" "`) instead of a number. All other 7,032 values
  are valid numbers. Because the value is a space and not truly empty,
  `df.isna()` reports **0** missing values, which hides the problem.
- **Investigation:** All 11 blank rows have `tenure = 0`, i.e. customers who
  signed up in the current month (10 on two-year contracts, 1 on a one-year
  contract, none churned). They have a monthly price but have not been billed
  yet, so the blank means "nothing charged so far", not "unknown".
- **What we changed:** Surrounding spaces are removed, empty text becomes
  missing, and the column is converted to a decimal number. Any value that is
  real text (e.g. `"abc"`) stops the pipeline with an error instead of being
  silently turned into a missing value, so a new kind of data problem cannot
  slip through unnoticed.

### 3. How missing numeric values are handled

| Situation | Action | Why |
|-----------|--------|-----|
| `TotalCharges` missing **and** `tenure = 0` (the 11 rows in this dataset) | Set to **0** | A customer who has not been billed has paid exactly 0. This is the true value, not a guess. |
| `TotalCharges` missing **and** `tenure > 0` (does not occur in this dataset) | Estimate as **`tenure × MonthlyCharges`** | On the 7,032 rows with a value, this estimate has a 0.9996 correlation with the real `TotalCharges` and a median error of 2%. |
| `tenure` or `MonthlyCharges` missing (does not occur) | **Stop with an error** | There is no reliable way to fill them, and filling with an average would invent customer information. Someone should look at the data. |

Alternatives that were considered and rejected:

- **Deleting the 11 rows:** this would remove every brand-new customer from
  training, but the model will still have to score new customers later.
- **Filling with the mean or median** (≈ $2,280 / $1,400): this would make a
  customer who has paid nothing look like a long-standing customer and
  contradicts `tenure = 0`.

**Rows removed: 0.**

### 4. `customerID` removed from the features but kept separately

- **What:** `customerID` is not part of the feature matrix. `clean()` returns
  it separately (`customer_ids`), in the same row order as the features, and
  it is the first column of the saved CSV.
- **Why:** The ID is a unique label with no relationship to churn. If the
  model saw it, it could only memorise individual customers, which does not
  help with new customers. It is still needed at the end so each prediction
  can be matched to a real customer.

### 5. `Churn` converted to a binary target

- **What:** `Yes` → **1** (customer left), `No` → **0** (customer stayed).
  Any other value stops the pipeline with an error. If a file has no `Churn`
  column (for example, new customers to score), cleaning still works and
  simply returns no target.
- **Why:** Models need a numeric target. Using 1 for the event we want to
  detect (churn) is the usual convention, so metrics such as precision and
  recall refer to churners. About 26.5% of customers churned (see Task 5).

### 6. Categorical features encoded with one consistent rule

| Column type | Columns | Encoding |
|-------------|---------|----------|
| Already 0/1 | `SeniorCitizen` | Kept as is |
| Yes / No only | `Partner`, `Dependents`, `PhoneService`, `PaperlessBilling` | `Yes` → 1, `No` → 0 (one column each) |
| Everything else | `gender`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaymentMethod` | **One-hot:** one 0/1 column per category, e.g. `Contract_Month-to-month`, `Contract_One year`, `Contract_Two year` |

Why this approach:

- **No false ordering.** Numbering categories (e.g. DSL = 1, Fiber = 2,
  No = 3) would tell the model that "No" is bigger than "DSL", which is
  meaningless. One-hot columns are just "yes, this customer has this value"
  or "no". Yes/No columns are already a single yes/no flag, so one column
  is enough.
- **Fixed list of categories.** The categories come from the expected-value
  list in `src/validation.py` (`EXPECTED_CATEGORIES`), not from whatever
  values happen to be in a file. The result always has the same 41 columns
  in the same order, even for a single customer or a file where some
  category does not appear. This matters because the training data and the
  data we predict on later must line up column by column. It also means
  nothing is "learned" from the data during cleaning, so cleaning before the
  train/test split does not leak test information into training.
- **Unknown values stop the pipeline.** A value outside the expected list
  (e.g. `"Three year"` or a typo) raises an error instead of being quietly
  encoded as all zeros. Task 3's validation report lists such values first.
- **"No internet service" / "No phone service" kept as their own category.**
  They could be merged with "No", but they describe a different customer
  (one who does not have the base service at all), and keeping them is more
  faithful to the raw data. The model can still combine them if useful.
- **All categories kept (no column dropped).** For a column with 3
  categories, the third one could be inferred from the other two, so some
  people drop one. We keep all of them because it is easier to read; if the
  model chosen in the training step is sensitive to this (plain,
  unregularised linear models), one column per feature can be dropped there.
- **Spaces around text are removed** before encoding, so `" Yes"` and
  `"Yes"` are treated the same.

### 7. What was deliberately *not* done during cleaning

- **No scaling of numeric columns.** `tenure`, `MonthlyCharges` and
  `TotalCharges` stay in their original units (months, dollars). Scaling, if
  a model needs it, will be fitted on the training split only, to avoid
  leaking information from the test data.
- **No outlier removal.** Validation found no negative values and tenure is
  within 0–72 months, so there was nothing to remove.
- **No duplicate removal.** Every `customerID` is unique.

### Result

| | Before | After |
|-|--------|-------|
| Rows | 7,043 | 7,043 |
| Feature columns | 19 (mixed text/number) | 41 (all numeric) |
| Missing values | 11 (hidden as blank text) | 0 |
| `TotalCharges` type | text | decimal number |
| `Churn` | `Yes` / `No` | 1 / 0 (kept separately as the target) |
| `customerID` | in the data | kept separately, not a feature |

## Task 5 — Exploratory analysis

`analyze()` in [src/eda.py](src/eda.py) answers all seven questions with
pandas, and `python main.py` prints the answers and saves them to
`output/eda_report.txt`, so every number below can be traced back to code.

**How the numbers are calculated**

- **Churn rate** of a group = customers in the group who churned ÷ all
  customers in the group. In pandas: `Churn` mapped to 1/0, then
  `churn.groupby(df[column]).mean()`. Because churned = 1 and stayed = 0,
  the average of the column is exactly the share of churned customers.
- **Tenure ranges:** `tenure` is a number of months, so it is first grouped
  with `pd.cut` into 0–12, 13–24, 25–48 and 49+ months (first year, second
  year, years 3–4, more than 4 years).
- **MonthlyCharges:** average and median for churned vs. stayed customers.
- The analysis uses the original text labels (e.g. `Month-to-month`) so the
  tables are readable. Every table also shows the number of customers, so
  small groups are not over-interpreted. The smallest group here has
  1,024 customers.

### Results

| Question | Result |
|----------|--------|
| 1. Overall churn | **26.5%** (1,869 of 7,043 customers) |
| 2. Contract | Month-to-month **42.7%** · One year **11.3%** · Two year **2.8%** |
| 3. InternetService | Fiber optic **41.9%** · DSL **19.0%** · No internet **7.4%** |
| 4. Tenure | 0–12 months **47.4%** · 13–24 **28.7%** · 25–48 **20.4%** · 49+ **9.5%** |
| 5. PaymentMethod | Electronic check **45.3%** · Mailed check **19.1%** · Bank transfer (automatic) **16.7%** · Credit card (automatic) **15.2%** |
| 6. MonthlyCharges | Churned: average **$74.44** (median $79.65) · Stayed: average **$61.27** (median $64.43) |
| 7. TechSupport | No tech support **41.6%** · With tech support **15.2%** · No internet service **7.4%** |

![Churn rate by contract type](output/churn_by_contract.png)

The chart is created by `plot_churn_by_contract()` and saved to
`output/churn_by_contract.png`. The dashed line is the overall churn rate
(26.5%); each bar shows the churn rate and the number of customers (n).

### Observations

These describe patterns in this dataset. They show which customer groups
churn more often, **not why** they churn. A higher churn rate in a group
does not mean that the group's characteristic causes churn.

1. **About one in four customers churned (26.5%).** Most customers stayed,
   so a model that always predicts "stays" would be right 73.5% of the time.
   Model results in later tasks need to be compared against that baseline,
   not against 50%.
2. **Churn is strongly associated with contract length.** 42.7% of
   month-to-month customers churned, compared with 11.3% on one-year and
   2.8% on two-year contracts, about 15 times lower than month-to-month.
   Month-to-month customers are also 55% of all customers, so they account
   for 1,655 of the 1,869 churned customers (89%).
3. **Churn is highest in the first year and falls as tenure grows.** 47.4%
   of customers in their first 12 months churned, compared with 9.5% of
   customers who have stayed more than 4 years. Tenure and contract type
   overlap: 91% of first-year customers are on month-to-month contracts,
   compared with 15% of customers with 49+ months. This table alone cannot
   separate the two.
4. **Fiber optic customers churn at about twice the rate of DSL customers**
   (41.9% vs. 19.0%), and customers without internet service churn least
   (7.4%). Fiber customers also pay more per month (average $91.50 vs.
   $58.10 for DSL), which is consistent with observation 6.
5. **Customers who pay by electronic check churn at about three times the
   rate of the automatic payment methods** (45.3% vs. 15–17%). Mailed check
   (19.1%) is close to the automatic methods, so the difference is specific
   to electronic check rather than "manual vs. automatic" in general.
6. **Customers who churned had higher monthly charges.** Their average was
   $74.44, compared with $61.27 for those who stayed, $13.18 more per month.
   The median shows the same direction ($79.65 vs. $64.43), so the gap is
   not caused by a few extreme values.
7. **Customers without tech support churn much more often** (41.6% vs.
   15.2% with tech support). The gap also appears within each internet
   type: DSL 27.8% vs. 9.7%, fiber optic 49.4% vs. 22.6%. So it is not only
   because fiber customers are less likely to have tech support. It is
   still an association: for example, customers who plan to stay longer may
   be more likely to buy add-on services in the first place.

**What this means for the next tasks:** contract type, tenure, internet
service, payment method, monthly charges and tech support all show clear
differences in churn rate, so they are likely to be useful features for the
model. Several of them overlap (contract and tenure, internet type and
monthly charges), so the model's view of each feature's importance may
differ from these one-at-a-time tables.

## Task 6 — Features and target

`get_features_and_target()` in [src/features.py](src/features.py) is the
**only** place that decides what the model sees. Training, evaluation and
prediction code must get X and y from it rather than dropping columns
themselves. It works on either the output of `clean()` or the saved
`data/processed/telco_churn_clean.csv`. The CSV is the easier place to make
a mistake, because it contains `customerID`, the features and `Churn` side
by side.

Conceptually it does what the task describes:

```python
X = cleaned_df.drop(columns=["Churn", "customerID"])
y = cleaned_df["Churn"]
```

It also returns `customer_ids` (same rows, same order as X and y) so every
prediction can be matched back to a customer.

### What goes where

| Role | Columns | Why |
|------|---------|-----|
| **Target (y)** | `Churn` (1 = churned, 0 = stayed) | This is what the model predicts. |
| **Features (X)** | 41 columns: `tenure`, `MonthlyCharges`, `TotalCharges`, `SeniorCitizen`, the 4 Yes/No columns, and the one-hot columns for the other 11 categorical fields (see Task 4) | Customer, service and billing information recorded for each customer. |
| **Excluded: identifier** | `customerID` | A unique label with no relationship to churn. A model could only memorise individual IDs, which is useless for new customers. Kept separately for the prediction output. |
| **Excluded: target** | `Churn` | Giving the model the answer as an input ("target leakage") would produce perfect but meaningless scores. |

`python main.py` prints this split, e.g.
`Features (X): 7,043 rows x 41 columns` and
`Target (y): Churn -> 1,869 churned (1), 5,174 stayed (0)`.

### Safeguards

Dropping two columns by name is easy to get wrong silently: a renamed
column, an extra ID-like column added later, or forgetting the drop
altogether would still produce a model. So `validate_features()` checks X
every time and **stops with an error** if:

- X contains `customerID` or `Churn`;
- X contains any column that is not one of the 41 expected features (for
  example a new `account_number` column). The expected list comes from
  `feature_columns()` in `src/cleaning.py`, so only columns created by the
  cleaning step can reach the model;
- an expected feature is missing, or the columns are in a different order
  (the model relies on the same columns in the same order for training and
  prediction);
- any feature is not numeric or has missing values.

The target must contain only 0 and 1 with no missing values, and
`customerID` must be present with no missing values. For new customers
without a known outcome, `get_features_and_target(data, require_target=False)`
returns X and the IDs with no y.

`tests/test_features.py` checks each of these cases, including that the
saved CSV and the in-memory cleaning result produce identical X, y and IDs.

## Task 7 — Train/test split

`split_data()` in [src/model.py](src/model.py) splits the data once, before
any model is trained. The model learns only from the training set and is
judged only on the test set, which it has never seen. This is how we
estimate how it will do on new customers.

```python
train_test_split(X, y, customer_ids, test_size=0.20, random_state=42, stratify=y)
```

| Setting | Value | Why |
|---------|-------|-----|
| `test_size` | **0.20** | 80% of customers for training, 20% held back for testing. Enough training data, and a test set large enough (1,409 customers, 374 churners) to give stable results. |
| `random_state` | **42** | Fixes the random shuffle, so every run puts exactly the same customers in each set and results are reproducible. The value itself is arbitrary; what matters is that it is fixed and recorded. |
| `stratify` | **`y` (Churn)** | Only 26.5% of customers churned. Stratifying keeps that share the same in both sets. Without it, a split can drift by chance (with another seed and no stratification the test set had 27.5% churners). |
| `customer_ids` | split together with X and y | Passed to the same `train_test_split` call, so each ID stays on the same row as its features and outcome and the predictions file can name the customer. |

**Result** (also printed by `python main.py` and saved in `output/model_report.txt`):

| | Customers | Churned | Churn share |
|-|-----------|---------|-------------|
| Training set | 5,634 | 1,495 | 26.54% |
| Test set | 1,409 | 374 | 26.54% |
| All | 7,043 | 1,869 | 26.54% |

Tests in `tests/test_model.py` check the 80/20 sizes and the equal churn
share. They also check that the split is identical on every run, that no
customer is in both sets, and that IDs stay aligned with their rows.

## Task 8 — Logistic Regression pipeline

`build_model()` in [src/model.py](src/model.py) creates a scikit-learn
`Pipeline` with two stages. `train_model()` fits it on the training set,
and `predict_churn()` returns a 0/1 prediction and a churn probability for
each customer.

```python
preprocessor = ColumnTransformer(
    [("scale_numeric", StandardScaler(), ["tenure", "MonthlyCharges", "TotalCharges"])],
    remainder="passthrough",          # the 38 encoded 0/1 columns are used as they are
)
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000)),
])
model.fit(X_train, y_train)
```

### What each stage does

**1. `preprocessor` — ColumnTransformer + StandardScaler**

- A `ColumnTransformer` applies different processing to different columns.
  Here only the three numeric columns are changed. The other 38 columns are
  already 0/1 from Task 4 and pass through untouched (a test checks this).
- `StandardScaler` rewrites each numeric value as "how many standard
  deviations from the average": `(value − mean) ÷ standard deviation`.
- **Why scale:** the numeric columns have very different ranges (tenure
  0–72, MonthlyCharges 18–119, TotalCharges 0–8,685) while every other
  column is 0 or 1. Logistic Regression's default regularisation shrinks all
  coefficients by the same rule, which is only fair when features are on a
  similar scale. Scaling also helps the optimiser: on this data the scaled
  model converges in 40 iterations, while the same model on unscaled data
  stops at the default 100 iterations with a `ConvergenceWarning`.
- **What it learns:** the mean and standard deviation of each column. These
  are learned during `fit` **from the training set only** (tenure: mean
  32.49, std 24.57; MonthlyCharges: 64.93 / 30.14; TotalCharges:
  2,299.33 / 2,279.00) and are reused unchanged when transforming the test
  set or new customers.

**Why the categorical columns are not encoded inside the pipeline.** The
task template shows a `OneHotEncoder` in the pipeline. In this project the
categorical columns were already one-hot encoded in Task 4, using a fixed
list of allowed categories. That encoding learns nothing from the data, so
doing it before the split cannot leak information from the test set, and
it keeps the 41-column check from Task 6. The only step that does learn
from data (the scaler) is inside the pipeline.

**2. `classifier` — LogisticRegression**

- Multiplies each (scaled) feature by a learned weight (coefficient), adds
  them up with an intercept, and passes the total through the sigmoid
  function, which turns any number into a probability between 0 and 1.
- `predict()` returns 1 (churn) when that probability is 0.5 or higher;
  `predict_proba()` returns the probability itself.
- Settings: scikit-learn defaults (solver `lbfgs`, L2 regularisation with
  `C=1.0`, `class_weight=None`) except **`max_iter=1000`**. The weights are
  found step by step, and `max_iter` caps the number of steps. 1000 leaves a
  wide margin over the 40 steps needed. `train_model()` turns a
  `ConvergenceWarning` into an error, so an unfinished model can never be
  used silently.
- All one-hot columns are kept (none dropped). With L2 regularisation this
  is safe; the effect is visible in the coefficients (see below).

### What `fit` and `predict` actually do

| Call | Stage 1: preprocessor | Stage 2: classifier |
|------|-----------------------|---------------------|
| `model.fit(X_train, y_train)` | learns mean/std from **training** data, then scales it | learns the coefficients |
| `model.predict(X_test)` | **reuses** the training mean/std to scale the test data (does not re-learn) | applies the learned coefficients |

This is why the pipeline is used. With separate manual steps it is easy
to accidentally fit the scaler on the test set or on all data, which
would let test information leak into training. A test confirms the
scaler's mean equals the training set mean and differs from the full-data
mean. The whole process is deterministic, so retraining gives identical
coefficients.

### Coefficients

`output/model_report.txt` lists the coefficients. A positive coefficient
means higher predicted churn odds, and a negative one means lower. For
numeric features it is the change per 1 standard deviation; for 0/1
features it is the change from 0 to 1, with the other features held
equal. **These are associations the model uses to predict, not causes.**

| Towards churn | Coef. | Towards staying | Coef. |
|---------------|-------|-----------------|-------|
| InternetService_Fiber optic | +0.598 | tenure | −1.239 |
| Contract_Month-to-month | +0.541 | Contract_Two year | −0.806 |
| TotalCharges | +0.515 | InternetService_DSL | −0.686 |
| PaperlessBilling | +0.370 | MonthlyCharges | −0.585 |
| PaymentMethod_Electronic check | +0.171 | *(each "No internet service" column)* | −0.336 |

How to read these, and why some look surprising:

- **The main signals match Task 5:** short tenure, month-to-month
  contracts, fiber optic and electronic check are linked to churn; long
  tenure and two-year contracts to staying.
- **`MonthlyCharges` is negative here, although churners paid more on
  average in Task 5.** A coefficient describes the effect *with all other
  features held equal*. Monthly charges are largely explained by the
  services a customer has (correlation with fiber optic: 0.79), and those
  services already have their own coefficients. The same applies to
  `TotalCharges` vs. `tenure` (correlation 0.83; TotalCharges ≈ tenure ×
  MonthlyCharges). When features overlap this much, individual coefficients
  can change sign and should not be read on their own.
- **Identical coefficients (−0.336) for the six "No internet service"
  columns:** these six columns are exactly the same as `InternetService_No`
  (verified on the data). The regularisation spreads one shared effect
  evenly across identical columns, so no single column should be
  interpreted on its own.

Model quality (accuracy, precision, recall, F1, confusion matrix) is
measured on the test set in Task 9.

## Task 9 — Model evaluation

`evaluate()` in [src/evaluation.py](src/evaluation.py) compares the model's
predictions with the real outcome for the **1,409 test-set customers**,
which the model never saw during training. `python main.py` prints the
results and saves them to `output/evaluation_report.txt`. A customer is
predicted to churn when the model's churn probability is 0.5 or higher
(scikit-learn's default).

```
Metric       Value
-----------  -------
Accuracy      0.8062
Precision     0.6583
Recall        0.5615
F1 Score      0.6061
```

### Confusion matrix

Every test customer falls into exactly one of four boxes:

| | Predicted: stays | Predicted: churns |
|-|------------------|-------------------|
| **Actually stayed** (1,035) | **TN = 926** — correctly expected to stay | **FP = 109** — false alarm |
| **Actually churned** (374) | **FN = 164** — missed churner | **TP = 210** — churner caught |

- **TP (true positive):** churned, and the model said they would.
- **TN (true negative):** stayed, and the model said they would.
- **FP (false positive):** the model said they would churn, but they stayed.
- **FN (false negative):** the model said they would stay, but they churned.

"Positive" means "predicted to churn", the event we are looking for. It
does not mean "good".

### What the metrics mean

- **Accuracy (80.6%)** — the share of all test customers the model
  classified correctly: (TP + TN) ÷ all = (210 + 926) ÷ 1,409. On its own
  this number flatters the model. Because most customers stay, simply
  predicting "everyone stays" would already be right **73.5%** of the time.
  The model is about 7 percentage points better than doing nothing.
- **Precision (65.8%)** — when the model flags a customer as a churner, how
  often it is right: TP ÷ (TP + FP) = 210 ÷ 319. About two out of three
  flagged customers really left. The rest are false alarms.
- **Recall (56.1%)** — of the customers who really left, how many the model
  caught: TP ÷ (TP + FN) = 210 ÷ 374. **The model finds just over half of
  the churners and misses the rest.**
- **F1 score (0.606)** — one number that combines precision and recall
  (their harmonic mean). It is only high when both are high, which makes it
  more useful than accuracy when the class we care about (churners) is the
  minority.

### What a false negative means for the business

A **false negative** is a customer who **actually churned but was predicted
as not churning**. In this test set that is **164 customers**, 44% of all
churners.

In practice, the model's output would be used to decide which customers
receive retention efforts, such as a call, a discount or a contract offer.
A false negative never appears on that list. The company does nothing, the
customer leaves, and their future revenue is lost without any attempt to
keep them.

A **false positive** (109 customers) is the opposite mistake. A customer
who would have stayed anyway receives a retention offer. That costs the
price of the offer, but the customer is not lost.

For churn, a missed churner usually costs more than an unnecessary offer.
So **recall is the metric to watch most closely**, and 56% recall is the
model's main weakness at the moment.

### Precision–recall trade-off (for later)

The 0.5 threshold is a default, not a business decision. A lower threshold
flags more customers, catching more churners but creating more false
alarms. Illustration on the same test set:

| Threshold | Flagged | TP | FP | FN | Precision | Recall |
|-----------|---------|----|----|----|-----------|--------|
| 0.3 | 543 | 282 | 261 | 92 | 0.519 | 0.754 |
| 0.4 | 440 | 250 | 190 | 124 | 0.568 | 0.668 |
| **0.5 (used)** | 319 | 210 | 109 | 164 | 0.658 | 0.561 |

This table only illustrates the trade-off. The model and its reported
metrics still use 0.5. Choosing a different threshold should depend on the
real cost of a lost customer versus a retention offer. It should also be
chosen on the training data (e.g. with cross-validation), not tuned on this
test set, otherwise the test results would no longer be an honest estimate.

## Task 10 — predictions.csv

`build_predictions()` in [src/evaluation.py](src/evaluation.py) writes
`output/predictions.csv`, with one row for each of the **1,409 test-set
customers**.

| Column | Meaning |
|--------|---------|
| `customerID` | Original customer identifier |
| `actual_churn` | What really happened: 1 = churned, 0 = stayed |
| `predicted_churn` | Model's prediction: 1 = churn, 0 = stay (probability ≥ 0.5) |
| `churn_probability` | Model's estimated churn probability (0–1), usable as a risk score |
| `outcome` | `TP`, `TN`, `FP` or `FN`, comparing the actual and predicted values (see Task 9) |

Example rows:

```
customerID,actual_churn,predicted_churn,churn_probability,outcome
4376-KFVRS,0,0,0.04541712331769346,TN
2754-SDJRD,0,1,0.6838138537270784,FP
0365-GXEZS,0,0,0.4016993034176392,TN
```

Decisions:

- **Only test-set customers.** These are the only customers whose
  predictions are an honest test. The model never saw them during
  training, and their real outcome is known, so `actual_churn` can be
  compared with the prediction. Training-set customers are excluded
  because the model has already seen their answers.
- **Test-set order is kept** (the order produced by the Task 7 split, which
  is reproducible with `random_state=42`). Customer IDs were split together
  with X and y, so each row's ID, actual value and prediction belong to the
  same customer. A test checks this alignment.
- **`outcome` column added.** It is not required, but it makes the file
  easy to use in Excel. Filtering on `FN` lists the 164 churners the model
  missed, and filtering on `FP` lists the false alarms. The counts match the
  confusion matrix above (TP 210, TN 926, FP 109, FN 164).
- **Probabilities are not rounded.** Rounding could show 0.5000 for a
  customer whose real probability is just below 0.5 and who is therefore
  predicted 0, which would look inconsistent.
- To rank customers by risk, sort the file by `churn_probability`
  (highest first) in Excel or pandas.
- `main.py` refuses to run if `--output` points at the raw input CSV.