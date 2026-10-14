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
python main.py                # validate -> clean -> explore -> (train/evaluate/predict: coming next)
python -m pytest tests        # run the unit tests
```

| Option         | Default                               | What it is                          |
|----------------|---------------------------------------|-------------------------------------|
| `--input`      | `data/raw/Telco-Customer-Churn.csv`   | Raw data (read only, never changed) |
| `--report`     | `output/validation_report.txt`        | Data validation report              |
| `--processed`  | `data/processed/telco_churn_clean.csv`| Cleaned and encoded data            |
| `--eda-report` | `output/eda_report.txt`               | Exploratory analysis report         |
| `--chart`      | `output/churn_by_contract.png`        | Churn rate by contract type chart   |
| `--output`     | `output/predictions.csv`              | Predictions (not produced yet)      |

## Project structure

```
main.py                 Command-line entry point
src/validation.py       Task 3: checks the raw data and writes a plain-language report
src/cleaning.py         Task 4: converts the raw data into a numeric feature matrix
src/eda.py              Task 5: churn rates by customer group, report and chart
tests/                  Unit tests for validation, cleaning and analysis
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