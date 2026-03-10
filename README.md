# AI-Assisted Resolution Quality

Causal analysis of the impact of AI assistance on customer-support resolution quality. Uses propensity-score matching (PSM) and difference-in-differences (DiD) to estimate treatment effects on resolution time, satisfaction score, and escalation rate.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Generate synthetic data, then explore:

```bash
python -m src.data.generate     # writes data/synthetic_conversations.csv
jupyter lab notebooks/          # open EDA and causal notebooks
streamlit run src/dashboard/app.py
```

Run tests:

```bash
pytest tests/ -v
```

## Experiment Design

| | |
|---|---|
| **Treatment** | AI-assisted response (`ai_assisted = 1`) |
| **Control** | Human-only response (`ai_assisted = 0`) |
| **Outcomes** | `resolution_time`, `satisfaction_score`, `escalated` |
| **Confounders** | `issue_severity`, `customer_tenure`, `time_of_day`, `agent_experience` |
| **Identification** | PSM for cross-sectional balance; DiD for pre/post rollout |
| **Robustness** | Covariate balance (SMD), placebo permutation test, E-value |

## Project Structure

```
ai-resolution-quality/
├── data/
│   └── synthetic_conversations.csv    pre-generated dataset (2 000 rows × 11 cols)
├── notebooks/
│   ├── 00-data-exploration.ipynb      EDA, outcome distributions, pre-matching Love plot
│   ├── 10-causal-psm.ipynb            PSM pipeline, ATT estimates, sensitivity analysis
│   └── 11-causal-did.ipynb            DiD analysis, parallel trends check, regression DiD
├── src/
│   ├── data/
│   │   └── generate.py                synthetic conversation generator
│   ├── analysis/
│   │   ├── psm.py                     propensity-score estimation & nearest-neighbor matching
│   │   ├── did.py                     simple and regression-based DiD estimators
│   │   └── balance.py                 SMD computation and balance table
│   ├── models/
│   │   └── propensity.py              logistic and gradient-boosted propensity score models
│   ├── sensitivity/
│   │   └── robustness.py              placebo permutation test, E-value
│   └── dashboard/
│       └── app.py                     Streamlit interactive dashboard
├── tests/
│   ├── test_generate.py               data generation unit tests
│   ├── test_balance.py                SMD and balance table tests
│   ├── test_psm.py                    propensity score and matching tests
│   ├── test_did.py                    DiD estimator tests
│   └── test_robustness.py             placebo test and E-value tests
├── .github/workflows/ci.yml           CI: lint (ruff) + pytest on Python 3.10 & 3.11
└── requirements.txt                   project dependencies
```

## Notebooks

| Notebook | Purpose |
|---|---|
| `00-data-exploration` | Load data, plot outcome distributions by treatment group, covariate correlations, pre-matching Love plot, weekly volume time series |
| `10-causal-psm` | Estimate propensity scores, common support check, 1:1 nearest-neighbor matching with caliper, pre/post-match balance, ATT estimates (t-tests), placebo test, E-value |
| `11-causal-did` | Parallel trends visual check, simple and regression DiD on all outcomes, coefficient plot, placebo period test |

## Source Modules

### `src/data/generate.py`
Generates a synthetic customer-support dataset with a confounded AI-assistance rollout.

- **Pre-period** (2023): all human-only
- **Post-period** (2024): `ai_rate` fraction AI-assisted (default 50%), with treatment probability increasing with `issue_severity`
- Known causal effects: AI reduces `resolution_time` by ~15 min, increases `satisfaction_score` by ~0.8, and halves escalation probability

### `src/analysis/psm.py`
- `estimate_propensity_score(df, treatment_col, covariates)` — logistic regression propensity scores
- `match_nearest_neighbor(df, ps, treatment_col, caliper)` — 1:1 greedy matching with caliper (default 0.05)

### `src/analysis/did.py`
- `difference_in_differences(df, time_col, period_cutoff, treatment_col, outcome_col)` — group-means DiD, returns estimate and cell means
- `regression_did(df, ...)` — OLS with `treatment × period` interaction, HC1 robust SEs, optional covariates

### `src/analysis/balance.py`
- `compute_smd(df, treatment_col, covariates)` — pooled-SD standardized mean differences
- `balance_table(df, treatment_col, covariates)` — per-covariate summary (mean_treated, mean_control, smd)

### `src/models/propensity.py`
- `logistic_propensity(df, treatment_col, covariates)` — logistic regression with StandardScaler
- `gradient_boosting_propensity(df, treatment_col, covariates, calibrate)` — gradient-boosted classifier with optional isotonic calibration

### `src/sensitivity/robustness.py`
- `placebo_test(df, treatment_col, outcome_col, n_runs, seed)` — permutation test; returns (observed_effect, null_distribution, p_value)
- `e_value(rr)` — minimum unmeasured confounding strength needed to explain away a risk ratio

### `src/dashboard/app.py`
Streamlit app: loads `synthetic_conversations.csv`, filters by issue severity, displays outcome summaries and an Altair density plot of resolution-time distributions by treatment group.

## Data Schema

| Column | Type | Description |
|---|---|---|
| `conversation_id` | int | Unique row identifier |
| `created_at` | datetime | Conversation date |
| `period` | 0/1 | Pre (2023) or post (2024) rollout |
| `ai_assisted` | 0/1 | Treatment indicator |
| `issue_severity` | 1–5 | Severity of the support issue |
| `customer_tenure` | 0–120 | Customer tenure in months |
| `time_of_day` | 0–23 | Hour of conversation |
| `agent_experience` | 1–10 | Agent experience in years |
| `resolution_time` | float | Minutes to resolve (clipped ≥ 5) |
| `satisfaction_score` | float | CSAT score 1–5 |
| `escalated` | 0/1 | Whether the conversation was escalated |
