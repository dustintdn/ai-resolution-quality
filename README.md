# AI-Assisted Resolution Quality

Causal analysis of the impact of AI assistance on customer-support resolution quality. Uses propensity-score matching (PSM) and difference-in-differences (DiD) to estimate treatment effects on resolution time, satisfaction, and escalation rate.

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
data/                          raw & synthetic datasets
notebooks/
  00-data-exploration.ipynb    EDA and distributions
  10-causal-psm.ipynb          propensity-score matching pipeline
  11-causal-did.ipynb          difference-in-differences analysis
src/
  data/generate.py             synthetic conversation generator
  analysis/
    psm.py                     PSM matching utilities
    did.py                     DiD estimators
    balance.py                 SMD / balance table
  sensitivity/robustness.py    placebo tests, E-value
  models/propensity.py         logistic & gradient-boosted PS models
  dashboard/app.py             Streamlit interactive dashboard
  utils.py                     shared helpers
tests/                         pytest unit tests
.github/workflows/ci.yml       CI (lint + test on Python 3.10 & 3.11)
```
