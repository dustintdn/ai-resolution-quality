# AI-assisted Resolution Quality — Repo Scaffold

This repo contains starter code to simulate and analyze the causal impact of AI assistance on support resolution quality.

Quickstart

1. Create & activate venv:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Generate synthetic data:

```bash
python -m src.data.generate
```

3. Run dashboard:

```bash
streamlit run src/dashboard/app.py
```

Structure

- `src/data/generate.py`: synthetic data generator
- `src/analysis`: PSM and DiD utilities
- `src/sensitivity`: placebo & E-value helpers
- `src/dashboard`: Streamlit app
# ai-resolution-quality

A data science project for exploring AI resolution quality.

## Setup

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## Structure

```
notebooks/   - Jupyter notebooks for exploration
src/         - Reusable Python modules
```
