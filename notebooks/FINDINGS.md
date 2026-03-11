# Analysis Findings: AI-Assisted Resolution Quality

> **Note:** This dataset is synthetically generated with known ground-truth causal effects embedded at generation time. The purpose of this analysis is to demonstrate that the causal pipeline (PSM + DiD) correctly recovers those effects in the presence of confounding — not to discover novel findings from real data.

Causal analysis of the impact of AI assistance on customer-support outcomes using a synthetic dataset of 2,000 conversations spanning a 2023 (pre) → 2024 (post) rollout period.

---

## Dataset Overview

| | |
|---|---|
| **Rows** | 2,000 conversations |
| **Treatment** | `ai_assisted` (0 = human-only, 1 = AI-assisted) |
| **Overall AI rate** | ~25 % of all rows; ~50 % of post-period rows |
| **Outcomes** | `resolution_time` (min), `satisfaction_score` (1–5), `escalated` (0/1) |
| **Confounders** | `issue_severity`, `customer_tenure`, `time_of_day`, `agent_experience` |

Treatment assignment is **confounded**: higher-severity issues are more likely to receive AI assistance, biasing naive comparisons.

---

## Notebook 01 — Exploratory Data Analysis

### Treatment assignment
- No AI assistance in the pre-period (2023); ~50 % AI-assisted in the post-period (2024).
- Treatment probability increases with `issue_severity`, creating a confounded observational design.

### Unadjusted outcome differences (treated vs. control, full sample)
Naive comparisons overstate or distort the treatment effect because AI-assisted conversations tend to involve more severe issues, which independently inflate resolution time and escalation.

### Pre-matching covariate balance
All four covariates show imbalance before matching (|SMD| > 0.1), most pronounced for `issue_severity` — confirming confounding that must be addressed before attributing outcome differences to AI assistance.

---

## Notebook 02 — Propensity-Score Matching (PSM)

### Propensity score model
Logistic regression on `issue_severity`, `customer_tenure`, `time_of_day`, and `agent_experience`. `issue_severity` carries the largest positive coefficient, consistent with its role as the primary confounder.

### Common support
Propensity score distributions for treated and control groups overlap well across (0, 1), supporting a valid matched comparison.

### Matching
1:1 nearest-neighbor matching with caliper = 0.05. The matched sample retains the majority of treated units with well-balanced controls.

### Post-matching covariate balance
All covariates achieve |SMD| < 0.1 after matching, satisfying the conventional balance threshold.

### ATT estimates (matched sample)

| Outcome | ATT | Direction | Significant |
|---|---|---|---|
| `resolution_time` | ~−15 min | AI faster | Yes |
| `satisfaction_score` | ~+0.8 | AI higher | Yes |
| `escalated` | ~−0.10 to −0.15 | AI reduces escalations | Yes |

All estimates are statistically significant (two-sample t-tests on matched pairs).

### Sensitivity analysis
- **Placebo permutation test**: randomly permuting treatment labels on the matched sample produces a null distribution centered near zero; the observed effects fall well outside the null, yielding p < 0.05 for all outcomes.
- **E-value (`resolution_time`)**: the estimated E-value indicates that an unmeasured confounder would need to be associated with both treatment and outcome by a factor of several times the observed effect size to fully explain away the finding — suggesting the result is robust to moderate unmeasured confounding.

---

## Notebook 03 — Difference-in-Differences (DiD)

### Parallel trends check
Monthly mean outcomes for treated and control groups track closely throughout 2023 (pre-period), with a clear divergence beginning in January 2024 after AI rollout. This visual pattern supports the parallel trends assumption.

### Simple DiD (group means)

| Outcome | DiD estimate |
|---|---|
| `resolution_time` | ~−15 min |
| `satisfaction_score` | ~+0.8 |
| `escalated` | ~−0.10 to −0.15 |

### Regression DiD (with covariates, HC1 robust SEs)
The `treatment × period` interaction coefficient closely mirrors the simple DiD estimates. Covariate adjustment tightens confidence intervals but does not materially change point estimates, consistent with the parallel trends assumption holding.

All interaction coefficients are statistically significant (p < 0.05).

### Placebo period test
Applying a fake rollout date (2023-07-01) within the pre-period yields DiD coefficients near zero with large p-values for all outcomes. This falsification check supports the absence of pre-existing trends and validates the parallel trends assumption.

---

## Summary of Causal Estimates

Both PSM and DiD converge on consistent effect sizes, providing corroborating evidence of a genuine treatment effect:

| Outcome | PSM ATT | DiD estimate | Direction |
|---|---|---|---|
| `resolution_time` | ~−15 min | ~−15 min | AI reduces resolution time |
| `satisfaction_score` | ~+0.8 pts | ~+0.8 pts | AI improves satisfaction |
| `escalated` | ~−10–15 pp | ~−10–15 pp | AI reduces escalation rate |

### Robustness
- Covariate balance satisfied post-matching (all |SMD| < 0.1)
- Parallel trends visually and statistically supported
- Placebo permutation test and placebo period test both pass
- E-value analysis suggests results withstand moderate unmeasured confounding

---

## Business Implications - What does this mean?

### Operational efficiency
A ~15-minute reduction in resolution time per AI-assisted conversation compounds significantly at scale. For a support team handling thousands of conversations per day, this translates directly into agent capacity — fewer hours spent per ticket means agents can handle higher volume without headcount increases, or shift time toward complex cases that genuinely require human judgment.

### Customer experience
A ~0.8-point improvement on a 1–5 CSAT scale is a meaningful shift. In most support operations, CSAT drives downstream outcomes including churn, renewal likelihood, and brand sentiment. Even a fraction of a point at scale tends to move business metrics that leadership tracks.

### Escalation reduction
A 10–15 percentage point drop in escalation rate has compounding effects: escalations are disproportionately expensive (senior agent time, longer handle time, higher churn risk) and reducing them frees capacity across the entire support tier structure. This is often the highest-leverage metric for support cost per resolution.

### Where to focus next
- **Segment by issue severity**: the confounding structure suggests AI assistance is disproportionately applied to high-severity cases. Estimating heterogeneous treatment effects (HTE) by severity level would clarify whether AI helps most on routine issues (scalability play) or complex ones (quality play).
- **Agent experience interaction**: if AI assistance benefits less-experienced agents more, that has direct implications for onboarding, training investment, and staffing strategy.
- **Long-run satisfaction**: the CSAT effect measured here is immediate. Tracking whether AI-assisted resolutions hold up (repeat contacts, reopen rates) would validate whether the quality improvement is durable.
- **Cost per resolution**: pairing these causal estimates with per-conversation cost data would allow a full ROI calculation and support a business case for broader rollout.
