<!-- academy:title -->
# Uncertainty, Interpretability and Model Validation

عدم اليقين وقابلية التفسير والتحقق من صحة النماذج

Data Science Track · SDAIA Academy · DS-305 · Level 300

REPLACE: one line under the title: what participants can do after the course

Participants produce the evidence an independent validator needs about a trained model: whether its probabilities can be trusted, how wide its errors are, what drives its predictions, where it stops working and whom it fails. For Mizan Community Lending, a fictional microfinance lender, they repair calibration, wrap models in conformal intervals and prediction sets, test coverage by subgroup and under shift, show where explanations mislead, tabulate subgroup error and write a validation report against a published checklist. Fairness mitigation and documentation are DS-309.

**Format:** Intensive: 4 teaching days of four sessions and a capstone day, with guided pre-work and a tracked follow-up task

**Course page:** https://sdaiaacademy.github.io/ds-305/
<!-- /academy:title -->

## Outcomes

| | Outcomes | Level | Evidence |
|---|---|---|---|
| LO1 | Measure calibration with proper scores and reliability diagrams, recalibrate, and report the effect by subgroup, given a trained default classifier and a held-out set | Applied | before-and-after diagrams, Brier scores and a subgroup table |
| LO2 | Implement conformal intervals and prediction sets and test coverage overall, by subgroup and under simulated shift, given a loss-amount regression and a three-grade risk classifier | Applied | a table of target against achieved coverage with widths and set sizes |
| LO3 | Apply permutation, accumulated-local-effect and Shapley-value explanations and test their stability across refits, given a boosted model with two correlated income features | Applied | an explanation report with a stability panel and one documented misleading result |
| LO4 | Diagnose why a model degrades on a later batch from slice metrics, coverage loss and changed explanations, given a shifted batch with a seeded cause | Performance | a memo naming the cause and an action |
| LO5 | Audit a model package for conceptual soundness, outcome performance and monitoring readiness and decide whether it is fit for use, given another team's model, documentation and data, with a reporting checklist and the national bias guide | Performance | a validation report with rated findings, subgroup gaps and conditions of use, defended before a panel |

<!-- academy:units -->
## Units

| Unit | Title | Sessions | Slides |
|---|---|---|---|
| Day 1 | Can the probabilities be trusted | What a validator needs; Proper scores and reliability; Recalibration; Calibration by subgroup | [`slides/day1.html`](slides/day1.html) |
| Day 2 | Intervals with guarantees | Conformal intervals; Prediction sets; Coverage by group; Coverage under shift | [`slides/day2.html`](slides/day2.html) |
| Day 3 | What drives the predictions | Permutation importance; Effect plots; Shapley values; When explanations mislead | [`slides/day3.html`](slides/day3.html) |
| Day 4 | Validate and decide | Subgroup validation; Diagnosing degradation; Checklists and model cards; The fit-for-use decision | [`slides/day4.html`](slides/day4.html) |
| Capstone | Capstone project presentations |  | [`slides/capstone.html`](slides/capstone.html) |
<!-- /academy:units -->

## Case study

**REPLACE: the name of the fictional organisation every lab works on.** REPLACE: two sentences on the case study and its synthetic data

<!-- academy:labs -->
## Labs

Every lab is a notebook in `labs/`. Open it in Google Colab from the course page, or run it locally with `uv`. Exercises raise `NotImplementedError` until they are done; the reference versions are in `labs/solutions/`.

| | Session | Lab | File |
|---|---|---|---|
| 1.1 | Day 1 · Session 1 | Mark six model-card claims supported, unsupported or untestable | [`D1_1_mark_six_model.ipynb`](labs/D1_1_mark_six_model.ipynb) |
| 1.2 | Day 1 · Session 2 | Compute Brier score and log loss and draw a reliability diagram | [`D1_2_compute_brier_score.ipynb`](labs/D1_2_compute_brier_score.ipynb) |
| 1.3 | Day 1 · Session 3 | Recalibrate with two methods on a held-out set | [`D1_3_recalibrate_two_methods.ipynb`](labs/D1_3_recalibrate_two_methods.ipynb) |
| 1.4 | Day 1 · Session 4 | Break calibration down by region and age band and name the worst group | [`D1_4_break_calibration_down.ipynb`](labs/D1_4_break_calibration_down.ipynb) |
| 2.1 | Day 2 · Session 1 | Wrap the loss regression in split conformal intervals and check 90% coverage | [`D2_1_wrap_loss_regression.ipynb`](labs/D2_1_wrap_loss_regression.ipynb) |
| 2.2 | Day 2 · Session 2 | Build prediction sets for three risk grades and measure set sizes | [`D2_2_build_prediction_sets.ipynb`](labs/D2_2_build_prediction_sets.ipynb) |
| 2.3 | Day 2 · Session 3 | Measure subgroup coverage and apply a group-conditional variant | [`D2_3_measure_subgroup_coverage.ipynb`](labs/D2_3_measure_subgroup_coverage.ipynb) |
| 2.4 | Day 2 · Session 4 | Simulate an income shift and measure the coverage lost | [`D2_4_simulate_income_shift.ipynb`](labs/D2_4_simulate_income_shift.ipynb) |
| 3.1 | Day 3 · Session 1 | Compute permutation importance and its spread over repeats | [`D3_1_compute_permutation_importance.ipynb`](labs/D3_1_compute_permutation_importance.ipynb) |
| 3.2 | Day 3 · Session 2 | Show where partial dependence and accumulated local effects disagree for income | [`D3_2_show_where_partial.ipynb`](labs/D3_2_show_where_partial.ipynb) |
| 3.3 | Day 3 · Session 3 | Explain ten applicants and check stability over five refits | [`D3_3_explain_ten_applicants.ipynb`](labs/D3_3_explain_ten_applicants.ipynb) |
| 3.4 | Day 3 · Session 4 | Show how a correlated proxy shifts attributions and write a caution | [`D3_4_show_how_correlated.ipynb`](labs/D3_4_show_how_correlated.ipynb) |
| 4.1 | Day 4 · Session 1 | Tabulate calibration, coverage and error by subgroup and tag each gap with a bias type from the national guide | [`D4_1_tabulate_calibration_coverage.ipynb`](labs/D4_1_tabulate_calibration_coverage.ipynb) |
| 4.2 | Day 4 · Session 2 | Find the seeded cause of decline in a later batch | [`D4_2_find_seeded_cause.ipynb`](labs/D4_2_find_seeded_cause.ipynb) |
| 4.3 | Day 4 · Session 3 | Score a model package against a checklist and list missing evidence | [`D4_3_score_model_package.ipynb`](labs/D4_3_score_model_package.ipynb) |
| 4.4 | Day 4 · Session 4 | Draft a decision with conditions of use and choose the capstone package | [`D4_4_draft_decision_conditions.ipynb`](labs/D4_4_draft_decision_conditions.ipynb) |
<!-- /academy:labs -->

## Capstone

An independent validation of another participant's or a course model package, in a public repository whose README reports calibration, coverage, explanation stability, subgroup findings and a fit-for-use decision; eight-minute panel defence plus questions

**Submission:** REPLACE: what participants submit and how: a public repository link, or a shared folder for a course without code

## Prerequisites

- DS-204 Machine Learning for Tabular Data: Baselines and Validation
- DS-301 Regression and Generalised Linear Modelling
- Trains and cross-validates a gradient-boosted classifier

<!-- academy:tools -->
## Tools

Each line is a capability; the examples are interchangeable.

- **Model training and calibration:** scikit-learn, LightGBM, statsmodels
- **Conformal prediction:** MAPIE, course-provided split conformal functions
- **Model explanation:** SHAP, scikit-learn inspection tools, DALEX
- **Subgroup metrics:** Fairlearn, course-provided subgroup audit functions
- **Reporting checklist and documentation:** REFORMS checklist, model cards
<!-- /academy:tools -->

<!-- academy:sources -->
## Sources

- REPLACE: every source the course cites, with its address, in course.json under "sources"; this list is written from there
<!-- /academy:sources -->

### National references

- SDAIA AI Ethics Principles: REPLACE: the address of the official text
- SDAIA AI Bias Reference Guide: REPLACE: the address of the official text
- National AI Risk Management Framework: REPLACE: the address of the official text
- Personal Data Protection Law and its implementing regulations: REPLACE: the address of the official text
