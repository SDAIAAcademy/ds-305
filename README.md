<!-- academy:title -->
# Uncertainty, Interpretability and Model Validation

عدم اليقين وقابلية التفسير والتحقق من صحة النماذج

Data Science Track · SDAIA Academy · DS-305 · Level 300

Produce the evidence an independent validator needs: calibrated probabilities, intervals with guarantees, honest explanations and a fit-for-use decision

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
| Capstone | Capstone: an independent validation |  | [`slides/capstone.html`](slides/capstone.html) |
<!-- /academy:units -->

## Case study

**REPLACE: the name of the fictional organisation every lab works on.** REPLACE: two sentences on the case study and its synthetic data

<!-- academy:labs -->
## Labs

Every lab is a notebook in `labs/`. Open it in Google Colab from the course page, or run it locally with `uv`. Exercises raise `NotImplementedError` until they are done; the reference versions are in `labs/solutions/`.

| | Session | Lab | File |
|---|---|---|---|
| 1.1 | Day 1 · Session 1 | Six claims, three verdicts | [`D1_1_mark_six_model.ipynb`](labs/D1_1_mark_six_model.ipynb) |
| 1.2 | Day 1 · Session 2 | Scores and a reliability diagram | [`D1_2_compute_brier_score.ipynb`](labs/D1_2_compute_brier_score.ipynb) |
| 1.3 | Day 1 · Session 3 | Platt and isotonic on held-out rows | [`D1_3_recalibrate_two_methods.ipynb`](labs/D1_3_recalibrate_two_methods.ipynb) |
| 1.4 | Day 1 · Session 4 | Calibration by region and age band | [`D1_4_break_calibration_down.ipynb`](labs/D1_4_break_calibration_down.ipynb) |
| 2.1 | Day 2 · Session 1 | Loss intervals with 90% coverage | [`D2_1_wrap_loss_regression.ipynb`](labs/D2_1_wrap_loss_regression.ipynb) |
| 2.2 | Day 2 · Session 2 | Sets for three risk grades | [`D2_2_build_prediction_sets.ipynb`](labs/D2_2_build_prediction_sets.ipynb) |
| 2.3 | Day 2 · Session 3 | Coverage by group, then by group | [`D2_3_measure_subgroup_coverage.ipynb`](labs/D2_3_measure_subgroup_coverage.ipynb) |
| 2.4 | Day 2 · Session 4 | An income shift, measured | [`D2_4_simulate_income_shift.ipynb`](labs/D2_4_simulate_income_shift.ipynb) |
| 3.1 | Day 3 · Session 1 | Importance and its spread | [`D3_1_compute_permutation_importance.ipynb`](labs/D3_1_compute_permutation_importance.ipynb) |
| 3.2 | Day 3 · Session 2 | Partial dependence against local effects | [`D3_2_show_where_partial.ipynb`](labs/D3_2_show_where_partial.ipynb) |
| 3.3 | Day 3 · Session 3 | Ten applicants, five refits | [`D3_3_explain_ten_applicants.ipynb`](labs/D3_3_explain_ten_applicants.ipynb) |
| 3.4 | Day 3 · Session 4 | A proxy takes the credit | [`D3_4_show_how_correlated.ipynb`](labs/D3_4_show_how_correlated.ipynb) |
| 4.1 | Day 4 · Session 1 | Gaps by group, tagged | [`D4_1_tabulate_calibration_coverage.ipynb`](labs/D4_1_tabulate_calibration_coverage.ipynb) |
| 4.2 | Day 4 · Session 2 | The cause of the decline | [`D4_2_find_seeded_cause.ipynb`](labs/D4_2_find_seeded_cause.ipynb) |
| 4.3 | Day 4 · Session 3 | The package against a checklist | [`D4_3_score_model_package.ipynb`](labs/D4_3_score_model_package.ipynb) |
| 4.4 | Day 4 · Session 4 | The decision and the capstone choice | [`D4_4_draft_decision_conditions.ipynb`](labs/D4_4_draft_decision_conditions.ipynb) |
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

### National references

- [AI Ethics Principles (SDAIA)](https://sdaia.gov.sa/en/SDAIA/about/Documents/ai-principles.pdf)
- [AI Bias Reference Guide, first edition (SDAIA, May 2026)](https://sdaia.gov.sa/en/MediaCenter/KnowledgeCenter/ResearchLibrary/EN-AI-BIAS.pdf)
- [National AI Risk Management Framework, English executive summary (SDAIA, April 2026)](https://sdaia.gov.sa/en/MediaCenter/KnowledgeCenter/ResearchLibrary/ExecutiveSummaryEn.pdf)
- [Personal Data Protection Law (SDAIA)](https://sdaia.gov.sa/en/SDAIA/about/Documents/PersonalDataProtectionLaw.pdf)
- [Implementing Regulation of the Personal Data Protection Law (SDAIA)](https://sdaia.gov.sa/en/SDAIA/about/Documents/ImplementingRegulationPersonalDataProtectionLaw.pdf)

### Validation and reporting

- [Board of Governors of the Federal Reserve System, SR 11-7: Guidance on Model Risk Management (2011)](https://www.federalreserve.gov/boarddocs/srletters/2011/sr1107.htm)
- [Kapoor and others, REFORMS: Reporting Standards for Machine Learning Based Science](https://arxiv.org/abs/2308.07832)
- [REFORMS checklist and guidelines](https://reforms.cs.princeton.edu/)
- [Mitchell and others, Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993)

### Calibration

- [Gneiting and Raftery, Strictly Proper Scoring Rules, Prediction, and Estimation (JASA, 2007)](https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf)
- [Niculescu-Mizil and Caruana, Predicting Good Probabilities with Supervised Learning (ICML, 2005)](https://www.cs.cornell.edu/~alexn/papers/calibration.icml05.crc.rev3.pdf)
- [scikit-learn user guide: probability calibration](https://scikit-learn.org/stable/modules/calibration.html)

### Conformal prediction

- [Angelopoulos and Bates, A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification](https://arxiv.org/abs/2107.07511)
- [Romano, Sesia and Candes, Classification with Valid and Adaptive Coverage](https://arxiv.org/abs/2006.02544)
- [Tibshirani, Barber, Candes and Ramdas, Conformal Prediction Under Covariate Shift](https://arxiv.org/abs/1904.06019)
- [MAPIE documentation](https://mapie.readthedocs.io/en/stable/)

### Explanations

- [Apley and Zhu, Visualizing the Effects of Predictor Variables in Black Box Supervised Learning Models](https://arxiv.org/abs/1612.08468)
- [Hooker, Mentch and Zhou, Unrestricted Permutation forces Extrapolation](https://arxiv.org/abs/1905.03151)
- [Lundberg and Lee, A Unified Approach to Interpreting Model Predictions](https://arxiv.org/abs/1705.07874)
- [scikit-learn user guide: permutation feature importance](https://scikit-learn.org/stable/modules/permutation_importance.html)
- [scikit-learn user guide: partial dependence and individual conditional expectation](https://scikit-learn.org/stable/modules/partial_dependence.html)
- [Molnar, Interpretable Machine Learning](https://christophm.github.io/interpretable-ml-book/)

### Tools

- [LightGBM documentation](https://lightgbm.readthedocs.io/en/stable/)
- [SHAP documentation](https://shap.readthedocs.io/en/latest/)
- [Fairlearn](https://fairlearn.org/)
- [DALEX](https://dalex.drwhy.ai/)
<!-- /academy:sources -->

### National references

- SDAIA AI Ethics Principles: REPLACE: the address of the official text
- SDAIA AI Bias Reference Guide: REPLACE: the address of the official text
- National AI Risk Management Framework: REPLACE: the address of the official text
- Personal Data Protection Law and its implementing regulations: REPLACE: the address of the official text
