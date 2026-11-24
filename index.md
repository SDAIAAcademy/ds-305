---
course: "Uncertainty, Interpretability and Model Validation"
code: "DS-305"
version: "2027.2.0"
language: "en"
view: "trainee"
source: "https://sdaiaacademy.github.io/ds-305/index.html"
---

# Uncertainty, Interpretability and Model Validation

عدم اليقين وقابلية التفسير والتحقق من صحة النماذج

Data Science Track · 4 teaching days + capstone day · 26 hours · Level 300

Produce the evidence an independent validator needs: calibrated probabilities, intervals with guarantees, honest explanations and a fit-for-use decision

Participants produce the evidence an independent validator needs about a trained model: whether its probabilities can be trusted, how wide its errors are, what drives its predictions, where it stops working and whom it fails. For Mizan Community Lending, a fictional microfinance lender, they repair calibration, wrap models in conformal intervals and prediction sets, test coverage by subgroup and under shift, show where explanations mislead, tabulate subgroup error and write a validation report against a published checklist. Fairness mitigation and documentation are DS-309.

**Who it is for:** Specialists. Analysts and data scientists who have completed the level 200 set or can show equivalent work: an analysis or model they built, validated and can explain.

**Format:** Intensive: 4 teaching days of four sessions and a capstone day, with guided pre-work and a tracked follow-up task

## The case study

**Mizan Community Lending.** Mizan Community Lending is a fictional microfinance lender whose credit analytics team hands its model package Amana (a default model, a three-grade model and a loss model) to the model risk team for independent validation. Participants are that team: on 50,000 synthetic applications, a held-out set and a later batch from 2025, they test the package's claims, repair calibration, wrap the models in conformal intervals and prediction sets, test what the explanations say and how stable they are, find the seeded cause of a decline, and decide whether Amana is fit for use and under which conditions. All data is synthetic and generated with a fixed seed.

## The course

### Day 1: Can the probabilities be trusted

Slides: https://sdaiaacademy.github.io/ds-305/slides/day1.md

- **Session 1: What a validator needs**
  - Name the three kinds of evidence an independent validation must produce
  - Separate a claim the data can test from one it cannot
  - Recompute a headline metric on rows the developers never saw
  - [Lab 1.1: Six claims, three verdicts (25 min)](https://sdaiaacademy.github.io/ds-305/labs/D1_1_mark_six_model.ipynb)
- **Session 2: Proper scores and reliability**
  - Compute the Brier score and log loss by hand and against a constant forecast
  - Draw a reliability diagram from equal-count bins
  - Explain why a model can rank well and still be miscalibrated
  - [Lab 1.2: Scores and a reliability diagram (25 min)](https://sdaiaacademy.github.io/ds-305/labs/D1_2_compute_brier_score.ipynb)
- **Session 3: Recalibration**
  - Fit Platt scaling and isotonic regression on a calibration split
  - Score both on separate held-out rows, never on the rows that fitted them
  - Choose a method and state what it cannot repair
  - [Lab 1.3: Platt and isotonic on held-out rows (25 min)](https://sdaiaacademy.github.io/ds-305/labs/D1_3_recalibrate_two_methods.ipynb)
- **Session 4: Calibration by subgroup**
  - Break calibration down by a grouping column with an interval for each group
  - Tell a real gap from noise in a small group
  - Name the worst group and the evidence that it is not chance
  - [Lab 1.4: Calibration by region and age band (25 min)](https://sdaiaacademy.github.io/ds-305/labs/D1_4_break_calibration_down.ipynb)

### Day 2: Intervals with guarantees

Slides: https://sdaiaacademy.github.io/ds-305/slides/day2.md

- **Session 1: Conformal intervals**
  - Compute the split conformal threshold with its finite-sample correction
  - Wrap a fitted regression in intervals and check coverage on new rows
  - State the one assumption the guarantee rests on
  - [Lab 2.1: Loss intervals with 90% coverage (25 min)](https://sdaiaacademy.github.io/ds-305/labs/D2_1_wrap_loss_regression.ipynb)
- **Session 2: Prediction sets**
  - Build LAC prediction sets from calibrated scores
  - Build adaptive (APS) sets and compare their sizes
  - Read a set of size one, two or three as a decision
  - [Lab 2.2: Sets for three risk grades (25 min)](https://sdaiaacademy.github.io/ds-305/labs/D2_2_build_prediction_sets.ipynb)
- **Session 3: Coverage by group**
  - Measure coverage and width by subgroup
  - Explain why marginal coverage says nothing about a group
  - Apply group-conditional (Mondrian) conformal and read its cost in width
  - [Lab 2.3: Coverage by group, then per group (25 min)](https://sdaiaacademy.github.io/ds-305/labs/D2_3_measure_subgroup_coverage.ipynb)
- **Session 4: Coverage under shift**
  - Simulate a covariate shift by reweighting a held-out sample
  - Measure the coverage lost for intervals and for sets
  - Explain why the guarantee needs exchangeability and what restores it
  - [Lab 2.4: An income shift, measured (25 min)](https://sdaiaacademy.github.io/ds-305/labs/D2_4_simulate_income_shift.ipynb)

### Day 3: What drives the predictions

Slides: https://sdaiaacademy.github.io/ds-305/slides/day3.md

- **Session 1: Permutation importance**
  - Compute permutation importance on held-out rows with repeats
  - Report its spread, not a single number
  - Permute correlated features together and explain the difference
  - [Lab 3.1: Importance and its spread (25 min)](https://sdaiaacademy.github.io/ds-305/labs/D3_1_compute_permutation_importance.ipynb)
- **Session 2: Effect plots**
  - Compute a partial dependence curve and individual curves
  - Compute accumulated local effects for the same feature
  - Show where the two disagree and which one to report
  - [Lab 3.2: Partial dependence against local effects (25 min)](https://sdaiaacademy.github.io/ds-305/labs/D3_2_show_where_partial.ipynb)
- **Session 3: Shapley values**
  - Compute exact tree Shapley values and check that they add up
  - Read the top reasons for one applicant
  - Measure whether the reasons survive five refits
  - [Lab 3.3: Ten applicants, five refits (25 min)](https://sdaiaacademy.github.io/ds-305/labs/D3_3_explain_ten_applicants.ipynb)
- **Session 4: When explanations mislead**
  - Show how a correlated proxy moves attributions
  - Tell a proxy from a cause with the data you have
  - Write a caution that a validator can attach to the explanations
  - [Lab 3.4: A proxy takes the credit (25 min)](https://sdaiaacademy.github.io/ds-305/labs/D3_4_show_how_correlated.ipynb)

### Day 4: Validate and decide

Slides: https://sdaiaacademy.github.io/ds-305/slides/day4.md

- **Session 1: Subgroup validation**
  - Tabulate calibration, coverage and error by subgroup in one table
  - Separate a gap from noise with an interval
  - Tag each gap with a bias type from the SDAIA AI Bias Reference Guide
  - [Lab 4.1: Gaps by group, tagged (25 min)](https://sdaiaacademy.github.io/ds-305/labs/D4_1_tabulate_calibration_coverage.ipynb)
- **Session 2: Diagnosing degradation**
  - Find the slice and the month where performance fell
  - Check feature distributions, coverage and explanations in that slice
  - Replay the same applications through the training and serving paths to tell a broken path from a changed population
  - Write a memo that names one cause and one action
  - [Lab 4.2: The cause of the decline (25 min)](https://sdaiaacademy.github.io/ds-305/labs/D4_2_find_seeded_cause.ipynb)
- **Session 3: Checklists and model cards**
  - Score a package against a reporting checklist item by item
  - Distinguish missing evidence from evidence of a problem
  - List what the developers must supply before a decision
  - [Lab 4.3: The package against a checklist (25 min)](https://sdaiaacademy.github.io/ds-305/labs/D4_3_score_model_package.ipynb)
- **Session 4: The fit-for-use decision**
  - Rate findings by severity and link each to evidence
  - Draft a decision with conditions of use and human review
  - Choose the capstone package and plan its validation
  - [Lab 4.4: The decision and the capstone choice (25 min)](https://sdaiaacademy.github.io/ds-305/labs/D4_4_draft_decision_conditions.ipynb)

### Capstone: Capstone: an independent validation

Slides: https://sdaiaacademy.github.io/ds-305/slides/capstone.md

## Capstone

An independent validation of another participant's or a course model package, in a public repository whose README reports calibration, coverage, explanation stability, subgroup findings and a fit-for-use decision; eight-minute panel defence plus questions

**What you submit:** A link to a public GitHub repository holding the validation notebooks, the run logs, and a README that reports calibration, coverage, explanation stability, subgroup findings and the fit-for-use decision

**Rubric:**

- Calibration and coverage: 25%
- Explanations: 20%
- Subgroup and shift findings: 25%
- Decision: 20%
- Reproducibility: 10%

## Outcomes

- **LO1** Measure calibration with proper scores and reliability diagrams, recalibrate, and report the effect by subgroup, given a trained default classifier and a held-out set (Applied) Evidence: before-and-after diagrams, Brier scores and a subgroup table
- **LO2** Implement conformal intervals and prediction sets and test coverage overall, by subgroup and under simulated shift, given a loss-amount regression and a three-grade risk classifier (Applied) Evidence: a table of target against achieved coverage with widths and set sizes
- **LO3** Apply permutation, accumulated-local-effect and Shapley-value explanations and test their stability across refits, given a boosted model with two correlated income features (Applied) Evidence: an explanation report with a stability panel and one documented misleading result
- **LO4** Diagnose why a model degrades on a later batch from slice metrics, coverage loss and changed explanations, given a shifted batch with a seeded cause (Performance) Evidence: a memo naming the cause and an action
- **LO5** Audit a model package for conceptual soundness, outcome performance and monitoring readiness and decide whether it is fit for use, given another team's model, documentation and data, with a reporting checklist and the national bias guide (Performance) Evidence: a validation report with rated findings, subgroup gaps and conditions of use, defended before a panel

## Tools

- **Model training and calibration:** scikit-learn, LightGBM, statsmodels
- **Conformal prediction:** MAPIE, course-provided split conformal functions
- **Model explanation:** SHAP, scikit-learn inspection tools, DALEX
- **Subgroup metrics:** Fairlearn, course-provided subgroup audit functions
- **Reporting checklist and documentation:** REFORMS checklist, model cards

## Prerequisites

- DS-204 Machine Learning for Tabular Data: Baselines and Validation
- DS-301 Regression and Generalised Linear Modelling
- Trains and cross-validates a gradient-boosted classifier

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

---

SDAIA Academy · Data Science Track · Uncertainty, Interpretability and Model Validation

Course version 2027.2.0 · engine 1.0.0 · built 2026-11-24 · commit 0baeea3
