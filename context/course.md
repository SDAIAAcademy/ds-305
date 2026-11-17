---
course: "Uncertainty, Interpretability and Model Validation"
code: "DS-305"
version: "2027.1.0"
scope: "course"
language: "en"
view: "trainee"
source: "https://sdaiaacademy.github.io/ds-305/"
---

# Uncertainty, Interpretability and Model Validation

Produce the evidence an independent validator needs: calibrated probabilities, intervals with guarantees, honest explanations and a fit-for-use decision

Participants produce the evidence an independent validator needs about a trained model: whether its probabilities can be trusted, how wide its errors are, what drives its predictions, where it stops working and whom it fails. For Mizan Community Lending, a fictional microfinance lender, they repair calibration, wrap models in conformal intervals and prediction sets, test coverage by subgroup and under shift, show where explanations mislead, tabulate subgroup error and write a validation report against a published checklist. Fairness mitigation and documentation are DS-309.

- **Audience:** Specialists. Analysts and data scientists who have completed the level 200 set or can show equivalent work: an analysis or model they built, validated and can explain.
- **Prerequisites:** DS-204 Machine Learning for Tabular Data: Baselines and Validation; DS-301 Regression and Generalised Linear Modelling; Trains and cross-validates a gradient-boosted classifier
- **Case study:** Mizan Community Lending: Mizan Community Lending is a fictional microfinance lender whose credit analytics team hands its model package Amana (a default model, a three-grade model and a loss model) to the model risk team for independent validation. Participants are that team: on 50,000 synthetic applications, a held-out set and a later batch from 2025, they test the package's claims, repair calibration, wrap the models in conformal intervals and prediction sets, test what the explanations say and how stable they are, find the seeded cause of a decline, and decide whether Amana is fit for use and under which conditions. All data is synthetic and generated with a fixed seed.

## Outcomes

- LO1 (Applied): Measure calibration with proper scores and reliability diagrams, recalibrate, and report the effect by subgroup, given a trained default classifier and a held-out set
- LO2 (Applied): Implement conformal intervals and prediction sets and test coverage overall, by subgroup and under simulated shift, given a loss-amount regression and a three-grade risk classifier
- LO3 (Applied): Apply permutation, accumulated-local-effect and Shapley-value explanations and test their stability across refits, given a boosted model with two correlated income features
- LO4 (Performance): Diagnose why a model degrades on a later batch from slice metrics, coverage loss and changed explanations, given a shifted batch with a seeded cause
- LO5 (Performance): Audit a model package for conceptual soundness, outcome performance and monitoring readiness and decide whether it is fit for use, given another team's model, documentation and data, with a reporting checklist and the national bias guide

## Day 1: Can the probabilities be trusted

Read the package as a validator, score its probabilities with proper scores, repair them and check the repair in every group

#### [day1#title] Can the probabilities be trusted

*SDAIA Academy · Uncertainty, Interpretability and Model Validation*

Day 1: Read the package as a validator, score its probabilities with proper scores, repair them and check the repair in every group

Data Science Track · 4 teaching days + capstone day · 4 sessions · 4 labs · DS-305

#### [day1#four-days-one-validation] Four days of evidence, then a decision you defend

*The course at a glance*

- **Day 1 · Probabilities.** Read the claims, score the probabilities, repair them, check every group.
- **Day 2 · Intervals.** Conformal intervals and prediction sets, by group and under shift.
- **Day 3 · Explanations.** Importance, effect plots, Shapley values, and where they mislead.
- **Day 4 · Decide.** Subgroup findings, a decline diagnosed, a checklist, a decision.

> **Day 5 · Capstone.** An independent validation of another package, in a public repository, defended before a panel for eight minutes plus questions.

#### [day1#evidence-not-opinion] A validator's job is evidence, and every number is rerun

*How this course works*

- **Rows nobody trained on.** Every claim is checked on rows the developers never saw: the calib and holdout splits.
- **Guarantees with assumptions.** Calibration, coverage and explanations each hold under conditions. The report names them.
- **A decision at the end.** Fit, fit with conditions, or not fit. Evidence without a decision is not a validation.

> **AI policy.** AI off for the entry and exit checks and for your first reading of a diagram or a coverage table. In labs an AI tutor may explain, not write code or findings. In the capstone anything goes, defended live.

#### [day1#mizan-and-amana] Mizan Community Lending hands you Amana to validate

*One case study all week*

A fictional microfinance lender. Its credit analytics team built **Amana** (أمانة); you are the independent model risk team.

- **Three models.** Probability of default, a grade (on time, late, default) and the loss in riyals if a loan defaults.
- **Your data.** 50,000 applications from 2023 and 2024: 30,000 the developers trained on, 10,000 calib and 10,000 holdout rows that are yours.
- **A later batch.** 10,000 applications from early 2025, with their outcomes. Something in it changed.

#### [day1#setup-and-reference-state] Open the notebooks in Colab, or run them locally with uv

*Lab environment*

1. **Colab:** open a notebook from the course page. Its first cell clones the repository and installs one package.
2. **Locally:** the commands on the right, then `00_setup_check.ipynb` once.
3. Every lab starts from a **supplied reference state**: the package, the data and the course functions in `mizan.py`. No lab needs your earlier run.

```bash
# file: terminal (local option)
git clone https://github.com/SDAIAAcademy/ds-305
cd ds-305/labs
uv sync
uv run jupyter lab
```

#### [day1#agenda] Can the probabilities be trusted

*Today*

- **Session 1 · What a validator needs**: Validation is evidence on conceptual soundness, outcomes and monitoring, and each claim in a model card is supported, unsupported or untestable
- **Session 2 · Proper scores and reliability**: A proper score rewards honest probabilities; the Brier score, log loss and a reliability diagram show what AUC hides
- **Session 3 · Recalibration**: Platt scaling and isotonic regression repair probabilities on rows the model never saw, and leave the ranking untouched
- **Session 4 · Calibration by subgroup**: A model calibrated on average can be miscalibrated for a group it never saw as a feature; small groups need intervals

### Session 1: What a validator needs

#### [day1#s1] What a validator needs

*Session 1*

- Name the three kinds of evidence an independent validation must produce
- Separate a claim the data can test from one it cannot
- Recompute a headline metric on rows the developers never saw

#### [day1#three-kinds-of-evidence] A validation answers three questions, not one

*What validation must show*

- **Conceptual soundness.** Is the design fit for the purpose: the target, the data, the features, the training choices?
- **Outcome analysis.** Does it perform on rows it never saw: discrimination, calibration, intervals, every group?
- **Monitoring.** Will anyone notice when it stops working: which metric, which threshold, who acts?

> **This week.** Day 1 and 2 are outcome analysis; Day 3 tests soundness of the explanations; Day 4 adds monitoring and the decision.

#### [day1#independent-means-other-rows] Independent means other people, other rows and other questions

*Who validates*

|  | The developers | The validator |
| --- | --- | --- |
| Question | Does my model work? | Where does it stop working, and for whom? |
| Rows | Their training and test split | Rows they never touched: calib and holdout |
| Output | A model and a card | Rated findings, conditions of use, a decision |
| At Mizan | Rasil's credit analytics team | Your model risk team, reporting to Dr. Sameera |

> **What goes wrong.** A validator who retrains the model has become a second developer. Repairs are recommended, and the developers own them.

#### [day1#claims-evidence-verdicts] Every claim gets one of three verdicts

*Reading a model card*

- **Supported.** The package's evidence, or your recomputation, backs the claim as written.
- **Unsupported.** The evidence is missing, or it contradicts the claim. Missing counts.
- **Untestable.** No data you have, or could have today, can test it as written. Rewrite it as a test.

> **The habit.** "Supported" needs evidence. Silence is not support.

#### [day1#amana-six-claims] Amana's card makes six claims

*The package you received*

1. It discriminates well: **AUC 0.864** on a held-out sample of 4,000 applications.
2. Its probabilities are well calibrated and can be read as default rates.
3. It uses no gender, age or location information.
4. Performance is consistent across all customer groups.
5. It will remain accurate for the next three years.
6. The reasons shown to officers are the causes of an applicant's risk.

> **Also in the package.** `metadata.json` (what they did) and `evidence.json` (what they measured). Read all three before computing anything.

#### [day1#headline-auc-rows] The claimed AUC came from rows the shipped model had seen

*One recomputation*

| Item | Value |
| --- | --- |
| Claimed in the card | 0.864 |
| On the training rows | 0.858 |
| On the holdout | 0.755 |

- The developers held out 4,000 rows, measured, then **retrained on all rows** and shipped that model.
- The shipped model has seen the 4,000 rows: its 0.864 is an in-sample number.
- On 10,000 rows it never saw: 0.755.

#### [day1#guarantees-assumptions] Every guarantee this week holds under a condition you must state

*Guarantees and assumptions*

| Evidence | It guarantees | If this holds |
| --- | --- | --- |
| A holdout score | Performance on similar new applicants | New applicants look like the holdout |
| Recalibration | Average calibration | The calib rows look like the applicants |
| Conformal intervals | 90% coverage on average | Rows are exchangeable |
| An explanation | What the model does | Not what causes default |

> **What goes wrong.** A report that quotes the guarantee and drops the condition. The condition is where the next failure comes from.

#### [day1#credit-data-personal] Validation data is applicants' personal data, even in a model risk team

*Responsible use*

- Income, loans and missed payments of named people fall under the **Personal Data Protection Law**.
- A validator needs outcomes and predictions, not names: ask for the minimum, and ask the data protection officer.
- Subgroup columns (age band, gender, region) are needed to audit the model and must not leak into it.

- **In this course.** Every applicant is synthetic. The capstone uses the course packages or a peer's synthetic package; no real borrower data enters a repository.

#### [day1#check-supported-claim] Which verdict does claim 4 get?

*Checkpoint*

**Question:** Amana's card says "performance is consistent across all customer groups". The evidence file has an overall AUC and no subgroup results. You have not yet computed anything by group. What is the verdict today?

- A. Supported, since no gap has been found
- B. Unsupported, since the package has no evidence for it
- C. Untestable, since groups were not measured
- D. No verdict until the subgroup tables of Lab 1.4 exist and are read

#### [day1#lab-1-1] Six claims, three verdicts

Lab 1.1 · 25 min

Read Amana's model card, metadata and evidence, recompute the claimed AUC on the validator's held-out rows, and mark each of six claims supported, unsupported or untestable with the evidence that decides it.

[D1_1_mark_six_model.ipynb](https://sdaiaacademy.github.io/ds-305/labs/D1_1_mark_six_model.ipynb)

- [ ] Open the package: card, metadata, evidence (5 min)
- [ ] Recompute the AUC on the holdout (6 min)
- [ ] Compare mean prediction with the default rate (4 min)
- [ ] Mark the six claims with their evidence (7 min)
- [ ] Rewrite one untestable claim as a test (3 min)
- [ ] ✱ Stretch: Find out on which rows the developers measured their AUC, and say why it is higher

### Session 2: Proper scores and reliability

#### [day1#s2] Proper scores and reliability

*Session 2*

- Compute the Brier score and log loss by hand and against a constant forecast
- Draw a reliability diagram from equal-count bins
- Explain why a model can rank well and still be miscalibrated

#### [day1#ranking-not-probability] A model can rank applicants well and still state the wrong risk

*Two different properties*

- **0.755** AUC on the holdout: who is riskier than whom
- **0.40** mean predicted default probability, against 0.10 observed

- **Discrimination**: do defaulters get higher scores? AUC answers it, and ignores the scale.
- **Calibration**: when the model says 20%, do 20% default? AUC is blind to it.
- Officers read the score as a probability, so both matter.

#### [day1#proper-scores] A proper score rewards the forecaster for saying what they believe

*Scoring a probability*

```python
# file: Lab 1.2 · step 1
def brier(y, p):
    return np.mean((p - y) ** 2)

def log_loss(y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -np.mean(y * np.log(p)
                    + (1 - y) * np.log(1 - p))
```

- Both are **proper**: the expected score is best when the forecast equals the true probability.
- Accuracy at 0.5 is not: 0.51 and 0.99 score the same.
- Log loss punishes a confident miss hard; Brier is bounded and easier to explain.

#### [day1#beaten-by-base-rate] As a probability, Amana loses to the base rate

*A forecast that knows nothing*

| Forecast on the holdout | Brier | Log loss | AUC |
| --- | --- | --- | --- |
| Amana, as shipped | 0.199 | 0.586 | 0.755 |
| Everyone gets the calib default rate | 0.090 | 0.324 | 0.500 |

> **What goes wrong.** Reporting AUC alone. The model that ranks better is the worse probability by both proper scores.

#### [day1#reliability-diagram-bins] A reliability diagram puts predicted against observed, bin by bin

*Seeing calibration*

| Bin (1,000 each) | Predicted | Observed |
| --- | --- | --- |
| Lowest | 0.076 | 0.020 |
| Fourth | 0.262 | 0.042 |
| Seventh | 0.484 | 0.108 |
| Highest | 0.854 | 0.314 |

- Sort by prediction, cut into **equal-count** bins, compare the means.
- Perfect calibration lies on the diagonal.
- Every Amana bin lies far below it. Expected calibration error: **0.30**.

#### [day1#rebalancing-inflates] Trained on a 50/50 sample, applied where one in ten defaults

*Why the curve is so low*

1. **1 · Rebalance** Non-defaults dropped until half the rows default
2. **2 · Train** The model learns odds for a world with 50% defaults
3. **3 · Apply** About 10% default: every probability is inflated

> **A quick check.** Rescaling the odds back to the true base rate brings the Brier score from 0.199 to 0.082. The proper repair is fitted on held-out rows: Session 3.

#### [day1#brier-decomposition] Brier = reliability minus resolution plus uncertainty

*What the Brier score is made of*

| Item | Value |
| --- | --- |
| Reliability (miscalibration) | 0.117 |
| Uncertainty (the base rate) | 0.089 |
| Resolution (separation) | 0.008 |

> **Read it.** Amana loses more to miscalibration than it gains from separating applicants. Recalibration removes most of the first term and leaves the other two.

#### [day1#check-proper-score] Which number shows that the probabilities are too high?

*Checkpoint*

**Question:** A second team's model has AUC 0.78 on your holdout and accuracy 90% at a threshold of 0.5. Its mean predicted default probability is 0.31; the default rate is 0.09. Which result tells the committee the probabilities cannot be read as rates?

- A. The AUC of 0.78, which is high for a credit model
- B. The accuracy of 90%
- C. The model's training loss, reported by the developers at the end of fitting
- D. A reliability diagram and a Brier score against the base rate

#### [day1#lab-1-2] Scores and a reliability diagram

Lab 1.2 · 25 min

Write the Brier score, log loss and equal-count reliability bins, score Amana against the constant base-rate forecast on the holdout, and draw the diagram that shows its probabilities are about four times too high.

[D1_2_compute_brier_score.ipynb](https://sdaiaacademy.github.io/ds-305/labs/D1_2_compute_brier_score.ipynb)

- [ ] Write the Brier score and log loss (5 min)
- [ ] Score Amana and the base-rate forecast (5 min)
- [ ] Write equal-count reliability bins (6 min)
- [ ] Draw the reliability diagram (5 min)
- [ ] Name the cause in the metadata (4 min)
- [ ] ✱ Stretch: Split the Brier score into reliability, resolution and uncertainty

### Session 3: Recalibration

#### [day1#s3] Recalibration

*Session 3*

- Fit Platt scaling and isotonic regression on a calibration split
- Score both on separate held-out rows, never on the rows that fitted them
- Choose a method and state what it cannot repair

#### [day1#recap-ds204-calibration] You calibrated your own model before; now you repair someone else's

*From DS-204*

- **DS-204, Day 4.** Calibration measured by decile on your own model, repaired on unseen rows, lost again under drift.
- **Here.** Two maps compared with proper scores, the rows each may touch, what no map can repair, and the check by group.

#### [day1#repair-without-retraining] A recalibration map sits after the model and changes only the meaning of the score

*Recalibration*

**Amana** score, unchanged → **Map** fitted on the calib rows → **Probability** judged on the holdout

- The map is monotone, so the ranking and the AUC do not move.
- The developers can adopt it without retraining: it is a recommendation with evidence.

#### [day1#platt-and-isotonic] Platt scaling fits one smooth curve; isotonic regression can step anywhere

*Two methods*

- **Platt scaling.** A logistic regression of the outcome on the logit of the score: two parameters, a smooth S-shaped map. Steady with few rows.
- **Isotonic regression.** Any non-decreasing step function of the score. Flexible, needs many rows, can overfit a small calib set.

```python
# file: Lab 1.3 · step 1
lr = LogisticRegression(C=1e6).fit(logit(p_cal).reshape(-1, 1), y_cal)
platt = lambda q: lr.predict_proba(logit(q).reshape(-1, 1))[:, 1]
```

#### [day1#recalibration-results] Both maps cut the Brier score by more than half, and the AUC stays at 0.755

*Lab 1.3 on the holdout*

| Holdout | Brier | Log loss | ECE | AUC |
| --- | --- | --- | --- | --- |
| Raw Amana | 0.199 | 0.586 | 0.304 | 0.755 |
| Platt | 0.081 | 0.284 | 0.011 | 0.755 |
| Isotonic | 0.081 | 0.287 | 0.012 | 0.754 |

Both now beat the base rate (Brier 0.090, log loss 0.324).

#### [day1#map-on-wrong-rows] A map judged on the rows that fitted it looks perfect

*What goes wrong*

- **0.002** ECE of isotonic fitted and scored on the holdout
- **0.012** ECE of isotonic fitted on calib, scored on the holdout

- Isotonic regression can follow the noise of the rows it was fitted on.
- The honest error is six times larger.
- Same mistake as the claimed AUC of Lab 1.1, one step later.

#### [day1#what-no-map-repairs] A map of the score cannot see what the score cannot see

*The limit of the repair*

- It fixes the **average** relation between score and outcome.
- It cannot fix a group whose risk differs at the **same score**: a column the model never had.
- It cannot improve the ranking.
- It holds while new applicants resemble the calib rows; a shift breaks it again.

> **Next session.** Amana excludes region and age band on purpose. Is it calibrated within them?

#### [day1#check-recalibration-rows] Where should the recalibration map be fitted?

*Checkpoint*

**Question:** You recommend Platt scaling for Amana. You have the 30,000 training rows, 10,000 calib rows and 10,000 holdout rows. Where do you fit the map, and where do you report its effect?

- A. Fit on the calib rows, report on the holdout
- B. Fit on the training rows, report on the holdout
- C. Fit and report on the holdout
- D. Fit and report on the calib rows

#### [day1#lab-1-3] Platt and isotonic on held-out rows

Lab 1.3 · 25 min

Fit Platt scaling and isotonic regression on the calib rows, score both on the holdout against the raw model, show that AUC does not move, and see what fitting on the wrong rows would have reported.

[D1_3_recalibrate_two_methods.ipynb](https://sdaiaacademy.github.io/ds-305/labs/D1_3_recalibrate_two_methods.ipynb)

- [ ] Fit Platt scaling on the calib rows (6 min)
- [ ] Fit isotonic regression on the same rows (4 min)
- [ ] Score raw, Platt and isotonic on the holdout (6 min)
- [ ] Draw the three reliability curves (5 min)
- [ ] Choose one and write why (4 min)
- [ ] ✱ Stretch: Recalibrate with only 500 calib rows and compare the two methods again

### Session 4: Calibration by subgroup

#### [day1#s4] Calibration by subgroup

*Session 4*

- Break calibration down by a grouping column with an interval for each group
- Tell a real gap from noise in a small group
- Name the worst group and the evidence that it is not chance

#### [day1#calibrated-on-average] Calibrated on average can hide a group the model under-predicts

*Calibration by subgroup*

- After Platt, mean predicted 0.090 against 0.099 observed on the holdout.
- Region and age band are **not features** of Amana, by design.
- If risk differs between them at the same score, the average hides it.

- **Why it matters at Mizan.** An under-predicted group gets loans priced below its risk; an over-predicted one is declined more than it should be. Both are harms to people.

#### [day1#intervals-for-small-groups] A gap in a group of 500 needs an interval before it is a finding

*Noise in small groups*

```python
# file: mizan.py
def wilson(k, n, z=1.96):
    ph = k / n
    c = (ph + z*z/(2*n)) / (1 + z*z/n)
    h = z * np.sqrt(ph*(1-ph)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return c - h, c + h
```

- The Northern region has 508 holdout applicants; 78 defaulted.
- The 95% Wilson interval for its rate: **12.5% to 18.8%**.
- A finding is a prediction **outside** the interval, not any gap.

#### [day1#calibration-by-region] The Northern region defaults 6.7 points more often than predicted

*Lab 1.4 · by region*

| Region | n | Predicted | Observed | 95% interval |
| --- | --- | --- | --- | --- |
| Central | 3,448 | 8.9% | 9.8% | 8.9 to 10.9 |
| Western | 2,707 | 8.9% | 9.8% | 8.8 to 11.0 |
| Eastern | 1,951 | 9.2% | 9.0% | 7.8 to 10.4 |
| Southern | 1,386 | 9.2% | 9.7% | 8.2 to 11.3 |
| Northern | 508 | 8.7% | 15.4% | 12.5 to 18.8 |

#### [day1#calibration-by-age] Applicants aged 18 to 24 are under-predicted by about 5 points

*Lab 1.4 · by age band*

| Item | Value |
| --- | --- |
| 18-24 · predicted | 12.7% |
| 18-24 · observed | 17.5% |
| 35-44 · predicted | 8.2% |
| 35-44 · observed | 8.4% |

18-24: 1,201 applicants, observed interval 15.4% to 19.7%, prediction outside. The other bands sit inside their intervals.

#### [day1#report-gap-not-fix] Report the gap, its interval and its likely cause; leave the fix to the owners

*What the validator writes*

- **The finding.** Northern applicants: 15.4% observed against 8.7% predicted, interval 12.5 to 18.8.
- **The likely cause.** Region is excluded and the North is 5% of the data: the model never learned it.
- **The options.** A condition of use, a group-aware calibration, more data: a policy choice for Talib and Dr. Sameera.

#### [day1#check-group-gap] Which gap is a finding?

*Checkpoint*

**Question:** After recalibration, the Southern region shows 9.2% predicted and 9.7% observed (interval 8.2 to 11.3). The Northern region shows 8.7% predicted and 15.4% observed (interval 12.5 to 18.8). What do you report?

- A. Neither: the model was recalibrated
- B. Both, as under-predicted regions
- C. The Northern gap, with its interval
- D. Neither: the Northern group is too small

#### [day1#lab-1-4] Calibration by region and age band

Lab 1.4 · 25 min

Break the recalibrated probabilities down by region and age band, put a 95% interval on each observed rate, and name the worst group: the young applicants and the small Northern region, which the model cannot see.

[D1_4_break_calibration_down.ipynb](https://sdaiaacademy.github.io/ds-305/labs/D1_4_break_calibration_down.ipynb)

- [ ] Write the calibration table by group (7 min)
- [ ] Add a 95% interval to each observed rate (5 min)
- [ ] Run it for region and age band (4 min)
- [ ] Plot predicted against observed per group (5 min)
- [ ] Name the worst group and why (4 min)
- [ ] ✱ Stretch: Run the same table for gender and channel and say which gap you would report

#### [day1#wrap-up] What today showed about Amana's probabilities

*Day 1 · three things to keep*

- **Claims need rows.** The 0.864 was in-sample. On the holdout, 0.755.
- **Ranking is not risk.** Probabilities four times too high, repaired by Platt on calib rows: Brier 0.199 to 0.081.
- **Average is not every group.** The North and the 18 to 24 band stay under-predicted after the repair.

#### [day1#next] Day 2: intervals with guarantees

*Tomorrow*

- A point prediction of a loss amount says nothing about how wrong it can be.
- Tomorrow: conformal intervals and prediction sets, and the condition their 90% rests on.
- Prepare: reread the bootstrap pre-work; bring one question about today's diagram.

## Day 2: Intervals with guarantees

Wrap the loss model and the grade model in conformal intervals and prediction sets, then test the guarantee by group and under shift

#### [day2#title] Intervals with guarantees

*SDAIA Academy · Uncertainty, Interpretability and Model Validation*

Day 2: Wrap the loss model and the grade model in conformal intervals and prediction sets, then test the guarantee by group and under shift

Data Science Track · 4 teaching days + capstone day · 4 sessions · 4 labs · DS-305

#### [day2#agenda] Intervals with guarantees

*Today*

- **Session 1 · Conformal intervals**: Split conformal prediction turns any regression into intervals with 90% coverage, as long as the rows are exchangeable
- **Session 2 · Prediction sets**: For the three risk grades, conformal prediction returns a set of grades, and the size of the set is the uncertainty
- **Session 3 · Coverage by group**: The 90% is an average over everyone: large loans and the Northern region can sit far below it, and a group-conditional variant repairs that
- **Session 4 · Coverage under shift**: When the population moves, the rows are no longer exchangeable and the guarantee quietly fails; a simulated shift measures how much

#### [day2#review-day1] Which statement about Amana after Platt scaling is right?

*Review of Day 1*

**Question:** Platt scaling was fitted on the calib rows. On the holdout, the Brier score fell from 0.199 to 0.081 and the mean prediction is 9.0% against 9.9% observed. Which statement holds?

- A. The AUC rose with the Brier score
- B. Every region is now calibrated, since the map was fitted on all of them
- C. The ranking is unchanged and the average is repaired
- D. The developers' model was retrained

### Session 1: Conformal intervals

#### [day2#s1] Conformal intervals

*Session 1*

- Compute the split conformal threshold with its finite-sample correction
- Wrap a fitted regression in intervals and check coverage on new rows
- State the one assumption the guarantee rests on

#### [day2#point-prediction-silent] A loss prediction of 12,000 riyals says nothing about how wrong it can be

*The problem*

- Amana's loss model predicts the riyals lost **if** a loan defaults.
- Finance provisions on it; collections plan on it.
- A point prediction is silent about its error: 12,000 could mean 11,000 to 13,000 or 2,000 to 30,000.

- **What the validator adds.** An interval around every prediction that contains the real loss 90% of the time, with the condition under which that holds.

#### [day2#exchangeability] Conformal prediction needs only one thing: new rows drawn like the calib rows

*The one assumption*

- **Exchangeable**: the calib rows and the new row could be shuffled without changing their joint distribution. Independent draws from one population are exchangeable.
- No assumption on the model, the shape of the errors or the features.
- The guarantee breaks when the population moves: a new product, a new region, a shift in incomes.

> **What goes wrong.** Quoting "90% guaranteed" for applicants who no longer look like the calib rows. Session 4 measures how much is lost.

#### [day2#split-conformal-steps] Score the calib rows, take a quantile, widen every prediction by it

*Split conformal, in three steps*

1. **1 · Score** On calib defaults, the error `|loss - prediction|`
2. **2 · Threshold** The `ceil((n+1)(1-α))`-th smallest score
3. **3 · Interval** `prediction ± q`, clipped at zero

> **No retraining.** The model is used as delivered; only its predictions on calib rows are needed.

#### [day2#finite-sample-correction] Counting the new loan turns "about 90%" into "at least 90%"

*Why "n + 1"*

```python
# file: Lab 2.1 · step 2
def conformal_quantile(scores, alpha=0.1):
    s = np.sort(scores)
    k = ceil((len(s) + 1) * (1 - alpha))
    if k > len(s):
        return inf          # too few rows
    return s[k - 1]
```

- The new loan's score is equally likely to fall at any of the **n + 1** ranks.
- So the chance it falls at or below rank k is k / (n + 1), at least 90%.
- With 8 calib rows, 90% needs a ninth: the interval is infinite. Honest, and a warning about small groups.

#### [day2#loss-interval-result] Plus or minus 8,134 riyals covers 91% of real losses

*Lab 2.1 on the holdout*

- **91.0%** coverage on 993 holdout defaults, target 90%
- **8,134** threshold q in riyals, from 895 calib defaults
- **15,667** mean interval width in riyals; median loss 10,520

> **Same answer from a library.** MAPIE's `SplitConformalRegressor` with `prefit=True` returns the same half-width, 8,134.

#### [day2#coverage-over-splits] Coverage is a promise about the average over draws, not about one interval

*What the 90% promises*

- **0.900** mean coverage over 100 random calib and test splits
- **0.867 to 0.935** the range across those splits

- One calib set can give 87% or 93% on one test set.
- "This loan has a 90% chance to fall in its interval" is not the promise.
- The promise is marginal: **over** loans and calib draws, on average.

#### [day2#check-conformal-promise] What does the 90% interval promise?

*Checkpoint*

**Question:** Split conformal on 895 calib defaults gives Amana's loss model intervals of plus or minus 8,134 riyals. A provisioning analyst asks what the "90%" means. What do you tell her?

- A. Each loan has a 90% chance of a loss inside its own interval
- B. Nine in ten losses fall inside, whoever applies to Mizan next year
- C. The loss model is now 90% accurate
- D. Over new loans like the calib loans, at least 90% fall inside

#### [day2#lab-2-1] Loss intervals with 90% coverage

Lab 2.1 · 25 min

Wrap Amana's loss model in split conformal intervals from the calib defaults, check 90% coverage and width on the holdout defaults, and confirm the same numbers with MAPIE.

[D2_1_wrap_loss_regression.ipynb](https://sdaiaacademy.github.io/ds-305/labs/D2_1_wrap_loss_regression.ipynb)

- [ ] Score the calib defaults: absolute residuals (4 min)
- [ ] Write the conformal threshold (6 min)
- [ ] Intervals for the holdout defaults (4 min)
- [ ] Coverage and mean width (5 min)
- [ ] The same with MAPIE (6 min)
- [ ] ✱ Stretch: Repeat over 100 random calib and test splits and plot the coverage you get

### Session 2: Prediction sets

#### [day2#s2] Prediction sets

*Session 2*

- Build LAC prediction sets from calibrated scores
- Build adaptive (APS) sets and compare their sizes
- Read a set of size one, two or three as a decision

#### [day2#top-grade-hides] An officer who sees only the top grade cannot see the model's doubt

*Classification*

- Amana's grade model gives three probabilities: on time (A), late but repaid (B), default (C).
- The top grade is right for **74.4%** of holdout applicants.
- For some it is a near-certainty; for others three grades are close. The top grade looks the same.

- **A prediction set.** The grades the model cannot rule out, chosen so the true grade is in the set 90% of the time. Its size is the doubt.

#### [day2#lac-sets] LAC keeps every grade whose probability clears one bar

*The simplest sets*

```python
# file: Lab 2.2 · step 1
score = 1 - P_cal[i, true_grade]     # calib rows
q = conformal_quantile(score, 0.1)    # 0.834
in_set = P_hold >= 1 - q              # 0.166
```

- Score: one minus the probability of the true grade.
- On Amana: a grade enters the set if its probability is at least **0.166**.
- Smallest sets on average; can be empty.

#### [day2#aps-sets] APS adds grades from most to least likely until their total reaches the threshold

*Adaptive sets*

1. **Sort** grades by probability
2. **Accumulate** add the next grade's probability
3. **Stop** when the total passes q (with a random share of the last grade)

- Unsure applicants, whose probabilities are spread, get larger sets.
- The randomisation makes coverage exact rather than conservative.

#### [day2#set-results] Both cover 90%; APS gives fewer one-grade sets

*Lab 2.2 on the holdout*

| Method | Coverage | Mean size | Size 1 | Size 2 | Size 3 |
| --- | --- | --- | --- | --- | --- |
| LAC | 89.2% | 1.56 | 56.7% | 30.6% | 12.7% |
| APS | 90.3% | 1.67 | 43.1% | 46.4% | 10.5% |

No empty sets with either method on this holdout.

#### [day2#coverage-by-true-grade] The sets miss the true grade of more than half the defaulters

*What the average hides*

| Item | Value |
| --- | --- |
| A · on time | 99.8% / 99.4% |
| B · late | 70.7% / 79.6% |
| C · default | 44.9% / 43.0% |

Coverage by true grade, LAC / APS. Marginal 90% is reached by over-covering the 72% who pay on time.

#### [day2#set-size-as-doubt] Applicants whose set holds all three grades default three times as often

*Reading set size*

- **1,046** holdout applicants (10.5%) with the set {A, B, C}
- **24.8% vs 8.2%** their default rate, against everyone else's

- **A use for the set.** A set of three grades could route an application to Fuad's officers for review instead of an automatic decision. Day 4 turns this into a condition of use.

#### [day2#check-set-size] What does a set of size one tell the officer?

*Checkpoint*

**Question:** An applicant's 90% APS set is {A}. The officer asks whether the model is certain this applicant will repay on time. What is the right answer?

- A. No: across applicants, about one in ten sets misses the true grade
- B. Yes: only one grade could be the true one
- C. Yes, with exactly 90% probability for this applicant
- D. No: a set of one means the model failed to find enough grades to include

#### [day2#lab-2-2] Sets for three risk grades

Lab 2.2 · 25 min

Build LAC and adaptive prediction sets for grades A, B and C from the grade model, check 90% coverage on the holdout, and measure how set sizes differ by method and by true grade.

[D2_2_build_prediction_sets.ipynb](https://sdaiaacademy.github.io/ds-305/labs/D2_2_build_prediction_sets.ipynb)

- [ ] Write the LAC score and its sets (7 min)
- [ ] Coverage and set sizes (4 min)
- [ ] Adaptive sets (APS) from the helper (5 min)
- [ ] Coverage and size by true grade (5 min)
- [ ] Which applicants get all three grades (4 min)
- [ ] ✱ Stretch: Make sets at 80% and 95% and plot the mean set size against the level

### Session 3: Coverage by group

#### [day2#s3] Coverage by group

*Session 3*

- Measure coverage and width by subgroup
- Explain why marginal coverage says nothing about a group
- Apply group-conditional (Mondrian) conformal and read its cost in width

#### [day2#marginal-not-conditional] 90% for everyone together can be 77% for the loans that matter

*Marginal and conditional*

- **Marginal** coverage: over all loans, 90%. This is what split conformal guarantees.
- **Conditional** coverage: within a group (large loans, a region), 90%. Not guaranteed.
- One width for all loans is too wide for loans whose losses vary little and too narrow for those that vary a lot.

- **The question at Mizan.** Talib provisions large loans separately. He needs their intervals to hold, not the average.

#### [day2#coverage-by-loan-size] Large loans are covered 77% of the time, the others 100%

*Lab 2.3 · one threshold for all*

| Loan size | Holdout defaults | Coverage | Mean width (SAR) |
| --- | --- | --- | --- |
| Large, above 25,000 | 382 | 76.7% | 16,268 |
| Medium, 10,000 to 25,000 | 532 | 100% | 15,737 |
| Small, up to 10,000 | 79 | 100% | 12,286 |

By region, the Northern region is lowest: 87.2% on 78 defaults, too few to call it a finding on its own.

#### [day2#group-conditional-conformal] One threshold per group gives each group its own guarantee

*The repair*

```python
# file: Lab 2.3 · step 2
def group_thresholds(s, groups, alpha=0.1):
    return {g: conformal_quantile(
                   s[groups == g], alpha)
            for g in sorted(set(groups))}
```

- Called **group-conditional** or **Mondrian** conformal.
- Thresholds: large 13,256, medium 4,332, small 1,536 riyals.
- Each group needs enough calib rows: 79 small-loan defaults give a threshold from 79 scores.

#### [day2#who-pays-in-width] Large loans get wider intervals; small loans get narrower ones

*Lab 2.3 · before and after*

| Loan size | Coverage before | after | Width before | after |
| --- | --- | --- | --- | --- |
| Large | 76.7% | 90.6% | 16,268 | 26,422 |
| Medium | 100% | 92.5% | 15,737 | 8,665 |
| Small | 100% | 86.1% | 12,286 | 3,072 |

Overall coverage stays at 91.2%. Small loans' 86.1% is on 79 defaults: within chance of 90%.

#### [day2#coverage-finding] A coverage finding names the group, the number and the cost of the repair

*What the report says*

> **Finding.** The loss intervals cover 91% of defaults overall but only 77% of loans above 25,000 SAR (382 holdout defaults). One threshold per loan-size band restores 91% for large loans, with intervals 1.6 times wider.

- **What.** Under-coverage of large loans
- **Evidence.** 382 holdout defaults, coverage table
- **Recommendation.** Per-band thresholds, recomputed when the loan mix changes

#### [day2#check-group-coverage] Why did one threshold under-cover large loans?

*Checkpoint*

**Question:** Split conformal with one threshold reaches 91% coverage overall on Amana's loss intervals, 77% on large loans and 100% on medium and small loans. What explains it?

- A. The conformal threshold was computed wrongly
- B. Losses on large loans vary more, and one width averages over groups
- C. Large loans are too rare in the calib rows
- D. The loss model under-predicts large losses, so their intervals sit too low

#### [day2#lab-2-3] Coverage by group, then per group

Lab 2.3 · 25 min

Measure the loss intervals' coverage by loan size and region, find the large loans near 77%, then compute one threshold per loan-size band and show every band back near 90% with wider intervals where they are needed.

[D2_3_measure_subgroup_coverage.ipynb](https://sdaiaacademy.github.io/ds-305/labs/D2_3_measure_subgroup_coverage.ipynb)

- [ ] Coverage and width by loan size and region (6 min)
- [ ] Write group-conditional thresholds (7 min)
- [ ] Coverage by group again (5 min)
- [ ] Width by group: who pays for the repair (4 min)
- [ ] Write one finding for the report (3 min)
- [ ] ✱ Stretch: Scale the residuals by the predicted loss and compare with the group-conditional repair

### Session 4: Coverage under shift

#### [day2#s4] Coverage under shift

*Session 4*

- Simulate a covariate shift by reweighting a held-out sample
- Measure the coverage lost for intervals and for sets
- Explain why the guarantee needs exchangeability and what restores it

#### [day2#when-population-moves] When applicants change, the interval does not know

*Coverage under shift*

- A conformal threshold is a fact about the **calib rows**.
- If next year's applicants earn more, borrow more or come from a new region, the rows are no longer exchangeable.
- The interval keeps its width and quietly loses coverage.

- **At Mizan.** A planned campaign for salaried applicants with higher incomes. Higher incomes mean larger loans.

#### [day2#simulating-a-shift] Tilt the holdout towards higher incomes and measure what is lost

*A shift you control*

```python
# file: Lab 2.4 · step 1
z = standardise(log(declared_income))
w = exp(strength * z)
shifted = hold.iloc[rng.choice(n, n, p=w / w.sum())]
```

- Strength 0: the holdout as it is.
- Strength 1: median declared income of defaulters from 9,400 to 16,100 riyals.
- The truth of every resampled row is known, so coverage can be measured exactly.

#### [day2#coverage-lost-under-shift] A strong income shift costs the loss intervals 14 points; the grade sets lose 1

*Lab 2.4*

| Shift strength | Loss interval coverage | Large-loan share | Grade set coverage |
| --- | --- | --- | --- |
| 0 (holdout) | 91.0% | 38.5% | 89.2% |
| 0.5 | 88.4% | 47.5% | 89.3% |
| 1.0 | 83.2% | 58.1% | 88.7% |
| 1.5 | 77.1% | 69.4% | 88.0% |

The loss intervals' mean width moves by less than 5% across the four rows.

#### [day2#weighted-conformal] Weighting the calib scores by the shift restores the coverage

*When the shift is known*

- **83.2%** plain split conformal at strength 1
- **93.6%** weighted conformal with the true weights

- Each calib score counts as much as its row is more likely after the shift.
- Needs the weights: in a simulation you know them; with real applicants you estimate them.
- The guarantee then rests on that estimate.

#### [day2#monitoring-rule] The result becomes a monitoring rule with a trigger and an action

*From a stress test to a condition*

| Watch | Monthly: the share of new loans above 25,000 riyals and the median declared income |
| --- | --- |
| Trigger | Large-loan share above 48% (holdout: 38%) |
| Action | Recompute the per-band thresholds on recent defaults, or stop quoting the 90% |

> **What goes wrong.** Monitoring coverage itself: losses are known only months after a default. Watch the inputs that predict the loss of coverage.

#### [day2#check-shift] Why did the grade sets survive the shift that broke the loss intervals?

*Checkpoint*

**Question:** Under the same income shift, loss interval coverage falls from 91% to 77% while grade set coverage falls from 89% to 88%. Which explanation is right?

- A. Prediction sets keep their guarantee under shift; intervals for amounts do not
- B. The grade model was calibrated on more rows
- C. Loss errors grow with income through loan size; grade errors barely do
- D. Richer applicants default less, so the sets are easier

#### [day2#lab-2-4] An income shift, measured

Lab 2.4 · 25 min

Tilt the holdout towards higher declared incomes at three strengths, measure the coverage the loss intervals and the grade sets lose, and see that the cost falls on the large loans.

[D2_4_simulate_income_shift.ipynb](https://sdaiaacademy.github.io/ds-305/labs/D2_4_simulate_income_shift.ipynb)

- [ ] Write the shifted sample (5 min)
- [ ] Loss interval coverage at three strengths (7 min)
- [ ] Grade set coverage at three strengths (5 min)
- [ ] Which loans lose coverage (5 min)
- [ ] A monitoring rule from the result (3 min)
- [ ] ✱ Stretch: Reweight the calib scores by the known shift and restore the coverage (weighted conformal)

#### [day2#wrap-up] What today showed about Amana's intervals and sets

*Day 2 · three things to keep*

- **The guarantee is marginal.** 91% overall; 77% for large loans; defaulters' grade covered 45% of the time.
- **Groups need their own threshold.** Per-band thresholds restore 91% for large loans, at 1.6 times the width.
- **Shift breaks it silently.** A strong income shift: 77% coverage, same width. Watch the loan mix.

#### [day2#next] Day 3: what drives the predictions

*Tomorrow*

- Officers see three reasons with every application. Are they right, and are they stable?
- Permutation importance, effect plots and Shapley values, on two income features that move together.
- Prepare: one question about the reasons a credit model should be allowed to give.

## Day 3: What drives the predictions

Explain the default model three ways, test how stable each explanation is, and find where explanations mislead

#### [day3#title] What drives the predictions

*SDAIA Academy · Uncertainty, Interpretability and Model Validation*

Day 3: Explain the default model three ways, test how stable each explanation is, and find where explanations mislead

Data Science Track · 4 teaching days + capstone day · 4 sessions · 4 labs · DS-305

#### [day3#agenda] What drives the predictions

*Today*

- **Session 1 · Permutation importance**: Permutation importance measures what the model loses without a feature, and with correlated features it measures something else
- **Session 2 · Effect plots**: Partial dependence averages over combinations that never occur; accumulated local effects stay inside the data
- **Session 3 · Shapley values**: Shapley values split one prediction among the features, add up exactly, and can change when the model is refitted
- **Session 4 · When explanations mislead**: A feature correlated with a protected attribute takes its attribution, and the reason shown to an applicant then names a proxy

#### [day3#review-day2] Which statement about the loss intervals is supported?

*Review of Day 2*

**Question:** Split conformal intervals for Amana's loss model cover 91% of holdout defaults, 77% of large loans, and 77% of all defaults after a strong simulated income shift. Which sentence can go in the report?

- A. Every loan's interval holds 90% of the time
- B. Coverage holds overall, not for large loans or a richer population
- C. The loss model must be retrained before any of its intervals can be used
- D. Under a shift the intervals widen to keep 90%

### Session 1: Permutation importance

#### [day3#s1] Permutation importance

*Session 1*

- Compute permutation importance on held-out rows with repeats
- Report its spread, not a single number
- Permute correlated features together and explain the difference

#### [day3#three-explanation-questions] Each explanation method answers a different question

*Explanations*

| Method | Its question | Session |
| --- | --- | --- |
| Permutation importance | How much worse does the model do without this feature? | 1 |
| Partial dependence, ALE | How does the prediction change as this feature changes? | 2 |
| Shapley values | How much did each feature move this one prediction? | 3 |

> **None of them asks** what causes default. All three describe the model, on data that may have features moving together.

#### [day3#permutation-idea] Shuffle one column on the holdout and measure how much worse the predictions get

*Permutation importance*

```python
# file: Lab 3.1 · step 1
def rise_one(df, col, rng):
    d = df.copy()
    d[col] = d[col].to_numpy()[rng.permutation(len(d))]
    return log_loss(y, predict(d)) - base
```

- On the **holdout**: importance for new applicants, not memorised noise.
- On the validator's model: Amana with the Platt repair, log loss 0.284.
- One shuffle is one draw: repeat and report the spread.

#### [day3#importance-with-spread] Two income features lead; the last two cannot be told from zero

*Lab 3.1 · ten repeats*

| Item | Value |
| --- | --- |
| bank_inflow | 0.068 |
| declared_income | 0.046 |
| term_months | 0.033 |
| loan_amount | 0.025 |
| prior_missed | 0.009 |
| household_size, channel | range includes 0 |

#### [day3#shuffle-creates-impossible] Shuffling one income creates applicants who do not exist

*The correlated pair*

- **0.84** correlation of log declared income and log bank inflow
- **36%** of rows outside the real declared-to-inflow range after shuffling declared income

- 98% of real applicants declare between 0.72 and 2.36 times their bank inflow.
- Shuffle declared income alone and a third of rows declare 20,000 against an inflow of 4,000.
- The model reads that as overstatement and predicts high risk: importance measures its reaction to fiction.

#### [day3#grouped-permutation] Shuffled together, the pair matters less than either one alone

*Lab 3.1 · step 4*

| Shuffled | Rise in log loss | Range over 10 repeats |
| --- | --- | --- |
| bank_inflow alone | 0.068 | 0.064 to 0.072 |
| declared_income alone | 0.046 | 0.043 to 0.050 |
| Both, same permutation | 0.034 | 0.030 to 0.037 |

> **For the report.** Give features that travel together one grouped importance. Do not rank them against each other.

#### [day3#what-importance-may-say] The table supports a statement about the model, never about the world

*Writing it up*

- **Supported.** Amana's holdout log loss depends most on the two income features, then on term and loan size.
- **Not supported.** "Bank inflow matters more than declared income", or "income drives default".

#### [day3#check-permutation] What do the two single-income importances tell you?

*Checkpoint*

**Question:** Shuffled alone, bank inflow raises the log loss by 0.068 and declared income by 0.046. Shuffled together, by 0.034. The two correlate at 0.84. Which reading is right?

- A. Bank inflow is the more important of the two
- B. Income accounts for 0.114 of the log loss
- C. Income matters; the single numbers measure reactions to impossible pairs
- D. Together they matter little, so income can be dropped from the next version

#### [day3#lab-3-1] Importance and its spread

Lab 3.1 · 25 min

Write permutation importance on the holdout with ten repeats, report the spread, then permute the two correlated income features together and see the pair matter less than either one shuffled alone.

[D3_1_compute_permutation_importance.ipynb](https://sdaiaacademy.github.io/ds-305/labs/D3_1_compute_permutation_importance.ipynb)

- [ ] Write permutation importance for one feature (7 min)
- [ ] All features, ten repeats, with spread (5 min)
- [ ] Plot the importances with their range (4 min)
- [ ] Permute the two incomes together (5 min)
- [ ] Write what the table may and may not say (4 min)
- [ ] ✱ Stretch: Use the Brier score in place of log loss and see whether the order changes

### Session 2: Effect plots

#### [day3#s2] Effect plots

*Session 2*

- Compute a partial dependence curve and individual curves
- Compute accumulated local effects for the same feature
- Show where the two disagree and which one to report

#### [day3#partial-dependence-idea] Partial dependence sets everyone's value to the same number and averages

*Partial dependence*

```python
# file: Lab 3.2 · step 1
for v in grid:
    d = sample.copy()
    d["declared_income"] = v      # everyone
    pd_curve.append(predict(d).mean())
```

- Answers: what if every applicant had declared **v**, all else unchanged?
- Fine when the feature is independent of the others.
- With bank inflow fixed, a high **v** means overstatement for most applicants.

#### [day3#ale-idea] ALE moves each applicant only a little, inside their own bin

*Accumulated local effects*

1. **1 · Bin** Cut declared income at the grid values
2. **2 · Nudge** Move only that bin's applicants from its lower to its upper edge
3. **3 · Accumulate** Add the mean changes from left to right, centre at zero

Every prediction is for a nearby, realistic applicant: their bank inflow already matches an income in that bin.

#### [day3#pd-ale-disagree] Both curves rise above 10,000 riyals; partial dependence rises 1.6 times as much

*Lab 3.2*

| Declared income (SAR) | 4,310 | 7,290 | 9,070 | 11,650 | 13,880 | 18,930 |
| --- | --- | --- | --- | --- | --- | --- |
| PD, centred | -0.036 | -0.055 | -0.033 | 0.035 | 0.073 | 0.133 |
| ALE, centred | -0.032 | -0.032 | -0.026 | 0.012 | 0.047 | 0.080 |
| PD rows outside the data | 75% | 29% | 17% | 17% | 27% | 53% |

From 9,070 to 18,930 riyals: PD rises by 0.166, ALE by 0.106 (change in predicted default probability).

#### [day3#counting-impossible-rows] Count the rows a curve was built from that no real applicant resembles

*Making the disagreement checkable*

- In the 4,000 sampled applicants, declared income is 0.72 to 2.34 times bank inflow (1st to 99th percentile).
- At 18,930 riyals, **53%** of PD rows lie outside that range; at 4,310, **75%**.
- A curve point built mostly from such rows describes the model's behaviour on fiction.

> **A habit for any effect plot.** Report, under the plot, the share of evaluated rows outside the data.

#### [day3#ice-curves] Individual curves show who the average hides

*Behind the average*

**37 of 50** applicants whose predicted risk moves more than 10 points across the grid

- An ICE curve is one applicant's partial dependence.
- Their average is the PD curve.
- The curves are not parallel: how much a declared income moves risk depends on the applicant's own bank inflow, through the overstatement signal.

#### [day3#which-curve-report] Put ALE in the report, and say what PD would have claimed

*What the report shows*

> Among realistic applicants, predicted risk changes little with declared income below about 12,000 riyals and rises above it, by 0.11 between 9,070 and 18,930 riyals. Partial dependence shows 1.6 times that rise because, at the top of the grid, 53% of the rows it averages over pair a high declared income with a bank inflow no real applicant has.

> **Neither is causal.** ALE does not say what would happen if an applicant earned more.

#### [day3#check-effect-plots] Why does partial dependence overstate the rise?

*Checkpoint*

**Question:** PD for declared income rises 1.6 times as much as ALE between 9,000 and 19,000 riyals. Declared income and bank inflow correlate at 0.84, and overstating income is a risk signal in the model. What explains the gap?

- A. PD pairs high declared incomes with inflows real applicants do not have
- B. ALE uses too few applicants in each bin, so its curve is flattened by noise
- C. The model overreacts to declared income
- D. PD is not centred and ALE is

#### [day3#lab-3-2] Partial dependence against local effects

Lab 3.2 · 25 min

Draw partial dependence and accumulated local effects for declared income, find where they disagree, and trace the disagreement to applicants whose declared income is set far from their bank inflow.

[D3_2_show_where_partial.ipynb](https://sdaiaacademy.github.io/ds-305/labs/D3_2_show_where_partial.ipynb)

- [ ] Write partial dependence for declared income (6 min)
- [ ] Accumulated local effects from the helper (4 min)
- [ ] Plot both, centred (5 min)
- [ ] Count the impossible rows partial dependence used (6 min)
- [ ] Say which curve goes in the report (4 min)
- [ ] ✱ Stretch: Draw individual conditional expectation curves for 50 applicants and find the ones that bend

### Session 3: Shapley values

#### [day3#s3] Shapley values

*Session 3*

- Compute exact tree Shapley values and check that they add up
- Read the top reasons for one applicant
- Measure whether the reasons survive five refits

#### [day3#shapley-accounting] Shapley values split one prediction exactly among the features

*Shapley values*

- Each feature's value is its average contribution over all orders of adding features.
- For trees it is computed exactly and fast (TreeSHAP).
- They **add up**: base value plus the features' values equals the raw score of that applicant.

```python
# file: Lab 3.3 · step 1
v = amana.predict(X10, pred_contrib=True)
# columns: one per feature, then the base
raw = amana.predict(X10, raw_score=True)
np.allclose(v.sum(axis=1), raw)   # True
```

#### [day3#reasons-to-officers] Officers see the three features that pushed risk up most

*How Mizan uses them*

| Applicant | Reasons shown |
| --- | --- |
| M008832 | bank_inflow, prior_loans, channel |
| M004026 | bank_inflow, prior_missed, years_in_business |
| M027921 | declared_income, channel, business_type |
| M046515 | term_months, years_in_business, channel |

> **The question.** Would the same applicant get the same reasons from a model the developers could equally have shipped?

#### [day3#refits-equally-shippable] Refit with the developers' own recipe and see whether the reasons survive

*Stability*

1. **Recipe** From metadata.json: rebalance to 50/50, 300 rounds, bagging
2. **Five refits** Bootstrap samples of the training rows, seeds 1 to 5
3. **Panel** Top reason per applicant per refit

Each refit is a model the developers could have shipped: same data source, same recipe, different random draws.

#### [day3#stability-panel] Two of ten applicants change their top reason; only one keeps all three

*Lab 3.3 · the stability panel*

| Applicant | Refit 1 | Refit 2 | Refit 3 | Refit 4 | Refit 5 |
| --- | --- | --- | --- | --- | --- |
| M004026 | bank_inflow | prior_missed | bank_inflow | prior_missed | bank_inflow |
| M008361 | loan_amount | years_in_business | loan_amount | loan_amount | years_in_business |
| M008832 | bank_inflow | bank_inflow | bank_inflow | bank_inflow | bank_inflow |
| Seven others | the same top reason in all five refits |  |  |  |  |

The set of three reasons is the same in all five refits for 1 of the 10 applicants.

#### [day3#incomes-trade-credit] The two incomes trade credit between refits; their sum moves less

*Why they move*

- **3.35** widest range of one income feature's value for one applicant, in log-odds
- **1.70** widest range of the two incomes' sum

- The model needs both incomes; how it splits the credit between them is partly chance.
- "Your bank inflow" and "your declared income" are one reason told two ways.
- Grouping them as "income" is the more honest reason.

#### [day3#reason-display-rule] Show a reason only when it survives refits

*From the panel to a rule*

- **Proposed rule.** Show a reason when it is the top reason in at least four of five refits; group correlated features into one reason; otherwise say the model has no stable main reason.
- **Why it matters.** An applicant told a reason that flips with a refit has been given an accident of training, not an explanation.

#### [day3#check-shapley] What does "the values add up" guarantee?

*Checkpoint*

**Question:** For every applicant, the base value plus the Shapley values of the ten features equals the model's raw score exactly. A colleague says this proves the reasons shown to officers are reliable. What do you answer?

- A. Yes: an exact decomposition cannot change
- B. Complete accounting for this model, not survival across a refit
- C. Yes: each value is the feature's causal effect
- D. No: Shapley values stop adding up once two features are strongly correlated

#### [day3#lab-3-3] Ten applicants, five refits

Lab 3.3 · 25 min

Explain ten applicants with LightGBM's exact tree Shapley values, check they add up to the prediction, then refit the model five times on bootstrap samples and build a stability panel of each applicant's top reason.

[D3_3_explain_ten_applicants.ipynb](https://sdaiaacademy.github.io/ds-305/labs/D3_3_explain_ten_applicants.ipynb)

- [ ] Shapley values for ten applicants (5 min)
- [ ] Check that they add up (4 min)
- [ ] Refit five times with the developers' recipe (6 min)
- [ ] The stability panel: top reason per refit (6 min)
- [ ] Mark the explanations you would not show (4 min)
- [ ] ✱ Stretch: Measure the spread of the declared-income and bank-inflow values across the refits

### Session 4: When explanations mislead

#### [day3#s4] When explanations mislead

*Session 4*

- Show how a correlated proxy moves attributions
- Tell a proxy from a cause with the data you have
- Write a caution that a validator can attach to the explanations

#### [day3#wallet-proposal] Version 2.2 adds mobile-wallet months and gains 0.004 of AUC

*A change request*

- Rasil's team proposes one new feature: how long the applicant has used a mobile wallet.
- Holdout AUC: 0.757 without, 0.761 with.
- Age is excluded from the model on purpose, and remains so.

- **The validator's question.** What does the new feature carry, and who will be told it is their reason?

#### [day3#what-a-proxy-is] A proxy is a permitted feature that carries an excluded one

*Proxies*

- Wallet months correlates with age band at **-0.44**: median 43 months for 18 to 24, 12 for 55 and over.
- Age is related to default in these data, so the model learns to use what wallet months carries of it.
- "Uses no age" stays true as written and becomes misleading in effect.

> **What goes wrong.** Checking only the feature list. Claim 3 of the card passed Lab 1.1 on exactly that check.

#### [day3#who-gets-wallet-reason] Applicants over 55 are told "short wallet history" more than twice as often

*Lab 3.4*

| Item | Value |
| --- | --- |
| 18-24 | 20.6% |
| 25-34 | 21.3% |
| 35-44 | 25.1% |
| 45-54 | 30.7% |
| 55+ | 46.5% |

Share of holdout applicants for whom wallet months is among the three reasons. For the 55+ band its mean value pushes risk up (+0.073 log-odds).

#### [day3#proxy-narrows-gap] The proxy narrows the age calibration gap a little, which is what a proxy does

*The tell-tale*

| Age band | Gap without wallet months | Gap with it |
| --- | --- | --- |
| 18-24 | 4.9 points | 4.7 points |
| 55+ | 1.2 points | 0.8 points |

Observed minus predicted default rate after Platt scaling, holdout rows.

#### [day3#writing-the-caution] A caution names what the feature carries, who receives it and what to do

*The documented misleading result*

> Wallet months correlates with age (r = -0.44) and is a top-three reason for 47% of applicants over 55 against 21% of those aged 18 to 24. Shown to officers or applicants, "short wallet history" would in effect tell older applicants that their age counts against them, in a model that excludes age.

> **Recommendation.** Do not add the feature for an AUC gain of 0.004; if it is added, do not show it as a reason, and audit by age band.

#### [day3#check-proxy] Is wallet months a proxy for age?

*Checkpoint*

**Question:** Wallet months correlates with age band at -0.44, is a top reason for 47% of applicants over 55 against 21% under 25, and narrows the age gap. What can you conclude?

- A. Wallet history causes default, so the feature is legitimate to use
- B. The correlation is too weak to matter
- C. Age is not in the model, so there is no age effect
- D. It carries age information and would deliver age-linked reasons

#### [day3#lab-3-4] A proxy takes the credit

Lab 3.4 · 25 min

Refit the model with mobile-wallet months added, show that the new feature becomes a top reason far more often for older applicants, check its link with age, and write the caution for the validation report.

[D3_4_show_how_correlated.ipynb](https://sdaiaacademy.github.io/ds-305/labs/D3_4_show_how_correlated.ipynb)

- [ ] Refit with wallet months added (5 min)
- [ ] Shapley values with and without it (5 min)
- [ ] How often it is a top reason, by age band (6 min)
- [ ] Its link with age (4 min)
- [ ] Write the caution (5 min)
- [ ] ✱ Stretch: Remove the proxy and check whether the age-band calibration gap of Lab 1.4 changed

#### [day3#wrap-up] What today showed about Amana's explanations

*Day 3 · three things to keep*

- **Correlated features.** Group them: one importance, one reason. Single numbers measure reactions to impossible rows.
- **Stability.** Top reasons survive five refits for 8 of 10 applicants; full sets of three for 1.
- **Proxies.** An excluded attribute comes back through a permitted feature, and so do its reasons.

#### [day3#next] Day 4: validate and decide

*Tomorrow*

- Every finding by subgroup in one table, tagged with a bias type from the national guide.
- A later batch where performance fell, with a cause to find.
- A checklist, a decision, and the choice of your capstone package.

## Day 4: Validate and decide

Turn the evidence into rated findings by subgroup, diagnose a decline, score the package against a checklist and decide

#### [day4#title] Validate and decide

*SDAIA Academy · Uncertainty, Interpretability and Model Validation*

Day 4: Turn the evidence into rated findings by subgroup, diagnose a decline, score the package against a checklist and decide

Data Science Track · 4 teaching days + capstone day · 4 sessions · 4 labs · DS-305

#### [day4#agenda] Validate and decide

*Today*

- **Session 1 · Subgroup validation**: Calibration, coverage and error by subgroup are validation evidence, and each gap gets a cause and a bias type from the national guide
- **Session 2 · Diagnosing degradation**: A decline is diagnosed from slices, coverage and explanations together, and the cause decides the action
- **Session 3 · Checklists and model cards**: A reporting checklist turns "is it documented?" into items with evidence, and the missing evidence is a finding
- **Session 4 · The fit-for-use decision**: A validation ends in a decision: fit, fit with conditions, or not fit, with conditions of use, human review and monitoring

#### [day4#review-day3] Which explanation statement can go in the report?

*Review of Day 3*

**Question:** Over five refits, 8 of 10 applicants keep their top reason and 1 of 10 keeps all three. Bank inflow and declared income correlate at 0.84. Which sentence is supported?

- A. Show officers one grouped income reason, and only reasons that survive refits
- B. The three reasons are exact, so they can be shown as they are
- C. Bank inflow is a stronger reason than declared income
- D. The model must be retrained until all three reasons are stable across refits

### Session 1: Subgroup validation

#### [day4#s1] Subgroup validation

*Session 1*

- Tabulate calibration, coverage and error by subgroup in one table
- Separate a gap from noise with an interval
- Tag each gap with a bias type from the SDAIA AI Bias Reference Guide

#### [day4#one-audit-table] One table, every group the lender answers for, every measure with its uncertainty

*Subgroup validation*

| Column | Measures per group |
| --- | --- |
| region, age band, gender, channel | n, mean predicted, observed rate with a 95% interval, Brier, AUC |
| the same groups, defaulters only | loss interval coverage and the number of defaults behind it |
| loan size band | loss coverage, because Talib provisions by loan size |

> **The validator's model.** Amana with the Platt repair from Day 1 and one conformal threshold of 8,134 SAR from Day 2.

#### [day4#region-table] The Northern region defaults at 15.4% against 8.7% predicted

*Lab 4.1 · region*

| Region | n | Predicted | Observed (95% interval) | AUC | Loss coverage |
| --- | --- | --- | --- | --- | --- |
| Central | 3,448 | 8.9% | 9.8% (8.9 to 10.9) | 0.775 | 90.0% |
| Eastern | 1,951 | 9.2% | 9.0% (7.8 to 10.4) | 0.766 | 93.2% |
| Northern | 508 | 8.7% | 15.4% (12.5 to 18.8) | 0.720 | 87.2% |
| Southern | 1,386 | 9.2% | 9.7% (8.2 to 11.3) | 0.751 | 93.3% |
| Western | 2,707 | 8.9% | 9.8% (8.8 to 11.0) | 0.740 | 91.0% |

#### [day4#flag-rules] Flag a gap only when an interval supports it

*From numbers to flags*

- **Calibration.** Flag when the mean prediction lies outside the observed rate's 95% interval.
- **Coverage.** Flag when loss coverage is below 85% on at least 50 defaults.

| Flagged | Value |
| --- | --- |
| Northern, calibration | +6.7 points |
| Age 18 to 24, calibration | +4.8 points |
| Age 55+, coverage | 83.1% on 83 defaults |
| Men; branch channel, calibration | +1.6; +1.0 points |
| Loans above 25,000 SAR, coverage | 76.7% on 382 defaults |

#### [day4#bias-guide-tags] A tag names how the gap arose, which points to the remedy

*SDAIA AI Bias Reference Guide*

| Gap | Tag | Guide | Mechanism in Amana |
| --- | --- | --- | --- |
| Northern | Aggregation Bias | p. 12 | 5% of rows, pooled with larger regions, fitted worst |
| Age 18 to 24 | Age Bias | p. 11 | age excluded, its differences not accounted for |
| Large loans | Evaluation Bias | p. 42 | 91% marginal coverage overstates how well they are covered |

> **A tag is not a verdict.** It names a mechanism from the guide. Whether a gap is unfair is the committee's judgement, made with the evidence.

#### [day4#rating-rule] Rate by what the gap does to people and how firmly it is supported

*Rating*

1. **High** Affects approval or price, interval supports it, 2 points or more
2. **Medium** Affects provisioning, or supported but small
3. **Low** Neither

Two high findings from this table: the Northern region and applicants aged 18 to 24.

#### [day4#small-cells] Crossing groups multiplies the cells and divides the evidence

*Lab 4.1 · stretch*

- **6 to 30** defaults per age band within the Northern region
- **9 to 27** points: width of the observed rate's interval per cell

- Young Northern applicants default at 39% in the sample, on 18 defaults.
- The interval runs from 26% to 54%: a reason to watch, not a finding.
- Report crossed cells with their counts and intervals, or not at all.

#### [day4#check-subgroup] How do you report the gap for men?

*Checkpoint*

**Question:** After recalibration, men default at 10.5% against 8.9% predicted, interval 9.6 to 11.4; women at 9.5% against 9.1%. Under the course rules, what goes in the report?

- A. Nothing: the gap is within noise
- B. A high finding of gender bias, since gender is a protected attribute
- C. Evidence that the model discriminates against men
- D. A medium finding: supported, 1.6 points, monitored by gender

#### [day4#lab-4-1] Gaps by group, tagged

Lab 4.1 · 25 min

Build one audit table of calibration, loss-interval coverage and error by region, age band, gender and channel, flag the gaps an interval supports, and tag each with a bias type from the national guide and a reason.

[D4_1_tabulate_calibration_coverage.ipynb](https://sdaiaacademy.github.io/ds-305/labs/D4_1_tabulate_calibration_coverage.ipynb)

- [ ] The audit table for one column (7 min)
- [ ] All four columns (4 min)
- [ ] Flag the gaps an interval supports (5 min)
- [ ] Tag each gap with a bias type (6 min)
- [ ] Rate each finding (3 min)
- [ ] ✱ Stretch: Compute the same table for the age band within the Northern region and say what the counts allow

### Session 2: Diagnosing degradation

#### [day4#s2] Diagnosing degradation

*Session 2*

- Find the slice and the month where performance fell
- Check feature distributions, coverage and explanations in that slice
- Write a memo that names one cause and one action

#### [day4#model-got-worse] Talib reports that Amana "got worse" in 2025

*Six months in production*

- **0.755 → 0.677** AUC, holdout 2023-24 to the 2025 batch
- **17.6% vs 9.2%** mean predicted against observed default, 2025

- **The tempting memo.** "The model has degraded and must be retrained on recent data."

#### [day4#where-and-since-when] Slice by month and channel: the failure is the app channel from March

*Lab 4.2 · step 1*

| Month | App predicted | App observed | App AUC | Branch AUC |
| --- | --- | --- | --- | --- |
| 2025-01 | 9.9% | 10.1% | 0.728 | 0.740 |
| 2025-02 | 9.8% | 9.9% | 0.723 | 0.770 |
| 2025-03 | 46.6% | 11.0% | 0.680 | 0.766 |
| 2025-04 to 06 | 46% | 9 to 11% | 0.53 to 0.68 | 0.69 to 0.77 |

#### [day4#psi-inputs] One input has moved: bank inflow, by a factor of a thousand

*Lab 4.2 · step 2*

| Item | Value |
| --- | --- |
| bank_inflow | 8.28 |
| declared_income | 0.037 |
| loan_amount | 0.016 |
| five others | below 0.011 |

- Population stability index against the holdout's app applicants: below 0.1 little change, above 0.25 large.
- Median app bank inflow: 8,065 SAR in January, **8.14** in March.

#### [day4#coverage-and-reasons] The explanations changed before anyone read the outcomes

*Lab 4.2 · step 3*

| Slice | Bank inflow is the top reason | Loss coverage |
| --- | --- | --- |
| Holdout, app | 21.6% | 92.4% |
| 2025, app from March | 98.4% | 94.8% |
| 2025, branch | 15.9% | 88.1% |

> **Use it.** Reasons are available on the day of scoring; outcomes take twelve months. A sudden change in the reasons officers see is an early alarm.

#### [day4#rule-out-decoy] Incomes also rose, in both channels; the branch channel is unharmed

*Lab 4.2 · step 4*

- Median declared income +9.3% (app) and +8.7% (branch).
- Branch channel 2025 AUC 0.747, as before.
- A cause must explain why the failure is in one channel and starts in one month.

- **Model drift or a broken input?** The guide's Model Drift (p. 66) and Concept Drift (p. 28) describe a changing world. This is Measurement Bias (p. 63): values recorded too low.

#### [day4#cause-memo] The memo: one cause, three numbers, one action, one thing not to do

*Lab 4.2 · step 5*

| **Cause** | From March 2025 the app channel's bank inflow arrives about a thousand times too small: a unit change in the data feed. |
| --- | --- |
| **Evidence** | App from March: 46% predicted against 10% observed; PSI of bank inflow 8.3; bank inflow the top reason for 98% of the slice. |
| **Action** | Fix the feed to riyals, rescore March to June, add an input range check before scoring. |
| **Not** | Do not retrain on the 2025 batch. |

#### [day4#check-degradation] What would have caught the fault in its first week?

*Checkpoint*

**Question:** The feed broke on 1 March 2025. Outcomes of March applications mature in March 2026. Which monitor raises the alarm in March 2025?

- A. Input range checks and top-reason shares, on the day of scoring
- B. Monthly AUC on scored applications, against the holdout AUC of 0.755
- C. Loss interval coverage on defaulted loans
- D. Retraining every quarter on recent applications

#### [day4#lab-4-2] The cause of the decline

Lab 4.2 · 25 min

Score the 2025 batch, find that the decline sits in the app channel from March, trace it to bank inflow arriving a thousand times too small, reject the income rise as a decoy, and write a memo with the cause and the action.

[D4_2_find_seeded_cause.ipynb](https://sdaiaacademy.github.io/ds-305/labs/D4_2_find_seeded_cause.ipynb)

- [ ] Score the batch by month and channel (6 min)
- [ ] Feature stability in the failing slice (5 min)
- [ ] Coverage and explanations in that slice (5 min)
- [ ] Rule out the decoy (4 min)
- [ ] Write the memo (5 min)
- [ ] ✱ Stretch: Write the input check that would have stopped the batch on the first day

### Session 3: Checklists and model cards

#### [day4#s3] Checklists and model cards

*Session 3*

- Score a package against a reporting checklist item by item
- Distinguish missing evidence from evidence of a problem
- List what the developers must supply before a decision

#### [day4#why-a-checklist] A checklist turns "is it documented?" into twelve answerable items

*Checklists*

- Each item is **present**, **partial** or **missing**, with the file and field that decide it.
- Two validators reach the same score; the developers know what to send.
- The course list adapts REFORMS (reporting standards for science that uses machine learning) and model cards to a credit model.

- **Three parts.** Conceptual soundness (S1 to S5), outcomes (O1 to O5), monitoring (M1, M2): the three elements of validation in SR 11-7.

#### [day4#checklist-items] Twelve items, each with the evidence that would satisfy it

*The course checklist*

| Soundness | Outcomes | Monitoring |
| --- | --- | --- |
| S1 intended and excluded uses | O1 AUC on unseen rows | M1 what is watched, thresholds, owner |
| S2 target and its window | O2 calibration, proper score | M2 conditions of use, human review |
| S3 training data, sampling | O3 intervals or sets, coverage |  |
| S4 features, when known, exclusions | O4 results by subgroup |  |
| S5 recipe, seeds, versions | O5 explanation and its stability |  |

#### [day4#amana-scored] Amana: two items present, five partial, nothing about monitoring

*Lab 4.3*

| Part | Present | Partial | Missing |
| --- | --- | --- | --- |
| Soundness | 2 | 3 | 0 |
| Outcomes | 0 | 2 | 3 |
| Monitoring | 0 | 0 | 2 |

Partial means the package answers part of the item, or answers it in a way you showed to be wrong.

#### [day4#partial-is-wrong] A number in the evidence file is not evidence if it answers a different question

*The hard calls*

- **O1 · partial.** AUC 0.864 is in evidence.json, measured on rows the shipped model trained on (Lab 1.1).
- **O2 · missing.** You computed calibration yourself. The package still lacks it; the report says who supplied the evidence.

#### [day4#request-to-developers] The request names each item and what would satisfy it

*Lab 4.3 · step 4*

| S4 | The reason for each exclusion and when each feature is known |
| --- | --- |
| S5 | Library versions, so the model can be refitted exactly |
| O1 | The shipped model's AUC on rows it never saw |
| M1 | Inputs and outcomes to monitor, thresholds and a named owner |
| M2 | Conditions of use and how officers review declined applicants |

#### [day4#personal-data-in-package] Validation evidence about people is personal data too

*Before you send it*

- Subgroup tables use gender, age band and region of real applicants: Sadeem Al-Oraini, the data protection officer, signs off the report's annexes.
- Report groups and rates, not rows; the stability panel uses application identifiers, never names.
- The Personal Data Protection Law and its implementing regulation apply to the validator's copy as to the developers'.

#### [day4#check-checklist] How do you score O5 for Amana?

*Checkpoint*

**Question:** metadata.json says officers see the three largest Shapley values. evidence.json says stability_checked: false. You tested stability yourself on Day 3. What is O5?

- A. Present, because the stability panel from Day 3 now exists
- B. Missing, because nothing about explanations is there
- C. Partial: the method is stated, its stability is not checked by the developers
- D. Not applicable, because the reasons were unstable

#### [day4#lab-4-3] The package against a checklist

Lab 4.3 · 25 min

Score Amana against a 12-item course checklist drawn from REFORMS and model cards, mark each item present, partial or missing with the file that shows it, and turn the gaps into a request to the developers.

[D4_3_score_model_package.ipynb](https://sdaiaacademy.github.io/ds-305/labs/D4_3_score_model_package.ipynb)

- [ ] Read the checklist (4 min)
- [ ] Score each item with its evidence (9 min)
- [ ] Count by part: soundness, outcomes, monitoring (4 min)
- [ ] Write the request for missing evidence (5 min)
- [ ] Which gap blocks a decision (3 min)
- [ ] ✱ Stretch: Score one of the two capstone packages with the same checklist

### Session 4: The fit-for-use decision

#### [day4#s4] The fit-for-use decision

*Session 4*

- Rate findings by severity and link each to evidence
- Draft a decision with conditions of use and human review
- Choose the capstone package and plan its validation

#### [day4#three-decisions] The committee needs one of three answers, and the conditions that go with it

*The fit-for-use decision*

- **Fit.** Use as documented. Rare for a first submission.
- **Fit with conditions.** Use only while named conditions hold, each with an owner.
- **Not fit.** Do not use. Say what would change the answer.

#### [day4#week-findings] Eleven findings from the week; six are high

*Lab 4.4 · step 1*

| High | Finding |
| --- | --- |
| F2 | Probabilities about four times too high; Platt on calib rows repairs them |
| F3, F4 | Northern region and ages 18 to 24 under-predicted after repair |
| F8 | Reasons shown to officers unstable over refits |
| F10 | App channel bank inflow in the wrong unit from March 2025 |
| F11 | No monitoring, owner, conditions of use or human review |

#### [day4#conditions-with-owners] Every condition says what must hold, who owns it and which findings it answers

*Lab 4.4 · step 2*

| Condition | Owner | Answers |
| --- | --- | --- |
| C1 Use probabilities only after the Platt map | Credit analytics | F2 |
| C2 No app scoring until the feed is fixed; rescore; range check | Data engineering | F10 |
| C3 Northern and 18 to 24 go to an officer | Branch officers | F3, F4 |
| C4 One grouped income reason; only stable reasons | Credit analytics | F8, F9 |
| C5 Monthly monitoring; two breaches return it to validation | Model risk | F11, F5, F7 |

#### [day4#human-review-real] Name which applicants an officer must see, and give the officer the evidence

*Human review*

- **A slogan.** "Borderline cases are reviewed by a human."
- **A rule.** Every application from the Northern region, aged 18 to 24, or with all three grades in its 90% set.

> **Automation Bias** (guide p. 19): officers who see only a score defer to it. They need the grade set, stable reasons, authority to decide differently, and their overrides monitored.

#### [day4#when-not-fit] "Fit with conditions" is only as good as its owners and its monitoring

*Judgement*

- The model ranks well (AUC 0.755) and its probabilities are repaired: the harms found have conditions that remove them while causes are fixed.
- If no one will own C2 or C5, the honest decision is **not fit**.
- A condition without an owner is a wish; a review rule without a list of applicants is a slogan.

#### [day4#capstone-packages] Two packages wait for the capstone; both claim more than they deliver

*Lab 4.4 · steps 4 and 5*

|  | Claimed AUC | Holdout AUC | Mean raw prediction | Default rate |
| --- | --- | --- | --- | --- |
| Amana-R · top-up loans for repeat borrowers | 0.902 | 0.711 | 0.412 | 0.089 |
| Amana-F · seasonal farm loans | 0.892 | 0.700 | 0.425 | 0.128 |

Each has applications, a later batch, three models, a card, metadata and evidence. Choose one, or a package agreed with your peer, and plan its first three checks.

#### [day4#check-decision] Which condition is written well enough to enforce?

*Checkpoint*

**Question:** Four drafts for the approval committee. Which one could an auditor check next year?

- A. Borderline applications receive careful human review by experienced officers in every branch
- B. Model risk checks calibration by region and age monthly; two breaches mean revalidation
- C. The model will be monitored for fairness
- D. The developers retrain the model every year

#### [day4#lab-4-4] The decision and the capstone choice

Lab 4.4 · 25 min

Collect the week's findings, rate them, draft Amana's decision with conditions of use, human review and monitoring, test that every high finding has a condition, and choose the package for the capstone.

[D4_4_draft_decision_conditions.ipynb](https://sdaiaacademy.github.io/ds-305/labs/D4_4_draft_decision_conditions.ipynb)

- [ ] Collect and rate the findings (6 min)
- [ ] Draft the decision and its conditions (8 min)
- [ ] Check every high finding has a condition (4 min)
- [ ] Compare the capstone packages (4 min)
- [ ] Choose one and plan it (3 min)
- [ ] ✱ Stretch: Write the two sentences you would say to the approval committee if it asked to approve without conditions

#### [day4#wrap-up] What a validation report gives the committee

*Day 4 · three things to keep*

- **Supported gaps.** By group, with intervals, tagged with the guide and rated by a rule stated before looking.
- **Causes, not symptoms.** Slice, then inputs, then reasons: a broken feed is not a degraded model.
- **Owned conditions.** A decision the committee can enforce, with monitoring and real human review.

#### [day4#next] Capstone: validate a package and defend the decision

*Next*

Bring the package you chose, your plan's first three checks, and the week's notebooks. Unfinished labs can be finished at home from their reference state.

## Capstone: Capstone: an independent validation

Eight-minute panel defence of a validation report, then four minutes of questions in which the panel picks one finding to rerun

#### [capstone#title] Capstone: an independent validation

*SDAIA Academy · Uncertainty, Interpretability and Model Validation*

Capstone: Eight-minute panel defence of a validation report, then four minutes of questions in which the panel picks one finding to rerun

Data Science Track · 4 teaching days + capstone day · 8 min talk and demo · 4 min Q&A · DS-305

#### [capstone#agenda] How the day runs

*Capstone*

- **The brief and the rubric**: what the committee expects, then a clean-clone check
- **Validation work**: the morning: finish the report on your package
- **Defences**: eight minutes each, four minutes of questions
- **Exit check and closing**: the AI-off exit check, what the room found, what comes next

#### [capstone#the-brief] Validate one package independently and defend a fit-for-use decision

*The brief*

- **Package.** Amana-R (top-up loans) or Amana-F (farm loans) from `labs/capstone/`, or a peer's model with its card, data and a later batch.
- **The week, repeated.** Recompute the claims, calibration, intervals and sets, explanation stability, the subgroup table, the later batch, the checklist.
- **The decision.** Fit, fit with conditions or not fit, with owned conditions and named human review.

- **Submission.** A link to a public GitHub repository holding the validation notebooks, the run logs, and a README that reports calibration, coverage, explanation stability, subgroup findings and the decision.

#### [capstone#report-sections] Seven sections, each with the number that supports it

*The README*

| Section | Evidence | From |
| --- | --- | --- |
| 1 · Claims | Each claim of the card: supported, unsupported or untestable | Lab 1.1 |
| 2 · Calibration | Brier and log loss before and after, reliability diagram | Labs 1.2 to 1.4 |
| 3 · Coverage | Target against achieved, by group and under shift, widths and set sizes | Day 2 |
| 4 · Explanations | Grouped importance, ALE, stability panel, one misleading result | Day 3 |
| 5 · Subgroups | The flagged, tagged and rated table | Lab 4.1 |
| 6 · Later batch | The memo: cause, evidence, action | Lab 4.2 |
| 7 · Checklist and decision | Twelve items scored; decision with owned conditions | Labs 4.3, 4.4 |

#### [capstone#rubric] Five criteria, 100 points

*How the validation is scored*

| Item | Value |
| --- | --- |
| Calibration and coverage | 25% |
| Explanations | 20% |
| Subgroup and shift findings | 25% |
| Decision | 20% |
| Reproducibility | 10% |

> **Full marks.** Every finding has a number, an interval where one applies, and the file that produced it.

#### [capstone#format] Eight minutes before the committee, then four minutes of questions

*The defence*

- **2 min · the package.** What it claims, and which claims failed on your numbers.
- **4 min · the findings.** The high findings, each with its evidence; the later batch's cause.
- **2 min · the decision.** The decision, the conditions and their owners.

> **The panel plays the committee.** In the questions it picks one finding and you rerun the cell that produced it.

#### [capstone#running-order] Who defends when

*Running order*

| # | Participant | Package and repository |
| --- | --- | --- |
| 1 |  |  |
| 2 |  |  |
| 3 |  |  |
| 4 |  |  |
| 5 |  |  |
| 6 |  |  |
| 7 |  |  |
| 8 |  |  |

#### [capstone#final-prep] Ten minutes: clone it clean, run one notebook

*Before the first defence*

Clone your public repository into a new folder and run one notebook from its README. If that fails, the panel's will too.

#### [capstone#now-presenting] Defence

*Now defending*

##### Questions

#### [capstone#review] What the validations showed

*After the defences*

- **Found by most.** Which planted problem did nearly everyone find, and with what evidence?
- **Missed by most.** Which one was quieter, and which check would have caught it?
- **Best condition.** Whose condition would you send to a real committee unchanged?

#### [capstone#what-next] Keep going

*After the course*

- **Follow-up task.** A one-page checklist review of a model from your workplace, or a published one, against the twelve items of Lab 4.3.
- **Keep the repository.** It is the evidence the DS Associate badge looks at.
- **Next Academy course:** DS-309, fairness mitigation and documentation, takes this week's findings to their remedies.

#### [capstone#thanks] Thank you

*SDAIA Academy*

Uncertainty, Interpretability and Model Validation
