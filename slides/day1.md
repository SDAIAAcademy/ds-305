---
course: "Uncertainty, Interpretability and Model Validation"
code: "DS-305"
version: "2027.2.0"
deck: "day1"
day: 1
language: "en"
view: "trainee"
source: "https://sdaiaacademy.github.io/ds-305/slides/day1.html"
---

# Uncertainty, Interpretability and Model Validation

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
