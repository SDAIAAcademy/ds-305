---
course: "Uncertainty, Interpretability and Model Validation"
code: "DS-305"
version: "2027.2.0"
deck: "day4"
day: 4
language: "en"
view: "trainee"
source: "https://sdaiaacademy.github.io/ds-305/slides/day4.html"
---

# Uncertainty, Interpretability and Model Validation

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
- Replay the same applications through the training and serving paths to tell a broken path from a changed population
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

> **Use it.** Reasons exist on the day of scoring, outcomes twelve months later: a sudden change in reasons is an early alarm.

#### [day4#rule-out-decoy] Incomes also rose, in both channels; the branch channel is unharmed

*Lab 4.2 · step 4*

- Median declared income +9.3% (app) and +8.7% (branch).
- Branch channel 2025 AUC 0.747, as before.

- **Model drift or a broken input?** The guide's Model Drift (p. 66) and Concept Drift (p. 28) describe a changing world. This is Measurement Bias (p. 63): values recorded too low.

#### [day4#cause-memo] The memo: one cause, three numbers, one action, one thing not to do

*Lab 4.2 · step 5*

| **Cause** | From March 2025 the app channel's bank inflow arrives about a thousand times too small: a unit change in the data feed. |
| --- | --- |
| **Evidence** | App from March: 46% predicted against 10% observed; PSI of bank inflow 8.3; bank inflow the top reason for 98% of the slice. |
| **Action** | Fix the feed to riyals, rescore March to June, add an input range check before scoring. |
| **Not** | Do not retrain on the 2025 batch. |

#### [day4#replay-same-records] PSI compares different applicants; a replay compares the same applicant twice

*Training and serving features*

- **Training path.** The job that built the training table, run again on the source records.
- **Serving path.** The values the scoring service received and logged on the day.

- **Drift:** different applicants, different values; the two paths agree.
- **Skew:** the same applicant gets different values from the two paths.
- **Replay:** take scored applications, rebuild them through the training path, compare column by column with a tolerance fixed before looking.

#### [day4#replay-diff-table] Before go-live, 200 replayed applications: the loan amount is 2% higher on every row

*Parallel example · a new branch tablet form*

| Input | Differ | Over tolerance | Serving ÷ training |
| --- | --- | --- | --- |
| Loan amount | 200 | 200 | 1.020 |
| Declared income | 183 | 0 | 1.000 |
| Six others | 0 | 0 | 1.000 |

> **Monitoring would miss it.** PSI of the served loan amount is 0.046, against 0.036 for the training values of the same rows. Yet 25 of 200 predictions rise by more than a point.

#### [day4#read-a-replay] The pattern of differences says which path to fix

*Reading a replay*

| Pattern | Likely cause | Action |
| --- | --- | --- |
| Every row, one constant ratio | A unit or a definition in one path | Fix that path, rescore |
| Some rows, one category or source | A code list, a join, a lookup | Fix the mapping, rescore those rows |
| Within tolerance only | Different precision | Nothing |
| No row differs, the distribution moved | The population changed | Review the model, not the feed |

> **Fix the serving path.** The training definition is the one the model learned.

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
- [ ] ✱ Stretch: Write the input check that would have stopped the batch on the first day, then replay 60 app applications through both feature paths

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
- **Causes, not symptoms.** Slice, inputs, reasons, then a replay: a broken feed is not a degraded model.
- **Owned conditions.** A decision the committee can enforce, with monitoring and real human review.

#### [day4#next] Capstone: validate a package and defend the decision

*Next*

Bring the package you chose, your plan's first three checks, and the week's notebooks. Unfinished labs can be finished at home from their reference state.
