---
course: "Uncertainty, Interpretability and Model Validation"
code: "DS-305"
version: "2027.2.0"
deck: "day2"
day: 2
language: "en"
view: "trainee"
source: "https://sdaiaacademy.github.io/ds-305/slides/day2.html"
---

# Uncertainty, Interpretability and Model Validation

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
