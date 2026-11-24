---
course: "Uncertainty, Interpretability and Model Validation"
code: "DS-305"
version: "2027.2.0"
deck: "day3"
day: 3
language: "en"
view: "trainee"
source: "https://sdaiaacademy.github.io/ds-305/slides/day3.html"
---

# Uncertainty, Interpretability and Model Validation

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
