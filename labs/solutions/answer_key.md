# Answer key for trainers

DS-305 Uncertainty, Interpretability and Model Validation. Everything here is synthetic:
`labs/prepare_data.py`, seed 305. The numbers below are those of the solution runs in full mode, which is
what participants run. Quick mode (`MIZAN_QUICK=1`) changes only Labs 3.1 to 3.4 (fewer repeats, a smaller
sample, refits with fewer rounds); their quick-mode numbers are given where they differ.

## What is planted

| Planted problem | Where | Found in |
|---|---|---|
| The default model was trained on a 50/50 rebalanced sample: probabilities about four times too high | `package/metadata.json`, `training_settings.rebalanced` | Labs 1.1, 1.2, 1.3 |
| The claimed AUC (0.864) was measured on 4,000 training rows the shipped model was then retrained on | `package/evidence.json` against `metadata.json` | Lab 1.1 |
| Region, age band and gender are not features; default risk is higher in the small Northern region (5% of applicants) and in the 18 to 24 band | the generator's risk terms | Labs 1.4, 4.1 |
| Loss given default is wider for large loans and in the North (heteroscedastic) | the generator's loss spread | Labs 2.1, 2.3, 2.4, 4.1 |
| Declared income and bank inflow correlate (about 0.84 on the log scale); overstating income (declared far above inflow) raises risk | the generator's risk terms | Labs 3.1, 3.2, 3.3 |
| Mobile-wallet months (not a feature of Amana) is longer for younger applicants and works as an age proxy when added | the generator | Lab 3.4 |
| Later batch: from 2025-03, bank inflow of the app channel arrives in thousands of riyals (a feed change); a harmless nominal income rise of 8% is a decoy | `data/later_batch.csv` | Lab 4.2 |
| Amana has no subgroup results, no calibration evidence, no stability check, no monitoring plan | `package/evidence.json`, `metadata.json` | Lab 4.3 |

## Day 1

**Lab 1.1.** Verdicts: claim 1 unsupported (holdout AUC 0.755, not 0.864; AUC on the training rows
0.858); claim 2 unsupported (mean prediction 0.403 against a default rate of 0.099; trained on a 50/50
sample); claim 3 supported as written (no gender, age band or region column among the features);
claim 4 unsupported (no subgroup evidence); claims 5 and 6 untestable. Trap: marking claim 3
unsupported because of proxies; as written it is supported, and Lab 3.4 is where the proxy shows.

**Lab 1.2.** Holdout: Brier 0.1986, log loss 0.5858, AUC 0.7555; base-rate forecast Brier 0.0895, log
loss 0.3241. Ten bins of 1,000: lowest 0.076 predicted against 0.020 observed, highest 0.854 against
0.314. ECE 0.304. Prior correction: Brier 0.0815. Murphy decomposition on the bins: reliability 0.117,
resolution 0.008, uncertainty 0.089.

**Lab 1.3.** Platt: Brier 0.0809, log loss 0.2840, ECE 0.0105, AUC 0.7555 (unchanged), mean predicted
0.090. Isotonic: Brier 0.0809, log loss 0.2866, ECE 0.0115, AUC 0.7536. Isotonic fitted and scored on
the holdout: ECE 0.0016. With 500 calib rows: log loss Platt 0.2861, isotonic 0.3060. Either choice is
accepted with a reason; the reference repair (`U.recalibrated`) is Platt on the calib rows.

**Lab 1.4.** After Platt, by region: Northern 508 rows, 8.66% predicted, 15.35% observed, interval
12.48 to 18.75, outside; the other regions inside. By age band: 18-24, 1,201 rows, 12.71% predicted,
17.49% observed, interval 15.44 to 19.74, outside; the others inside. Stretch: men 8.89% against 10.48%
(interval 9.59 to 11.44, just outside), branch channel 8.21% against 9.23% (just outside). The worst
group is the Northern region.

## Day 2

**Lab 2.1.** 895 calib and 993 holdout defaults. Threshold q = 8,134 SAR; coverage 91.0%; mean width
15,667 SAR (median loss 10,520). MAPIE's half-width equals q. Stretch: 100 random splits of the 1,888
defaults outside the training rows, mean coverage 0.900, range 0.867 to 0.935.

**Lab 2.2.** Top grade right for 74.4%. LAC threshold 0.834 (grade probability at least 0.166):
coverage 89.2%, mean size 1.56 (sizes 1, 2, 3: 56.7, 30.6, 12.7%). APS: 90.3%, mean size 1.67 (43.1,
46.4, 10.5%). By true grade (LAC / APS): A 99.8 / 99.4%, B 70.7 / 79.6%, C 44.9 / 43.0%. 1,046 applicants
get all three grades; their default rate is 24.8% against 8.2%. Stretch: mean APS size 1.33 at 80%,
1.67 at 90%, 2.03 at 95%. Trap: claiming APS repairs coverage of the defaulters; it does not.

**Lab 2.3.** One threshold: large 76.7% (382 defaults), medium and small 100%; Northern region 87.2%
on 78 defaults. Per-band thresholds: large 13,256, medium 4,332, small 1,536 SAR (calib defaults 340,
476, 79). After: large 90.6%, medium 92.5%, small 86.1%; overall 91.2%; large-loan width 16,268 to
26,422 SAR. Stretch (normalised score): large 86.1%, medium 97.6%, small 97.5%.

**Lab 2.4.** Loss interval coverage at strengths 0, 0.5, 1, 1.5: 91.0, 88.4, 83.2, 77.1%; large-loan
share 38.5, 47.5, 58.1, 69.4%; width moves less than 5%. Grade sets: 89.2, 89.3, 88.7, 88.0%. Within
large loans at strength 1: 71.1%. Weighted conformal at strength 1: 93.6%.

## Day 3

**Lab 3.1.** Validator's model (Platt on calib), holdout log loss 0.2840; correlation of the log incomes
0.84. Ten repeats: bank inflow 0.0677 (0.0639 to 0.0717), declared income 0.0455, term 0.0326, loan
amount 0.0253, prior missed 0.0085, years in business 0.0078, prior loans 0.0021, business type 0.0014,
household size and channel ranges include zero. Both incomes shuffled together: 0.0339 (0.0301 to
0.0368). Real declared-to-inflow ratio: 98% between 0.72 and 2.36; after shuffling declared income, 36%
of rows fall outside. The order is the same with the Brier score. Quick mode (five repeats): same order.
Trap: ranking bank inflow above declared income.

**Lab 3.2.** 4,000 holdout applicants (quick: 2,000). From 9,070 to 18,930 SAR, PD rises by 0.166 and
ALE by 0.106 (quick: 0.211 and 0.132). Share of PD rows outside the real ratio range: 75% at 4,310 SAR,
15% at 10,200, 53% at 18,930. ICE: 37 of 50 applicants move more than 10 points (quick: 43). The report
uses ALE with a caption that gives both rises.

**Lab 3.3.** Ten holdout applicants, drawn with seed 305. Values add up to the raw score.
Over five bootstrap refits, 8 of 10 keep their top reason; M004026 (bank inflow or prior missed) and
M008361 (loan amount or years in business) change. Same set of three reasons: 1 of 10 (quick: 2). Widest
range of one income value for one applicant 3.35 log-odds, of the two incomes' sum 1.70.

**Lab 3.4.** Refits with seed 1 on all training rows: AUC 0.757 without wallet months, 0.761 with
(quick: 0.767 and 0.768). Wallet months among the three reasons: 26.5% overall; by age band 20.6, 21.3,
25.1, 30.7, 46.5% (18-24 to 55+); mean value for 55+ +0.073 log-odds. Correlation with age band -0.44;
median wallet months 43, 37, 28, 19, 12. Age-band calibration gap after Platt, without and with: 18-24
4.9 and 4.7 points, 55+ 1.2 and 0.8.

## Day 4

**Lab 4.1.** Recalibrated mean 0.090 against 0.099; loss threshold 8,134 SAR. Flagged gaps: Northern
calibration +6.7 points (15.4% observed, 8.7% predicted, interval 12.5 to 18.8; 508 applicants);
18-24 calibration +4.8 points (17.5% against 12.7%, interval 15.4 to 19.7); 55+ loss coverage 83.1% on 83
defaults; men +1.6 points; branch channel +1.0 point; large loans loss coverage 76.7% on 382 defaults.
Ratings: Northern and 18-24 high; the rest medium. Defensible tags: Northern (Aggregation, Coverage or
Sample Size Bias), 18-24 (Age or Aggregation Bias), large loans (Evaluation or Overconfidence Bias).
Stretch: Northern by age band, 45 to 183 applicants and 6 to 30 defaults per cell, intervals 9 to 27
points wide; no finding per cell. Quick mode gives the same numbers.

**Lab 4.2.** Holdout AUC 0.755; later batch AUC 0.677, mean predicted 0.176 against 0.092 observed. App
channel from March 2025: predicted 46% against 9 to 11% observed, AUC 0.53 to 0.68; branch unharmed
(AUC 0.747 for 2025). PSI in the failing slice: bank inflow 8.28, declared income 0.037, the rest below
0.02. Median app bank inflow 8,065 and 8,240 SAR in January and February, about 8 from March. Bank
inflow is the top reason for 98.4% of the failing slice against 21.6% of holdout app applicants. Loss
coverage 92.4% (holdout app), 94.8% (failing slice). Decoy: declared incomes rose 9% in both channels.
Stretch: the input check alerts on bank inflow (and loan amount) in the first 40 March app rows, nothing
in February. Stretch replay (`data/replay_app.csv`, 60 app applications through both paths): nothing
differs beyond tolerance in February; in March bank inflow differs on 40 of 40 at a median serving to
training ratio of 0.001, declared income on most rows by at most 5 SAR (within tolerance): fix the serving
feed. Cause: unit change in the app channel's bank inflow feed. Bias type: Measurement Bias.

**Lab 4.3.** Key: S1 partial, S2 present, S3 present, S4 partial, S5 partial, O1 partial, O2 missing, O3
missing, O4 missing, O5 partial, M1 missing, M2 missing. By part: soundness 2 present, 3 partial;
outcomes 2 partial, 3 missing; monitoring 2 missing. Blocking gap: M1 (M2 accepted with a reason).

**Lab 4.4.** High findings: F2, F3, F4, F8, F10, F11. Reference decision: fit with conditions, five
conditions with owners. Capstone first look: renewal claimed AUC 0.902, holdout 0.711, mean raw
prediction 0.412 against 0.089, later batch AUC 0.713 with 0.081 predicted against 0.080 observed; farm
claimed 0.892, holdout 0.700, mean raw 0.425 against 0.128, later batch AUC 0.686 with 0.140 predicted
against 0.173 observed.

## Capstone packages (trainer only)

Both packages share Amana's planted problems: the claimed AUC was measured on training rows, the
probabilities come from a 50/50 rebalanced sample, and no calibration, subgroup or stability evidence is
supplied. In addition:

- **Amana-R (renewal).** From April 2025 the app channel's prior-missed-payments feed stops arriving
  (recorded as zero), so repeat borrowers who missed payments look clean. It is quieter than Amana's
  fault: the overall later batch is calibrated (0.081 against 0.080), but the app channel's AUC falls
  from about 0.70 to 0.64 by June while its mean prior missed payments drop from about 0.25 to 0.00
  (PSI 1.6). The slice-by-month table and the PSI of prior missed payments find it.
- **Amana-F (farm).** The later batch comes from a dry season: default rises for everyone (17.3% observed
  against 14.0% predicted) with no change in the inputs, a shift in the base rate rather than a broken
  feed. The Northern region is under-predicted (24.0% observed against 15.1%). A good report says that
  recalibration on recent matured loans, not a feed fix, is the remedy, and that the North needs review.
