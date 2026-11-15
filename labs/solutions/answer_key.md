# Answer key for trainers

DS-305 Uncertainty, Interpretability and Model Validation. Everything here is synthetic:
`labs/prepare_data.py`, seed 305. The numbers below are those of the solution runs in quick mode
(`MIZAN_QUICK=1`); the full mode changes only the labs that refit models (3.3, 3.4) and the number of
repeats (3.1), by a few units in the third decimal.

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
