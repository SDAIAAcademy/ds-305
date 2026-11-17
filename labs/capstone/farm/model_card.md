# Model card: Amana-F seasonal farm model, version 1.0

Prepared by the credit analytics team of Mizan Community Lending. Synthetic, for training only.

## Intended use

Scores new microfinance applications with the probability of default within 12 months of
disbursement. Branch credit officers see the score and the top three reasons with each application.

## Claims

1. The model discriminates well: AUC 0.892 on a held-out sample of 2000 applications.
2. The predicted probabilities are well calibrated and can be read as default rates.
3. The model uses no gender, age or location information.
4. Performance is consistent across all customer groups.
5. The model will remain accurate for the next three years.
6. The reasons shown to officers are the causes of an applicant's risk.

## Training

Applications from January 2023 to December 2024 (training rows only). See metadata.json for the
features and the training settings.

## Companion models

- Grade model: on time (A), late but repaid (B) or default (C).
- Loss model: the loss in riyals if a loan defaults, trained on defaulted loans.
