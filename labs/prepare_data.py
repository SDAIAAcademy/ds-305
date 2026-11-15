# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "pandas>=2.2", "lightgbm>=4.5"]
# ///
"""Write the synthetic data and the model packages of Mizan Community Lending (DS-305).

    cd labs
    uv run python prepare_data.py

Seed 305; the same files on every run. Everything is invented: the lender, the applicants and
the models. What it writes:

    data/applications.csv      50,000 applications, 2023 to 2024, with their outcomes and a split column:
                               train 30,000 (the developers' rows), calib 10,000 and holdout 10,000
                               (the validator's rows)
    data/later_batch.csv       10,000 applications from January to June 2025, with outcomes
    package/                   Amana, the package under validation: three LightGBM models as text files,
                               the model card, metadata.json and the developers' evidence
    capstone/renewal/          a second package for the capstone (repeat borrowers' top-up loans)
    capstone/farm/             a third package for the capstone (seasonal farm loans)

The planted problems and the numbers each lab should find are in solutions/answer_key.md.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

LABS = Path(__file__).resolve().parent
SEED = 305

REGIONS = ["Central", "Western", "Eastern", "Southern", "Northern"]
REGION_P = [0.34, 0.27, 0.20, 0.14, 0.05]
REGION_RISK = {"Central": 0.0, "Western": 0.05, "Eastern": -0.05, "Southern": 0.1, "Northern": 0.85}
AGES = ["18-24", "25-34", "35-44", "45-54", "55+"]
AGE_P = [0.12, 0.33, 0.28, 0.17, 0.10]
AGE_RISK = {"18-24": 0.55, "25-34": 0.12, "35-44": 0.0, "45-54": -0.05, "55+": 0.25}
BUSINESS = ["retail", "food", "services", "crafts", "agriculture"]
BUSINESS_P = [0.30, 0.24, 0.22, 0.12, 0.12]
BUSINESS_RISK = {"retail": 0.05, "food": 0.0, "services": -0.1, "crafts": 0.2, "agriculture": 0.3}

FEATURES = ["years_in_business", "household_size", "prior_loans", "prior_missed", "declared_income",
            "bank_inflow", "loan_amount", "term_months", "business_type", "channel"]
CATEGORICAL = ["business_type", "channel"]


def applicants(rng: np.random.Generator, n: int, months: list[str], *, base: float = -2.55,
               income_scale: float = 1.0, inflow_feed_fault_from: str | None = None,
               missed_feed_fault_from: str | None = None,
               farm: bool = False, renewal: bool = False) -> pd.DataFrame:
    """Draw n applications and their outcomes. All risk effects are on the log-odds scale."""
    region = rng.choice(REGIONS, n, p=REGION_P)
    age = rng.choice(AGES, n, p=AGE_P)
    age_i = np.array([AGES.index(a) for a in age])
    gender = np.where(rng.random(n) < 0.58, "F", "M")
    app_p = np.array([0.55, 0.45, 0.35, 0.25, 0.15])[age_i]
    channel = np.where(rng.random(n) < app_p, "app", "branch")
    business = rng.choice(BUSINESS, n, p=BUSINESS_P)
    if farm:
        business = np.full(n, "agriculture")
    years = np.clip(np.round(rng.gamma(1.2 + 0.9 * age_i, 2.2)), 0, 35).astype(int)
    household = np.clip(rng.poisson(4.2, n), 1, 12)
    prior_loans = np.clip(rng.poisson(0.4 + 0.5 * age_i), 0, 9)
    if renewal:
        prior_loans = prior_loans + 1
    prior_missed = rng.binomial(prior_loans, 0.12)
    wallet = np.clip(np.round(rng.gamma(2.0, 6 + 10 * (4 - age_i) / 4 * 0.2 + 4 * (4 - age_i))), 0, 96)
    wallet = np.where(channel == "app", wallet + 6, wallet).astype(int)

    true_income = np.exp(rng.normal(8.75 + 0.07 * age_i + 0.015 * years, 0.38)) * income_scale
    inflow = true_income * np.exp(rng.normal(0, 0.12, n))
    overstate = np.where(rng.random(n) < 0.16, rng.uniform(0.25, 0.75, n), 0.0)
    declared = true_income * np.exp(rng.normal(0.03, 0.10, n) + overstate)
    term = rng.choice([6, 12, 18, 24], n, p=[0.15, 0.45, 0.25, 0.15])
    loan = np.clip(np.round(true_income * rng.uniform(1.2, 4.0, n) * (term / 12) ** 0.5, -2), 3000, 90000)
    pti = loan / term / true_income                       # share of monthly income the instalment takes

    z = (base + 9.0 * (pti - 0.2) - 0.45 * np.log1p(years) + 0.85 * prior_missed - 0.12 * np.minimum(prior_loans, 4)
         + 3.2 * np.maximum(np.log(declared / inflow) - 0.12, 0)
         + np.vectorize(BUSINESS_RISK.get)(business) + np.vectorize(REGION_RISK.get)(region)
         + np.vectorize(AGE_RISK.get)(age) - 0.08 * (gender == "F") + 0.12 * (channel == "app")
         + 0.03 * (household - 4))
    p =1 / (1 + np.exp(-z))
    default = (rng.random(n) < p).astype(int)
    late = (rng.random(n) < 1 / (1 + np.exp(-(1.3 * z + 1.9)))).astype(int)
    grade = np.where(default == 1, "C", np.where(late == 1, "B", "A"))

    # loss given default: a share of the loan, wider for large loans and in the North
    size = np.log(loan / 15000)
    spread = 0.45 + 0.35 * np.clip(size, 0, None) + 0.35 * (region == "Northern")
    share = 1 / (1 + np.exp(-(0.15 + 0.25 * (business == "agriculture") - 0.15 * np.log1p(years) / 2
                              + rng.normal(0, spread))))
    loss = np.where(default == 1, np.round(loan * share, -1), np.nan)

    month = np.array(months)[rng.integers(0, len(months), n)]
    month.sort()
    df = pd.DataFrame({
        "month": month, "region": region, "age_band": age, "gender": gender, "channel": channel,
        "business_type": business, "years_in_business": years, "household_size": household,
        "prior_loans": prior_loans, "prior_missed": prior_missed, "wallet_months": wallet,
        "declared_income": np.round(declared, -1), "bank_inflow": np.round(inflow, -1),
        "loan_amount": loan, "term_months": term, "grade": grade, "default": default, "loss_sar": loss,
    })
    if inflow_feed_fault_from:
        # the seeded cause: a new feed reports the app channel's bank inflow in thousands of riyals
        hit = (df.month >= inflow_feed_fault_from) & (df.channel == "app")
        df.loc[hit, "bank_inflow"] = np.round(df.loc[hit, "bank_inflow"] / 1000, 2)
    if missed_feed_fault_from:
        # a feed fault: the app channel's missed-payment history stops loading and arrives as zero
        hit = (df.month >= missed_feed_fault_from) & (df.channel == "app")
        df.loc[hit, "prior_missed"] = 0
    return df


def X(df: pd.DataFrame) -> pd.DataFrame:
    out = df[FEATURES].copy()
    out["business_type"] = out.business_type.map({b: i for i, b in enumerate(BUSINESS)}).astype(int)
    out["channel"] = (out.channel == "app").astype(int)
    return out


PARAMS = dict(learning_rate=0.05, num_leaves=31, min_data_in_leaf=40, feature_fraction=0.9,
              bagging_fraction=0.9, bagging_freq=1, lambda_l2=1.0, seed=SEED, deterministic=True,
              num_threads=1, force_col_wise=True, verbose=-1)


def train_default(train: pd.DataFrame, seed: int = SEED, rounds: int = 300, rebalance: bool = True,
                  features: list[str] | None = None) -> lgb.Booster:
    """The developers' recipe: rebalance to 50/50 by dropping non-defaults, then boost."""
    rng = np.random.default_rng(seed)
    rows = train
    if rebalance:
        pos = train[train["default"] == 1]
        neg = train[train["default"] == 0]
        neg = neg.iloc[np.sort(rng.choice(len(neg), len(pos), replace=False))]
        rows = pd.concat([pos, neg]).sort_index()
    feats = features or FEATURES
    data = X(rows) if features is None else rows[feats]
    ds = lgb.Dataset(data, rows["default"], categorical_feature=[c for c in CATEGORICAL if c in feats],
                     free_raw_data=False)
    return lgb.train({**PARAMS, "objective": "binary", "seed": seed}, ds, num_boost_round=rounds)


def write_package(folder: Path, train: pd.DataFrame, holdout: pd.DataFrame, card: str, meta: dict,
                  evidence: dict, grade_and_loss: bool = True) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    m = train_default(train)
    prefix = "amana_" if folder.name == "package" else "model_"
    m.save_model(folder / f"{prefix}default.txt")
    if grade_and_loss:
        gcode = train.grade.map({"A": 0, "B": 1, "C": 2})
        g = lgb.train({**PARAMS, "objective": "multiclass", "num_class": 3},
                      lgb.Dataset(X(train), gcode, categorical_feature=CATEGORICAL), num_boost_round=250)
        g.save_model(folder / f"{prefix}grade.txt")
        d = train[train["default"] == 1]
        loss = lgb.train({**PARAMS, "objective": "regression", "min_data_in_leaf": 30},
                         lgb.Dataset(X(d), d.loss_sar, categorical_feature=CATEGORICAL), num_boost_round=250)
        loss.save_model(folder / f"{prefix}loss.txt")
    # the developers' own headline numbers, measured on their holdout sample
    p = m.predict(X(holdout))
    evidence = {**evidence, "auc_holdout": round(auc(holdout["default"].to_numpy(), p), 3),
                "holdout_rows": int(len(holdout))}
    (folder / "model_card.md").write_text(card.format(**evidence), encoding="utf-8", newline="\n")
    (folder / "metadata.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8", newline="\n")
    (folder / "evidence.json").write_text(json.dumps(evidence, indent=1) + "\n", encoding="utf-8", newline="\n")


def auc(y: np.ndarray, p: np.ndarray) -> float:
    ranks = pd.Series(p).rank(method="average").to_numpy()   # average ranks of ties
    n1 = y.sum()
    n0 = len(y) - n1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


AMANA_CARD = """# Model card: Amana default model, version 2.1

Prepared by the credit analytics team of Mizan Community Lending. Synthetic, for training only.

## Intended use

Scores new microfinance applications with the probability of default within 12 months of
disbursement. Branch credit officers see the score and the top three reasons with each application.

## Claims

1. The model discriminates well: AUC {auc_holdout} on a held-out sample of {holdout_rows} applications.
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
"""


def main() -> None:
    rng = np.random.default_rng(SEED)
    data = LABS / "data"
    data.mkdir(exist_ok=True)
    months = [f"{y}-{m:02d}" for y in (2023, 2024) for m in range(1, 13)]
    apps = applicants(rng, 50_000, months)
    apps.insert(0, "app_id", [f"M{i:06d}" for i in range(1, len(apps) + 1)])
    split = np.array(["train"] * 30_000 + ["calib"] * 10_000 + ["holdout"] * 10_000)
    rng.shuffle(split)
    apps.insert(2, "split", split)
    apps.to_csv(data / "applications.csv", index=False, lineterminator="\n")

    later_months = [f"2025-{m:02d}" for m in range(1, 7)]
    later = applicants(rng, 10_000, later_months, income_scale=1.08, inflow_feed_fault_from="2025-03")
    later.insert(0, "app_id", [f"L{i:06d}" for i in range(1, len(later) + 1)])
    later.to_csv(data / "later_batch.csv", index=False, lineterminator="\n")

    train = apps[apps.split == "train"]
    dev_holdout = train.sample(n=4000, random_state=SEED)       # the developers held out part of their own rows
    dev_train = train.drop(dev_holdout.index)
    meta = {
        "package": "Amana", "version": "2.1", "owner": "Credit analytics team, Mizan Community Lending",
        "target": "default within 12 months of disbursement (1) or not (0)",
        "features": FEATURES, "categorical": {"business_type": BUSINESS, "channel": ["branch", "app"]},
        "excluded_on_purpose": ["gender", "age_band", "region", "wallet_months"],
        "training_rows": "applications 2023-01 to 2024-12, split == train",
        "training_settings": {"rebalanced": "non-defaults dropped at random to a 50/50 sample",
                              "library": "LightGBM", "rounds": 300, **{k: v for k, v in PARAMS.items()
                                                                     if k not in ("verbose", "num_threads")}},
        "outputs": {"amana_default.txt": "probability of default", "amana_grade.txt": "probabilities of grades A, B, C",
                    "amana_loss.txt": "loss in riyals if the loan defaults"},
        "reasons_shown_to_officers": "the three features with the largest positive Shapley values",
        "monitoring": "none defined",
    }
    evidence = {"metric": "AUC", "evaluated_on": "a random 4,000 of the training rows, held out before training",
                "calibration_checked": False, "subgroup_results": None, "stability_checked": False}
    pkg = LABS / "package"
    if pkg.exists():
        shutil.rmtree(pkg)
    # the package itself is trained on all training rows; its evidence was measured by the developers
    write_package(pkg, train, dev_holdout, AMANA_CARD, meta, evidence)

    # capstone packages
    cap = LABS / "capstone"
    for name, kw, n in (("renewal", dict(renewal=True, base=-2.75), 16_000),
                        ("farm", dict(farm=True, base=-2.35), 12_000)):
        folder = cap / name
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True)
        d = applicants(rng, n, months, **kw)
        d.insert(0, "app_id", [f"{name[0].upper()}{i:06d}" for i in range(1, n + 1)])
        s = np.array(["train"] * (n * 6 // 10) + ["calib"] * (n * 2 // 10) + ["holdout"] * (n - n * 8 // 10))
        rng.shuffle(s)
        d.insert(2, "split", s)
        later_kw = dict(kw)
        if name == "farm":
            later_kw["base"] = kw["base"] + 0.45                # a dry season: risk rises for every farm loan
        if name == "renewal":
            later_kw["missed_feed_fault_from"] = "2025-04"     # missed-payment history lost for the app channel
        lb = applicants(rng, n // 4, later_months, income_scale=1.05, **later_kw)
        lb.insert(0, "app_id", [f"{name[0].upper()}L{i:05d}" for i in range(1, len(lb) + 1)])
        d.to_csv(folder / "applications.csv", index=False, lineterminator="\n")
        lb.to_csv(folder / "later_batch.csv", index=False, lineterminator="\n")
        tr = d[d.split == "train"]
        title = {"renewal": "Amana-R top-up model, version 1.0", "farm": "Amana-F seasonal farm model, version 1.0"}[name]
        card = AMANA_CARD.replace("Amana default model, version 2.1", title)
        m2 = dict(meta, package=title.split(",")[0], version="1.0",
                  outputs={"model_default.txt": "probability of default", "model_grade.txt": "probabilities of grades A, B, C",
                           "model_loss.txt": "loss in riyals if the loan defaults"})
        ev2 = dict(evidence, evaluated_on="a random 2,000 of the training rows, held out before training")
        write_package(folder, tr, tr.sample(n=2000, random_state=SEED), card, m2, ev2)
    print("applications:", len(apps), "default rate", round(apps["default"].mean(), 4),
          "| later batch:", len(later), "default rate", round(later["default"].mean(), 4))


if __name__ == "__main__":
    main()
