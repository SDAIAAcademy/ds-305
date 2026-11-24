"""Replay: the same applications through the training path and the serving path (DS-305, Day 4 Session 2).

    cd labs
    uv run python _tools/replay_examples.py

Writes data/replay_app.csv for the stretch of Lab 4.2 and prints the numbers of the two parallel
examples (the slides on replay and the practice question). Seed 3052; reads data/ only, so
prepare_data.py and every other file stay as they are. All values are synthetic.

The training path is the warehouse job that built Amana's training table: it rebuilds each feature from
the source records, in riyals, to the riyal. The serving path is what the scoring service received and
logged on the day. A replay takes applications that were scored, runs them through both paths and
compares the values row by row: the records are the same, so any difference is in the paths.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

LABS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LABS))
import mizan as U  # noqa: E402

SEED = 3052
NUMERIC = [f for f in U.FEATURES if f not in U.CATEGORICAL]
# the course's tolerance, fixed before looking: money within 0.5% (the feed rounds to tens), counts exact
TOL = {f: (0.005 if f in ("declared_income", "bank_inflow", "loan_amount") else 0.0) for f in NUMERIC}


def differs(train: pd.Series, serve: pd.Series, rel: float) -> pd.Series:
    t, s = train.astype(float), serve.astype(float)
    return (s - t).abs() > rel * t.abs()


def summary(train: pd.DataFrame, serve: pd.DataFrame, cols) -> pd.DataFrame:
    rows = []
    for f in cols:
        if f in NUMERIC:
            d = differs(train[f], serve[f], TOL[f])
            any_diff = (train[f].astype(float) != serve[f].astype(float))
            ratio = (serve[f].astype(float) / train[f].astype(float)).replace([np.inf, -np.inf], np.nan)
            rows.append({"feature": f, "rows": len(train), "differ_at_all": int(any_diff.sum()),
                         "beyond_tolerance": int(d.sum()),
                         "median_ratio": float(ratio[d].median()) if d.any() else 1.0})
        else:
            d = train[f] != serve[f]
            rows.append({"feature": f, "rows": len(train), "differ_at_all": int(d.sum()),
                         "beyond_tolerance": int(d.sum()), "median_ratio": np.nan})
    return pd.DataFrame(rows).set_index("feature")


def cat_psi(ref: pd.Series, cur: pd.Series) -> float:
    pe = ref.value_counts(normalize=True)
    pa = cur.value_counts(normalize=True).reindex(pe.index, fill_value=0)
    pe, pa = np.clip(pe, 1e-4, None), np.clip(pa, 1e-4, None)
    return float(np.sum((pa - pe) * np.log(pa / pe)))


def lab_file(rng: np.random.Generator) -> pd.DataFrame:
    """Lab 4.2 stretch: 20 February and 40 March app applications of the 2025 batch, through both paths."""
    later = U.load("later")
    app = later[later.channel == "app"]
    pick = pd.concat([app[app.month == "2025-02"].head(20), app[app.month == "2025-03"].head(40)])
    serve = pick.copy()
    train = pick.copy()
    # the warehouse keeps bank inflow in riyals; the new app feed of March sends thousands
    hit = train.month >= "2025-03"
    train.loc[hit, "bank_inflow"] = np.round(train.loc[hit, "bank_inflow"] * 1000, -1)
    # the warehouse keeps declared income to the riyal; the feed rounds it to tens
    train["declared_income"] = train["declared_income"] + rng.integers(-5, 5, len(train))
    cols = ["app_id", "month"] + NUMERIC + ["business_type"]
    out = pd.concat([train[cols].assign(path="training"), serve[cols].assign(path="serving")])
    out = out[["app_id", "month", "path"] + NUMERIC + ["business_type"]]
    return out.sort_values(["app_id", "path"], ascending=[True, False]).reset_index(drop=True)


def slide_example(rng: np.random.Generator) -> None:
    """Slides: before a new branch tablet form goes live, 200 holdout branch applications are replayed
    through it. The form adds the 2% administration fee to the loan amount; everything else agrees."""
    hold = U.load("holdout")
    br = hold[hold.channel == "branch"]
    sample = br.sample(n=200, random_state=SEED)
    train = sample.copy()
    serve = sample.copy()
    serve["loan_amount"] = np.round(train["loan_amount"] * 1.02, -1)
    # the warehouse keeps declared income to the riyal; the form rounds it to tens, as the data files do
    train["declared_income"] = train["declared_income"] + rng.integers(-5, 5, len(train))
    s = summary(train, serve, NUMERIC)
    print("SLIDE EXAMPLE: 200 holdout branch applications, training path against the new form")
    print(s.round(4).to_string())
    rest = br.drop(sample.index)
    print(f"PSI loan_amount against the other branch holdout rows: serving {U.psi(rest.loan_amount, serve.loan_amount):.3f}, "
          f"training {U.psi(rest.loan_amount, train.loan_amount):.3f}")
    di = (serve.declared_income - train.declared_income).abs()
    print(f"declared income: largest absolute difference {di.max():.0f} SAR, largest relative {(di / train.declared_income).max():.4f}")
    pt, ps = U.recalibrated(train), U.recalibrated(serve)
    print(f"mean recalibrated prediction: training {pt.mean():.4f}, serving {ps.mean():.4f}; "
          f"rows whose prediction rises: {(ps > pt + 1e-12).sum()}, by more than 1 point: {(ps - pt > 0.01).sum()}; "
          f"largest rise {(ps - pt).max():.4f}")


def question_example(rng: np.random.Generator) -> None:
    """Practice question: 150 applications replayed after a code-list update in the serving path
    that maps the business type crafts to retail."""
    hold = U.load("holdout")
    sample = hold.sample(n=150, random_state=SEED + 1)
    train = sample.copy()
    serve = sample.copy()
    serve.loc[serve.business_type == "crafts", "business_type"] = "retail"
    s = summary(train, serve, NUMERIC + ["business_type"])
    print("\nQUESTION EXAMPLE: 150 holdout applications, a code-list update in the serving path")
    print(s.to_string())
    pairs = pd.crosstab(train.business_type, serve.business_type)
    print(pairs.to_string())
    rest = hold.drop(sample.index)
    print(f"categorical PSI of business_type against the other holdout rows: serving {cat_psi(rest.business_type, serve.business_type):.3f}, "
          f"training {cat_psi(rest.business_type, train.business_type):.3f}")


def main() -> None:
    rng = np.random.default_rng(SEED)
    df = lab_file(rng)
    df.to_csv(U.DATA / "replay_app.csv", index=False, lineterminator="\n")
    tr = df[df.path == "training"].set_index("app_id")
    sv = df[df.path == "serving"].set_index("app_id").loc[tr.index]
    print("LAB FILE: data/replay_app.csv,", len(tr), "applications")
    for m in ("2025-02", "2025-03"):
        k = tr.month == m
        print(m, "\n", summary(tr[k], sv[k], NUMERIC).round(4).to_string())
    slide_example(rng)
    question_example(rng)


if __name__ == "__main__":
    main()
