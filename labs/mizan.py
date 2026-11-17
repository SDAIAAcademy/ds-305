"""Helpers shared by every lab of Uncertainty, Interpretability and Model Validation (DS-305).

The case: Mizan Community Lending, a fictional microfinance lender, and Amana, the model package its
credit analytics team hands to the model risk team for validation. All data is synthetic
(prepare_data.py, seed 305).

    import mizan as U
    hold = U.load("holdout")                  # the validator's held-out rows (also "calib", "train", "later")
    p = U.p_default(hold)                     # Amana's probability of default
    U.brier(hold["default"], p)               # proper scores, reliability, recalibration
    U.check(cond, "what is true")             # prints a tick, or stops with what went wrong

These are the course-provided functions the tool slots name ("course-provided split conformal
functions", "course-provided subgroup audit functions"): the labs ask you to write each one once,
and the safety net of a skipped exercise falls back to the version here.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

LABS = Path(__file__).resolve().parent
DATA = LABS / "data"
PKG = LABS / "package"
REF = DATA / "reference"
OUT = LABS / "out"
QUICK = os.getenv("MIZAN_QUICK", "0") == "1"
WRITE_REF = os.getenv("MIZAN_WRITE_REF", "0") == "1"

BUSINESS = ["retail", "food", "services", "crafts", "agriculture"]
FEATURES = ["years_in_business", "household_size", "prior_loans", "prior_missed", "declared_income",
            "bank_inflow", "loan_amount", "term_months", "business_type", "channel"]
CATEGORICAL = ["business_type", "channel"]
GRADES = ["A", "B", "C"]
GROUPS = ["region", "age_band", "gender", "channel"]
SEED = 305


def pick(full, quick):
    """The small value in quick mode (MIZAN_QUICK=1), the full one otherwise."""
    return quick if QUICK else full


def check(cond, msg: str) -> None:
    if not cond:
        raise AssertionError("✗ " + msg)
    print("✓", msg)


def style():
    import matplotlib
    if not os.getenv("MPLBACKEND") and "ipykernel" not in __import__("sys").modules:
        import warnings
        matplotlib.use("Agg")                                # run as a script: draw, do not open windows
        warnings.filterwarnings("ignore", message=".*non-interactive.*")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.figsize": (7, 4), "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.3, "font.size": 10})
    return plt


# ------------------------------------------------------------------ data
_cache: dict = {}


def load(part: str = "holdout") -> pd.DataFrame:
    """'train', 'calib', 'holdout' (rows of applications.csv), 'all', or 'later' (the 2025 batch)."""
    if part == "later":
        if "later" not in _cache:
            _cache["later"] = pd.read_csv(DATA / "later_batch.csv")
        return _cache["later"].copy()
    if "apps" not in _cache:
        _cache["apps"] = pd.read_csv(DATA / "applications.csv")
    df = _cache["apps"]
    return (df if part == "all" else df[df.split == part]).reset_index(drop=True).copy()


def X(df: pd.DataFrame, features: list[str] | None = None) -> pd.DataFrame:
    """The model's input table: the package's features, categories as integer codes."""
    out = df[features or FEATURES].copy()
    if "business_type" in out and not pd.api.types.is_numeric_dtype(out.business_type):
        out["business_type"] = out.business_type.map({b: i for i, b in enumerate(BUSINESS)}).astype(int)
    if "channel" in out and not pd.api.types.is_numeric_dtype(out.channel):
        out["channel"] = (out.channel == "app").astype(int)
    return out


def model(name: str = "default", folder: Path | None = None):
    """A model of the package as a LightGBM booster: 'default', 'grade' or 'loss'."""
    import lightgbm as lgb
    key = (name, str(folder))
    if key not in _cache:
        f = (folder or PKG) / (f"amana_{name}.txt" if folder is None else f"model_{name}.txt")
        _cache[key] = lgb.Booster(model_file=str(f))
    return _cache[key]


def p_default(df: pd.DataFrame) -> np.ndarray:
    return model("default").predict(X(df))


def p_grade(df: pd.DataFrame) -> np.ndarray:
    """n by 3: probabilities of grades A, B, C."""
    return model("grade").predict(X(df))


def loss_pred(df: pd.DataFrame) -> np.ndarray:
    return model("loss").predict(X(df))


def package_file(name: str) -> str:
    return (PKG / name).read_text(encoding="utf-8")


def keep(name: str, value) -> None:
    """Save a reference result (only when MIZAN_WRITE_REF=1, which the course authors use)."""
    if WRITE_REF:
        REF.mkdir(parents=True, exist_ok=True)
        (REF / f"{name}.json").write_text(json.dumps(value, indent=1, default=float) + "\n", encoding="utf-8",
                                          newline="\n")


def reference(name: str):
    return json.loads((REF / f"{name}.json").read_text(encoding="utf-8"))


# ------------------------------------------------------------------ Day 1: scores and calibration
def auc(y, p) -> float:
    y = np.asarray(y)
    ranks = pd.Series(np.asarray(p)).rank(method="average").to_numpy()
    n1 = y.sum()
    n0 = len(y) - n1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def brier(y, p) -> float:
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def log_loss(y, p, eps: float = 1e-12) -> float:
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    y = np.asarray(y)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def reliability(y, p, bins: int = 10) -> pd.DataFrame:
    """Equal-count bins of the prediction: mean predicted, observed rate and count per bin."""
    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    d["bin"] = pd.qcut(d.p.rank(method="first"), bins, labels=False)
    return d.groupby("bin").agg(predicted=("p", "mean"), observed=("y", "mean"), n=("y", "size")).reset_index()


def ece(y, p, bins: int = 10) -> float:
    """Expected calibration error: the count-weighted mean gap between predicted and observed."""
    r = reliability(y, p, bins)
    return float(np.sum(r.n * (r.predicted - r.observed).abs()) / r.n.sum())


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_platt(p, y):
    """Platt scaling: a logistic regression of the outcome on the logit of the score."""
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(logit(p).reshape(-1, 1), np.asarray(y))
    return lambda q: lr.predict_proba(logit(q).reshape(-1, 1))[:, 1]


def fit_isotonic(p, y):
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(np.asarray(p), np.asarray(y))
    return iso.predict


def recalibrated(df: pd.DataFrame) -> np.ndarray:
    """The reference repair of Day 1: Platt scaling fitted on the calib rows, applied to df."""
    if "platt" not in _cache:
        cal = load("calib")
        _cache["platt"] = fit_platt(p_default(cal), cal["default"])
    return _cache["platt"](p_default(df))


def wilson(k, n, z: float = 1.96):
    """95% Wilson interval of a proportion k/n."""
    k, n = np.asarray(k, float), np.asarray(n, float)
    ph = k / n
    centre = (ph + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def calibration_by(df: pd.DataFrame, p, col: str) -> pd.DataFrame:
    """Per group: n, mean predicted, observed rate with a 95% interval, and the gap (observed minus predicted)."""
    d = df.assign(p=np.asarray(p))
    t = d.groupby(col).agg(n=("default", "size"), predicted=("p", "mean"), observed=("default", "mean"),
                           defaults=("default", "sum"))
    t["lo"], t["hi"] = wilson(t.defaults, t.n)
    t["gap"] = t.observed - t.predicted
    t["outside"] = (t.predicted < t.lo) | (t.predicted > t.hi)
    return t.drop(columns="defaults").round(4)


# ------------------------------------------------------------------ Day 2: conformal prediction
def conformal_quantile(scores, alpha: float = 0.1) -> float:
    """The ceil((n+1)(1-alpha))-th smallest score: the split conformal threshold."""
    s = np.sort(np.asarray(scores))
    n = len(s)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return float(s[min(k, n) - 1]) if k <= n else float("inf")


def defaults(part: str) -> pd.DataFrame:
    """Defaulted loans of one part: the rows the loss model is about."""
    d = load(part)
    return d[d["default"] == 1].reset_index(drop=True)


def loss_interval(df: pd.DataFrame, q: float) -> tuple[np.ndarray, np.ndarray]:
    pred = loss_pred(df)
    return np.maximum(pred - q, 0), pred + q


def loan_band(df: pd.DataFrame) -> pd.Series:
    return pd.cut(df.loan_amount, [0, 10_000, 25_000, np.inf], labels=["small", "medium", "large"]).astype(str)


def mondrian_quantiles(scores, groups, alpha: float = 0.1) -> dict:
    """Group-conditional (Mondrian) split conformal: one threshold per group."""
    s = pd.Series(np.asarray(scores))
    g = pd.Series(np.asarray(groups))
    return {k: conformal_quantile(s[g == k], alpha) for k in sorted(g.unique())}


def lac_sets(proba, qhat: float) -> np.ndarray:
    """LAC prediction sets: every grade whose probability is at least 1 - qhat (boolean n by k)."""
    return np.asarray(proba) >= 1 - qhat


def lac_scores(proba, y_idx) -> np.ndarray:
    return 1 - np.asarray(proba)[np.arange(len(y_idx)), np.asarray(y_idx)]


def _uniform(n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).random(n)


def aps_scores(proba, y_idx, seed: int = SEED) -> np.ndarray:
    """Randomised APS: the probability of the grades ranked above the true one, plus a uniform share of its own."""
    proba = np.asarray(proba)
    order = np.argsort(-proba, axis=1)
    sorted_p = np.take_along_axis(proba, order, axis=1)
    before = np.cumsum(sorted_p, axis=1) - sorted_p
    rank = np.argmax(order == np.asarray(y_idx)[:, None], axis=1)
    i = np.arange(len(rank))
    return before[i, rank] + _uniform(len(rank), seed) * sorted_p[i, rank]


def aps_sets(proba, qhat: float, seed: int = SEED + 1) -> np.ndarray:
    """Randomised APS sets: a grade is in the set when its score would be at or below the threshold."""
    proba = np.asarray(proba)
    order = np.argsort(-proba, axis=1)
    sorted_p = np.take_along_axis(proba, order, axis=1)
    before = np.cumsum(sorted_p, axis=1) - sorted_p
    u = _uniform(len(proba), seed)[:, None]
    keep_sorted = before + u * sorted_p <= qhat
    keep_sorted[:, 0] |= ~keep_sorted.any(axis=1)          # never an empty set
    out = np.zeros_like(proba, dtype=bool)
    np.put_along_axis(out, order, keep_sorted, axis=1)
    return out


def grade_index(df: pd.DataFrame) -> np.ndarray:
    return df.grade.map({g: i for i, g in enumerate(GRADES)}).to_numpy()


def coverage_by(df: pd.DataFrame, covered, col: str, extra: dict | None = None) -> pd.DataFrame:
    d = df.assign(covered=np.asarray(covered, float), **(extra or {}))
    agg = {"n": ("covered", "size"), "coverage": ("covered", "mean")}
    for k in (extra or {}):
        agg[k] = (k, "mean")
    return d.groupby(col).agg(**agg).round(3)


def income_shift_weights(df: pd.DataFrame, strength: float = 1.0) -> np.ndarray:
    """Weights that tilt a sample towards higher declared income (a simulated shift)."""
    z = (np.log(df.declared_income) - np.log(df.declared_income).mean()) / np.log(df.declared_income).std()
    w = np.exp(strength * z)
    return (w / w.sum()).to_numpy()


def shifted_sample(df: pd.DataFrame, strength: float = 1.0, n: int | None = None, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(df), n or len(df), replace=True, p=income_shift_weights(df, strength))
    return df.iloc[idx].reset_index(drop=True)


# ------------------------------------------------------------------ Day 3: explanations
def permutation_importance(predict, df: pd.DataFrame, y, columns, repeats: int = 10, seed: int = SEED,
                           metric=log_loss) -> pd.DataFrame:
    """Rise in the metric when a column (or a tuple of columns, permuted together) is shuffled, over repeats."""
    rng = np.random.default_rng(seed)
    base = metric(y, predict(df))
    rows = []
    for col in columns:
        cols = list(col) if isinstance(col, tuple) else [col]
        for r in range(repeats):
            d = df.copy()
            perm = rng.permutation(len(d))
            d[cols] = d[cols].to_numpy()[perm]
            rows.append({"feature": " + ".join(cols), "repeat": r, "rise": metric(y, predict(d)) - base})
    t = pd.DataFrame(rows).groupby("feature").rise.agg(["mean", "std", "min", "max"])
    return t.sort_values("mean", ascending=False)


def pd_curve(predict, df: pd.DataFrame, col: str, grid) -> np.ndarray:
    """Partial dependence: set the column to each grid value for every row, average the predictions."""
    out = []
    for v in grid:
        d = df.copy()
        d[col] = v
        out.append(float(np.mean(predict(d))))
    return np.array(out)


def ale_curve(predict, df: pd.DataFrame, col: str, edges) -> np.ndarray:
    """First-order ALE at the bin edges: local differences inside each bin, accumulated and centred."""
    edges = np.asarray(edges, float)
    x = df[col].to_numpy()
    b = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(edges) - 2)
    effects, counts = [], []
    for k in range(len(edges) - 1):
        rows = df[b == k]
        if len(rows) == 0:
            effects.append(0.0)
            counts.append(0)
            continue
        lo, hi = rows.copy(), rows.copy()
        lo[col], hi[col] = edges[k], edges[k + 1]
        effects.append(float(np.mean(predict(hi) - predict(lo))))
        counts.append(len(rows))
    ale = np.concatenate([[0.0], np.cumsum(effects)])
    mids = (ale[:-1] + ale[1:]) / 2
    return ale - np.sum(mids * np.array(counts)) / max(np.sum(counts), 1)


def shap_values(booster, Xdf: pd.DataFrame) -> pd.DataFrame:
    """Exact TreeSHAP from LightGBM (pred_contrib), on the log-odds scale; the last column is the base value."""
    v = booster.predict(Xdf, pred_contrib=True)
    return pd.DataFrame(v, columns=list(Xdf.columns) + ["base"], index=Xdf.index)


def refit_default(seed: int, train: pd.DataFrame | None = None, rounds: int | None = None,
                  features: list[str] | None = None, bootstrap: bool = True):
    """The developers' recipe again (rebalance to 50/50, boost), on a bootstrap resample drawn with this seed."""
    import lightgbm as lgb
    train = load("train") if train is None else train
    rng = np.random.default_rng(seed)
    if bootstrap:
        train = train.iloc[rng.integers(0, len(train), len(train))]
    pos = train[train["default"] == 1]
    neg = train[train["default"] == 0]
    neg = neg.iloc[rng.choice(len(neg), len(pos), replace=False)]
    rows = pd.concat([pos, neg])
    feats = features or FEATURES
    params = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=40,
                  feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=1, lambda_l2=1.0, seed=seed,
                  deterministic=True, num_threads=1, force_col_wise=True, verbose=-1)
    ds = lgb.Dataset(X(rows, feats), rows["default"], categorical_feature=[c for c in CATEGORICAL if c in feats])
    return lgb.train(params, ds, num_boost_round=rounds or pick(300, 150))


def top_reasons(shap: pd.DataFrame, k: int = 3) -> list[list[str]]:
    """The k features with the largest positive contribution, per row: the reasons shown to officers."""
    v = shap.drop(columns="base")
    return [list(r.sort_values(ascending=False).index[:k]) for _, r in v.iterrows()]


# ------------------------------------------------------------------ Day 4: subgroups, shift, decision
def subgroup_audit(df: pd.DataFrame, p, covered, col: str) -> pd.DataFrame:
    """Calibration, coverage and error for one grouping column: the validator's evidence table."""
    d = df.assign(p=np.asarray(p), covered=np.asarray(covered, float))
    rows = []
    for g, part in d.groupby(col):
        lo, hi = wilson(part["default"].sum(), len(part))
        dd = part[part["default"] == 1]
        rows.append({col: g, "n": len(part), "predicted": part.p.mean(), "observed": part["default"].mean(),
                     "obs_lo": float(lo), "obs_hi": float(hi), "brier": brier(part["default"], part.p),
                     "auc": auc(part["default"], part.p) if part["default"].nunique() == 2 else np.nan,
                     "loss_coverage": dd.covered.mean() if len(dd) else np.nan, "defaults": len(dd)})
    t = pd.DataFrame(rows).set_index(col)
    t["calib_gap"] = t.observed - t.predicted
    t["calib_outside"] = (t.predicted < t.obs_lo) | (t.predicted > t.obs_hi)
    return t.round(4)


def psi(expected, actual, bins: int = 10) -> float:
    """Population stability index of one feature: the expected sample's deciles, applied to the actual one."""
    e = np.asarray(expected, float)
    a = np.asarray(actual, float)
    edges = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    pe = np.histogram(e, edges)[0] / len(e)
    pa = np.histogram(a, edges)[0] / len(a)
    pe, pa = np.clip(pe, 1e-4, None), np.clip(pa, 1e-4, None)
    return float(np.sum((pa - pe) * np.log(pa / pe)))


def checklist() -> list[dict]:
    return json.loads((DATA / "checklist.json").read_text(encoding="utf-8"))
