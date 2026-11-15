# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "pandas>=2.2", "matplotlib>=3.9", "scikit-learn>=1.6", "lightgbm>=4.5"]
# ///
# %% [markdown]
# # Lab 4.1 · Gaps by group, tagged
#
# Day 4 · Session 1 (Subgroup validation) · 25 min.
#
# The validation report needs one table that a committee can read: calibration, coverage and error for
# every group the lender is accountable to, with the gaps an interval supports flagged, each tagged with
# a bias type from the SDAIA AI Bias Reference Guide and rated.
#
# **Reference state.** The package with the Platt repair (`U.recalibrated`), one conformal threshold for
# the loss intervals from the calib defaults, the holdout rows and `data/bias_types.json` (names and pages
# of the guide's entries).
#
# **Tasks**
# - [ ] 1 · The audit table for one column (7')
# - [ ] 2 · All four columns (4')
# - [ ] 3 · Flag the gaps an interval supports (5')
# - [ ] 4 · Tag each gap with a bias type (6')
# - [ ] 5 · Rate each finding (3')
# - [ ] ✱ Stretch · Compute the same table for the age band within the Northern region and say what the counts allow

# %%
import os, sys, subprocess
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    if not Path("/content/ds-305").exists():
        subprocess.run(["git", "clone", "-q", "https://github.com/SDAIAAcademy/ds-305.git", "/content/ds-305"], check=True)
    os.chdir("/content/ds-305/labs")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-colab.txt"], check=True)
else:
    while not Path("mizan.py").exists() and Path.cwd() != Path.cwd().parent:
        os.chdir("..")  # solutions/ notebooks run from labs/
sys.path.insert(0, os.getcwd())

import json
import numpy as np
import pandas as pd
import mizan as U

hold = U.load("holdout")
p = U.recalibrated(hold)
cal_d = U.defaults("calib")
q = U.conformal_quantile(np.abs(cal_d.loss_sar - U.loss_pred(cal_d)), 0.1)
lo, hi = U.loss_interval(hold, q)                      # computed for every row; only defaults have a loss
covered = np.where(hold["default"] == 1, (hold.loss_sar >= lo) & (hold.loss_sar <= hi), np.nan)
guide = json.loads((U.DATA / "bias_types.json").read_text(encoding="utf-8"))
TYPES = {t["name"] for t in guide["types"]}
print(f"recalibrated mean {p.mean():.3f} against {hold['default'].mean():.3f}; loss threshold {q:,.0f} SAR")

# %% [markdown]
# ## 1 · The audit table for one column (7')
#
# ✏️ **Exercise** · For one grouping column, return one row per group with: `n`, `predicted`,
# `observed`, the 95% interval of the observed rate (`obs_lo`, `obs_hi`), `brier`, `auc`,
# `loss_coverage` (among the group's defaults) and `defaults`.

# %%
def audit(df: pd.DataFrame, p, covered, col: str) -> pd.DataFrame:
    # BEGIN SOLUTION
    d = df.assign(p=np.asarray(p), covered=np.asarray(covered, float))
    rows = []
    for g, part in d.groupby(col):
        lo_, hi_ = U.wilson(part["default"].sum(), len(part))
        rows.append({col: g, "n": len(part), "predicted": part.p.mean(), "observed": part["default"].mean(),
                     "obs_lo": float(lo_), "obs_hi": float(hi_), "brier": U.brier(part["default"], part.p),
                     "auc": U.auc(part["default"], part.p), "loss_coverage": part.covered.mean(),
                     "defaults": int(part["default"].sum())})
    return pd.DataFrame(rows).set_index(col)
    # END SOLUTION

# %%
try:
    t_region = audit(hold, p, covered, "region")
except NotImplementedError:
    t_region = U.subgroup_audit(hold, p, covered, "region")
ref = U.subgroup_audit(hold, p, covered, "region")
U.check(np.allclose(t_region.loss_coverage, ref.loss_coverage, atol=1e-3) and np.allclose(t_region.obs_lo, ref.obs_lo, atol=1e-3),
        "your audit table matches the course function")
print(t_region.round(3).to_string())

# %% [markdown]
# ## 2 · All four columns (4')

# %%
tables = {col: U.subgroup_audit(hold, p, covered, col) for col in U.GROUPS}
for col, t in tables.items():
    print(t[["n", "predicted", "observed", "obs_lo", "obs_hi", "auc", "loss_coverage", "defaults"]].round(3).to_string(), "\n")

# %% [markdown]
# ## 3 · Flag the gaps an interval supports (5')
#
# A calibration gap is flagged when the mean prediction lies outside the observed rate's interval. A
# coverage gap is flagged when coverage is below 85% on at least 50 defaults.
#
# ✏️ **Exercise** · Return a list of `(column, group, kind, value)` for every flagged gap, where kind is
# `"calibration"` (value: observed minus predicted) or `"coverage"` (value: the coverage).

# %%
def flag(tables: dict) -> list:
    # BEGIN SOLUTION
    out = []
    for col, t in tables.items():
        for g, r in t.iterrows():
            if r.calib_outside:
                out.append((col, g, "calibration", round(float(r.calib_gap), 4)))
            if r.defaults >= 50 and r.loss_coverage < 0.85:
                out.append((col, g, "coverage", round(float(r.loss_coverage), 3)))
    return out
    # END SOLUTION

# %%
try:
    flags = flag(tables)
except NotImplementedError:
    flags = [(c, g, "calibration", round(float(r.calib_gap), 4)) for c, t in tables.items() for g, r in t.iterrows() if r.calib_outside]
for f in flags:
    print(f)
keys = {(c, g) for c, g, _, _ in flags}
U.check({("region", "Northern"), ("age_band", "18-24")} <= keys, "the North and the youngest band are flagged")

# %% [markdown]
# **The loan-size view.** Loan size is not a protected group, but Day 2 found the worst coverage there.
# Add it, because Talib provisions by loan size.

# %%
hold_b = hold.assign(band=U.loan_band(hold))
t_band = U.subgroup_audit(hold_b, p, covered, "band")
print(t_band[["n", "loss_coverage", "defaults"]].round(3).to_string())
flags += [("band", g, "coverage", round(float(r.loss_coverage), 3)) for g, r in t_band.iterrows()
          if r.defaults >= 50 and r.loss_coverage < 0.85]

# %% [markdown]
# ## 4 · Tag each gap (6')
#
# Read the entries in the guide (pages in `data/bias_types.json`) and choose the type that best names
# **how** the gap arose, with a one-line reason.

# %%
for t in guide["types"]:
    print(f"{t['name']:<22} p. {t['page']:<4} {t['summary']}")

# %% [markdown]
# ✏️ **Exercise** · Return a dict from `(column, group)` to `(bias type, reason)` for every flagged gap.

# %%
def tag(flags: list) -> dict:
    # BEGIN SOLUTION
    tags = {}
    for col, g, kind, v in flags:
        if (col, g) == ("region", "Northern"):
            tags[(col, g)] = ("Aggregation Bias", "region excluded and the North is 5% of rows: pooled with larger regions, it is fitted worst")
        elif col == "age_band":
            tags[(col, g)] = ("Age Bias", "age excluded and its differences not accounted for: the youngest band's risk is under-predicted")
        elif col == "gender":
            tags[(col, g)] = ("Gender Bias", "gender excluded; men default slightly more than predicted, women slightly less")
        elif col == "channel":
            tags[(col, g)] = ("Aggregation Bias", "the two channels are pooled; a small residual gap remains")
        elif col == "band":
            tags[(col, g)] = ("Evaluation Bias", "marginal coverage of 91% overstates how well intervals hold for large loans")
    return tags
    # END SOLUTION

# %%
ACCEPT = {("region", "Northern"): {"Aggregation Bias", "Coverage Bias", "Sample Size Bias"},
          ("age_band", "18-24"): {"Age Bias", "Aggregation Bias"},
          ("band", "large"): {"Evaluation Bias", "Overconfidence Bias"}}
try:
    tags = tag(flags)
except NotImplementedError:
    tags = {k: (sorted(v)[0], "reference tag") for k, v in ACCEPT.items()}
U.check(all(t in TYPES for t, _ in tags.values()), "every tag is a type named in the guide")
for k, ok in ACCEPT.items():
    if k in tags:
        U.check(tags[k][0] in ok, f"{k}: '{tags[k][0]}' is one of the defensible tags")

# %% [markdown]
# ## 5 · Rate each finding (3')
#
# Rating rule of the course: **high** when the gap affects a decision about people (approval or price) and
# the interval supports it; **medium** when it affects provisioning or is supported but small (under 2
# points); **low** otherwise.

# %%
def rate(col, g, kind, v) -> str:
    if kind == "calibration":
        return "high" if abs(v) >= 0.02 else "medium"
    return "medium"


findings = pd.DataFrame([{"column": c, "group": g, "kind": k, "value": v, "tag": tags.get((c, g), ("", ""))[0],
                          "reason": tags.get((c, g), ("", ""))[1], "rating": rate(c, g, k, v)} for c, g, k, v in flags])
print(findings.drop(columns="reason").to_string(index=False))
U.check(set(findings.loc[findings.rating == "high", "group"]) >= {"Northern", "18-24"}, "the two large calibration gaps are rated high")
U.keep("d4_1", findings.to_dict(orient="records"))

# %% [markdown]
# **Stretch, done.** Age band within the Northern region.

# %%
north = hold[hold.region == "Northern"]
t_n = U.subgroup_audit(north, p[hold.region == "Northern"], covered[hold.region == "Northern"], "age_band")
print(t_n[["n", "defaults", "predicted", "observed", "obs_lo", "obs_hi"]].round(3).to_string())
w = (t_n.obs_hi - t_n.obs_lo) * 100
print(f"With {t_n.n.min()} to {t_n.n.max()} applicants and {t_n.defaults.min()} to {t_n.defaults.max()} defaults per cell, "
      f"the intervals are {w.min():.0f} to {w.max():.0f} points wide: the data support no finding per cell, only a reason to watch.")

# %% [markdown]
# **Look at.** The tags name how a gap arose, which points to the remedy: aggregation suggests group-aware
# calibration or more data from the group, evaluation bias a change in how results are reported. Choosing
# and costing the remedy is the owners' decision, and fairness mitigation is the subject of DS-309.
#
# ## Checkpoint
#
# <details><summary>1 · Men are under-predicted by 1.6 points, just outside the interval. High?</summary>
# Medium under the course rule: supported, small. Report it and monitor it.</details>
#
# <details><summary>2 · Why is "Sample Size Bias" also defensible for the North?</summary>
# The North is 5% of the training rows; too few rows to learn its difference is one way the gap arose.</details>
#
# <details><summary>3 · Is a tag a finding of discrimination?</summary>
# No: it names a mechanism from the guide. Whether a gap is unlawful or unfair is a judgement the report
# leaves to the committee, with the evidence.</details>
#
# ## Carry forward
#
# The tagged table is the subgroup section of the report. Lab 4.4 reads the week's findings from
# `data/findings.json`.
