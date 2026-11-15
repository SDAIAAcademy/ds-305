# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "pandas>=2.2", "matplotlib>=3.9", "scikit-learn>=1.6", "lightgbm>=4.5"]
# ///
# %% [markdown]
# # Lab 1.4 · Calibration by region and age band
#
# Day 1 · Session 4 (Calibration by subgroup) · 25 min.
#
# After Platt scaling, Amana's mean prediction matches the default rate. Does it match for every group of
# applicants? The model cannot see region or age band: they are excluded on purpose. That does not stop
# the risk from differing between them.
#
# **Reference state.** `U.recalibrated(df)`: the Platt repair of Lab 1.3, fitted on the calib rows.
#
# **Tasks**
# - [ ] 1 · Write the calibration table by group (7')
# - [ ] 2 · Add a 95% interval to each observed rate (5')
# - [ ] 3 · Run it for region and age band (4')
# - [ ] 4 · Plot predicted against observed per group (5')
# - [ ] 5 · Name the worst group and why (4')
# - [ ] ✱ Stretch · Run the same table for gender and channel and say which gap you would report

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

import numpy as np
import pandas as pd
import mizan as U

plt = U.style()
hold = U.load("holdout")
p = U.recalibrated(hold)
print(f"holdout: mean predicted {p.mean():.3f}, observed {hold['default'].mean():.3f}")

# %% [markdown]
# ## 1 · The calibration table (7')
#
# ✏️ **Exercise** · For one grouping column, return one row per group with `n`, `predicted` (mean of
# `p`), `observed` (default rate) and `gap` (observed minus predicted).

# %%
def calibration_table(df: pd.DataFrame, p, col: str) -> pd.DataFrame:
    # BEGIN SOLUTION
    d = df.assign(p=np.asarray(p))
    t = d.groupby(col).agg(n=("default", "size"), predicted=("p", "mean"), observed=("default", "mean"))
    t["gap"] = t.observed - t.predicted
    return t
    # END SOLUTION

# %%
try:
    t_region = calibration_table(hold, p, "region")
except NotImplementedError:
    t_region = U.calibration_by(hold, p, "region")[["n", "predicted", "observed", "gap"]]
print(t_region.round(4).to_string())
U.check(int(t_region.n.sum()) == len(hold), "every holdout row is in exactly one region")

# %% [markdown]
# ## 2 · An interval for each observed rate (5')
#
# The Northern region has about 500 holdout applicants; its observed rate could move by several points by
# chance. The **Wilson interval** is a 95% interval for a proportion that behaves well for small groups.
#
# ✏️ **Exercise** · Add `lo` and `hi` (use `U.wilson(defaults, n)`) and `outside`: True when the mean
# prediction falls outside the interval.

# %%
def with_interval(df: pd.DataFrame, p, col: str) -> pd.DataFrame:
    # BEGIN SOLUTION
    t = calibration_table(df, p, col)
    k = df.groupby(col)["default"].sum()
    t["lo"], t["hi"] = U.wilson(k, t.n)
    t["outside"] = (t.predicted < t.lo) | (t.predicted > t.hi)
    return t
    # END SOLUTION

# %% [markdown]
# ## 3 · Region and age band (4')

# %%
tables = {}
for col in ("region", "age_band"):
    try:
        tables[col] = with_interval(hold, p, col)
    except NotImplementedError:
        tables[col] = U.calibration_by(hold, p, col)
    print(tables[col].round(4).to_string(), "\n")
ref = U.calibration_by(hold, p, "age_band")
U.check(np.allclose(tables["age_band"].lo.to_numpy(), ref.lo.to_numpy(), atol=1e-4), "the intervals match the course function")

# %% [markdown]
# ## 4 · Predicted against observed, per group (5')

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, col in zip(axes, ("region", "age_band")):
    t = tables[col]
    xs = np.arange(len(t))
    ax.errorbar(xs, t.observed, yerr=[t.observed - t.lo, t.hi - t.observed], fmt="o", capsize=4, label="observed, 95% interval")
    ax.scatter(xs, t.predicted, marker="x", s=60, color="black", label="mean predicted", zorder=3)
    ax.set_xticks(xs, t.index); ax.set_title(col, loc="left")
axes[0].set_ylabel("default rate"); axes[0].legend(frameon=False, loc="upper left")
plt.show()

# %% [markdown]
# ## 5 · The worst group (4')
#
# ✏️ **Exercise** · Across both tables, return the group with the largest gap **among the groups whose
# prediction falls outside the interval**, as `(column, group, gap)`.

# %%
def worst_group(tables: dict) -> tuple[str, str, float]:
    # BEGIN SOLUTION
    rows = [(col, g, float(r.gap)) for col, t in tables.items() for g, r in t.iterrows() if r.outside]
    return max(rows, key=lambda r: abs(r[2]))
    # END SOLUTION

# %%
try:
    worst = worst_group(tables)
except NotImplementedError:
    rows = [(c, g, float(r.gap)) for c, t in tables.items() for g, r in t.iterrows() if r.outside]
    worst = max(rows, key=lambda r: abs(r[2]))
print("worst group:", worst)
U.check(worst[:2] == ("region", "Northern"), f"Northern applicants default {100 * worst[2]:.1f} points more often than predicted")
U.check(bool(tables["age_band"].loc["18-24", "outside"]), "the 18-24 band is also under-predicted, beyond chance")
U.keep("d1_4_gaps", {c: t[["n", "predicted", "observed", "lo", "hi", "gap"]].round(4).to_dict(orient="index") for c, t in tables.items()})

# %% [markdown]
# **Stretch, done.** Gender and channel.

# %%
for col in ("gender", "channel"):
    print(U.calibration_by(hold, p, col).to_string(), "\n")

# %% [markdown]
# **Look at.** The recalibration fixed the average and left the Northern region about 7 points and the
# youngest applicants about 5 points under-predicted. Region and age band are not model features, so no
# recalibration of the score can see them. Whether to fix this (and how) is a decision for Day 4.
#
# ## Checkpoint
#
# <details><summary>1 · The Southern region's gap is 0.5 points and its interval contains the prediction. Report it?</summary>
# Not as a finding: the data cannot tell it from chance. Report the table, flag only what the interval supports.</details>
#
# <details><summary>2 · Why can a model that never sees age be miscalibrated by age?</summary>
# Risk differs by age through things the features do not carry. Excluding a column removes it from the
# input, not from the outcome.</details>
#
# <details><summary>3 · Would adding region as a feature be the validator's fix?</summary>
# No: the validator reports and recommends; whether to use location in lending is a policy decision with
# its own risks. Day 4 turns this into a condition of use.</details>
#
# ## Carry forward
#
# The gap table is saved as `data/reference/d1_4_gaps.json`; Lab 4.1 builds the full audit table from the
# course functions, so it does not need this run.
