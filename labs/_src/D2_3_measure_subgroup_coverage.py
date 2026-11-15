# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "pandas>=2.2", "matplotlib>=3.9", "scikit-learn>=1.6", "lightgbm>=4.5"]
# ///
# %% [markdown]
# # Lab 2.3 · Coverage by group, then per group
#
# Day 2 · Session 3 (Coverage by group) · 25 min.
#
# The loss intervals of Lab 2.1 cover 91% of holdout defaults. Talib Al-Harbash, the chief credit
# officer, provisions for large loans separately: does the 90% hold for them? You measure coverage by
# loan size and by region, then compute one threshold per loan-size band (group-conditional, or
# *Mondrian*, conformal) and see who pays for the repair in width.
#
# **AI policy: AI-off for the first reading of the coverage table.**
#
# **Reference state.** `package/amana_loss.txt`; the defaulted loans of the calib and holdout rows.
#
# **Tasks**
# - [ ] 1 · Coverage and width by loan size and region (6')
# - [ ] 2 · Write group-conditional thresholds (7')
# - [ ] 3 · Coverage by group again (5')
# - [ ] 4 · Width by group: who pays for the repair (4')
# - [ ] 5 · Write one finding for the report (3')
# - [ ] ✱ Stretch · Scale the residuals by the predicted loss and compare with the group-conditional repair

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
cal, hold = U.defaults("calib"), U.defaults("holdout")
pred_cal, pred_hold = U.loss_pred(cal), U.loss_pred(hold)
s_cal = np.abs(cal.loss_sar.to_numpy() - pred_cal)
cal["band"], hold["band"] = U.loan_band(cal), U.loan_band(hold)
q = U.conformal_quantile(s_cal, 0.1)
lo, hi = np.maximum(pred_hold - q, 0), pred_hold + q
covered = (hold.loss_sar.to_numpy() >= lo) & (hold.loss_sar.to_numpy() <= hi)
print(f"one threshold for all: q = {q:,.0f} SAR, coverage {covered.mean():.1%}")

# %% [markdown]
# ## 1 · Coverage and width by group (6')
#
# Loan-size bands: small (up to 10,000 SAR), medium (10,000 to 25,000), large (above 25,000).

# %%
by_band = U.coverage_by(hold, covered, "band", {"width": hi - lo})
by_region = U.coverage_by(hold, covered, "region", {"width": hi - lo})
print(by_band.to_string(), "\n")
print(by_region.to_string())
U.check(by_band.loc["large", "coverage"] < 0.85, f"large loans: {by_band.loc['large', 'coverage']:.1%} coverage, far below 90%")
U.check(by_band.loc["medium", "coverage"] > 0.97, "medium loans are over-covered: the same width is too generous for them")

# %% [markdown]
# **Read the table yourself first.** Coverage of 91% overall is an average of near-certainty for small and
# medium loans and about three in four for large ones. Marginal coverage is a promise about everyone
# together, not about any group.
#
# ## 2 · One threshold per group (7')
#
# ✏️ **Exercise** · Return a dict from each band to the conformal threshold of the calib scores **in that
# band**.

# %%
def group_thresholds(scores, groups, alpha: float = 0.1) -> dict:
    # BEGIN SOLUTION
    s, g = np.asarray(scores), np.asarray(groups)
    return {k: U.conformal_quantile(s[g == k], alpha) for k in sorted(set(g))}
    # END SOLUTION

# %%
try:
    qs = group_thresholds(s_cal, cal.band, 0.1)
except NotImplementedError:
    qs = U.mondrian_quantiles(s_cal, cal.band, 0.1)
print({k: round(v) for k, v in qs.items()}, "| calib defaults per band:", cal.band.value_counts().to_dict())
U.check(qs["large"] > q > qs["small"], "large loans need a wider threshold, small loans a narrower one")

# %% [markdown]
# ## 3 · Coverage by group again (5')

# %%
qq = hold.band.map(qs).to_numpy()
lo2, hi2 = np.maximum(pred_hold - qq, 0), pred_hold + qq
covered2 = (hold.loss_sar.to_numpy() >= lo2) & (hold.loss_sar.to_numpy() <= hi2)
by_band2 = U.coverage_by(hold, covered2, "band", {"width": hi2 - lo2})
print(by_band2.to_string())
print(f"overall coverage {covered2.mean():.1%}")
U.check(by_band2.loc["large", "coverage"] > 0.87, "large loans are back near 90%")

# %% [markdown]
# ## 4 · Who pays for the repair (4')
#
# ✏️ **Exercise** · Build one table with, per band, the coverage and mean width before and after.

# %%
def before_after(b1: pd.DataFrame, b2: pd.DataFrame) -> pd.DataFrame:
    # BEGIN SOLUTION
    return pd.DataFrame({"n": b1.n, "coverage before": b1.coverage, "coverage after": b2.coverage,
                         "width before": b1.width.round(), "width after": b2.width.round()})
    # END SOLUTION

# %%
try:
    ba = before_after(by_band, by_band2)
except NotImplementedError:
    ba = pd.DataFrame({"n": by_band.n, "coverage before": by_band.coverage, "coverage after": by_band2.coverage,
                       "width before": by_band.width.round(), "width after": by_band2.width.round()})
print(ba.to_string())
U.check(ba.loc["large", "width after"] > ba.loc["large", "width before"], "the large loans' intervals widen")
U.check(ba.loc["small", "width after"] < ba.loc["small", "width before"], "the small loans' intervals narrow")
U.keep("d2_3", ba.to_dict())

fig, ax = plt.subplots(figsize=(6, 3.5))
x = np.arange(len(ba))
ax.bar(x - 0.2, ba["coverage before"], 0.4, label="one threshold")
ax.bar(x + 0.2, ba["coverage after"], 0.4, label="one per band")
ax.axhline(0.9, color="black", ls="--", lw=1); ax.set_xticks(x, ba.index); ax.set_ylim(0.5, 1.02)
ax.set_ylabel("coverage"); ax.legend(frameon=False, loc="lower left"); ax.set_title("Coverage by loan size", loc="left"); plt.show()

# %% [markdown]
# ## 5 · One finding (3')
#
# ✏️ **Exercise** · Write the finding as the report will carry it: what, how much, for whom, and the repair.

# %%
def finding() -> str:
    # BEGIN SOLUTION
    return (f"The loss intervals cover {covered.mean():.0%} of defaults overall but only {by_band.loc['large', 'coverage']:.0%} "
            f"of loans above 25,000 SAR ({int(by_band.loc['large', 'n'])} holdout defaults). One threshold per loan-size band "
            f"restores {by_band2.loc['large', 'coverage']:.0%} for large loans, with intervals {ba.loc['large', 'width after'] / ba.loc['large', 'width before']:.1f} times wider.")
    # END SOLUTION

# %%
try:
    text = finding()
except NotImplementedError:
    text = "reference finding: large loans under-covered; per-band thresholds repair it at the cost of width"
print(text)
U.check(len(text) > 40, "a finding with numbers")

# %% [markdown]
# **Stretch, done.** A normalised score `|y - pred| / pred` gives every loan an interval proportional to its
# predicted loss, without choosing groups.

# %%
s_norm = s_cal / np.maximum(pred_cal, 1.0)
qn = U.conformal_quantile(s_norm, 0.1)
lo3, hi3 = np.maximum(pred_hold * (1 - qn), 0), pred_hold * (1 + qn)
cov3 = (hold.loss_sar.to_numpy() >= lo3) & (hold.loss_sar.to_numpy() <= hi3)
print(U.coverage_by(hold, cov3, "band", {"width": hi3 - lo3}).to_string())

# %% [markdown]
# **Look at.** The Northern region is also below target (87% on 78 defaults), but with so few rows its
# coverage could be 80% or 95% by chance. Group-conditional conformal needs enough calib rows **in each
# group**: a group with 30 calib rows gets a threshold from 30 scores.
#
# ## Checkpoint
#
# <details><summary>1 · Overall coverage stayed near 91% after the repair. What changed?</summary>
# How it is shared: each band now has its own guarantee, at the cost of wider intervals for large loans and
# narrower ones for small loans.</details>
#
# <details><summary>2 · Why not one threshold per region and per band and per age band?</summary>
# Each cell needs its own calib rows. With many small cells the thresholds come from a handful of scores and
# the intervals become very wide or infinite.</details>
#
# <details><summary>3 · Is 77% coverage for large loans a fault of the loss model?</summary>
# It is a property of the residuals: losses on large loans vary more. The conformal wrapper with one width
# hides it; the group view reveals it.</details>
#
# ## Carry forward
#
# `U.mondrian_quantiles` and `U.loan_band` reproduce this lab's repair for Lab 4.1.
