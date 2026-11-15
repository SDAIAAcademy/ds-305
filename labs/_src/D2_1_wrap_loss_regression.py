# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "pandas>=2.2", "matplotlib>=3.9", "scikit-learn>=1.6", "lightgbm>=4.5", "mapie>=1.0"]
# ///
# %% [markdown]
# # Lab 2.1 · Loss intervals with 90% coverage
#
# Day 2 · Session 1 (Conformal intervals) · 25 min.
#
# Amana's loss model predicts how many riyals Mizan loses if a loan defaults: one number per loan, with
# no statement of how wrong it can be. You wrap it in split conformal intervals from the calib defaults
# and check, on the holdout defaults, that 90% of real losses fall inside. Then you confirm the same
# numbers with a library, MAPIE.
#
# **Reference state.** `package/amana_loss.txt`; the defaulted loans of the calib and holdout rows.
#
# **Tasks**
# - [ ] 1 · Score the calib defaults: absolute residuals (4')
# - [ ] 2 · Write the conformal threshold (6')
# - [ ] 3 · Intervals for the holdout defaults (4')
# - [ ] 4 · Coverage and mean width (5')
# - [ ] 5 · The same with MAPIE (6')
# - [ ] ✱ Stretch · Repeat over 100 random calib and test splits and plot the coverage you get

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
print(f"defaulted loans: calib {len(cal)}, holdout {len(hold)}; median loss {hold.loss_sar.median():,.0f} SAR")

# %% [markdown]
# ## 1 · Nonconformity scores (4')
#
# The **score** says how badly the model did on a calib loan. For a regression the simplest is the
# absolute residual `|y - prediction|`.
#
# ✏️ **Exercise** · Compute the scores of the calib defaults.

# %%
def residual_scores(y, pred) -> np.ndarray:
    # BEGIN SOLUTION
    return np.abs(np.asarray(y) - np.asarray(pred))
    # END SOLUTION

# %%
try:
    scores = residual_scores(cal.loss_sar, pred_cal)
except NotImplementedError:
    scores = np.abs(cal.loss_sar.to_numpy() - pred_cal)
U.check(len(scores) == len(cal) and scores.min() >= 0, f"{len(scores)} scores, median {np.median(scores):,.0f} SAR")

# %% [markdown]
# ## 2 · The conformal threshold (6')
#
# With `n` calib scores and a miscoverage `alpha` (0.1 for 90%), the threshold is the
# `ceil((n + 1)(1 - alpha))`-th smallest score. The `+ 1` counts the new loan itself: it is what turns
# "about 90%" into "at least 90%" when the new loan is exchangeable with the calib loans.
#
# ✏️ **Exercise** · Write it. Return infinity if `ceil((n + 1)(1 - alpha)) > n`.

# %%
def conformal_quantile(scores, alpha: float = 0.1) -> float:
    # BEGIN SOLUTION
    s = np.sort(np.asarray(scores))
    k = int(np.ceil((len(s) + 1) * (1 - alpha)))
    return float(s[k - 1]) if k <= len(s) else float("inf")
    # END SOLUTION

# %%
try:
    q = conformal_quantile(scores, 0.1)
    U.check(conformal_quantile(np.arange(1, 9), 0.1) == float("inf"), "with 8 scores, 90% needs the 9th: no finite interval")
except NotImplementedError:
    q = U.conformal_quantile(scores, 0.1)
U.check(abs(q - U.conformal_quantile(scores, 0.1)) < 1e-6, f"threshold q = {q:,.0f} SAR")

# %% [markdown]
# ## 3 · Intervals (4')
#
# The interval for a new loan is `[prediction - q, prediction + q]`. A loss cannot be negative, so clip
# the lower end at zero.
#
# ✏️ **Exercise** · Return the lower and upper ends for the holdout defaults.

# %%
def intervals(pred, q: float):
    # BEGIN SOLUTION
    pred = np.asarray(pred)
    return np.maximum(pred - q, 0), pred + q
    # END SOLUTION

# %%
try:
    lo, hi = intervals(pred_hold, q)
except NotImplementedError:
    lo, hi = np.maximum(pred_hold - q, 0), pred_hold + q
U.check(np.all(hi - lo <= 2 * q + 1e-6), "every interval is at most 2q wide")

# %% [markdown]
# ## 4 · Coverage and width (5')
#
# **Coverage** is the share of holdout losses inside their interval; **width** is what the interval costs
# the reader. Report both: an interval from zero to a million covers everything and says nothing.

# %%
covered = (hold.loss_sar.to_numpy() >= lo) & (hold.loss_sar.to_numpy() <= hi)
coverage, width = float(covered.mean()), float(np.mean(hi - lo))
print(f"target 90% | achieved {coverage:.1%} on {len(hold)} holdout defaults | mean width {width:,.0f} SAR")
U.check(0.88 <= coverage <= 0.94, "coverage is close to the 90% target")

fig, ax = plt.subplots()
order = np.argsort(pred_hold)[:: max(1, len(hold) // 120)]
ax.vlines(pred_hold[order], lo[order], hi[order], color="#9ab", lw=1, label="90% interval")
ax.scatter(pred_hold[order], hold.loss_sar.to_numpy()[order], s=8, c=np.where(covered[order], "#14866d", "#d9534f"),
           label="actual loss (red: outside)")
ax.set_xlabel("predicted loss (SAR)"); ax.set_ylabel("loss (SAR)"); ax.legend(frameon=False)
ax.set_title("Split conformal intervals, holdout defaults", loc="left"); plt.show()

# %% [markdown]
# ## 5 · The same with MAPIE (6')
#
# MAPIE expects a scikit-learn regressor. `prefit=True` tells it the model is already trained, so it only
# conformalizes. The wrapper below hands it Amana's loss model.
#
# ✏️ **Exercise** · Conformalize on the calib defaults and return MAPIE's intervals for the holdout
# defaults, shape `(n, 2)`.

# %%
from sklearn.base import BaseEstimator, RegressorMixin


class AmanaLoss(RegressorMixin, BaseEstimator):
    """Amana's loss model, already trained, seen by scikit-learn as a fitted regressor."""

    def fit(self, X, y=None):
        self.fitted_ = True
        return self

    def predict(self, X):
        return U.model("loss").predict(X)


def mapie_intervals(cal: pd.DataFrame, hold: pd.DataFrame) -> np.ndarray:
    # BEGIN SOLUTION
    from mapie.regression import SplitConformalRegressor
    scr = SplitConformalRegressor(estimator=AmanaLoss().fit(None), confidence_level=0.9, prefit=True)
    scr.conformalize(U.X(cal), cal.loss_sar)
    _, iv = scr.predict_interval(U.X(hold))
    return iv[:, :, 0]
    # END SOLUTION

# %%
try:
    iv = mapie_intervals(cal, hold)
except (NotImplementedError, ImportError):
    iv = np.column_stack([pred_hold - q, pred_hold + q])
half = (iv[:, 1] - iv[:, 0]) / 2
U.check(np.allclose(half, q, rtol=1e-6), f"MAPIE's half-width equals your threshold: {half[0]:,.0f} SAR")
U.keep("d2_1", {"q": round(q), "coverage": round(coverage, 3), "width": round(width)})

# %% [markdown]
# **Stretch, done.** The guarantee is about the average over draws of the calib set. Redraw the calib and
# test split 100 times from all defaulted loans the developers did not train on.

# %%
pool = pd.concat([cal, hold], ignore_index=True)
pool_pred = U.loss_pred(pool)
rng = np.random.default_rng(U.SEED)
covs = []
for _ in range(100):
    idx = rng.permutation(len(pool))
    c, t = idx[: len(pool) // 2], idx[len(pool) // 2:]
    qq = U.conformal_quantile(np.abs(pool.loss_sar.to_numpy()[c] - pool_pred[c]), 0.1)
    covs.append(np.mean(np.abs(pool.loss_sar.to_numpy()[t] - pool_pred[t]) <= qq))
covs = np.array(covs)
print(f"{len(covs)} splits: mean coverage {covs.mean():.3f}, from {covs.min():.3f} to {covs.max():.3f}")
U.check(abs(covs.mean() - 0.9) < 0.02, "on average over splits, coverage is at the target")
fig, ax = plt.subplots(figsize=(6, 3))
ax.hist(covs, bins=15, color="#5aa"); ax.axvline(0.9, color="black", ls="--")
ax.set_xlabel("coverage on the test half"); ax.set_title("Coverage varies from split to split", loc="left"); plt.show()

# %% [markdown]
# **Look at.** One interval width for every loan: a 5,000-riyal loan and an 80,000-riyal loan both get
# plus or minus the same amount. The 90% holds on average. Lab 2.3 asks for whom it holds.
#
# ## Checkpoint
#
# <details><summary>1 · Why the "+ 1" in the threshold's rank?</summary>
# The new loan is one more exchangeable draw; counting it gives coverage of at least 1 - alpha in finite
# samples, not only approximately.</details>
#
# <details><summary>2 · What does the 90% promise, exactly?</summary>
# That over new loans drawn like the calib loans, at least 90% of losses fall inside, on average. Not
# that each interval has a 90% chance for that particular loan.</details>
#
# <details><summary>3 · Did you need to retrain or open the loss model?</summary>
# No: split conformal needs only predictions on calib rows. That is why a validator can use it.</details>
#
# ## Carry forward
#
# The threshold is recomputed in each lab from the calib defaults (`U.conformal_quantile`), so later labs
# do not depend on this run.
