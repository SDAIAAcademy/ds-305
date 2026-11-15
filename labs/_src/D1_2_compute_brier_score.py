# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "pandas>=2.2", "matplotlib>=3.9", "scikit-learn>=1.6", "lightgbm>=4.5"]
# ///
# %% [markdown]
# # Lab 1.2 · Scores and a reliability diagram
#
# Day 1 · Session 2 (Proper scores and reliability) · 25 min.
#
# Amana ranks applicants reasonably well (AUC 0.755 on the holdout). Credit officers, though, read its
# output as a probability. You write the two proper scores and the reliability diagram by hand, score
# Amana against the simplest honest forecast, and find out why its probabilities are so high.
#
# **AI policy: AI-off for the first reading of the diagram.** Write down what it shows before you ask anyone.
#
# **Reference state.** `package/amana_default.txt` and the holdout rows. Nothing from another lab.
#
# **Tasks**
# - [ ] 1 · Write the Brier score and log loss (5')
# - [ ] 2 · Score Amana and the base-rate forecast (5')
# - [ ] 3 · Write equal-count reliability bins (6')
# - [ ] 4 · Draw the reliability diagram (5')
# - [ ] 5 · Name the cause in the metadata (4')
# - [ ] ✱ Stretch · Split the Brier score into reliability, resolution and uncertainty

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

plt = U.style()
hold = U.load("holdout")
y = hold["default"].to_numpy()
p = U.p_default(hold)
print(f"{len(hold):,} holdout applications, {y.mean():.1%} defaulted")

# %% [markdown]
# ## 1 · Two proper scores (5')
#
# A scoring rule is **proper** when the forecaster gets the best expected score by reporting what they
# believe. Both of these are proper; accuracy at a threshold is not.
#
# - Brier score: `mean((p - y)^2)`, between 0 and 1, lower is better.
# - Log loss: `-mean(y log p + (1 - y) log(1 - p))`, lower is better, and very harsh on a confident miss.
#
# ✏️ **Exercise** · Write both. Clip `p` to `[1e-12, 1 - 1e-12]` in the log loss.

# %%
def brier(y, p) -> float:
    # BEGIN SOLUTION
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))
    # END SOLUTION


def log_loss(y, p) -> float:
    # BEGIN SOLUTION
    p = np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
    y = np.asarray(y)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    # END SOLUTION

# %%
try:
    b, ll = brier([0, 1, 1], [0.2, 0.7, 0.9]), log_loss([0, 1, 1], [0.2, 0.7, 0.9])
except NotImplementedError:
    brier, log_loss = U.brier, U.log_loss
    b, ll = brier([0, 1, 1], [0.2, 0.7, 0.9]), log_loss([0, 1, 1], [0.2, 0.7, 0.9])
U.check(abs(b - 0.0467) < 1e-3 and abs(ll - 0.2284) < 1e-3, f"toy case: Brier {b:.4f}, log loss {ll:.4f}")

# %% [markdown]
# ## 2 · Amana against the base rate (5')
#
# The **base-rate forecast** gives every applicant the default rate of the calib rows. It knows nothing
# about anyone, and it is perfectly calibrated on average. A model that loses to it on a proper score is
# doing harm as a probability, whatever its AUC.

# %%
base_rate = U.load("calib")["default"].mean()
scores = pd.DataFrame({
    "Amana": {"Brier": brier(y, p), "log loss": log_loss(y, p), "AUC": U.auc(y, p)},
    "base rate": {"Brier": brier(y, np.full(len(y), base_rate)), "log loss": log_loss(y, np.full(len(y), base_rate)), "AUC": 0.5},
}).T.round(4)
print(scores.to_string())
U.check(scores.loc["Amana", "Brier"] > 2 * scores.loc["base rate", "Brier"], "Amana's Brier score is more than twice the base rate's")
U.keep("d1_2_scores", scores.to_dict())

# %% [markdown]
# ## 3 · Reliability bins (6')
#
# Sort the applicants by predicted probability and cut them into `bins` groups of equal size. In each
# group, compare the **mean predicted** probability with the **observed** default rate.
#
# ✏️ **Exercise** · Return a table with columns `predicted`, `observed`, `n`, one row per bin.
# (`pd.qcut(pd.Series(p).rank(method="first"), bins, labels=False)` gives the bin of each row.)

# %%
def reliability(y, p, bins: int = 10) -> pd.DataFrame:
    # BEGIN SOLUTION
    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    d["bin"] = pd.qcut(d.p.rank(method="first"), bins, labels=False)
    return d.groupby("bin").agg(predicted=("p", "mean"), observed=("y", "mean"), n=("y", "size")).reset_index()
    # END SOLUTION

# %%
try:
    rel = reliability(y, p)
except NotImplementedError:
    rel = U.reliability(y, p)
print(rel.round(3).to_string(index=False))
U.check(len(rel) == 10 and rel.n.sum() == len(y), "ten bins of 1,000 applicants")
U.check((rel.predicted > rel.observed).all(), "every bin predicts more defaults than it has")
ece = float(np.sum(rel.n * (rel.predicted - rel.observed).abs()) / rel.n.sum())
print(f"expected calibration error (ECE): {ece:.3f}")
U.keep("d1_2_reliability", {"ece": round(ece, 3), "top_bin": rel.iloc[-1][["predicted", "observed"]].round(3).to_dict()})

# %% [markdown]
# ## 4 · The reliability diagram (5')
#
# ✏️ **Exercise** · Plot observed against predicted for the bins, with the diagonal. Then write, in
# the cell after it, one sentence on what it shows.

# %%
fig, ax = plt.subplots(figsize=(5, 5))
ax.plot([0, 1], [0, 1], color="grey", lw=1, ls="--", label="perfect calibration")
ax.plot(rel.predicted, rel.observed, marker="o", label="Amana, holdout")
ax.set_xlabel("mean predicted probability"); ax.set_ylabel("observed default rate")
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend(frameon=False)
ax.set_title("Reliability: ten equal-count bins", loc="left")
plt.show()
reading = "Every bin sits far below the diagonal: the scores rank applicants but overstate the risk about fourfold."

# %% [markdown]
# ## 5 · The cause (4')
#
# A model that ranks well but overstates every probability by the same factor has usually been trained
# on a different base rate than the one it meets.
#
# ✏️ **Exercise** · From the metadata, return the training setting that explains the curve, and the
# default rate the model was trained to expect.

# %%
meta = json.loads(U.package_file("metadata.json"))


def cause(meta: dict) -> tuple[str, float]:
    # BEGIN SOLUTION
    return meta["training_settings"]["rebalanced"], 0.5
    # END SOLUTION

# %%
try:
    setting, trained_rate = cause(meta)
except NotImplementedError:
    setting, trained_rate = meta["training_settings"]["rebalanced"], 0.5
print("training setting:", setting)
U.check("50/50" in setting and trained_rate == 0.5, "trained on a 50/50 sample, applied where about 10% default")
# a rebalanced model's odds are inflated by the ratio of the sampling rates: undo it for intuition
odds = p / (1 - p) * (base_rate / (1 - base_rate))
p_prior = odds / (1 + odds)
print(f"after the prior correction: mean predicted {p_prior.mean():.3f}, Brier {brier(y, p_prior):.4f}")

# %% [markdown]
# **Stretch, done.** Murphy's decomposition: Brier = reliability - resolution + uncertainty, computed on
# the ten bins (it holds exactly when every row of a bin has the same prediction, so here approximately).

# %%
ybar = y.mean()
reliab = float(np.sum(rel.n * (rel.predicted - rel.observed) ** 2) / len(y))
resol = float(np.sum(rel.n * (rel.observed - ybar) ** 2) / len(y))
unc = float(ybar * (1 - ybar))
print(f"reliability {reliab:.4f} | resolution {resol:.4f} | uncertainty {unc:.4f} | sum {reliab - resol + unc:.4f} | Brier {brier(y, p):.4f}")
U.check(reliab > resol, "Amana loses more to miscalibration than it gains from resolution")

# %% [markdown]
# **Look at.** The AUC (0.755) and the reliability diagram describe two different properties. The prior
# correction shows the size of the problem; Lab 1.3 repairs it properly, on rows set aside for that.
#
# ## Checkpoint
#
# <details><summary>1 · Why is accuracy at 0.5 not a proper score?</summary>
# It only asks which side of 0.5 a forecast is on. A forecaster who reports 0.51 where they believe 0.9
# loses nothing, so it does not reward honest probabilities.</details>
#
# <details><summary>2 · Why equal-count bins rather than equal-width bins?</summary>
# Most predictions crowd into a narrow range; equal-width bins leave some bins with a handful of rows and
# a noisy observed rate.</details>
#
# <details><summary>3 · Can a model with a worse Brier score than the base rate still be useful?</summary>
# As a ranking, yes (its AUC is 0.755). As a probability that officers read as a default rate, no: it
# must be recalibrated first.</details>
#
# ## Carry forward
#
# The before numbers of the report: Brier, log loss, ECE and the diagram. Lab 1.3 starts again from the package.
