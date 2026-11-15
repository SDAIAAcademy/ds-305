# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "pandas>=2.2", "matplotlib>=3.9", "scikit-learn>=1.6", "lightgbm>=4.5"]
# ///
# %% [markdown]
# # Lab 1.3 · Platt and isotonic on held-out rows
#
# Day 1 · Session 3 (Recalibration) · 25 min.
#
# A validator does not retrain another team's model; a recalibration map placed after it is a repair the
# developers can adopt without touching the model. You fit two maps on the **calib** rows and judge them
# on the **holdout** rows, which neither map has seen.
#
# **Reference state.** `package/amana_default.txt`, the calib and holdout rows. Nothing from another lab.
#
# **Tasks**
# - [ ] 1 · Fit Platt scaling on the calib rows (6')
# - [ ] 2 · Fit isotonic regression on the same rows (4')
# - [ ] 3 · Score raw, Platt and isotonic on the holdout (6')
# - [ ] 4 · Draw the three reliability curves (5')
# - [ ] 5 · Choose one and write why (4')
# - [ ] ✱ Stretch · Recalibrate with only 500 calib rows and compare the two methods again

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
cal, hold = U.load("calib"), U.load("holdout")
p_cal, y_cal = U.p_default(cal), cal["default"].to_numpy()
p_hold, y_hold = U.p_default(hold), hold["default"].to_numpy()
print(f"calib {len(cal):,} rows, holdout {len(hold):,} rows")

# %% [markdown]
# ## 1 · Platt scaling (6')
#
# Platt scaling is a one-feature logistic regression: the outcome on the **logit of the score**,
# `log(p / (1 - p))`. It can shift and stretch the curve, never bend it.
#
# ✏️ **Exercise** · Fit it on `p_cal, y_cal` and return a function that maps new scores to probabilities.
# (`LogisticRegression(C=1e6)` is an unpenalised fit; `U.logit` clips and transforms.)

# %%
from sklearn.linear_model import LogisticRegression


def fit_platt(p, y):
    # BEGIN SOLUTION
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(U.logit(p).reshape(-1, 1), np.asarray(y))
    return lambda q: lr.predict_proba(U.logit(q).reshape(-1, 1))[:, 1]
    # END SOLUTION

# %%
try:
    platt = fit_platt(p_cal, y_cal)
except NotImplementedError:
    platt = U.fit_platt(p_cal, y_cal)
p_platt = platt(p_hold)
U.check(abs(p_platt.mean() - y_hold.mean()) < 0.01, f"after Platt: mean predicted {p_platt.mean():.3f}, observed {y_hold.mean():.3f}")

# %% [markdown]
# ## 2 · Isotonic regression (4')
#
# Isotonic regression fits any non-decreasing step function of the score. It can bend, so it needs more
# rows and it can overfit a small calibration set.
#
# ✏️ **Exercise** · Fit `IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)` on the calib rows and
# return its `predict`.

# %%
from sklearn.isotonic import IsotonicRegression


def fit_isotonic(p, y):
    # BEGIN SOLUTION
    return IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(np.asarray(p), np.asarray(y)).predict
    # END SOLUTION

# %%
try:
    iso = fit_isotonic(p_cal, y_cal)
except NotImplementedError:
    iso = U.fit_isotonic(p_cal, y_cal)
p_iso = iso(p_hold)
U.check(np.all(np.diff(iso(np.sort(p_hold))) >= 0), "the isotonic map never reverses the order of two applicants")

# %% [markdown]
# ## 3 · Score all three on the holdout (6')

# %%
def score(p):
    return {"Brier": U.brier(y_hold, p), "log loss": U.log_loss(y_hold, p), "ECE": U.ece(y_hold, p), "AUC": U.auc(y_hold, p),
            "mean predicted": float(np.mean(p))}


table = pd.DataFrame({"raw": score(p_hold), "Platt": score(p_platt), "isotonic": score(p_iso)}).T.round(4)
print(table.to_string())
U.check(table.loc["Platt", "Brier"] < table.loc["raw", "Brier"] / 2, "Platt more than halves the Brier score")
U.check(abs(table.loc["Platt", "AUC"] - table.loc["raw", "AUC"]) < 1e-6, "a monotone map leaves the AUC exactly where it was")
U.keep("d1_3_table", table.to_dict())

# %% [markdown]
# **The wrong rows.** What if the map were fitted and judged on the same rows? Fit isotonic on the holdout
# and score it there: the number looks better than anything an applicant will ever see.

# %%
iso_same = U.fit_isotonic(p_hold, y_hold)
print(f"isotonic fitted and scored on the holdout: ECE {U.ece(y_hold, iso_same(p_hold)):.4f} (honest: {table.loc['isotonic', 'ECE']:.4f})")

# %% [markdown]
# ## 4 · Three reliability curves (5')

# %%
fig, ax = plt.subplots(figsize=(5, 5))
ax.plot([0, 1], [0, 1], color="grey", lw=1, ls="--")
for name, q in (("raw", p_hold), ("Platt", p_platt), ("isotonic", p_iso)):
    r = U.reliability(y_hold, q)
    ax.plot(r.predicted, r.observed, marker="o", label=name)
ax.set_xlabel("mean predicted probability"); ax.set_ylabel("observed default rate"); ax.legend(frameon=False)
ax.set_xlim(0, 0.9); ax.set_ylim(0, 0.9); ax.set_title("Holdout reliability, before and after", loc="left")
plt.show()

# %% [markdown]
# ## 5 · Choose (4')
#
# ✏️ **Exercise** · Return the method you recommend and one sentence of reasons that cites a number from
# `table`.

# %%
def choose(table: pd.DataFrame) -> tuple[str, str]:
    # BEGIN SOLUTION
    return "Platt", (f"log loss {table.loc['Platt', 'log loss']:.4f} against {table.loc['isotonic', 'log loss']:.4f} for isotonic: "
                     "as good, with two parameters instead of a step function, so it is steadier on a new batch")
    # END SOLUTION

# %%
try:
    method, why = choose(table)
except NotImplementedError:
    method, why = "Platt", "reference choice"
print(method, "|", why)
U.check(method in ("Platt", "isotonic"), "a method is chosen; the reference repair used in Labs 1.4 and 4.1 is Platt on the calib rows")

# %% [markdown]
# **Stretch, done.** With 500 calib rows, isotonic has fewer points per step.

# %%
rng = np.random.default_rng(U.SEED)
small = rng.choice(len(cal), 500, replace=False)
pl500, is500 = U.fit_platt(p_cal[small], y_cal[small]), U.fit_isotonic(p_cal[small], y_cal[small])
print(f"500 calib rows: Brier Platt {U.brier(y_hold, pl500(p_hold)):.4f}, isotonic {U.brier(y_hold, is500(p_hold)):.4f}; "
      f"log loss Platt {U.log_loss(y_hold, pl500(p_hold)):.4f}, isotonic {U.log_loss(y_hold, is500(p_hold)):.4f}")

# %% [markdown]
# **Look at.** The AUC did not move: recalibration changes what a score *means*, not who ranks above
# whom. And a map scored on its own rows is as untrustworthy as the developers' AUC in Lab 1.1.
#
# ## Checkpoint
#
# <details><summary>1 · Why fit the map on calib rows, not on the training rows?</summary>
# The model has seen the training rows and is overconfident on them; a map fitted there learns the
# wrong correction. Calib rows behave like new applicants.</details>
#
# <details><summary>2 · Isotonic has a lower log loss on 10,000 rows. When would you still pick Platt?</summary>
# When the calibration set is small or the map must travel to a later batch: two parameters overfit less
# than a step function.</details>
#
# <details><summary>3 · What can neither map repair?</summary>
# The ranking, and any miscalibration that differs between groups the score cannot see. Lab 1.4 shows
# the second.</details>
#
# ## Carry forward
#
# The reference repair is `U.recalibrated(df)`: Platt fitted on the calib rows. Every later lab uses it.
