# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "pandas>=2.2", "matplotlib>=3.9", "scikit-learn>=1.6", "lightgbm>=4.5"]
# ///
# %% [markdown]
# # Lab 2.2 · Sets for three risk grades
#
# Day 2 · Session 2 (Prediction sets) · 25 min.
#
# Amana's grade model gives each applicant three probabilities: on time (A), late but repaid (B) and
# default (C). A branch officer sees the top grade only. You turn the probabilities into **prediction
# sets** that contain the true grade 90% of the time, and read the size of each set as the model's
# uncertainty about that applicant.
#
# **Reference state.** `package/amana_grade.txt`; the calib and holdout rows.
#
# **Tasks**
# - [ ] 1 · Write the LAC score and its sets (7')
# - [ ] 2 · Coverage and set sizes (4')
# - [ ] 3 · Adaptive sets (APS) from the helper (5')
# - [ ] 4 · Coverage and size by true grade (5')
# - [ ] 5 · Which applicants get all three grades (4')
# - [ ] ✱ Stretch · Make sets at 80% and 95% and plot the mean set size against the level

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
P_cal, P_hold = U.p_grade(cal), U.p_grade(hold)
y_cal, y_hold = U.grade_index(cal), U.grade_index(hold)
print("grades in the holdout:", hold.grade.value_counts(normalize=True).round(3).to_dict())
print("the top grade is right for", f"{np.mean(P_hold.argmax(1) == y_hold):.1%}", "of holdout applicants")

# %% [markdown]
# ## 1 · LAC: the simplest sets (7')
#
# The LAC score of a calib applicant is `1 - probability the model gave to their true grade`. With the
# conformal threshold `q` of those scores, the set of a new applicant is every grade whose probability is
# at least `1 - q`.
#
# ✏️ **Exercise** · Write the scores and the sets (a boolean array, one row per applicant, one column per grade).

# %%
def lac_scores(P, y) -> np.ndarray:
    # BEGIN SOLUTION
    return 1 - P[np.arange(len(y)), y]
    # END SOLUTION


def lac_sets(P, q: float) -> np.ndarray:
    # BEGIN SOLUTION
    return P >= 1 - q
    # END SOLUTION

# %%
try:
    q_lac = U.conformal_quantile(lac_scores(P_cal, y_cal), 0.1)
    S_lac = lac_sets(P_hold, q_lac)
except NotImplementedError:
    q_lac = U.conformal_quantile(U.lac_scores(P_cal, y_cal), 0.1)
    S_lac = U.lac_sets(P_hold, q_lac)
print(f"LAC threshold {q_lac:.3f}: a grade enters the set when its probability is at least {1 - q_lac:.3f}")
U.check(S_lac.shape == P_hold.shape and S_lac.dtype == bool, "one row per applicant, one column per grade")

# %% [markdown]
# ## 2 · Coverage and set sizes (4')

# %%
def coverage_and_size(S, y) -> dict:
    sizes = S.sum(1)
    return {"coverage": float(S[np.arange(len(y)), y].mean()), "mean size": float(sizes.mean()),
            **{f"size {k}": float(np.mean(sizes == k)) for k in (0, 1, 2, 3)}}


lac = coverage_and_size(S_lac, y_hold)
print({k: round(v, 3) for k, v in lac.items()})
U.check(0.88 <= lac["coverage"] <= 0.92, f"LAC covers the true grade for {lac['coverage']:.1%} of applicants")

# %% [markdown]
# ## 3 · Adaptive sets (5')
#
# LAC gives the smallest sets on average, but it can be empty or too small exactly where the model is
# unsure. **APS** (adaptive prediction sets) adds grades from the most to the least likely until their
# total probability reaches the threshold, so unsure applicants get larger sets. The helper has it.
#
# ✏️ **Exercise** · Compute the APS threshold from the calib rows with `U.aps_scores`, then the holdout
# sets with `U.aps_sets`.

# %%
def aps(P_cal, y_cal, P_hold):
    # BEGIN SOLUTION
    q = U.conformal_quantile(U.aps_scores(P_cal, y_cal), 0.1)
    return q, U.aps_sets(P_hold, q)
    # END SOLUTION

# %%
try:
    q_aps, S_aps = aps(P_cal, y_cal, P_hold)
except NotImplementedError:
    q_aps = U.conformal_quantile(U.aps_scores(P_cal, y_cal), 0.1)
    S_aps = U.aps_sets(P_hold, q_aps)
aps_r = coverage_and_size(S_aps, y_hold)
table = pd.DataFrame({"LAC": lac, "APS": aps_r}).T.round(3)
print(table.to_string())
U.check(0.88 <= aps_r["coverage"] <= 0.92, "APS also covers about 90%")
U.check(aps_r["mean size"] > lac["mean size"], "APS pays for adaptivity with larger sets on average")
U.keep("d2_2", table.to_dict())

# %% [markdown]
# ## 4 · By true grade (5')
#
# Marginal coverage can hide a grade that is rarely covered. The defaulters (C) are the ones that matter most.

# %%
rows = []
for name, S in (("LAC", S_lac), ("APS", S_aps)):
    for g, gname in enumerate(U.GRADES):
        m = y_hold == g
        rows.append({"method": name, "true grade": gname, "n": int(m.sum()),
                     "coverage": S[m, g].mean(), "mean size": S[m].sum(1).mean()})
by_grade = pd.DataFrame(rows).pivot(index="true grade", columns="method", values=["coverage", "mean size"]).round(3)
print(by_grade.to_string())
U.check(by_grade[("coverage", "LAC")]["C"] < 0.6 and by_grade[("coverage", "APS")]["C"] < 0.6, "both methods cover the true grade of fewer than 6 in 10 defaulters")

# %% [markdown]
# ## 5 · Who gets all three grades (4')
#
# ✏️ **Exercise** · Return the holdout applicants whose APS set has all three grades, and compare their
# observed default rate with everyone else's.

# %%
def all_three(hold: pd.DataFrame, S) -> pd.DataFrame:
    # BEGIN SOLUTION
    return hold[S.sum(1) == 3]
    # END SOLUTION

# %%
try:
    unsure = all_three(hold, S_aps)
except NotImplementedError:
    unsure = hold[S_aps.sum(1) == 3]
rest = hold.drop(unsure.index)
print(f"{len(unsure)} applicants ({len(unsure) / len(hold):.1%}) get all three grades; default rate {unsure['default'].mean():.1%} "
      f"against {rest['default'].mean():.1%} for the others")
U.check(unsure["default"].mean() > rest["default"].mean(), "the model is least sure about the riskier applicants")

# %% [markdown]
# **Stretch, done.** Sets at three levels.

# %%
levels = [0.8, 0.9, 0.95]
sizes = []
for lev in levels:
    qq = U.conformal_quantile(U.aps_scores(P_cal, y_cal), 1 - lev)
    sizes.append(U.aps_sets(P_hold, qq).sum(1).mean())
print(dict(zip(levels, np.round(sizes, 3))))
fig, ax = plt.subplots(figsize=(5, 3))
ax.plot(levels, sizes, marker="o"); ax.set_xlabel("target coverage"); ax.set_ylabel("mean APS set size")
ax.set_title("Asking for more coverage buys larger sets", loc="left"); plt.show()

# %% [markdown]
# **Look at.** Set size is a per-applicant statement of uncertainty that the top grade alone hides. An
# officer who sees "{A, B, C}" knows the model cannot tell; one who sees "A" is told it can.
#
# ## Checkpoint
#
# <details><summary>1 · Why can LAC sets be empty?</summary>
# When no grade's probability reaches 1 - q. The coverage guarantee still holds on average; an empty set
# says the applicant is unlike the calib rows.</details>
#
# <details><summary>2 · What does APS buy at the cost of larger sets?</summary>
# Sets that grow where the model is unsure. It does not promise coverage within each grade: here both
# methods cover the true grade of defaulters less than half the time. One threshold per grade would.</details>
#
# <details><summary>3 · Is a set of size one a certainty?</summary>
# No: it is the set that, over many applicants, contains the truth 90% of the time. About one in ten is wrong.</details>
#
# ## Carry forward
#
# The thresholds are recomputed from the calib rows in each lab that needs them.
