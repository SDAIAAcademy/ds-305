# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "pandas>=2.2", "matplotlib>=3.9", "scikit-learn>=1.6", "lightgbm>=4.5"]
# ///
# %% [markdown]
# # Lab 3.1 · Importance and its spread
#
# Day 3 · Session 1 (Permutation importance) · 25 min.
#
# Which features does Amana rely on? Permutation importance answers by breaking one feature at a time on
# the holdout and measuring how much worse the predictions get. You write it, repeat it to see its
# spread, and then meet what it does with two income features that move together.
#
# **Reference state.** `package/amana_default.txt` with the Platt repair (`U.recalibrated`), the holdout rows.
#
# **Tasks**
# - [ ] 1 · Write permutation importance for one feature (7')
# - [ ] 2 · All features, ten repeats, with spread (5')
# - [ ] 3 · Plot the importances with their range (4')
# - [ ] 4 · Permute the two incomes together (5')
# - [ ] 5 · Write what the table may and may not say (4')
# - [ ] ✱ Stretch · Use the Brier score in place of log loss and see whether the order changes

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
y = hold["default"].to_numpy()
predict = U.recalibrated                     # the validator's version of Amana: Platt on the calib rows
base = U.log_loss(y, predict(hold))
REPEATS = U.pick(10, 5)
print(f"holdout log loss {base:.4f}; corr(log declared income, log bank inflow) = "
      f"{np.corrcoef(np.log(hold.declared_income), np.log(hold.bank_inflow))[0, 1]:.2f}")

# %% [markdown]
# ## 1 · Permutation importance for one feature (7')
#
# Shuffle one column across the holdout rows, keep every other column, and score again. The **rise in log
# loss** is the importance: how much the model loses when that column stops carrying information.
#
# ✏️ **Exercise** · Return the rise for one column and one random permutation.

# %%
def rise_one(df: pd.DataFrame, col: str, rng) -> float:
    # BEGIN SOLUTION
    d = df.copy()
    d[col] = d[col].to_numpy()[rng.permutation(len(d))]
    return U.log_loss(y, predict(d)) - base
    # END SOLUTION

# %%
try:
    r = rise_one(hold, "prior_missed", np.random.default_rng(U.SEED))
except NotImplementedError:
    r = U.permutation_importance(predict, hold, y, ["prior_missed"], repeats=1)["mean"].iloc[0]
U.check(r > 0, f"shuffling prior_missed raises the log loss by {r:.4f}")

# %% [markdown]
# ## 2 · All features, with repeats (5')
#
# One shuffle is one draw. Repeat it and report the mean and the range: an importance smaller than its
# own spread is not a finding.

# %%
imp = U.permutation_importance(predict, hold, y, U.FEATURES, repeats=REPEATS)
print(imp.round(4).to_string())
top2 = set(imp.index[:2])
U.check(top2 == {"declared_income", "bank_inflow"}, "the two income features are the top two, each on its own")
U.keep("d3_1", imp.round(4).to_dict(orient="index"))

# %% [markdown]
# ## 3 · Plot with the range (4')

# %%
fig, ax = plt.subplots(figsize=(7, 4))
t = imp.iloc[::-1]
ax.barh(t.index, t["mean"], xerr=[t["mean"] - t["min"], t["max"] - t["mean"]], color="#5aa", capsize=3)
ax.set_xlabel(f"rise in log loss when shuffled ({REPEATS} repeats, range)")
ax.set_title("Permutation importance on the holdout", loc="left"); plt.show()

# %% [markdown]
# ## 4 · The two incomes together (5')
#
# Declared income and bank inflow move together. Shuffling only one of them creates applicants who
# declare 20,000 riyals and receive 4,000 a month: combinations that hardly occur, and that the model reads
# as heavy overstatement.
#
# ✏️ **Exercise** · Return the importance of the two columns **shuffled together** (the same permutation
# for both), with the same repeats. `U.permutation_importance` accepts a tuple of columns.

# %%
def grouped_income(repeats: int) -> pd.DataFrame:
    # BEGIN SOLUTION
    return U.permutation_importance(predict, hold, y, [("declared_income", "bank_inflow")], repeats=repeats)
    # END SOLUTION

# %%
try:
    g = grouped_income(REPEATS)
except NotImplementedError:
    g = U.permutation_importance(predict, hold, y, [("declared_income", "bank_inflow")], repeats=REPEATS)
both = pd.concat([imp.loc[["declared_income", "bank_inflow"]], g])
print(both.round(4).to_string())
U.check(g["mean"].iloc[0] < imp.loc[["declared_income", "bank_inflow"], "mean"].min(),
        "shuffled together, the two incomes matter less than either shuffled alone")

# check how unusual the single-shuffle rows are
rng = np.random.default_rng(U.SEED)
ratio = np.log(hold.declared_income / hold.bank_inflow)
shuf = np.log(hold.declared_income.to_numpy()[rng.permutation(len(hold))] / hold.bank_inflow.to_numpy())
lo, hi = np.quantile(ratio, [0.01, 0.99])
outside = float(np.mean((shuf < lo) | (shuf > hi)))
print(f"declared/inflow ratio of real applicants: 98% between {np.exp(lo):.2f} and {np.exp(hi):.2f}; "
      f"after shuffling declared income, {outside:.0%} of rows fall outside that range")
U.check(outside > 0.3, "single-feature shuffling sends many rows outside the data")

# %% [markdown]
# ## 5 · What the table may say (4')
#
# ✏️ **Exercise** · Write two sentences for the explanation report: one the table supports, one it does not.

# %%
def may_and_may_not() -> dict:
    # BEGIN SOLUTION
    return {"supported": "Amana's holdout log loss depends most on the two income features, and next on loan size and term.",
            "not supported": "Bank inflow matters more than declared income: each single-feature score mostly measures "
                             "how badly the model handles impossible combinations of the two."}
    # END SOLUTION

# %%
try:
    text = may_and_may_not()
except NotImplementedError:
    text = {"supported": "income features matter most", "not supported": "ranking between the two incomes"}
print(text)
U.check(set(text) == {"supported", "not supported"}, "one supported and one unsupported reading")

# %% [markdown]
# **Stretch, done.** The same with the Brier score.

# %%
imp_b = U.permutation_importance(predict, hold, y, U.FEATURES, repeats=REPEATS, metric=U.brier)
print(pd.DataFrame({"rank, log loss": imp.reset_index().reset_index().set_index("feature")["index"] + 1,
                    "rank, Brier": imp_b.reset_index().reset_index().set_index("feature")["index"] + 1}).sort_values("rank, log loss").to_string())

# %% [markdown]
# **Look at.** Permutation importance measures what the **model** loses on these rows, not what drives
# default in the world. With correlated features it also measures how the model behaves on rows that do
# not exist. Report the grouped number for features that travel together.
#
# ## Checkpoint
#
# <details><summary>1 · Why compute importance on the holdout, not on the training rows?</summary>
# On training rows the model has memorised some noise; shuffling a feature it overfitted shows an
# importance that does not hold for new applicants.</details>
#
# <details><summary>2 · A feature's importance is 0.002 with a range of -0.001 to 0.005. Report it?</summary>
# As "no measurable effect": the range includes zero.</details>
#
# <details><summary>3 · Why is the grouped importance smaller than each single one?</summary>
# Shuffled together, each row keeps a realistic pair of incomes. Shuffled alone, rows become
# overstatements that the model punishes heavily.</details>
#
# ## Carry forward
#
# The importance table is the first panel of the explanation report. Lab 3.2 starts again from the package.
