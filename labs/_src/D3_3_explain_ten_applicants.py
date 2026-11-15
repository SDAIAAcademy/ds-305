# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "pandas>=2.2", "matplotlib>=3.9", "scikit-learn>=1.6", "lightgbm>=4.5"]
# ///
# %% [markdown]
# # Lab 3.3 · Ten applicants, five refits
#
# Day 3 · Session 3 (Shapley values) · 25 min.
#
# Branch officers see the three features with the largest positive Shapley values as "the reasons" for
# each applicant. You compute those values for ten holdout applicants, check that they add up, and then
# ask whether the reasons would be the same if the developers had trained the same model again on a
# slightly different sample.
#
# **Reference state.** `package/amana_default.txt`, the training rows (for the refits) and the holdout rows.
# The refits follow the developers' recipe in `metadata.json` (`U.refit_default`).
#
# **Tasks**
# - [ ] 1 · Shapley values for ten applicants (5')
# - [ ] 2 · Check that they add up (4')
# - [ ] 3 · Refit five times with the developers' recipe (6')
# - [ ] 4 · The stability panel: top reason per refit (6')
# - [ ] 5 · Mark the explanations you would not show (4')
# - [ ] ✱ Stretch · Measure the spread of the declared-income and bank-inflow values across the refits

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
ten = U.load("holdout").sample(n=10, random_state=U.SEED).reset_index(drop=True)
X10 = U.X(ten)
amana = U.model("default")
print(ten[["app_id", "declared_income", "bank_inflow", "loan_amount", "term_months", "prior_missed"]].to_string(index=False))

# %% [markdown]
# ## 1 · Shapley values (5')
#
# A Shapley value splits one prediction among the features: how much each feature moved this applicant's
# score away from the average score. For tree models they can be computed exactly and fast (TreeSHAP).
# LightGBM does it itself with `pred_contrib=True`; the SHAP library (`shap.TreeExplainer`) gives the same
# numbers. They are on the **log-odds** scale of the raw model.
#
# ✏️ **Exercise** · Return a table with one column per feature plus `base`, one row per applicant.

# %%
def shapley(booster, X: pd.DataFrame) -> pd.DataFrame:
    # BEGIN SOLUTION
    v = booster.predict(X, pred_contrib=True)
    return pd.DataFrame(v, columns=list(X.columns) + ["base"], index=X.index)
    # END SOLUTION

# %%
try:
    sv = shapley(amana, X10)
except NotImplementedError:
    sv = U.shap_values(amana, X10)
print(sv.round(2).to_string())

# %% [markdown]
# ## 2 · They add up (4')
#
# The feature values plus the base value equal the model's raw score (log-odds) for each applicant:
# the property that makes Shapley values an accounting of one prediction.

# %%
raw = amana.predict(X10, raw_score=True)
U.check(np.allclose(sv.sum(axis=1), raw, atol=1e-6), "for every applicant, the values add up to the raw score")
reasons = U.top_reasons(sv, 3)
for app, r in zip(ten.app_id, reasons):
    print(app, "reasons shown to the officer:", ", ".join(r))

# %% [markdown]
# ## 3 · Five refits (6')
#
# The developers' recipe has randomness in it: which non-defaults are dropped to rebalance, which rows are
# bagged. A refit on a bootstrap sample of the training rows is the model they might equally have shipped.
#
# ✏️ **Exercise** · Refit five times (`U.refit_default(seed)` for seeds 1 to 5) and return the Shapley
# tables of the ten applicants under each refit, in a list.

# %%
def refit_shapley(seeds) -> list:
    # BEGIN SOLUTION
    return [U.shap_values(U.refit_default(s), X10) for s in seeds]
    # END SOLUTION

# %%
seeds = [1, 2, 3, 4, 5]
try:
    tables = refit_shapley(seeds)
except NotImplementedError:
    tables = [U.shap_values(U.refit_default(s), X10) for s in seeds]
U.check(len(tables) == 5 and all(t.shape == sv.shape for t in tables), "five Shapley tables, ten applicants each")

# %% [markdown]
# ## 4 · The stability panel (6')
#
# ✏️ **Exercise** · Build the panel: one row per applicant, one column per refit, each cell the top reason
# (the feature with the largest positive value), plus a column `stable` that is True when all five agree.

# %%
def stability_panel(tables: list) -> pd.DataFrame:
    # BEGIN SOLUTION
    panel = pd.DataFrame({f"refit {i + 1}": [r[0] for r in U.top_reasons(t, 1)] for i, t in enumerate(tables)})
    panel.insert(0, "app_id", ten.app_id)
    panel["stable"] = panel.filter(like="refit").nunique(axis=1) == 1
    return panel
    # END SOLUTION

# %%
try:
    panel = stability_panel(tables)
except NotImplementedError:
    panel = pd.DataFrame({f"refit {i + 1}": [r[0] for r in U.top_reasons(t, 1)] for i, t in enumerate(tables)})
    panel.insert(0, "app_id", ten.app_id)
    panel["stable"] = panel.filter(like="refit").nunique(axis=1) == 1
print(panel.to_string(index=False))
n_stable = int(panel.stable.sum())
print(f"{n_stable} of 10 applicants keep the same top reason in all five refits")
U.check(3 <= n_stable < 10, "some top reasons change when the model is refitted")
# the full three-reason list is less stable still
same3 = sum(len({tuple(sorted(U.top_reasons(t.iloc[[i]], 3)[0])) for t in tables}) == 1 for i in range(10))
print(f"{same3} of 10 keep the same set of three reasons")
U.keep("d3_3", {"stable_top": n_stable, "stable_three": same3})

# %% [markdown]
# ## 5 · Which explanations not to show (4')
#
# ✏️ **Exercise** · Return the `app_id`s whose top reason changes between refits, and the rule you would
# propose for showing reasons to officers.

# %%
def do_not_show(panel: pd.DataFrame) -> tuple[list, str]:
    # BEGIN SOLUTION
    return (panel.loc[~panel.stable, "app_id"].tolist(),
            "Show a reason only when it is the top reason in at least four of five refits; otherwise say the model has no stable main reason.")
    # END SOLUTION

# %%
try:
    unstable, rule = do_not_show(panel)
except NotImplementedError:
    unstable, rule = panel.loc[~panel.stable, "app_id"].tolist(), "reference rule"
print(unstable, "|", rule)
U.check(set(unstable) == set(panel.loc[~panel.stable, "app_id"]), "the unstable applicants are listed")

# %% [markdown]
# **Stretch, done.** The two income features, applicant by applicant, across the refits.

# %%
spread = pd.DataFrame({
    "declared_income: min": np.min([t.declared_income for t in tables], 0),
    "declared_income: max": np.max([t.declared_income for t in tables], 0),
    "bank_inflow: min": np.min([t.bank_inflow for t in tables], 0),
    "bank_inflow: max": np.max([t.bank_inflow for t in tables], 0),
}, index=ten.app_id).round(2)
print(spread.to_string())
sum_inc = np.array([t.declared_income + t.bank_inflow for t in tables])
print(f"widest range for one income feature: {np.max(spread.max(axis=1) - spread.min(axis=1)):.2f} log-odds; "
      f"the sum of the two varies by at most {np.max(sum_inc.max(0) - sum_inc.min(0)):.2f}")

# %% [markdown]
# **Look at.** The two incomes trade attribution between refits while their sum stays steadier: the model
# needs both, and how it splits the credit between them is partly chance. A reason shown to an officer
# should not depend on which refit was shipped.
#
# ## Checkpoint
#
# <details><summary>1 · What does "the values add up" guarantee?</summary>
# That the explanation accounts for the prediction exactly. It does not guarantee the explanation is
# stable or causal.</details>
#
# <details><summary>2 · Why refit on bootstrap samples rather than on the same rows?</summary>
# The developers' recipe has random steps and their training rows are one sample; a bootstrap refit is a
# model they could equally have shipped.</details>
#
# <details><summary>3 · An applicant's top reason is "bank inflow" in three refits and "declared income" in two. What do you show?</summary>
# Income, as one combined reason, or no single reason: splitting the credit between the two is not stable.</details>
#
# ## Carry forward
#
# The panel is the stability panel of the explanation report (outcome LO3).
