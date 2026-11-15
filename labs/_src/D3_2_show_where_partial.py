# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "pandas>=2.2", "matplotlib>=3.9", "scikit-learn>=1.6", "lightgbm>=4.5"]
# ///
# %% [markdown]
# # Lab 3.2 · Partial dependence against local effects
#
# Day 3 · Session 2 (Effect plots) · 25 min.
#
# How does Amana's predicted risk change with declared income? Two plots answer that question in
# different ways. **Partial dependence** sets every applicant's declared income to the same value and
# averages; **accumulated local effects** (ALE) only move each applicant a little, inside the data. With
# a feature that moves with another, they can tell different stories.
#
# **Reference state.** `package/amana_default.txt` with the Platt repair, the holdout rows.
#
# **Tasks**
# - [ ] 1 · Write partial dependence for declared income (6')
# - [ ] 2 · Accumulated local effects from the helper (4')
# - [ ] 3 · Plot both, centred (5')
# - [ ] 4 · Count the impossible rows partial dependence used (6')
# - [ ] 5 · Say which curve goes in the report (4')
# - [ ] ✱ Stretch · Draw individual conditional expectation curves for 50 applicants and find the ones that bend

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
sample = hold.sample(n=U.pick(4000, 2000), random_state=U.SEED).reset_index(drop=True)
predict = U.recalibrated
grid = np.quantile(sample.declared_income, np.linspace(0.05, 0.95, 10)).round(-1)
print("grid of declared income (SAR):", grid.astype(int).tolist())

# %% [markdown]
# ## 1 · Partial dependence (6')
#
# For each grid value `v`: copy the sample, set **every** applicant's declared income to `v`, predict, and
# average.
#
# ✏️ **Exercise** · Return one average prediction per grid value.

# %%
def partial_dependence(df: pd.DataFrame, col: str, grid) -> np.ndarray:
    # BEGIN SOLUTION
    out = []
    for v in grid:
        d = df.copy()
        d[col] = v
        out.append(float(np.mean(predict(d))))
    return np.array(out)
    # END SOLUTION

# %%
try:
    pdp = partial_dependence(sample, "declared_income", grid)
except NotImplementedError:
    pdp = U.pd_curve(predict, sample, "declared_income", grid)
U.check(np.allclose(pdp, U.pd_curve(predict, sample, "declared_income", grid)), "partial dependence matches the course function")
print(pd.Series(pdp.round(4), index=grid.astype(int)).to_string())

# %% [markdown]
# ## 2 · Accumulated local effects (4')
#
# ALE cuts declared income into bins at the grid values. Inside each bin it moves **only the applicants in
# that bin** from the bin's lower edge to its upper edge, averages the change, and adds the changes up
# from left to right. Every prediction it uses is for a nearby, realistic applicant.
#
# ✏️ **Exercise** · Call `U.ale_curve(predict, sample, "declared_income", grid)`.

# %%
def ale(df, col, grid):
    # BEGIN SOLUTION
    return U.ale_curve(predict, df, col, grid)
    # END SOLUTION

# %%
try:
    ale_v = ale(sample, "declared_income", grid)
except NotImplementedError:
    ale_v = U.ale_curve(predict, sample, "declared_income", grid)
U.check(len(ale_v) == len(grid), "one ALE value per grid edge")

# %% [markdown]
# ## 3 · Both curves, centred (5')
#
# ALE is centred on zero; centre the partial dependence on its mean so the two shapes can be compared.

# %%
pd_c = pdp - pdp.mean()
ale_c = ale_v - ale_v.mean()
curves = pd.DataFrame({"declared income": grid.astype(int), "PD (centred)": pd_c.round(4), "ALE (centred)": ale_c.round(4)})
print(curves.to_string(index=False))
fig, ax = plt.subplots()
ax.plot(grid, pd_c, marker="o", label="partial dependence")
ax.plot(grid, ale_c, marker="s", label="accumulated local effects")
ax.axhline(0, color="grey", lw=0.8)
ax.set_xlabel("declared income (SAR per month)"); ax.set_ylabel("change in predicted default probability")
ax.legend(frameon=False); ax.set_title("Two answers to one question", loc="left"); plt.show()
pd_rise, ale_rise = pd_c[-1] - pd_c[5], ale_c[-1] - ale_c[5]
print(f"from {grid[5]:,.0f} to {grid[-1]:,.0f} SAR: PD rises by {pd_rise:+.3f}, ALE by {ale_rise:+.3f}")
U.check(pd_rise > 1.3 * ale_rise > 0, "both rise at the top, and partial dependence rises much more")
U.keep("d3_2", curves.to_dict(orient="list"))

# %% [markdown]
# ## 4 · The impossible rows (6')
#
# Partial dependence at 19,000 riyals gives every applicant a declared income of 19,000, including the
# ones whose bank inflow is 4,000. Real applicants almost never look like that.
#
# ✏️ **Exercise** · For each grid value, return the share of the partial-dependence rows whose ratio of
# declared income to bank inflow lies outside the range of 98% of real applicants (the 1st to the 99th
# percentile of the real ratio).

# %%
real = sample.declared_income / sample.bank_inflow
lo, hi = np.quantile(real, [0.01, 0.99])


def outside_share(df: pd.DataFrame, grid) -> np.ndarray:
    # BEGIN SOLUTION
    return np.array([np.mean(((v / df.bank_inflow) < lo) | ((v / df.bank_inflow) > hi)) for v in grid])
    # END SOLUTION

# %%
try:
    share = outside_share(sample, grid)
except NotImplementedError:
    share = np.array([np.mean(((v / sample.bank_inflow) < lo) | ((v / sample.bank_inflow) > hi)) for v in grid])
print(f"real ratio range (1st to 99th percentile): {lo:.2f} to {hi:.2f}")
print(pd.Series((share * 100).round(1), index=grid.astype(int), name="% of PD rows outside the data").to_string())
U.check(share[-1] > 0.5 and share[0] > 0.5, "at both ends of the grid, most PD rows are applicants who do not exist")

# %% [markdown]
# ## 5 · Which curve for the report (4')
#
# ✏️ **Exercise** · Return the curve you would put in the explanation report and the sentence that goes
# under it.

# %%
def report_choice() -> tuple[str, str]:
    # BEGIN SOLUTION
    return "ALE", (f"Among realistic applicants, predicted risk changes little with declared income below about 12,000 SAR and "
                   f"rises above it, by {ale_rise:.2f} between {grid[5]:,.0f} and {grid[-1]:,.0f} SAR. Partial dependence shows "
                   f"{pd_rise / ale_rise:.1f} times that rise because, at the top of the grid, {share[-1]:.0%} of the rows it "
                   "averages over pair a high declared income with a bank inflow no real applicant has.")
    # END SOLUTION

# %%
try:
    choice, caption = report_choice()
except NotImplementedError:
    choice, caption = "ALE", "reference choice"
print(choice, "|", caption)
U.check(choice == "ALE", "the report uses the curve that stays inside the data")

# %% [markdown]
# **Stretch, done.** Individual conditional expectation (ICE): the partial-dependence curve of one
# applicant at a time. Their average is the PD curve.

# %%
few = sample.head(50)
ice = np.array([[predict(r.assign(declared_income=v))[0] for v in grid] for _, r in
                ((i, few.iloc[[i]]) for i in range(len(few)))])
fig, ax = plt.subplots()
ax.plot(grid, ice.T, color="#5aa", alpha=0.3, lw=1)
ax.plot(grid, ice.mean(0), color="black", lw=2, label="average (PD of these 50)")
ax.set_xlabel("declared income (SAR)"); ax.set_ylabel("predicted probability"); ax.legend(frameon=False)
ax.set_title("ICE curves: each applicant's own response", loc="left"); plt.show()
bend = int(np.sum(ice.max(1) - ice.min(1) > 0.1))
print(f"{bend} of 50 applicants move by more than 10 points across the grid")

# %% [markdown]
# **Look at.** Both curves are correct answers to different questions. PD asks "what if this applicant
# had declared a different income, with everything else unchanged?", which for income means "what if they
# overstated it?". ALE asks "how does risk change between similar real applicants?".
#
# ## Checkpoint
#
# <details><summary>1 · Why does PD rise with declared income?</summary>
# Holding bank inflow fixed while raising declared income creates overstatement, which the model has
# learnt to read as risk.</details>
#
# <details><summary>2 · When do PD and ALE agree?</summary>
# When the feature is roughly independent of the others: then setting it to any value creates realistic
# rows.</details>
#
# <details><summary>3 · Is the ALE curve a causal effect of income?</summary>
# No: it describes the model's predictions among similar applicants, not what would happen if an
# applicant earned more.</details>
#
# ## Carry forward
#
# The two curves are the second panel of the explanation report.
