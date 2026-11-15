# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "pandas>=2.2", "matplotlib>=3.9", "scikit-learn>=1.6", "lightgbm>=4.5"]
# ///
# %% [markdown]
# # Lab 2.4 · An income shift, measured
#
# Day 2 · Session 4 (Coverage under shift) · 25 min.
#
# Mizan plans to market to salaried applicants with higher incomes. If next year's applicants earn more
# than the calib rows, are the loss intervals and the grade sets still right 90% of the time? You
# simulate the shift by **tilting** the holdout towards higher declared incomes at three strengths and
# measure the coverage lost.
#
# **Reference state.** `package/`; the calib and holdout rows. Nothing from another lab.
#
# **Tasks**
# - [ ] 1 · Write the shifted sample (5')
# - [ ] 2 · Loss interval coverage at three strengths (7')
# - [ ] 3 · Grade set coverage at three strengths (5')
# - [ ] 4 · Which loans lose coverage (5')
# - [ ] 5 · A monitoring rule from the result (3')
# - [ ] ✱ Stretch · Reweight the calib scores by the known shift and restore the coverage (weighted conformal)

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
cal_d, hold_d = U.defaults("calib"), U.defaults("holdout")
q_loss = U.conformal_quantile(np.abs(cal_d.loss_sar - U.loss_pred(cal_d)), 0.1)
cal, hold = U.load("calib"), U.load("holdout")
q_grade = U.conformal_quantile(U.lac_scores(U.p_grade(cal), U.grade_index(cal)), 0.1)
print(f"thresholds from the calib rows: loss {q_loss:,.0f} SAR, grade LAC {q_grade:.3f}")

# %% [markdown]
# ## 1 · A shifted sample (5')
#
# Give each holdout row a weight `exp(strength * z)`, where `z` is its standardised log declared income,
# and resample rows with those weights. Strength 0 is the holdout as it is; strength 1 makes a row one
# standard deviation richer about 2.7 times as likely to be drawn.
#
# ✏️ **Exercise** · Return a resampled frame of the same size, drawn with replacement with these weights
# (`rng.choice(n, n, replace=True, p=w)`).

# %%
def shifted(df: pd.DataFrame, strength: float, seed: int = U.SEED) -> pd.DataFrame:
    # BEGIN SOLUTION
    z = np.log(df.declared_income)
    z = (z - z.mean()) / z.std()
    w = np.exp(strength * z).to_numpy()
    rng = np.random.default_rng(seed)
    return df.iloc[rng.choice(len(df), len(df), replace=True, p=w / w.sum())].reset_index(drop=True)
    # END SOLUTION

# %%
try:
    s1 = shifted(hold_d, 1.0)
except NotImplementedError:
    shifted = U.shifted_sample
    s1 = shifted(hold_d, 1.0)
print(f"median declared income: holdout defaults {hold_d.declared_income.median():,.0f}, shifted (1.0) {s1.declared_income.median():,.0f} SAR")
U.check(s1.declared_income.median() > 1.2 * hold_d.declared_income.median(), "the shifted sample earns more")

# %% [markdown]
# ## 2 · Loss interval coverage under shift (7')
#
# ✏️ **Exercise** · For strengths 0, 0.5, 1 and 1.5, return the coverage and mean width of the loss
# intervals (same threshold `q_loss`) on the shifted holdout defaults, and the share of large loans.

# %%
def loss_under_shift(strengths) -> pd.DataFrame:
    # BEGIN SOLUTION
    rows = []
    for st in strengths:
        d = shifted(hold_d, st) if st else hold_d
        lo, hi = U.loss_interval(d, q_loss)
        rows.append({"strength": st, "coverage": float(((d.loss_sar >= lo) & (d.loss_sar <= hi)).mean()),
                     "width": float(np.mean(hi - lo)), "large share": float((U.loan_band(d) == "large").mean())})
    return pd.DataFrame(rows).set_index("strength")
    # END SOLUTION

# %%
strengths = [0, 0.5, 1.0, 1.5]
try:
    loss_t = loss_under_shift(strengths)
except NotImplementedError:
    rows = []
    for st in strengths:
        d = U.shifted_sample(hold_d, st) if st else hold_d
        lo, hi = U.loss_interval(d, q_loss)
        rows.append({"strength": st, "coverage": float(((d.loss_sar >= lo) & (d.loss_sar <= hi)).mean()),
                     "width": float(np.mean(hi - lo)), "large share": float((U.loan_band(d) == "large").mean())})
    loss_t = pd.DataFrame(rows).set_index("strength")
print(loss_t.round(3).to_string())
U.check(loss_t.loc[1.5, "coverage"] < loss_t.loc[0, "coverage"] - 0.1, "a strong shift costs the loss intervals more than 10 points of coverage")
U.check(loss_t.width.max() < 1.05 * loss_t.width.min(), "and the width barely moves: nothing tells the interval the population changed")

# %% [markdown]
# ## 3 · Grade set coverage under shift (5')

# %%
grade_rows = []
for st in strengths:
    d = U.shifted_sample(hold, st) if st else hold
    S = U.lac_sets(U.p_grade(d), q_grade)
    y = U.grade_index(d)
    grade_rows.append({"strength": st, "coverage": S[np.arange(len(y)), y].mean(), "mean size": S.sum(1).mean()})
grade_t = pd.DataFrame(grade_rows).set_index("strength")
print(grade_t.round(3).to_string())
U.check(grade_t.coverage.min() > 0.86, "the grade sets lose little: their errors do not grow with income")
table = loss_t[["coverage", "width"]].add_prefix("loss ").join(grade_t.add_prefix("grade "))
U.keep("d2_4", table.round(3).to_dict())

fig, ax = plt.subplots(figsize=(6, 3.5))
ax.plot(strengths, loss_t.coverage, marker="o", label="loss intervals")
ax.plot(strengths, grade_t.coverage, marker="s", label="grade sets")
ax.axhline(0.9, color="black", ls="--", lw=1); ax.set_xlabel("shift strength (towards higher income)")
ax.set_ylabel("coverage"); ax.legend(frameon=False); ax.set_title("The guarantee under a simulated shift", loc="left"); plt.show()

# %% [markdown]
# ## 4 · Which loans lose coverage (5')
#
# ✏️ **Exercise** · On the strength-1.0 sample, return coverage by loan-size band. Is the loss of coverage
# a change *within* the bands, or a change in how many large loans there are?

# %%
def by_band_under_shift(st: float = 1.0) -> pd.DataFrame:
    # BEGIN SOLUTION
    d = shifted(hold_d, st)
    lo, hi = U.loss_interval(d, q_loss)
    return U.coverage_by(d.assign(band=U.loan_band(d)), (d.loss_sar >= lo) & (d.loss_sar <= hi), "band")
    # END SOLUTION

# %%
try:
    bb = by_band_under_shift(1.0)
except NotImplementedError:
    d = U.shifted_sample(hold_d, 1.0)
    lo, hi = U.loss_interval(d, q_loss)
    bb = U.coverage_by(d.assign(band=U.loan_band(d)), (d.loss_sar >= lo) & (d.loss_sar <= hi), "band")
print(bb.to_string())
U.check(bb.loc["large", "coverage"] < 0.85, "within the large band coverage stays low; the shift adds more large loans")

# %% [markdown]
# ## 5 · A monitoring rule (3')
#
# ✏️ **Exercise** · Write the rule a validator would attach as a condition of use: what is watched, the
# trigger and the action.

# %%
def monitoring_rule() -> dict:
    # BEGIN SOLUTION
    return {"watch": "share of new loans above 25,000 SAR and the median declared income, monthly",
            "trigger": f"large-loan share above {loss_t.loc[0, 'large share'] + 0.1:.0%} (holdout: {loss_t.loc[0, 'large share']:.0%})",
            "action": "recompute the thresholds per loan-size band on recent defaults, or stop quoting the 90%"}
    # END SOLUTION

# %%
try:
    rule = monitoring_rule()
except NotImplementedError:
    rule = {"watch": "large-loan share", "trigger": "10 points above the holdout", "action": "recompute thresholds"}
print(rule)
U.check(set(rule) == {"watch", "trigger", "action"}, "a rule with what to watch, a trigger and an action")

# %% [markdown]
# **Stretch, done.** When the shift is **known** (here we made it), weighted conformal prediction restores
# the guarantee: each calib score is weighted by how much more likely its row is under the shift, and the
# threshold is a weighted quantile.

# %%
def weighted_threshold(scores, w_cal, w_test, alpha=0.1):
    order = np.argsort(scores)
    s, w = np.asarray(scores)[order], np.asarray(w_cal)[order]
    cum = np.cumsum(w) / (w.sum() + w_test)
    k = np.searchsorted(cum, 1 - alpha)
    return float(s[k]) if k < len(s) else float("inf")


st = 1.0
z_ref = np.log(hold_d.declared_income)
mu, sd = z_ref.mean(), z_ref.std()
w_of = lambda d: np.exp(st * (np.log(d.declared_income) - mu) / sd).to_numpy()
d = U.shifted_sample(hold_d, st)
s_cal = np.abs(cal_d.loss_sar - U.loss_pred(cal_d)).to_numpy()
w_cal, w_test = w_of(cal_d), w_of(d)
pred = U.loss_pred(d)
cov_w = np.mean([abs(y - p) <= weighted_threshold(s_cal, w_cal, wt) for y, p, wt in zip(d.loss_sar, pred, w_test)])
print(f"strength {st}: plain conformal {loss_t.loc[st, 'coverage']:.3f}, weighted conformal {cov_w:.3f}")
U.check(cov_w > loss_t.loc[st, "coverage"] + 0.03, "weighting by the known shift recovers most of the coverage")

# %% [markdown]
# **Look at.** The intervals did not widen when the population changed: a conformal interval knows only
# the calib rows. Coverage fell because large loans, whose losses vary most, became more common. This is
# why a validator writes conditions of use and a monitoring rule, not only an interval.
#
# ## Checkpoint
#
# <details><summary>1 · Which assumption did the shift break?</summary>
# Exchangeability: the new loans are no longer drawn like the calib loans.</details>
#
# <details><summary>2 · Why did the grade sets survive the same shift?</summary>
# Their errors do not depend much on income, so the calib scores still describe the shifted applicants.
# A shift breaks a guarantee only where it moves the errors.</details>
#
# <details><summary>3 · Can weighted conformal be used on next year's real applicants?</summary>
# Only with an estimate of the shift (a model of which applicants are more likely now), and the guarantee
# then depends on that estimate.</details>
#
# ## Carry forward
#
# Lab 4.2 meets a real shift in the 2025 batch; it starts from the package and the batch.
