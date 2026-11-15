# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "pandas>=2.2", "matplotlib>=3.9", "scikit-learn>=1.6", "lightgbm>=4.5"]
# ///
# %% [markdown]
# # Lab 4.2 · The cause of the decline
#
# Day 4 · Session 2 (Diagnosing degradation) · 25 min.
#
# Amana has scored applications since January 2025. Now the outcomes of the first half of 2025 are known,
# and Talib Al-Harbash reports that the model "got worse". You find where, since when and why, from slice
# metrics, feature stability, coverage and explanations, and you write a memo with one cause and one action.
#
# **Reference state.** `data/later_batch.csv` (10,000 applications, January to June 2025, with outcomes),
# the package with the Platt repair, and the holdout rows as the reference period.
#
# **Tasks**
# - [ ] 1 · Score the batch by month and channel (6')
# - [ ] 2 · Feature stability in the failing slice (5')
# - [ ] 3 · Coverage and explanations in that slice (5')
# - [ ] 4 · Rule out the decoy (4')
# - [ ] 5 · Write the memo (5')
# - [ ] ✱ Stretch · Write the input check that would have stopped the batch on the first day

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
hold, later = U.load("holdout"), U.load("later")
p_hold, p_later = U.recalibrated(hold), U.recalibrated(later)
print(f"holdout 2023-24: AUC {U.auc(hold['default'], p_hold):.3f}, Brier {U.brier(hold['default'], p_hold):.4f}")
print(f"later batch 2025: AUC {U.auc(later['default'], p_later):.3f}, Brier {U.brier(later['default'], p_later):.4f}, "
      f"mean predicted {p_later.mean():.3f} against {later['default'].mean():.3f} observed")

# %% [markdown]
# ## 1 · Where and since when (6')
#
# ✏️ **Exercise** · Return a table with one row per month and channel: `n`, `auc`, `predicted`, `observed`.

# %%
def slice_metrics(df: pd.DataFrame, p) -> pd.DataFrame:
    # BEGIN SOLUTION
    d = df.assign(p=np.asarray(p))
    rows = [{"month": m, "channel": c, "n": len(g), "auc": U.auc(g["default"], g.p), "predicted": g.p.mean(),
             "observed": g["default"].mean()} for (m, c), g in d.groupby(["month", "channel"])]
    return pd.DataFrame(rows).set_index(["month", "channel"])
    # END SOLUTION

# %%
try:
    sl = slice_metrics(later, p_later)
except NotImplementedError:
    d = later.assign(p=p_later)
    sl = d.groupby(["month", "channel"]).apply(lambda g: pd.Series({"n": len(g), "auc": U.auc(g["default"], g.p),
                                                                       "predicted": g.p.mean(), "observed": g["default"].mean()}))
print(sl.round(3).unstack("channel").to_string())
bad = sl[(sl.predicted > 2 * sl.observed)]
print("slices predicting more than twice the observed rate:", list(bad.index))
U.check(set(bad.index.get_level_values("channel")) == {"app"} and bad.index.get_level_values("month").min() == "2025-03",
        "the failure is in the app channel, from March 2025")

fig, ax = plt.subplots(figsize=(7, 3.5))
for c in ("app", "branch"):
    s = sl.xs(c, level="channel")
    ax.plot(s.index, s.predicted, marker="o", label=f"{c}: predicted")
    ax.plot(s.index, s.observed, marker="x", ls="--", label=f"{c}: observed")
ax.set_ylabel("default rate"); ax.legend(frameon=False, ncol=2); ax.set_title("Later batch by month and channel", loc="left"); plt.show()

# %% [markdown]
# ## 2 · What changed in the inputs (5')
#
# The population stability index (PSI) compares a feature's distribution in a slice with the reference
# period: below 0.1 little change, above 0.25 a large one.
#
# ✏️ **Exercise** · For the failing slice (app, March onwards), return the PSI of every numeric feature
# against the holdout's app applicants, sorted from largest.

# %%
fail = later[(later.channel == "app") & (later.month >= "2025-03")]
ref_app = hold[hold.channel == "app"]
NUMERIC = [f for f in U.FEATURES if f not in U.CATEGORICAL]


def stability(ref: pd.DataFrame, cur: pd.DataFrame) -> pd.Series:
    # BEGIN SOLUTION
    return pd.Series({f: U.psi(ref[f], cur[f]) for f in NUMERIC}).sort_values(ascending=False)
    # END SOLUTION

# %%
try:
    psi = stability(ref_app, fail)
except NotImplementedError:
    psi = pd.Series({f: U.psi(ref_app[f], fail[f]) for f in NUMERIC}).sort_values(ascending=False)
print(psi.round(3).to_string())
print("\nbank inflow, median by month and channel (SAR):")
print(later.groupby(["month", "channel"]).bank_inflow.median().unstack().to_string())
U.check(psi.index[0] == "bank_inflow" and psi.iloc[0] > 1, "bank inflow has moved far more than anything else")

# %% [markdown]
# ## 3 · Coverage and explanations in the slice (5')

# %%
cal_d = U.defaults("calib")
q = U.conformal_quantile(np.abs(cal_d.loss_sar - U.loss_pred(cal_d)), 0.1)
rows = []
for name, part in (("holdout app", ref_app), ("2025 app from March", fail), ("2025 branch", later[later.channel == "branch"])):
    dd = part[part["default"] == 1]
    lo, hi = U.loss_interval(dd, q)
    sv = U.shap_values(U.model("default"), U.X(part))
    top = pd.Series([r[0] for r in U.top_reasons(sv, 1)]).value_counts(normalize=True)
    rows.append({"slice": name, "loss coverage": ((dd.loss_sar >= lo) & (dd.loss_sar <= hi)).mean(),
                 "defaults": len(dd), "bank_inflow top reason": top.get("bank_inflow", 0.0),
                 "mean bank_inflow value": sv.bank_inflow.mean()})
evidence = pd.DataFrame(rows).set_index("slice")
print(evidence.round(3).to_string())
U.check(evidence.loc["2025 app from March", "bank_inflow top reason"] > 2 * evidence.loc["holdout app", "bank_inflow top reason"],
        "in the failing slice bank inflow has become everyone's top reason")
U.keep("d4_2", {"psi": psi.round(3).to_dict(), "evidence": evidence.round(3).to_dict(orient="index")})

# %% [markdown]
# **Look at.** Loss interval coverage barely moves: the loss model's predictions use bank inflow less. The
# default model's explanation, on the other hand, has changed completely: bank inflow is now the top
# reason for most app applicants. A changed explanation is often the fastest clue.
#
# ## 4 · Rule out the decoy (4')
#
# Declared incomes are also higher in 2025. Is that the cause?
#
# ✏️ **Exercise** · Return the median declared income of the holdout and of the 2025 batch, by channel,
# and the AUC of the branch channel in 2025.

# %%
def decoy_check():
    # BEGIN SOLUTION
    med = pd.DataFrame({"holdout": hold.groupby("channel").declared_income.median(),
                        "2025": later.groupby("channel").declared_income.median()})
    br = later.channel == "branch"
    return med, U.auc(later.loc[br, "default"], p_later[br])
    # END SOLUTION

# %%
try:
    med, auc_branch = decoy_check()
except NotImplementedError:
    med = pd.DataFrame({"holdout": hold.groupby("channel").declared_income.median(),
                        "2025": later.groupby("channel").declared_income.median()})
    auc_branch = U.auc(later.loc[later.channel == "branch", "default"], p_later[later.channel == "branch"])
print(med.assign(change=(med["2025"] / med.holdout - 1).round(3)).to_string())
print(f"branch channel 2025 AUC {auc_branch:.3f}")
U.check(auc_branch > 0.72, "incomes rose in both channels, and the branch channel is unharmed: the rise is not the cause")

# %% [markdown]
# ## 5 · The memo (5')
#
# ✏️ **Exercise** · Fill the memo: the cause, the evidence (three numbers), the action, and what not to do.

# %%
def memo() -> dict:
    # BEGIN SOLUTION
    return {
        "cause": "From March 2025 the app channel's bank inflow arrives about a thousand times too small "
                 "(medians near 8 instead of about 8,000 SAR): a unit change in the data feed.",
        "evidence": f"app from March: predicted {fail.pipe(U.recalibrated).mean():.0%} against {fail['default'].mean():.0%} observed; "
                    f"PSI of bank inflow {psi['bank_inflow']:.1f}; bank inflow the top reason for "
                    f"{evidence.loc['2025 app from March', 'bank_inflow top reason']:.0%} of the slice.",
        "action": "Fix the feed to riyals, rescore the affected applications, and add an input range check before scoring.",
        "not": "Do not retrain on the 2025 batch: the model would learn the broken unit.",
    }
    # END SOLUTION

# %%
try:
    m = memo()
except NotImplementedError:
    m = {"cause": "bank inflow unit change in the app feed from March 2025", "evidence": "reference",
         "action": "fix the feed and rescore", "not": "do not retrain"}
for k, v in m.items():
    print(f"{k.upper()}: {v}")
U.check("bank inflow" in m["cause"] and "March" in m["cause"], "the memo names the feature and the start")

# %% [markdown]
# **Stretch, done.** A range check on each numeric input, fitted on the reference period: flag a day's
# batch when more than 1% of rows fall below the reference's 0.1th percentile or above its 99.9th.

# %%
bounds = {f: (hold[f].quantile(0.001), hold[f].quantile(0.999)) for f in NUMERIC}


def input_check(batch: pd.DataFrame) -> list:
    return [f for f, (a, b) in bounds.items() if np.mean((batch[f] < a) | (batch[f] > b)) > 0.01]


first_day = later[(later.month == "2025-03") & (later.channel == "app")].head(40)
print("first scored rows of March, app channel: alerts on", input_check(first_day))
print("February, app channel: alerts on", input_check(later[(later.month == "2025-02") & (later.channel == "app")]))

# %% [markdown]
# ## Checkpoint
#
# <details><summary>1 · Why look at slices before features?</summary>
# The slice tells you where and since when; that narrows which feature or feed to inspect.</details>
#
# <details><summary>2 · Overall AUC fell from 0.755 to about 0.68. Why is "the model degraded" the wrong memo?</summary>
# The model did not change; its input did. The branch channel in the same months is as good as before.</details>
#
# <details><summary>3 · Which bias type from the guide names this cause?</summary>
# Measurement bias: values systematically recorded too low.</details>
#
# ## Carry forward
#
# The memo is the evidence of outcome LO4 and finding F10 in `data/findings.json`.
