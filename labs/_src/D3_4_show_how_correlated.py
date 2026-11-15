# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "pandas>=2.2", "matplotlib>=3.9", "scikit-learn>=1.6", "lightgbm>=4.5"]
# ///
# %% [markdown]
# # Lab 3.4 · A proxy takes the credit
#
# Day 3 · Session 4 (When explanations mislead) · 25 min.
#
# Rasil's team proposes version 2.2 of Amana with one new feature: **mobile-wallet months**, how long the
# applicant has used a mobile wallet. It slightly improves the AUC. Age is excluded from the model on
# purpose. You check what the new feature does to the reasons officers see, and for whom.
#
# **Reference state.** The training and holdout rows; `U.refit_default` refits with the developers' recipe.
#
# **Tasks**
# - [ ] 1 · Refit with wallet months added (5')
# - [ ] 2 · Shapley values with and without it (5')
# - [ ] 3 · How often it is a top reason, by age band (6')
# - [ ] 4 · Its link with age (4')
# - [ ] 5 · Write the caution (5')
# - [ ] ✱ Stretch · Remove the proxy and check whether the age-band calibration gap of Lab 1.4 changed

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
F_OLD = U.FEATURES
F_NEW = U.FEATURES + ["wallet_months"]
print("the proposal adds:", [f for f in F_NEW if f not in F_OLD])

# %% [markdown]
# ## 1 · Refit with and without the new feature (5')
#
# Same recipe, same seed, all training rows: the only difference is the feature list.
#
# ✏️ **Exercise** · Return the two boosters.

# %%
def two_models():
    # BEGIN SOLUTION
    return (U.refit_default(1, features=F_OLD, bootstrap=False), U.refit_default(1, features=F_NEW, bootstrap=False))
    # END SOLUTION

# %%
try:
    m_old, m_new = two_models()
except NotImplementedError:
    m_old, m_new = U.refit_default(1, features=F_OLD, bootstrap=False), U.refit_default(1, features=F_NEW, bootstrap=False)
auc_old = U.auc(hold["default"], m_old.predict(U.X(hold, F_OLD)))
auc_new = U.auc(hold["default"], m_new.predict(U.X(hold, F_NEW)))
print(f"holdout AUC without wallet months {auc_old:.3f}, with {auc_new:.3f}")
U.check(auc_new >= auc_old - 0.002, "the new feature does not hurt the ranking")

# %% [markdown]
# ## 2 · Shapley values with and without (5')

# %%
sv_old = U.shap_values(m_old, U.X(hold, F_OLD))
sv_new = U.shap_values(m_new, U.X(hold, F_NEW))
top_old = U.top_reasons(sv_old, 3)
top_new = U.top_reasons(sv_new, 3)
in_top = pd.Series(["wallet_months" in r for r in top_new], index=hold.index)
print(f"wallet months is one of the three reasons for {in_top.mean():.1%} of holdout applicants")
mean_abs = sv_new.drop(columns="base").abs().mean().sort_values(ascending=False).round(3)
print("mean absolute Shapley value, new model:\n" + mean_abs.to_string())

# %% [markdown]
# ## 3 · Who is told "short wallet history" (6')
#
# ✏️ **Exercise** · Return, by age band, the share of applicants for whom wallet months is among the three
# reasons, and the mean Shapley value of wallet months.

# %%
def by_age(hold: pd.DataFrame, in_top: pd.Series, sv_new: pd.DataFrame) -> pd.DataFrame:
    # BEGIN SOLUTION
    return pd.DataFrame({"n": hold.groupby("age_band").size(),
                         "share with wallet reason": in_top.groupby(hold.age_band).mean(),
                         "mean wallet value": sv_new.wallet_months.groupby(hold.age_band).mean()})
    # END SOLUTION

# %%
try:
    t_age = by_age(hold, in_top, sv_new)
except NotImplementedError:
    t_age = pd.DataFrame({"n": hold.groupby("age_band").size(),
                          "share with wallet reason": in_top.groupby(hold.age_band).mean(),
                          "mean wallet value": sv_new.wallet_months.groupby(hold.age_band).mean()})
print(t_age.round(3).to_string())
U.check(t_age.loc["55+", "share with wallet reason"] > 1.5 * t_age.loc["18-24", "share with wallet reason"],
        "applicants over 55 are told 'short wallet history' far more often than the youngest")
U.check(t_age.loc["55+", "mean wallet value"] > 0, "for the oldest band the feature pushes risk up on average")
U.keep("d3_4", t_age.round(3).to_dict(orient="index"))

# %% [markdown]
# ## 4 · Its link with age (4')

# %%
age_idx = hold.age_band.map({a: i for i, a in enumerate(["18-24", "25-34", "35-44", "45-54", "55+"])})
r = np.corrcoef(hold.wallet_months, age_idx)[0, 1]
print(f"correlation of wallet months with age band: {r:.2f}")
print(hold.groupby("age_band").wallet_months.median().rename("median wallet months").to_string())
U.check(r < -0.3, "older applicants have much shorter wallet histories")

fig, ax = plt.subplots(figsize=(6, 3.5))
ax.bar(t_age.index, t_age["share with wallet reason"], color="#c75")
ax.set_ylabel("share told 'wallet months'"); ax.set_title("Who gets the new reason", loc="left"); plt.show()

# %% [markdown]
# ## 5 · The caution (5')
#
# ✏️ **Exercise** · Write the caution for the validation report: what the feature carries, who receives
# the reason, and what you recommend.

# %%
def caution() -> str:
    # BEGIN SOLUTION
    return (f"Wallet months correlates with age (r = {r:.2f}) and is a top-three reason for "
            f"{t_age.loc['55+', 'share with wallet reason']:.0%} of applicants over 55 against "
            f"{t_age.loc['18-24', 'share with wallet reason']:.0%} of those aged 18 to 24. Shown to officers or applicants, "
            "'short wallet history' would in effect tell older applicants that their age counts against them, in a model "
            "that excludes age. Recommend: do not add the feature for an AUC gain of "
            f"{auc_new - auc_old:+.4f}; if added, do not show it as a reason, and audit by age band.")
    # END SOLUTION

# %%
try:
    text = caution()
except NotImplementedError:
    text = "reference caution: wallet months is an age proxy; do not show it as a reason"
print(text)
U.check("age" in text, "the caution names what the feature carries")

# %% [markdown]
# **Stretch, done.** Calibration by age band after Platt scaling, with and without the proxy.

# %%
cal = U.load("calib")
gaps = {}
for name, m, F in (("without", m_old, F_OLD), ("with", m_new, F_NEW)):
    platt = U.fit_platt(m.predict(U.X(cal, F)), cal["default"])
    p = platt(m.predict(U.X(hold, F)))
    gaps[name] = U.calibration_by(hold, p, "age_band").gap
print(pd.DataFrame(gaps).round(4).to_string())

# %% [markdown]
# **Look at.** The new feature narrows the age gap a little because it carries some of what age carries:
# that is exactly why it is a proxy. "The model uses no age" (claim 3 of the card) stays true as written
# and becomes misleading in effect.
#
# ## Checkpoint
#
# <details><summary>1 · Why does the proxy raise the AUC?</summary>
# Age is related to default in these data, and the proxy brings part of that information back.</details>
#
# <details><summary>2 · Can the validator prove wallet months is a proxy and not a cause?</summary>
# Not from these data alone. What can be shown is its link with age and who receives the reason; that is
# enough for a caution.</details>
#
# <details><summary>3 · Is removing the feature the end of the matter?</summary>
# No: other features can also carry age (years in business, prior loans). The audit by age band stays a
# condition of use.</details>
#
# ## Carry forward
#
# The caution is the "documented misleading result" of the explanation report (outcome LO3).
