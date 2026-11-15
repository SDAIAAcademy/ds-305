# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "pandas>=2.2", "matplotlib>=3.9", "scikit-learn>=1.6", "lightgbm>=4.5"]
# ///
# %% [markdown]
# # Lab 4.4 · The decision and the capstone choice
#
# Day 4 · Session 4 (The fit-for-use decision) · 25 min.
#
# Dr. Sameera Al-Fadda's approval committee meets on Amana next week. It needs one of three decisions
# (fit, fit with conditions, not fit), the conditions, who reviews which applicants, and what is monitored.
# You draft it from the week's findings, test that every high finding is answered, and then choose the
# package you will validate in the capstone.
#
# **Reference state.** `data/findings.json` (the week's findings, F1 to F11, as the labs found them) and
# the two capstone packages in `capstone/`.
#
# **Tasks**
# - [ ] 1 · Collect and rate the findings (6')
# - [ ] 2 · Draft the decision and its conditions (8')
# - [ ] 3 · Check every high finding has a condition (4')
# - [ ] 4 · Compare the capstone packages (4')
# - [ ] 5 · Choose one and plan it (3')
# - [ ] ✱ Stretch · Write the two sentences you would say to the approval committee if it asked to approve without conditions

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

findings = pd.DataFrame(json.loads((U.DATA / "findings.json").read_text(encoding="utf-8"))).set_index("id")
print(findings[["lab", "impact", "supported"]].to_string())

# %% [markdown]
# ## 1 · Rate the findings (6')
#
# The course rule: **high** when the impact is high and the evidence supports it; **medium** when one of
# the two holds; **low** otherwise.
#
# ✏️ **Exercise** · Add a column `rating`.

# %%
def rate(f: pd.DataFrame) -> pd.Series:
    # BEGIN SOLUTION
    hi = f.impact == "high"
    return pd.Series(np.where(hi & f.supported, "high", np.where(hi | f.supported, "medium", "low")), index=f.index)
    # END SOLUTION

# %%
try:
    findings["rating"] = rate(findings)
except NotImplementedError:
    hi = findings.impact == "high"
    findings["rating"] = np.where(hi & findings.supported, "high", np.where(hi | findings.supported, "medium", "low"))
print(findings[["rating", "finding"]].to_string())
HIGH = set(findings.index[findings.rating == "high"])
U.check(HIGH == {"F2", "F3", "F4", "F8", "F10", "F11"}, f"six high findings: {sorted(HIGH)}")

# %% [markdown]
# ## 2 · The decision and its conditions (8')
#
# Each condition says what must hold, who owns it and which findings it answers. Human review names which
# applicants an officer must look at, not "borderline cases".
#
# ✏️ **Exercise** · Fill the decision.

# %%
def draft_decision() -> dict:
    # BEGIN SOLUTION
    return {
        "decision": "fit with conditions",
        "conditions": [
            {"id": "C1", "answers": ["F2"], "owner": "credit analytics",
             "text": "Probabilities are used only after the Platt map fitted on the calib rows; the raw score is never shown."},
            {"id": "C2", "answers": ["F10"], "owner": "data engineering",
             "text": "No app-channel application is scored until the bank inflow feed is in riyals and the March to June applications are rescored; an input range check runs before every batch."},
            {"id": "C3", "answers": ["F3", "F4"], "owner": "branch credit officers",
             "text": "Applications from the Northern region and from applicants aged 18 to 24 go to an officer before any decision, until a group-aware recalibration is validated."},
            {"id": "C4", "answers": ["F8", "F9"], "owner": "credit analytics",
             "text": "Officers see one grouped income reason and only reasons stable over five refits; wallet months is not added."},
            {"id": "C5", "answers": ["F11", "F5", "F7"], "owner": "model risk",
             "text": "Monthly monitoring with a named owner: input ranges, the large-loan share, and calibration by region and age band on matured loans; a breach two months running returns the model to validation."},
        ],
        "human_review": "Officers review every application in the Northern region, aged 18 to 24, or with all three grades in its 90% set.",
        "review_date": "in six months, or at the first monitoring breach",
    }
    # END SOLUTION

# %%
try:
    decision = draft_decision()
except NotImplementedError:
    decision = {"decision": "fit with conditions", "conditions": [{"id": "C1", "answers": sorted(HIGH), "owner": "model risk", "text": "reference"}],
                "human_review": "reference", "review_date": "six months"}
print("DECISION:", decision["decision"])
for c in decision["conditions"]:
    print(f"  {c['id']} ({', '.join(c['answers'])}; {c['owner']}): {c['text']}")
print("HUMAN REVIEW:", decision["human_review"])
U.check(decision["decision"] in ("fit", "fit with conditions", "not fit"), "one of the three decisions")

# %% [markdown]
# ## 3 · Every high finding answered (4')

# %%
answered = {f for c in decision["conditions"] for f in c["answers"]}
missing = sorted(HIGH - answered)
print("high findings without a condition:", missing or "none")
U.check(not missing, "every high finding has a condition")
U.check(all(c.get("owner") for c in decision["conditions"]), "every condition has an owner")
U.keep("d4_4", decision)

# %% [markdown]
# ## 4 · The capstone packages (4')
#
# Two course packages wait in `capstone/`: **Amana-R** (top-up loans for repeat borrowers) and **Amana-F**
# (seasonal farm loans). Each has applications, a later batch, three models, a card, metadata and evidence.
# A first look tells you where each one's validation will spend its time.
#
# ✏️ **Exercise** · For each package, return the claimed AUC, the holdout AUC, the mean raw prediction
# against the default rate, and the later batch's AUC after a Platt map fitted on its calib rows.

# %%
import lightgbm as lgb


def first_look(name: str) -> dict:
    # BEGIN SOLUTION
    f = U.LABS / "capstone" / name
    d = pd.read_csv(f / "applications.csv")
    later = pd.read_csv(f / "later_batch.csv")
    m = lgb.Booster(model_file=str(f / "model_default.txt"))
    ev = json.loads((f / "evidence.json").read_text(encoding="utf-8"))
    h, c = d[d.split == "holdout"], d[d.split == "calib"]
    p_h = m.predict(U.X(h))
    platt = U.fit_platt(m.predict(U.X(c)), c["default"])
    p_l = platt(m.predict(U.X(later)))
    return {"claimed AUC": ev["auc_holdout"], "holdout AUC": round(U.auc(h["default"], p_h), 3),
            "mean raw prediction": round(float(p_h.mean()), 3), "default rate": round(float(h["default"].mean()), 3),
            "later AUC": round(U.auc(later["default"], p_l), 3),
            "later predicted": round(float(p_l.mean()), 3), "later observed": round(float(later["default"].mean()), 3)}
    # END SOLUTION

# %%
try:
    look = pd.DataFrame({n: first_look(n) for n in ("renewal", "farm")})
except NotImplementedError:
    look = pd.DataFrame({"renewal": {"claimed AUC": "see evidence.json"}, "farm": {"claimed AUC": "see evidence.json"}})
print(look.to_string())

# %% [markdown]
# ## 5 · Choose and plan (3')
#
# ✏️ **Exercise** · Return your choice (`"renewal"`, `"farm"` or `"peer"`) and the first three checks of
# your plan, in order.

# %%
def choose() -> tuple[str, list]:
    # BEGIN SOLUTION
    return "farm", ["recompute the claimed AUC and the calibration on the holdout",
                    "calibration and loss coverage by region, with intervals",
                    "the later batch by month: is the decline in every slice or one?"]
    # END SOLUTION

# %%
try:
    pick, plan = choose()
except NotImplementedError:
    pick, plan = "peer", ["agree the package with your peer"]
print(pick, plan)
U.check(pick in ("renewal", "farm", "peer") and len(plan) >= 1, "a package and a plan")

# %% [markdown]
# **Stretch, done.**

# %%
print("If the committee approves Amana without conditions, Northern and young applicants will keep being "
      "priced on risks 5 to 7 points lower than they are, and every app application since March is being "
      "scored on a broken input. The conditions are what make the model's evidence true in use.")

# %% [markdown]
# **Look at.** "Fit with conditions" is only as good as its owners and its monitoring. A condition without an
# owner is a wish; a review rule without a list of applicants is a slogan.
#
# ## Checkpoint
#
# <details><summary>1 · Why not "not fit" with two high calibration gaps and a broken feed?</summary>
# Each has a condition that removes the harm while the cause is fixed; the model ranks well and its
# probabilities are repaired. If no one will own the conditions, "not fit" is the honest decision.</details>
#
# <details><summary>2 · Why route the Northern region to officers rather than adjust its scores?</summary>
# A group adjustment is a modelling change that needs its own validation; review is available now.</details>
#
# <details><summary>3 · What makes the human review real rather than automation bias waiting to happen?</summary>
# Officers see the evidence (the set of grades, stable reasons), have authority to decide differently, and
# their overrides are monitored.</details>
#
# ## Carry forward
#
# The decision is the last section of the report; the capstone repeats the week on the package you chose.
