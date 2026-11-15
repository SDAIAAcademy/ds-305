# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "pandas>=2.2", "matplotlib>=3.9", "scikit-learn>=1.6", "lightgbm>=4.5"]
# ///
# %% [markdown]
# # Lab 4.3 · The package against a checklist
#
# Day 4 · Session 3 (Checklists and model cards) · 25 min.
#
# A reporting checklist turns "is it documented?" into items that are each present, partial or missing,
# with the file that shows it. The course checklist has twelve items in three parts (conceptual soundness,
# outcomes, monitoring), written for this course from the REFORMS reporting standards and model cards. You
# score Amana against it and turn the gaps into a request to the developers.
#
# **Reference state.** `package/` and `data/checklist.json`.
#
# **Tasks**
# - [ ] 1 · Read the checklist (4')
# - [ ] 2 · Score each item with its evidence (9')
# - [ ] 3 · Count by part: soundness, outcomes, monitoring (4')
# - [ ] 4 · Write the request for missing evidence (5')
# - [ ] 5 · Which gap blocks a decision (3')
# - [ ] ✱ Stretch · Score one of the two capstone packages with the same checklist

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
import pandas as pd
import mizan as U

checklist = U.checklist()
card = U.package_file("model_card.md")
meta = json.loads(U.package_file("metadata.json"))
evidence = json.loads(U.package_file("evidence.json"))

# %% [markdown]
# ## 1 · The checklist (4')

# %%
print(pd.DataFrame(checklist).to_string(index=False))
print("\n--- metadata.json keys:", list(meta))
print("--- evidence.json:", evidence)

# %% [markdown]
# ## 2 · Score each item (9')
#
# Three statuses: **present** (the package answers the item fully), **partial** (it answers part, or
# answers it in a way you showed to be wrong), **missing**. Name the file and field that decides it.
#
# ✏️ **Exercise** · Fill `scores` for all twelve items.

# %%
def score_package() -> dict:
    # BEGIN SOLUTION
    return {
        "S1": ("partial", "model_card.md: intended use stated; no uses it must not serve"),
        "S2": ("present", "metadata.json target: default within 12 months"),
        "S3": ("present", "metadata.json training_rows and training_settings.rebalanced"),
        "S4": ("partial", "features and excluded_on_purpose listed; no reason, no time each is known"),
        "S5": ("partial", "training_settings has recipe and seed; no library version"),
        "O1": ("partial", "evidence.json AUC measured on rows the shipped model trained on"),
        "O2": ("missing", "evidence.json calibration_checked: false"),
        "O3": ("missing", "no interval or set for any of the three models"),
        "O4": ("missing", "evidence.json subgroup_results: null"),
        "O5": ("partial", "metadata.json names Shapley reasons; stability_checked: false"),
        "M1": ("missing", "metadata.json monitoring: none defined"),
        "M2": ("missing", "no conditions of use or human review anywhere in the package"),
    }
    # END SOLUTION

# %%
KEY = {"S1": "partial", "S2": "present", "S3": "present", "S4": "partial", "S5": "partial", "O1": "partial",
       "O2": "missing", "O3": "missing", "O4": "missing", "O5": "partial", "M1": "missing", "M2": "missing"}
try:
    scores = score_package()
except NotImplementedError:
    scores = {k: (v, "reference") for k, v in KEY.items()}
table = pd.DataFrame([{**c, "status": scores[c["id"]][0], "evidence": scores[c["id"]][1]} for c in checklist])
print(table[["id", "status", "evidence"]].to_string(index=False))
diff = [k for k in KEY if scores[k][0] != KEY[k]]
print("differences from the trainer's key:", diff or "none")
U.check(len(diff) <= 2, "at most two items differ from the key (discuss them with the trainer)")

# %% [markdown]
# ## 3 · Count by part (4')

# %%
counts = table.groupby(["part", "status"]).size().unstack(fill_value=0).reindex(["soundness", "outcomes", "monitoring"])
print(counts.to_string())
U.check(counts.loc["monitoring"].get("present", 0) == 0, "nothing about monitoring is present")
U.keep("d4_3", counts.to_dict(orient="index"))

# %% [markdown]
# ## 4 · The request to the developers (5')
#
# Some gaps you have filled yourself this week (calibration, intervals, subgroups, stability); the report
# says so. Others only the developers or the business can fill.
#
# ✏️ **Exercise** · Return the list of items you request from the developers, each with what is needed.

# %%
def request(table: pd.DataFrame) -> list:
    # BEGIN SOLUTION
    asks = {"S1": "uses the model must not serve (for example pricing without review)",
            "S4": "the reason for each exclusion and when each feature is known",
            "S5": "library versions, so the model can be refitted exactly",
            "O1": "the AUC of the shipped model on rows it never saw, or a statement that 0.864 belongs to another model",
            "M1": "inputs and outcomes to monitor, thresholds and a named owner",
            "M2": "conditions of use and how officers review declined applicants"}
    return [(i, asks[i]) for i in table.loc[table.status != "present", "id"] if i in asks]
    # END SOLUTION

# %%
try:
    asks = request(table)
except NotImplementedError:
    asks = [("M1", "monitoring plan"), ("M2", "conditions of use")]
for i, a in asks:
    print(f"{i}: {a}")
U.check({"M1", "M2"} <= {i for i, _ in asks}, "monitoring and conditions of use are requested")

# %% [markdown]
# ## 5 · The blocking gap (3')
#
# ✏️ **Exercise** · Return the one item without which no "fit with conditions" decision can be enforced,
# and why.

# %%
def blocking() -> tuple[str, str]:
    # BEGIN SOLUTION
    return "M1", "conditions of use need someone named to watch inputs and outcomes and act on a threshold"
    # END SOLUTION

# %%
try:
    item, why = blocking()
except NotImplementedError:
    item, why = "M1", "reference"
print(item, "|", why)
U.check(item in ("M1", "M2"), "the blocking gap is in monitoring")

# %% [markdown]
# **Stretch, done.** The renewal package's evidence file, against the same items that the evidence decides.

# %%
ev_r = json.loads((U.LABS / "capstone" / "renewal" / "evidence.json").read_text(encoding="utf-8"))
print(ev_r)
print("O2:", "present" if ev_r["calibration_checked"] else "missing", "| O4:", "present" if ev_r["subgroup_results"] else "missing",
      "| O5:", "partial" if not ev_r["stability_checked"] else "present")

# %% [markdown]
# **Look at.** Five items missing and five partial is not unusual for a first submission. The checklist
# makes the conversation with Rasil's team specific: twelve items, each with a file.
#
# ## Checkpoint
#
# <details><summary>1 · O1 has a number in the evidence file. Why "partial" and not "present"?</summary>
# The number belongs to rows the shipped model trained on (Lab 1.1).</details>
#
# <details><summary>2 · You computed calibration yourself. Is O2 still missing?</summary>
# For the package, yes; the report states that the validator supplied the evidence and that the
# developers must keep it current.</details>
#
# <details><summary>3 · Why is the checklist the course's, not REFORMS itself?</summary>
# REFORMS is written for research that uses machine learning; the course adapts the relevant items to a
# credit model package and adds monitoring. The report names both sources.</details>
#
# ## Carry forward
#
# The scored checklist is a section of the report and finding F11 in `data/findings.json`.
