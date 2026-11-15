# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "pandas>=2.2", "matplotlib>=3.9", "scikit-learn>=1.6", "lightgbm>=4.5"]
# ///
# %% [markdown]
# # Lab 1.1 · Six claims, three verdicts
#
# Day 1 · Session 1 (What a validator needs) · 25 min.
#
# You are Aseel Al-Rumaithi of the model risk team at Mizan Community Lending. The credit analytics
# team has sent you **Amana**, its credit model package, for independent validation. Before any method,
# read what the package claims and decide, for each claim, whether the evidence supports it, contradicts
# it or could never test it.
#
# **Reference state.** `package/` (model card, metadata, evidence, three models) and the validator's
# held-out rows in `data/applications.csv`. Nothing from another lab.
#
# **Tasks**
# - [ ] 1 · Open the package: card, metadata, evidence (5')
# - [ ] 2 · Recompute the AUC on the holdout (6')
# - [ ] 3 · Compare mean prediction with the default rate (4')
# - [ ] 4 · Mark the six claims with their evidence (7')
# - [ ] 5 · Rewrite one untestable claim as a test (3')
# - [ ] ✱ Stretch · Find out on which rows the developers measured their AUC, and say why it is higher

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

# %% [markdown]
# ## 1 · Open the package (5')
#
# A package is three things: what the developers **claim** (the model card), what they **did**
# (the metadata) and what they **measured** (the evidence). Read all three before you compute anything.

# %%
print(U.package_file("model_card.md"))
meta = json.loads(U.package_file("metadata.json"))
evidence = json.loads(U.package_file("evidence.json"))
print(json.dumps(evidence, indent=1))
print("features:", meta["features"])
print("excluded on purpose:", meta["excluded_on_purpose"])
print("training settings:", meta["training_settings"]["rebalanced"])

# %% [markdown]
# ✏️ **Exercise** · Return the set of columns the default model uses that are on the list of protected
# or location columns `{"gender", "age_band", "region"}`. An empty set supports claim 3.

# %%
def protected_in_features(meta: dict) -> set:
    # BEGIN SOLUTION
    return set(meta["features"]) & {"gender", "age_band", "region"}
    # END SOLUTION

# %%
# Check, with a safety net: the cells below still run when the exercise is not done yet
try:
    used = protected_in_features(meta)
except NotImplementedError:
    used = set(meta["features"]) & {"gender", "age_band", "region"}
U.check(used == set(), "the default model uses no gender, age band or region column")

# %% [markdown]
# ## 2 · Recompute the AUC on the holdout (6')
#
# The validator's **holdout** rows were never seen by the developers. The AUC is the probability that
# a randomly chosen defaulter gets a higher score than a randomly chosen non-defaulter. With ranks:
#
# `AUC = (sum of the ranks of the defaulters - n1 (n1 + 1) / 2) / (n1 n0)`
#
# ✏️ **Exercise** · Write it (average ranks for ties: `pd.Series(p).rank()`).

# %%
def auc(y, p) -> float:
    # BEGIN SOLUTION
    y = np.asarray(y)
    ranks = pd.Series(np.asarray(p)).rank(method="average").to_numpy()
    n1, n0 = y.sum(), len(y) - y.sum()
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    # END SOLUTION

# %%
hold = U.load("holdout")
p = U.p_default(hold)
try:
    auc_hold = auc(hold["default"], p)
except NotImplementedError:
    auc_hold = U.auc(hold["default"], p)
U.check(abs(auc_hold - U.auc(hold["default"], p)) < 1e-9, f"holdout AUC {auc_hold:.3f} on {len(hold):,} rows")
U.check(auc_hold < evidence["auc_holdout"] - 0.05, f"the card claims {evidence['auc_holdout']}: the holdout gives {auc_hold:.3f}")
U.keep("d1_1_auc", {"claimed": evidence["auc_holdout"], "holdout": round(auc_hold, 3)})

# %% [markdown]
# ## 3 · Mean prediction against the default rate (4')
#
# If the probabilities could be "read as default rates" (claim 2), the average predicted probability
# would be close to the share of applicants who defaulted.

# %%
mean_p, rate = float(p.mean()), float(hold["default"].mean())
print(f"mean predicted probability {mean_p:.3f} | observed default rate {rate:.3f} | ratio {mean_p / rate:.1f}")
U.check(mean_p > 3 * rate, "the model predicts about four times more defaults than happen")
U.keep("d1_1_mean", {"mean_p": round(mean_p, 3), "rate": round(rate, 3)})

# %% [markdown]
# ## 4 · Mark the six claims (7')
#
# Three verdicts:
#
# - **supported**: the package's evidence or your recomputation backs the claim as written;
# - **unsupported**: the evidence is missing, or it contradicts the claim;
# - **untestable**: no data you have, or could have today, can test the claim as written.
#
# ✏️ **Exercise** · Fill `verdicts` with one verdict and one line of evidence per claim.

# %%
def mark_claims() -> dict:
    # BEGIN SOLUTION
    return {
        1: ("unsupported", f"holdout AUC {auc_hold:.3f}, not {evidence['auc_holdout']}"),
        2: ("unsupported", f"mean prediction {mean_p:.2f} against a default rate of {rate:.2f}; trained on a 50/50 sample"),
        3: ("supported", "metadata: no gender, age band or region among the features"),
        4: ("unsupported", "evidence.json has no subgroup results"),
        5: ("untestable", "no data from the next three years exists"),
        6: ("untestable", "the model is fitted to observed associations; causes are not in the data"),
    }
    # END SOLUTION

# %%
KEY = {1: "unsupported", 2: "unsupported", 3: "supported", 4: "unsupported", 5: "untestable", 6: "untestable"}
try:
    verdicts = mark_claims()
except NotImplementedError:
    verdicts = {k: (v, "reference verdict") for k, v in KEY.items()}
table = pd.DataFrame([(k, v, e) for k, (v, e) in verdicts.items()], columns=["claim", "verdict", "evidence"])
print(table.to_string(index=False))
wrong = [k for k in KEY if verdicts[k][0] != KEY[k]]
U.check(not wrong, "every verdict matches the trainer's key" if not wrong else f"look again at claim(s) {wrong}")

# %% [markdown]
# ## 5 · Rewrite an untestable claim as a test (3')
#
# "The model will remain accurate for the next three years" cannot be tested today. A validator turns
# it into a condition that can be checked every month.
#
# ✏️ **Exercise** · Fill the four fields of a monitoring test for claim 5.

# %%
def claim5_as_test() -> dict:
    # BEGIN SOLUTION
    return {"metric": "AUC and calibration gap on scored applications whose outcome is known",
            "threshold": "AUC at least 0.72 and mean predicted within 1 point of the observed rate",
            "data": "every month's applications, scored 12 months later",
            "action": "refer the model back to validation when either fails two months running"}
    # END SOLUTION

# %%
try:
    test5 = claim5_as_test()
except NotImplementedError:
    test5 = {"metric": "AUC on matured applications", "threshold": "at least 0.72", "data": "monthly", "action": "revalidate"}
U.check(set(test5) == {"metric", "threshold", "data", "action"} and all(len(v) > 5 for v in test5.values()),
        "claim 5 rewritten as a test with a metric, a threshold, data and an action")

# %% [markdown]
# **Stretch, done.** The evidence says the AUC was measured on "a random 4,000 of the training rows,
# held out before training"; the metadata says the package was trained on **all** training rows. So the
# 4,000 rows were seen by the model that was shipped.

# %%
train = U.load("train")
auc_train = U.auc(train["default"], U.p_default(train))
print(f"AUC on the training rows: {auc_train:.3f} | claimed: {evidence['auc_holdout']} | holdout: {auc_hold:.3f}")
U.check(auc_train > auc_hold + 0.05, "the claimed AUC is close to the in-sample AUC, not to the holdout's")

# %% [markdown]
# **Look at.** Two of the six claims fail on numbers you computed in five minutes; two more could never
# be tested. Claim 3 is supported *as written*: Day 3 will show that excluding a column does not exclude
# what it carries.
#
# ## Checkpoint
#
# <details><summary>1 · Why is the developers' AUC not evidence for claim 1?</summary>
# It was measured on rows the shipped model was trained on. Only rows the model never saw can support a
# claim about new applicants.</details>
#
# <details><summary>2 · Claim 4 is not contradicted by anything yet. Why is it still "unsupported"?</summary>
# A claim is supported by evidence, not by the absence of contrary evidence. The package has no subgroup
# results at all.</details>
#
# <details><summary>3 · What makes claim 6 untestable rather than wrong?</summary>
# A model fitted to observational data shows associations. Whether a feature causes default needs a
# design (an experiment or a causal argument) that the package does not have.</details>
#
# ## Carry forward
#
# The six verdicts are the first section of your validation report. Lab 4.4 collects them from
# `data/reference/findings.json`, so it does not depend on this run.
