# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "pandas>=2.2", "matplotlib>=3.9", "scikit-learn>=1.6", "lightgbm>=4.5", "mapie>=1.0"]
# ///
# %% [markdown]
# # Setup check
#
# Run this once before Day 1, in Colab or locally. It imports every library the labs use, reads the
# data, loads the three models of the package and scores one applicant. No key and no account are needed:
# no lab of this course calls a language model, and everything runs on a laptop processor.

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

# %%
import importlib

missing = []
for name in ("numpy", "pandas", "matplotlib", "sklearn", "lightgbm", "mapie"):
    try:
        mod = importlib.import_module(name)
        print(f"✓ {name} {getattr(mod, '__version__', '')}")
    except ImportError:
        missing.append(name)
        print(f"✗ {name} is missing")
print("Python", sys.version.split()[0])

# %%
import mizan as U

apps, later = U.load("all"), U.load("later")
U.check(len(apps) == 50_000 and len(later) == 10_000, "data: 50,000 applications and a later batch of 10,000")
one = U.load("holdout").head(1)
print("one applicant:", one[["app_id", "business_type", "loan_amount", "term_months"]].to_dict(orient="records")[0])
print(f"default probability {U.p_default(one)[0]:.3f} | grades A, B, C {U.p_grade(one)[0].round(3)} | loss if default {U.loss_pred(one)[0]:,.0f} SAR")
U.check(not missing, "every library is installed" if not missing else f"install: {missing}")
