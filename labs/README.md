# Uncertainty, Interpretability and Model Validation: labs

Every lab works on **Amana**, the credit model package that Mizan Community Lending (a fictional
microfinance lender) has sent to its model risk team for validation. All data are synthetic, written by
`prepare_data.py` with seed 305 (the replay file by `_tools/replay_examples.py` from the later batch), and every lab runs on a laptop CPU or in Colab. No lab calls a language
model, downloads a model or needs a key.

## Quick start

- **Google Colab:** open a notebook from the table below; its first cell clones this repository and
  installs `requirements-colab.txt` (only MAPIE is missing from Colab's image).
- **Locally, with [uv](https://docs.astral.sh/uv/):** from `labs/`, run `uv sync`, then
  `uv run jupyter lab`. Python 3.12 and the exact versions are pinned in `pyproject.toml` and `uv.lock`.
  On Windows, if a path is too long for the environment, set `UV_PROJECT_ENVIRONMENT` to a short folder
  outside the repository (for example `C:\uv305`) before `uv sync`.
- **Check the setup first:** `00_setup_check.ipynb` loads the data and the package in under a minute.

**Quick mode.** Set the environment variable `MIZAN_QUICK=1` to run the slower labs (refits, repeats)
with fewer rounds; the order of results is the same and the answer key notes the numbers that change.

**Each lab starts from a reference state** that it rebuilds itself from `data/` and `package/`, so a
participant who missed a lab can start the next one. Solutions are in `solutions/`; the trainer's
numbers and the capstone packages' planted problems are in `solutions/answer_key.md`.

| Folder | Contents |
|---|---|
| `data/` | applications (train, calib, holdout), the 2025 later batch, a replay of 60 of its app applications through both feature paths, the checklist, the bias guide's types, the week's findings |
| `package/` | Amana: three LightGBM models (default, grade, loss), model card, metadata, evidence |
| `capstone/` | two more packages for the capstone: `renewal` (Amana-R) and `farm` (Amana-F) |
| `mizan.py` | the course functions the labs import as `U` |

| Unit | Session | Notebook |
|---|---|---|
| Day 1 | Session 1 · What a validator needs | [`D1_1_mark_six_model.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-305/blob/main/labs/D1_1_mark_six_model.ipynb) |
| Day 1 | Session 2 · Proper scores and reliability | [`D1_2_compute_brier_score.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-305/blob/main/labs/D1_2_compute_brier_score.ipynb) |
| Day 1 | Session 3 · Recalibration | [`D1_3_recalibrate_two_methods.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-305/blob/main/labs/D1_3_recalibrate_two_methods.ipynb) |
| Day 1 | Session 4 · Calibration by subgroup | [`D1_4_break_calibration_down.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-305/blob/main/labs/D1_4_break_calibration_down.ipynb) |
| Day 2 | Session 1 · Conformal intervals | [`D2_1_wrap_loss_regression.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-305/blob/main/labs/D2_1_wrap_loss_regression.ipynb) |
| Day 2 | Session 2 · Prediction sets | [`D2_2_build_prediction_sets.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-305/blob/main/labs/D2_2_build_prediction_sets.ipynb) |
| Day 2 | Session 3 · Coverage by group | [`D2_3_measure_subgroup_coverage.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-305/blob/main/labs/D2_3_measure_subgroup_coverage.ipynb) |
| Day 2 | Session 4 · Coverage under shift | [`D2_4_simulate_income_shift.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-305/blob/main/labs/D2_4_simulate_income_shift.ipynb) |
| Day 3 | Session 1 · Permutation importance | [`D3_1_compute_permutation_importance.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-305/blob/main/labs/D3_1_compute_permutation_importance.ipynb) |
| Day 3 | Session 2 · Effect plots | [`D3_2_show_where_partial.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-305/blob/main/labs/D3_2_show_where_partial.ipynb) |
| Day 3 | Session 3 · Shapley values | [`D3_3_explain_ten_applicants.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-305/blob/main/labs/D3_3_explain_ten_applicants.ipynb) |
| Day 3 | Session 4 · When explanations mislead | [`D3_4_show_how_correlated.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-305/blob/main/labs/D3_4_show_how_correlated.ipynb) |
| Day 4 | Session 1 · Subgroup validation | [`D4_1_tabulate_calibration_coverage.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-305/blob/main/labs/D4_1_tabulate_calibration_coverage.ipynb) |
| Day 4 | Session 2 · Diagnosing degradation | [`D4_2_find_seeded_cause.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-305/blob/main/labs/D4_2_find_seeded_cause.ipynb) |
| Day 4 | Session 3 · Checklists and model cards | [`D4_3_score_model_package.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-305/blob/main/labs/D4_3_score_model_package.ipynb) |
| Day 4 | Session 4 · The fit-for-use decision | [`D4_4_draft_decision_conditions.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-305/blob/main/labs/D4_4_draft_decision_conditions.ipynb) |

The notebooks are generated from the percent-format sources in `labs/_src/` by `labs/_tools/py2nb.py` (`uv run python _tools/py2nb.py` from `labs/`), which also writes the solutions to `labs/solutions/`. Edit the sources, not the notebooks.
