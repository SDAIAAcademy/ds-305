# Uncertainty, Interpretability and Model Validation: labs

## Quick start

- Google Colab: open a notebook from the course page; its first cell clones this repository.
- Locally: clone the repository, then open the notebooks in `labs/` with Jupyter.

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
