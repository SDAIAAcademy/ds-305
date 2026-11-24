---
course: "Uncertainty, Interpretability and Model Validation"
code: "DS-305"
version: "2027.2.0"
deck: "capstone"
day: 5
language: "en"
view: "trainee"
source: "https://sdaiaacademy.github.io/ds-305/slides/capstone.html"
---

# Uncertainty, Interpretability and Model Validation

## Capstone: Capstone: an independent validation

Eight-minute panel defence of a validation report, then four minutes of questions in which the panel picks one finding to rerun

#### [capstone#title] Capstone: an independent validation

*SDAIA Academy · Uncertainty, Interpretability and Model Validation*

Capstone: Eight-minute panel defence of a validation report, then four minutes of questions in which the panel picks one finding to rerun

Data Science Track · 4 teaching days + capstone day · 8 min talk and demo · 4 min Q&A · DS-305

#### [capstone#agenda] How the day runs

*Capstone*

- **The brief and the rubric**: what the committee expects, then a clean-clone check
- **Validation work**: the morning: finish the report on your package
- **Defences**: eight minutes each, four minutes of questions
- **Exit check and closing**: the AI-off exit check, what the room found, what comes next

#### [capstone#the-brief] Validate one package independently and defend a fit-for-use decision

*The brief*

- **Package.** Amana-R (top-up loans) or Amana-F (farm loans) from `labs/capstone/`, or a peer's model with its card, data and a later batch.
- **The week, repeated.** Recompute the claims, calibration, intervals and sets, explanation stability, the subgroup table, the later batch, the checklist.
- **The decision.** Fit, fit with conditions or not fit, with owned conditions and named human review.

- **Submission.** A link to a public GitHub repository holding the validation notebooks, the run logs, and a README that reports calibration, coverage, explanation stability, subgroup findings and the decision.

#### [capstone#report-sections] Seven sections, each with the number that supports it

*The README*

| Section | Evidence | From |
| --- | --- | --- |
| 1 · Claims | Each claim of the card: supported, unsupported or untestable | Lab 1.1 |
| 2 · Calibration | Brier and log loss before and after, reliability diagram | Labs 1.2 to 1.4 |
| 3 · Coverage | Target against achieved, by group and under shift, widths and set sizes | Day 2 |
| 4 · Explanations | Grouped importance, ALE, stability panel, one misleading result | Day 3 |
| 5 · Subgroups | The flagged, tagged and rated table | Lab 4.1 |
| 6 · Later batch | The memo: cause, evidence, action | Lab 4.2 |
| 7 · Checklist and decision | Twelve items scored; decision with owned conditions | Labs 4.3, 4.4 |

#### [capstone#rubric] Five criteria, 100 points

*How the validation is scored*

| Item | Value |
| --- | --- |
| Calibration and coverage | 25% |
| Explanations | 20% |
| Subgroup and shift findings | 25% |
| Decision | 20% |
| Reproducibility | 10% |

> **Full marks.** Every finding has a number, an interval where one applies, and the file that produced it.

#### [capstone#format] Eight minutes before the committee, then four minutes of questions

*The defence*

- **2 min · the package.** What it claims, and which claims failed on your numbers.
- **4 min · the findings.** The high findings, each with its evidence; the later batch's cause.
- **2 min · the decision.** The decision, the conditions and their owners.

> **The panel plays the committee.** In the questions it picks one finding and you rerun the cell that produced it.

#### [capstone#running-order] Who defends when

*Running order*

| # | Participant | Package and repository |
| --- | --- | --- |
| 1 |  |  |
| 2 |  |  |
| 3 |  |  |
| 4 |  |  |
| 5 |  |  |
| 6 |  |  |
| 7 |  |  |
| 8 |  |  |

#### [capstone#final-prep] Ten minutes: clone it clean, run one notebook

*Before the first defence*

Clone your public repository into a new folder and run one notebook from its README. If that fails, the panel's will too.

#### [capstone#now-presenting] Defence

*Now defending*

##### Questions

#### [capstone#review] What the validations showed

*After the defences*

- **Found by most.** Which planted problem did nearly everyone find, and with what evidence?
- **Missed by most.** Which one was quieter, and which check would have caught it?
- **Best condition.** Whose condition would you send to a real committee unchanged?

#### [capstone#what-next] Keep going

*After the course*

- **Follow-up task.** A one-page checklist review of a model from your workplace, or a published one, against the twelve items of Lab 4.3.
- **Keep the repository.** It is the evidence the DS Associate badge looks at.
- **Next Academy course:** DS-309, fairness mitigation and documentation, takes this week's findings to their remedies.

#### [capstone#thanks] Thank you

*SDAIA Academy*

Uncertainty, Interpretability and Model Validation
