# Directional recovery obstructions in sparsely controlled mixed-autonomy traffic

## Overview

This repository contains code and numerical source data for a study of finite-amplitude recovery in directed mixed-autonomy vehicle chains. The primary nonlinear study has 1,848 closed-loop runs. A separate 24-run OVM/FVD extension probes the prefix-obstruction mechanism under another human-driving law. Sampled-linear support-function calculations use a distinct model and disturbance family.

## Repository structure

- src/sparse_recovery/: simulation model, experiment definitions and sampled-linear calculations.
- configs/: configuration for the primary nonlinear grid.
- scripts/: verification, figure generation, analysis and optional simulation entry points.
- results/nonlinear_revision/: primary run table, event logs, representative trajectories and boundary records.
- results/ovm_prefix_extension/: separate OVM/FVD trajectories, gap traces and witness validation.
- results/revision_package/: sampled-linear calculations and response coefficients.
- derived_results/: classification, paired statistics, predictor diagnostics and figure source tables.
- source_data/README.md and SOURCE_DATA_README.md: mapping from every display to its numerical source.
- manuscript/: LaTeX main text, supplement and bibliography.
- figures/results/: PDF figures used by the manuscript. The scripts regenerate these from the included data.

## Main datasets

The 1,848-run nonlinear table is results/nonlinear_revision/raw_runs.csv. The 24-run OVM/FVD extension is results/ovm_prefix_extension/raw_added_runs.csv and is not appended to the primary table. These two datasets should be analyzed separately. The sampled-linear model and its response coefficients are in results/revision_package/.

The original run table is preserved as a scientific input. Verification scripts inspect it without rerunning the full simulation grid.

## Reproducing verification and figures

Tested with Python 3.9.2, NumPy, pandas, Matplotlib and SciPy. Install dependencies from requirements.txt. From the repository root:

~~~text
python scripts/check_v6_evidence.py
python scripts/check_v7_extension.py
python scripts/draw_framework.py
python scripts/plot_v5_evidence.py
~~~

The first two commands verify 198/198 IDM prefix-witness records, three OVM/FVD trace witnesses, the primary count of 1,848 runs, and the representative wave's pre-collision normalized distance deficit. The figure scripts regenerate five main and six supplementary figure groups from the included numerical records; they do not rerun traffic simulations. The framework drawing is vector-native and contains no synthetic data curves.

For an optional independent recreation of only the 24-run extension, run python scripts/run_ovm_prefix_extension.py. This writes to results/ovm_prefix_extension/ and does not replace the 1,848-run primary table. The large-grid entry point scripts/run_revision.py is not needed for manuscript reproduction.

## Compiling the manuscript

Compile supplement.tex first, then main.tex, because main.tex imports supplementary labels. TeX Live 2026 with latexmk was used for the submitted PDF build. From manuscript/:

~~~text
latexmk -pdf -interaction=nonstopmode -halt-on-error '-outdir=../build/supplement' supplement.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error '-outdir=../build/main' main.tex
~~~

The command examples use PowerShell quoting. The combined-review source includes the resulting two PDFs; copy them to the repository root as supplement_submission.pdf and main_submission.pdf before compiling combined_review.tex.

## Source-data mapping and data dictionary

SOURCE_DATA_README.md maps each main and supplementary display to the files used to generate it. In source code and historical data fields, predictive means endpoint-screened control, and margin means margin-priority endpoint-screened control. The legacy metric field false_recoverable_fraction denotes the manuscript's recorded-endpoint exceedance fraction, not a collision probability. A Class B record is a positive-gap crossing below the 4-m minimum in the uncontrolled preceding prefix before prefix collision. Class C is an observed controller failure without that witness; Class D is an unevaluated grid cell.

Fixed seeds regenerate matched human-driver parameter sequences, but the realized parameter vectors were not separately stored in the primary archive. The source-data map identifies the seed records and derived parameter reconstruction.

## Known scope

The nonlinear observations concern synthetic car-following laws, prescribed pulses, fixed single-lane order and a 60-s horizon. Scan-ceiling observations at requested 4.5 m s^-2 are right-censored. The sampled-linear reserve is exact only for its stated model, fixed policy and disturbance box; it is not a nonlinear certificate. Predictor errors are based on held configurations with coarse or censored endpoint labels. Time frames and repeated amplitudes are correlated observations, not independent samples. Post-collision continuation is retained in logs but is excluded from physical terminal interpretation.

## Citation

Citation metadata will be added after publication. Until then, cite the manuscript title and this repository's release tag when available.
