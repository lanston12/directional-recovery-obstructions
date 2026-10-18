# Communications Engineering final submission snapshot

Article: Directional recovery obstructions in sparsely controlled mixed-autonomy traffic

Authors: Jianbo Feng (corresponding author) and Shahid Mumtaz.

This final snapshot is based solely on the CE-transfer snapshot v1.0.1-ce-transfer. It updates figure geometry/lettering, schematic annotations, manuscript presentation and the author-confirmed NTU Clifton Campus address (Nottingham NG11 8NS, UK). It adds no simulations, controller changes, parameter changes or scientific claims. All scientific numerical files, model code and configurations preserve their original hashes.

- Scientific data and original code: https://github.com/lanston12/directional-recovery-obstructions/releases/tag/v1.0.0
- CE transfer: https://github.com/lanston12/directional-recovery-obstructions/releases/tag/v1.0.1-ce-transfer
- Final manuscript/plot snapshot: https://github.com/lanston12/directional-recovery-obstructions/releases/tag/v1.0.2-ce-final
- File-level scientific hashes: FINAL_REPRODUCIBILITY_MANIFEST.json
- Display-to-data mapping: SOURCE_DATA_README.md

Use a checkout of the final tag, with Python dependencies in requirements.txt:

    python scripts/check_v6_evidence.py
    python scripts/check_v7_extension.py
    python scripts/draw_framework.py
    python scripts/plot_v5_evidence.py

The first two commands only read saved scientific records. Plotting reads saved arrays; it does not rerun the original nonlinear grid. For final formatting-only QA, the local preparation directory also contains qa/figure_checks.py and qa/validate_freeze.py; these QA materials are not part of the journal upload ZIP.

From manuscript/, compile in order:

    latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=../build/supplement supplement.tex
    latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=../build/main main.tex
    latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=../build/cover_letter cover_letter.tex

The main article reads supplement labels from build/supplement/supplement.aux. All figures stay in the article body; main display legends are collected after references. The source ZIP includes every PDF required for compilation. The public scientific archive remains the source of the original data, not a moving default branch. No final journal submission is performed by preparing this package.
