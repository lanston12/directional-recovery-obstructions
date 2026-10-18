# Communications Engineering transfer snapshot

Title: Directional recovery obstructions in sparsely controlled mixed-autonomy traffic

Authors: Jianbo Feng and Shahid Mumtaz.

The original scientific data and code are preserved in v1.0.0. The v1.0.1-ce-transfer snapshot changes the manuscript's engineering interpretation, adds a compact placement-screening table from existing counts, restores the author-approved declarations and records full AI assistance. It corrects the described bootstrap resample count to 20,000, matching the unchanged analysis script and saved confidence intervals.

The only plotting-code modification moves the representative-gap legend outside its axes and shortens the descriptive title. No simulation arrays, constraints, seeds, response coefficients, statistics or recovery outcomes were modified. Fig. 1 remains the original native vector diagram. TRANSFER_REPRODUCIBILITY_MANIFEST.json records the frozen numerical hashes and revised source hashes.

Release: https://github.com/lanston12/directional-recovery-obstructions/releases/tag/v1.0.1-ce-transfer

Build the supplement before the main manuscript, from manuscript/:

    latexmk -pdf -outdir=../build/supplement supplement.tex
    latexmk -pdf -outdir=../build/main main.tex

Read-only checks from the repository root:

    python scripts/check_v6_evidence.py
    python scripts/check_v7_extension.py

Figure generation reads saved records:

    python scripts/draw_framework.py
    python scripts/plot_v5_evidence.py

The release source archive includes required vector figures and bibliography. Original numerical records remain in results/ and derived_results/ as in v1.0.0. This snapshot is a transfer-submission preparation, not a statement of journal acceptance.
