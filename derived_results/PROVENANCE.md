# Derived numerical source records

The files in this directory are derived from the primary 1,848-run nonlinear dataset, the separate 24-run OVM/FVD extension, or the sampled-linear calculation. They retain the original numerical values used by the submitted manuscript.

- nonlinear_directional_mechanism_runs.csv joins the primary raw_runs.csv with failure_log.jsonl by run ID. It records the declared layout cells, active indices, uncontrolled prefix and first prefix event.
- nonlinear_directional_mechanism_counts.csv summarizes A/B/C/D categories. D denotes an unevaluated declared cell.
- v6_prefix_witness_checks.csv validates all 198 Class-B records against the earliest logged prefix safety or collision event. Every first prefix crossing is below 4 m at positive gap and precedes prefix collision.
- v6_check_summary.json contains the category counts and the representative wave's pre-collision normalized distance deficit. The deficit integrates the trajectory through 19.3 s and is 0.009046928373130274.
- endpoint_predictor_decisions.csv contains 56,544 eligible selected-command comparisons from 24 paired trajectories, with 589 eligible time indices and four active vehicles per trajectory. The comparison holds the selected command under the recorded predecessor path; it is not the subsequent closed-loop ego trajectory.
- paired_alpha45_effects.csv and paired_alpha45_replays.csv support 12-seed paired controller comparisons at the requested scan ceiling.
- source_protocol_replays_alpha21.csv, finite_horizon_chain_reach_alpha40_seed97.csv and mean_equilibrium_jacobians.csv support Supplementary Fig. S6.

The representative wave trajectory is illustrative and distinct from the 1,848-run statistical table. Post-collision numerical continuation is retained in the source records but excluded from physical terminal interpretation. Exact figure-to-file mapping is in SOURCE_DATA_README.md at the repository root.
