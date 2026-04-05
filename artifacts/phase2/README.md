# Phase 2 Evidence Pack

This directory contains the tracked evidence bundle for the public Phase 2 repo.

It exists because the full raw `runs/` tree is large and remains a local research workspace, while GitHub-facing documentation needs stable, commit-able artifacts.

## Contents

- [static_kv](static_kv)
  - compact metadata for the original symbolic KV circuit analysis
- [story_negative](story_negative)
  - canonical summaries and visuals for the failed story benchmark
- [textual_kv](textual_kv)
  - canonical summaries and visuals for the successful textual KV benchmark

## Source of Truth

These files were copied from the canonical run outputs and visualizations recorded in:

- `runs/story_text_circuit_run_001`
- `runs/story_text_circuit_run_001_visuals`
- `runs/kv_textual_balanced_v1/pilot_seed0_curriculum_on_d64_l2`
- `runs/kv_textual_balanced_v1/pilot_seed0_curriculum_on_d64_l2_visuals_v4`
- `runs/kv_textual_balanced_v1/curriculum_on`
- `runs/kv_textual_balanced_v1/curriculum_off`

The long-form interpretation remains in:

- [results.md](../../results.md)

The public documentation entrypoint is:

- [docs/index.md](../../docs/index.md)
