# KV Symbolic Factor Screen V1

This experiment reworks Phase 3 from a single narrow baseline into a first-wave factor screen.

The question is no longer only:

**what motif forms?**

It is:

**which control factors change whether the retrieval motif forms, when it forms, and whether shortcut families win instead?**

## Wave 1 Factors

The screen sweeps four factors while keeping the optimizer and schedule fixed:

- initialization scale `gamma`
- weight decay `lambda`
- width `d_model`
- query-slot policy / shortcut availability `C`

`query_slot_policy=fixed_first` is the deliberate shortcut regime: the queried key is always taken from slot `0`, so a positional copy strategy can compete with a genuine query-conditioned retrieval mechanism.

## Fixed Controls

- optimizer: `Adam`
- learning rate: `0.001`
- depth: `2`
- heads: `2`
- curriculum: `off`
- train context: `2` pairs only
- OOD context: `3` pairs

## State Variables

The reduced state tracked during training remains role-level, not head-identity-level:

- `Q(t)`: query-support quality
- `R(t)`: routing quality
- `W(t)`: write quality
- `S(t)`: support-to-retrieval path gain
- `M(t)`: shortcut pressure

`Q/R/W/S` are measured directly by the Phase 3 stack. `M` is induced by dataset regime and should be summarized from family-conditioned behavior and rival-family projection statistics.

## Mathematical Target

The intended reduced law is:

```text
z_{t+1} = z_t + Phi(z_t, F) + epsilon_t
```

with:

```text
z_t = (Q_t, R_t, W_t, S_t, M_t)
F = (gamma, lambda, d_model, C, seed)
```

The microscopic object underneath it is still the optimizer step:

```text
Delta theta_t = -eta P_t grad L(theta_t; B_t) - eta lambda theta_t
```

Phase 3 should explain motif selection by connecting those parameter-space updates to role-level state changes and rival circuit-family outcomes.

## Generated Artifacts

Use the factor-screen builder to emit:

- dataset build commands
- dataset matrix CSV
- manifest matrix CSV
- one manifest JSON per run condition

```bash
python -m research.phase3.scripts.build_kv_factor_screen
```

By default this writes under:

- `manifests/phase3/kv_symbolic_factor_screen_v1`
- `dataset/phase3/kv_symbolic_factor_screen_v1`
- `runs/phase3/kv_symbolic_factor_screen_v1`

## Recommended Workflow

1. Build the plan with the `ml` conda environment. The builder records the exact Python executable you used, so the generated scripts keep using that interpreter:

```bash
/opt/miniconda3/envs/ml/bin/python -m research.phase3.scripts.build_kv_factor_screen --device mps
```

Use `--device auto` if you want runtime device selection instead of strict `mps`.

2. Build the dataset variants:

```bash
bash manifests/phase3/kv_symbolic_factor_screen_v1/dataset_build_commands.sh
```

3. Launch the generated manifests sequentially:

```bash
bash manifests/phase3/kv_symbolic_factor_screen_v1/run_manifests_sequential.sh
```

4. Launch one manifest manually when you only want a single run:

```bash
python -m scripts.train_run --manifest manifests/phase3/kv_symbolic_factor_screen_v1/seed0_slot_balanced_d64_wd0_init1.json
```

5. Summarize one run or one seed-only condition block:

```bash
python -m scripts.summarize_training_dynamics --target-dir runs/phase3/kv_symbolic_factor_screen_v1/seed0_slot_balanced_d64_wd0_init1
```

6. Compare formation births, family gradients, and final role identities across factors before introducing second-wave optimizer or architecture sweeps.

## What Counts As Success

Wave 1 is successful if it identifies which of the four factors most strongly shift:

- behavior birth
- `R` birth
- role-level support/retrieval stability
- shortcut-vs-retrieval family outcomes
- OOD retention

Only after that should Phase 3 widen to optimizer family, batch noise, or deeper architecture sweeps.
