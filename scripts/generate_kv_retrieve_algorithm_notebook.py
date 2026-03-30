#!/usr/bin/env python3
"""Generate the unified KV-retrieve analysis notebook."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebook" / "kv_retrieve_algorithm_analysis.ipynb"


def _normalize_cell_source(text: str) -> list[str]:
    normalized = textwrap.dedent(text).strip("\n") + "\n"
    return normalized.splitlines(keepends=True)


def markdown_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": _normalize_cell_source(text),
    }


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _normalize_cell_source(source),
    }


def build_notebook() -> dict:
    cells = [
        markdown_cell(
            """
            # KV Retrieve Unified Analysis

            This notebook replaces the split circuit / feature / neuron / algorithm notebooks with one canonical artifact.

            It keeps all of the tracking machinery that was already built, but organizes it once in a single flow:

            1. baseline prompt inspection
            2. clean/corrupt behavior
            3. activation patching, path patching, causal ablations
            4. QK / OV analysis and circuit tracing
            5. sparse autoencoder feature analysis
            6. neuron-level analysis
            7. register-style summary of the current internal program

            The point is not to pretend this is already a solved algorithm notebook. The point is to have one place where every analysis object is visible, non-duplicated, and explained.
            """
        ),
        markdown_cell(
            """
            ## Setup

            This cell loads the model, dataset, saved SAE artifacts, and one anchor prompt used throughout the notebook.

            Why this matters:

            - every later cell reuses the same model state and prompt-local caches
            - the feature sections load saved SAE checkpoints instead of retraining in the notebook
            - failure is explicit: if an artifact is missing, the notebook stops instead of hiding the error
            """
        ),
        code_cell(
            """
            from __future__ import annotations

            import json
            import math
            import sys
            from pathlib import Path

            import pandas as pd
            import torch
            import torch.nn.functional as F
            from IPython.display import Markdown, display

            pd.set_option("display.max_colwidth", 200)
            pd.set_option("display.max_columns", None)
            pd.set_option("display.width", 220)

            if Path.cwd().name == "notebook":
                PROJECT_ROOT = Path.cwd().resolve().parent
            else:
                PROJECT_ROOT = Path.cwd().resolve()

            if str(PROJECT_ROOT) not in sys.path:
                sys.path.insert(0, str(PROJECT_ROOT))

            from scripts.kv_retrieve_analysis import (
                apply_rope,
                build_attention_tables,
                build_head_role_summary_table,
                build_head_source_contribution_table,
                build_head_source_write_table,
                build_kv_algorithm_variable_table,
                build_kv_prompt_layout_table,
                build_layer_feature_readout_table,
                build_mlp_neuron_batch_ablation_table,
                build_mlp_neuron_clean_corrupt_head_patch_table,
                build_mlp_neuron_clean_corrupt_source_patch_table,
                build_mlp_neuron_contribution_table,
                build_mlp_neuron_group_summary_table,
                build_mlp_neuron_read_comparison_table,
                build_mlp_neuron_upstream_head_effect_table,
                build_top_mlp_neuron_examples,
                build_ov_topk_table,
                build_path_patched_attention_table,
                build_qk_table,
                build_qkv_patched_attention_table,
                build_query_swap_head_role_comparison_table,
                build_query_swap_slot_routing_comparison_table,
                build_single_prompt_head_role_table,
                build_single_prompt_slot_routing_table,
                build_slot_routing_summary_table,
                build_stage_variable_readout_table,
                build_stage_variable_summary_table,
                collect_head_role_attention_table,
                collect_mlp_neuron_activation_table,
                collect_slot_routing_table,
                collect_stage_variable_readout_table,
                compute_path_patched_head_details,
                head_residual_contribution,
                load_checkpoint_model,
                load_dataset_bundle,
                make_query_swap_prompt,
                make_query_swap_row,
                ov_source_logits,
                residual_vector_to_logits,
                run_prompt,
                score_mlp_neuron_ablation_prompt,
                score_patched_prompt,
                score_path_patched_prompt,
                score_qkv_patched_prompt,
                score_rows_with_optional_ablation,
            )
            from scripts.kv_retrieve_features import (
                build_feature_activation_table,
                build_feature_encoder_contribution_table,
                build_feature_group_summary_table,
                collect_split_activations,
                intervene_on_sae_features,
                load_sae_checkpoint,
                score_feature_intervention,
                score_query_feature_intervention,
                select_feature_panel,
            )

            DEVICE = torch.device("cpu")
            DATASET_DIR = PROJECT_ROOT / "dataset" / "kv_retrieve_3"
            CHECKPOINT_PATH = PROJECT_ROOT / "models" / "kv_retrieve_3" / "selected_checkpoint.pt"
            L1H0_SAE_PATH = PROJECT_ROOT / "models" / "kv_retrieve_3" / "sae_block1_final_l1h0_l1e1.pt"
            L2Q_SAE_PATH = PROJECT_ROOT / "models" / "kv_retrieve_3" / "sae_l2h0_final_q_l1e1.pt"
            OUTPUT_DIR = PROJECT_ROOT / "notebook" / "outputs"
            L1H0_SUMMARY_PATH = OUTPUT_DIR / "kv_retrieve_feature_basis_l1h0_site.json"
            RESID_SUMMARY_PATH = OUTPUT_DIR / "kv_retrieve_feature_basis_l1e1.json"
            L2Q_SUMMARY_PATH = OUTPUT_DIR / "kv_retrieve_feature_basis_l2h0_q.json"

            for required_path in [
                DATASET_DIR,
                CHECKPOINT_PATH,
                L1H0_SAE_PATH,
                L2Q_SAE_PATH,
                L1H0_SUMMARY_PATH,
                RESID_SUMMARY_PATH,
                L2Q_SUMMARY_PATH,
            ]:
                if not required_path.exists():
                    raise FileNotFoundError(f"Missing required artifact: {required_path}")

            bundle = load_dataset_bundle(DATASET_DIR)
            checkpoint, analysis_model = load_checkpoint_model(CHECKPOINT_PATH, device=DEVICE)
            _, feature_sae = load_sae_checkpoint(L1H0_SAE_PATH, device=DEVICE)
            _, l2q_sae = load_sae_checkpoint(L2Q_SAE_PATH, device=DEVICE)

            with L1H0_SUMMARY_PATH.open() as handle:
                l1h0_data = json.load(handle)
            with RESID_SUMMARY_PATH.open() as handle:
                resid_data = json.load(handle)
            with L2Q_SUMMARY_PATH.open() as handle:
                l2q_data = json.load(handle)

            feature_df = pd.DataFrame(l1h0_data["feature_summary_top10"])
            resid_feature_df = pd.DataFrame(resid_data["feature_summary_top10"])
            l2q_feature_df = pd.DataFrame(l2q_data["feature_summary_top10"])

            EXAMPLE_SPLIT = "test"
            EXAMPLE_INDEX = 0
            BATCH_LIMIT = 64
            BATCH_PAIR_LIMIT = 64
            NEURON_BATCH_LIMIT = 64

            example = bundle.raw_splits[EXAMPLE_SPLIT][EXAMPLE_INDEX]
            clean_prompt = example["prompt"]
            clean_target = example["target"]
            corrupt_prompt, corrupt_target = make_query_swap_prompt(example)
            corrupt_row = make_query_swap_row(example)

            clean_result, clean_cache = run_prompt(
                analysis_model,
                bundle,
                clean_prompt,
                DEVICE,
                expected_target=clean_target,
                return_cache=True,
            )
            corrupt_result, corrupt_cache = run_prompt(
                analysis_model,
                bundle,
                corrupt_prompt,
                DEVICE,
                expected_target=corrupt_target,
                return_cache=True,
            )

            if clean_cache is None or corrupt_cache is None:
                raise ValueError("Expected clean and corrupt caches in the unified notebook")

            prompt_tokens = clean_prompt.split()
            final_position_index = len(prompt_tokens) - 1
            query_position_index = len(prompt_tokens) - 2
            foil_token = clean_result["foil_token"]

            target_positions = [index for index, token in enumerate(prompt_tokens) if token == clean_target]
            if len(target_positions) != 1:
                raise ValueError(f"Expected exactly one clean target position, found {target_positions}")
            clean_value_position = target_positions[0]

            corrupt_value_positions = [index for index, token in enumerate(prompt_tokens) if token == corrupt_target]
            if len(corrupt_value_positions) != 1:
                raise ValueError(f"Expected exactly one corrupt target position, found {corrupt_value_positions}")
            corrupt_value_position = corrupt_value_positions[0]

            clean_l1h0_source = head_residual_contribution(analysis_model, clean_cache, layer_index=0, head_index=0)
            corrupt_l1h0_source = head_residual_contribution(analysis_model, corrupt_cache, layer_index=0, head_index=0)
            clean_l1h0_vector = clean_l1h0_source[0, final_position_index].detach().cpu()
            corrupt_l1h0_vector = corrupt_l1h0_source[0, final_position_index].detach().cpu()
            clean_l2h0_q = clean_cache["blocks"][1]["attention"]["q"][0, 0, final_position_index, :].detach().cpu()
            corrupt_l2h0_q = corrupt_cache["blocks"][1]["attention"]["q"][0, 0, final_position_index, :].detach().cpu()
            clean_l2h0_pattern = clean_cache["blocks"][1]["attention"]["pattern"][0, 0, final_position_index, :].detach().cpu()
            corrupt_l2h0_pattern = corrupt_cache["blocks"][1]["attention"]["pattern"][0, 0, final_position_index, :].detach().cpu()

            available_example_feature_ids = [int(key) for key in l1h0_data["top_examples"].keys()]
            feature_panel = select_feature_panel(
                feature_df,
                available_example_feature_ids=available_example_feature_ids,
                support_count=2,
                control_count=1,
            )
            support_features = feature_panel["support_features"]
            control_features = feature_panel["control_features"]
            selected_features = feature_panel["panel_features"]
            selected_feature_roles = {feature_index: "support" for feature_index in support_features}
            selected_feature_roles.update({feature_index: "control" for feature_index in control_features})

            l2q_panel = select_feature_panel(
                l2q_feature_df,
                support_count=2,
                control_count=1,
            )
            downstream_feature_set = l2q_panel["support_features"]

            display(pd.DataFrame([
                {
                    "clean_prompt": clean_prompt,
                    "clean_target": clean_target,
                    "clean_prediction": clean_result["predicted_token"],
                    "clean_margin": clean_result["margin"],
                    "corrupt_prompt": corrupt_prompt,
                    "corrupt_target": corrupt_target,
                    "corrupt_prediction": corrupt_result["predicted_token"],
                    "corrupt_margin": corrupt_result["margin"],
                }
            ]))
            """
        ),
        markdown_cell(
            """
            ## Prompt Layout And Single-Prompt Inspection

            This cell fixes the anchor prompt in symbolic form and shows the raw final-position attention patterns.

            Why this helps:

            - it grounds the rest of the notebook in one concrete prompt
            - it reveals the first query-head and value-head candidates without any patching
            - it gives the token-level view before we move to features, neurons, and registers
            """
        ),
        code_cell(
            """
            variable_df = build_kv_algorithm_variable_table(example)
            layout_df = build_kv_prompt_layout_table(example)

            display(variable_df)
            display(layout_df)

            for title, attention_df in build_attention_tables(clean_prompt, clean_cache):
                print(title)
                display(attention_df)

            head_focus_rows = []
            for layer_index, block_cache in enumerate(clean_cache["blocks"]):
                pattern = block_cache["attention"]["pattern"][0, :, -1, :]
                for head_index in range(pattern.shape[0]):
                    head_focus_rows.append(
                        {
                            "component": f"L{layer_index + 1}H{head_index}",
                            "layer_index": layer_index,
                            "head_index": head_index,
                            "attention_to_query": float(pattern[head_index, query_position_index].item()),
                            "attention_to_target_value": float(pattern[head_index, clean_value_position].item()),
                        }
                    )

            head_focus_df = pd.DataFrame(head_focus_rows)
            display(head_focus_df.sort_values(["attention_to_query", "attention_to_target_value"], ascending=False).reset_index(drop=True))

            query_head_row = head_focus_df.sort_values("attention_to_query", ascending=False).iloc[0]
            value_head_row = head_focus_df.sort_values("attention_to_target_value", ascending=False).iloc[0]

            display(
                pd.DataFrame(
                    [
                        {
                            "query_head_candidate": query_head_row["component"],
                            "query_attention": query_head_row["attention_to_query"],
                            "value_head_candidate": value_head_row["component"],
                            "value_attention": value_head_row["attention_to_target_value"],
                        }
                    ]
                )
            )
            """
        ),
        markdown_cell(
            """
            ## Clean / Corrupt Query Swap

            This is the simplest sanity check: change only the query key while leaving the context pairs fixed.

            Why this helps:

            - it shows whether the model is actually doing keyed retrieval instead of memorizing values
            - it gives the clean/corrupt pair used by most later patching and tracing cells
            """
        ),
        code_cell(
            """
            query_swap_df = pd.DataFrame(
                [
                    {"case": "clean", "prompt": clean_prompt, **clean_result},
                    {"case": "corrupt_query_swap", "prompt": corrupt_prompt, **corrupt_result},
                ]
            )[
                [
                    "case",
                    "prompt",
                    "predicted_token",
                    "target_token",
                    "foil_token",
                    "target_logit",
                    "foil_logit",
                    "margin",
                    "correct",
                ]
            ]

            display(query_swap_df)
            """
        ),
        markdown_cell(
            """
            ## Activation Patching

            This cell asks a broad causal question: if a clean activation is restored inside the corrupt run, does the clean answer come back?

            Why this helps:

            - it identifies which whole-block or whole-head activations are strong enough to rescue the behavior
            - it is a coarse localization step before path-specific patching
            """
        ),
        code_cell(
            """
            patch_specs = [
                {"label": "clean baseline", "mode": "clean"},
                {"label": "corrupt no patch", "mode": "corrupt", "patch": None},
                {"label": "patch resid after block 1", "mode": "patch", "patch": {"kind": "resid_after_block", "layer_index": 0}},
                {"label": "patch resid after block 2", "mode": "patch", "patch": {"kind": "resid_after_block", "layer_index": 1}},
                {"label": "patch L1H0 head_out", "mode": "patch", "patch": {"kind": "head_out", "layer_index": 0, "head_index": 0}},
                {"label": "patch L1H1 head_out", "mode": "patch", "patch": {"kind": "head_out", "layer_index": 0, "head_index": 1}},
                {"label": "patch L2H0 head_out", "mode": "patch", "patch": {"kind": "head_out", "layer_index": 1, "head_index": 0}},
                {"label": "patch L2H1 head_out", "mode": "patch", "patch": {"kind": "head_out", "layer_index": 1, "head_index": 1}},
                {"label": "patch block 1 mlp_out", "mode": "patch", "patch": {"kind": "mlp_out", "layer_index": 0}},
                {"label": "patch block 2 mlp_out", "mode": "patch", "patch": {"kind": "mlp_out", "layer_index": 1}},
            ]

            patch_rows = []
            for spec in patch_specs:
                if spec["mode"] == "clean":
                    result, _ = run_prompt(
                        analysis_model,
                        bundle,
                        clean_prompt,
                        DEVICE,
                        expected_target=clean_target,
                        return_cache=False,
                    )
                elif spec["mode"] == "corrupt":
                    result = score_patched_prompt(
                        analysis_model,
                        bundle,
                        clean_prompt,
                        corrupt_prompt,
                        clean_target,
                        DEVICE,
                        patch=None,
                        clean_cache=clean_cache,
                    )
                else:
                    result = score_patched_prompt(
                        analysis_model,
                        bundle,
                        clean_prompt,
                        corrupt_prompt,
                        clean_target,
                        DEVICE,
                        patch=spec["patch"],
                        clean_cache=clean_cache,
                    )

                patch_rows.append(
                    {
                        "case": spec["label"],
                        "predicted_token": result["predicted_token"],
                        "target_token": result["target_token"],
                        "foil_token": result["foil_token"],
                        "margin": result["margin"],
                        "restores_clean_answer": result["predicted_token"] == clean_target,
                    }
                )

            patch_df = pd.DataFrame(patch_rows).sort_values("margin", ascending=False).reset_index(drop=True)
            display(patch_df)
            """
        ),
        markdown_cell(
            """
            ## Path Patching And Position-Resolved Path Patching

            This section narrows the question from whole activations to specific source-to-destination paths.

            Why this helps:

            - it asks which upstream component matters specifically for the downstream value head
            - the position-resolved view turns a broad head-to-head path into a token-local path
            - this is the main bridge from coarse patching into a concrete retrieval circuit
            """
        ),
        code_cell(
            """
            if int(value_head_row["layer_index"]) == 0:
                raise RuntimeError("Path patching requires a destination head above layer 1")

            destination = {
                "layer_index": int(value_head_row["layer_index"]),
                "head_index": int(value_head_row["head_index"]),
            }
            source_layer_index = destination["layer_index"] - 1
            source_layer_label = source_layer_index + 1

            path_specs = [
                {"label": "clean baseline", "mode": "clean"},
                {"label": "corrupt no patch", "mode": "corrupt"},
                {
                    "label": f"path block {source_layer_label} residual -> {value_head_row['component']}",
                    "mode": "path",
                    "source_patch": {"kind": "resid_after_block", "layer_index": source_layer_index},
                },
            ]
            for head_index in range(checkpoint["config"]["n_heads"]):
                path_specs.append(
                    {
                        "label": f"path L{source_layer_label}H{head_index} -> {value_head_row['component']}",
                        "mode": "path",
                        "source_patch": {
                            "kind": "head_resid",
                            "layer_index": source_layer_index,
                            "head_index": head_index,
                        },
                    }
                )
            path_specs.append(
                {
                    "label": f"path block {source_layer_label} mlp_out -> {value_head_row['component']}",
                    "mode": "path",
                    "source_patch": {"kind": "mlp_out", "layer_index": source_layer_index},
                }
            )

            path_rows = []
            for spec in path_specs:
                if spec["mode"] == "clean":
                    result, _ = run_prompt(
                        analysis_model,
                        bundle,
                        clean_prompt,
                        DEVICE,
                        expected_target=clean_target,
                        return_cache=False,
                    )
                elif spec["mode"] == "corrupt":
                    result = score_patched_prompt(
                        analysis_model,
                        bundle,
                        clean_prompt,
                        corrupt_prompt,
                        clean_target,
                        DEVICE,
                        patch=None,
                        clean_cache=clean_cache,
                    )
                else:
                    result = score_path_patched_prompt(
                        analysis_model,
                        bundle,
                        clean_prompt,
                        corrupt_prompt,
                        clean_target,
                        DEVICE,
                        source_patch=spec["source_patch"],
                        destination=destination,
                        clean_cache=clean_cache,
                        corrupt_cache=corrupt_cache,
                    )

                path_rows.append(
                    {
                        "case": spec["label"],
                        "predicted_token": result["predicted_token"],
                        "foil_token": result["foil_token"],
                        "margin": result["margin"],
                        "restores_clean_answer": result["predicted_token"] == clean_target,
                    }
                )

            path_df = pd.DataFrame(path_rows).sort_values("margin", ascending=False).reset_index(drop=True)
            display(path_df)

            source_head_candidates = path_df[path_df["case"].str.contains(r"^path L\\d+H\\d+ ->", regex=True)].copy()
            if source_head_candidates.empty:
                raise RuntimeError("No source-head path rows found")

            source_head_candidates["source_head_index"] = (
                source_head_candidates["case"].str.extract(r"^path L\\d+H(\\d+) ->")[0].astype(int)
            )
            best_source_head_row = source_head_candidates.sort_values("margin", ascending=False).iloc[0]
            best_source_head_index = int(best_source_head_row["source_head_index"])

            position_path_rows = []
            for source_position, source_token in enumerate(clean_prompt.split()):
                result = score_path_patched_prompt(
                    analysis_model,
                    bundle,
                    clean_prompt,
                    corrupt_prompt,
                    clean_target,
                    DEVICE,
                    source_patch={
                        "kind": "head_resid",
                        "layer_index": source_layer_index,
                        "head_index": best_source_head_index,
                        "source_positions": [source_position],
                    },
                    destination=destination,
                    clean_cache=clean_cache,
                    corrupt_cache=corrupt_cache,
                )
                position_path_rows.append(
                    {
                        "source_position": source_position,
                        "source_token": source_token,
                        "predicted_token": result["predicted_token"],
                        "foil_token": result["foil_token"],
                        "margin": result["margin"],
                        "restores_clean_answer": result["predicted_token"] == clean_target,
                    }
                )

            position_path_df = pd.DataFrame(position_path_rows).sort_values("margin", ascending=False).reset_index(drop=True)
            display(position_path_df)
            """
        ),
        markdown_cell(
            """
            ## Destination Routing And Q / K / V Tests

            This cell asks two tighter questions:

            - which downstream head actually receives the useful source signal?
            - inside that head, is the critical channel `Q`, `K`, or `V`?

            Why this helps:

            - it turns a broad source-to-layer statement into a source-to-head statement
            - it isolates whether the mechanism is query retargeting or value overwriting
            """
        ),
        code_cell(
            """
            destination_specific_rows = []
            for destination_head_index in range(checkpoint["config"]["n_heads"]):
                destination_spec = {
                    "layer_index": destination["layer_index"],
                    "head_index": destination_head_index,
                }
                source_specs = [
                    {
                        "source_label": f"block {source_layer_label} residual at final position",
                        "source_patch": {
                            "kind": "resid_after_block",
                            "layer_index": source_layer_index,
                            "source_positions": [final_position_index],
                        },
                    },
                    {
                        "source_label": f"L{source_layer_label}H{best_source_head_index} at final position",
                        "source_patch": {
                            "kind": "head_resid",
                            "layer_index": source_layer_index,
                            "head_index": best_source_head_index,
                            "source_positions": [final_position_index],
                        },
                    },
                ]

                for spec in source_specs:
                    result = score_path_patched_prompt(
                        analysis_model,
                        bundle,
                        clean_prompt,
                        corrupt_prompt,
                        clean_target,
                        DEVICE,
                        source_patch=spec["source_patch"],
                        destination=destination_spec,
                        clean_cache=clean_cache,
                        corrupt_cache=corrupt_cache,
                    )
                    destination_specific_rows.append(
                        {
                            "source": spec["source_label"],
                            "destination": f"L{destination_spec['layer_index'] + 1}H{destination_spec['head_index']}",
                            "predicted_token": result["predicted_token"],
                            "foil_token": result["foil_token"],
                            "margin": result["margin"],
                            "restores_clean_answer": result["predicted_token"] == clean_target,
                        }
                    )

            final_position_destination_df = (
                pd.DataFrame(destination_specific_rows)
                .sort_values(["source", "margin"], ascending=[True, False])
                .reset_index(drop=True)
            )
            display(final_position_destination_df)

            qkv_specs = [
                {"label": "clean baseline", "mode": "clean"},
                {"label": "corrupt no patch", "mode": "corrupt"},
                {"label": "patch Q only", "mode": "qkv", "components": ["q"]},
                {"label": "patch K only", "mode": "qkv", "components": ["k"]},
                {"label": "patch V only", "mode": "qkv", "components": ["v"]},
                {"label": "patch QK", "mode": "qkv", "components": ["q", "k"]},
                {"label": "patch QV", "mode": "qkv", "components": ["q", "v"]},
                {"label": "patch KV", "mode": "qkv", "components": ["k", "v"]},
                {"label": "patch QKV", "mode": "qkv", "components": ["q", "k", "v"]},
            ]

            qkv_rows = []
            for spec in qkv_specs:
                if spec["mode"] == "clean":
                    result, _ = run_prompt(
                        analysis_model,
                        bundle,
                        clean_prompt,
                        DEVICE,
                        expected_target=clean_target,
                        return_cache=False,
                    )
                elif spec["mode"] == "corrupt":
                    result = score_patched_prompt(
                        analysis_model,
                        bundle,
                        clean_prompt,
                        corrupt_prompt,
                        clean_target,
                        DEVICE,
                        patch=None,
                        clean_cache=clean_cache,
                    )
                else:
                    result = score_qkv_patched_prompt(
                        analysis_model,
                        bundle,
                        clean_prompt,
                        corrupt_prompt,
                        clean_target,
                        DEVICE,
                        destination=destination,
                        components=spec["components"],
                        clean_cache=clean_cache,
                        corrupt_cache=corrupt_cache,
                    )

                qkv_rows.append(
                    {
                        "case": spec["label"],
                        "predicted_token": result["predicted_token"],
                        "foil_token": result["foil_token"],
                        "margin": result["margin"],
                        "restores_clean_answer": result["predicted_token"] == clean_target,
                    }
                )

            qkv_df = pd.DataFrame(qkv_rows).sort_values("margin", ascending=False).reset_index(drop=True)
            display(qkv_df)
            """
        ),
        markdown_cell(
            """
            ## Batch Causal Checks

            The previous cells are prompt-local. This cell checks whether the same causal story survives across a batch.

            Why this helps:

            - it prevents overfitting to the anchor prompt
            - it keeps both kinds of evidence in view:
              - path-specific restoration on clean/corrupt pairs
              - direct head ablations on baseline-correct prompts
            """
        ),
        code_cell(
            """
            batch_pairs = []
            for row in bundle.raw_splits["test"]:
                clean_result_batch, clean_cache_batch = run_prompt(
                    analysis_model,
                    bundle,
                    row["prompt"],
                    DEVICE,
                    expected_target=row["target"],
                    return_cache=True,
                )
                corrupt_prompt_batch, corrupt_target_batch = make_query_swap_prompt(row)
                corrupt_result_batch, corrupt_cache_batch = run_prompt(
                    analysis_model,
                    bundle,
                    corrupt_prompt_batch,
                    DEVICE,
                    expected_target=corrupt_target_batch,
                    return_cache=True,
                )
                if clean_result_batch["predicted_token"] != row["target"]:
                    continue
                batch_pairs.append(
                    {
                        "clean_prompt": row["prompt"],
                        "clean_target": row["target"],
                        "corrupt_prompt": corrupt_prompt_batch,
                        "corrupt_target": corrupt_target_batch,
                        "clean_cache": clean_cache_batch,
                        "corrupt_cache": corrupt_cache_batch,
                    }
                )

            if len(batch_pairs) < BATCH_PAIR_LIMIT:
                raise RuntimeError(
                    f"Expected at least {BATCH_PAIR_LIMIT} clean/corrupt test pairs, found {len(batch_pairs)}"
                )
            batch_pairs = batch_pairs[:BATCH_PAIR_LIMIT]

            batch_case_specs = [
                {"label": "corrupt no patch", "kind": "corrupt"},
                {
                    "label": "patch L1H0 final-pos -> destination",
                    "kind": "path",
                    "source_patch": {"kind": "head_resid", "layer_index": 0, "head_index": 0, "source_positions": [final_position_index]},
                    "destination": destination,
                },
                {
                    "label": "patch Q into destination",
                    "kind": "qkv",
                    "destination": destination,
                    "components": ["q"],
                },
                {
                    "label": "patch V into destination",
                    "kind": "qkv",
                    "destination": destination,
                    "components": ["v"],
                },
            ]

            batch_rows = []
            for spec in batch_case_specs:
                restore_flags = []
                margins = []
                for pair in batch_pairs:
                    if spec["kind"] == "corrupt":
                        result = score_patched_prompt(
                            analysis_model,
                            bundle,
                            pair["clean_prompt"],
                            pair["corrupt_prompt"],
                            pair["clean_target"],
                            DEVICE,
                            patch=None,
                            clean_cache=pair["clean_cache"],
                        )
                    elif spec["kind"] == "path":
                        result = score_path_patched_prompt(
                            analysis_model,
                            bundle,
                            pair["clean_prompt"],
                            pair["corrupt_prompt"],
                            pair["clean_target"],
                            DEVICE,
                            source_patch=spec["source_patch"],
                            destination=spec["destination"],
                            clean_cache=pair["clean_cache"],
                            corrupt_cache=pair["corrupt_cache"],
                        )
                    else:
                        result = score_qkv_patched_prompt(
                            analysis_model,
                            bundle,
                            pair["clean_prompt"],
                            pair["corrupt_prompt"],
                            pair["clean_target"],
                            DEVICE,
                            destination=spec["destination"],
                            components=spec["components"],
                            clean_cache=pair["clean_cache"],
                            corrupt_cache=pair["corrupt_cache"],
                        )
                    restore_flags.append(result["predicted_token"] == pair["clean_target"])
                    margins.append(result["margin"])

                batch_rows.append(
                    {
                        "case": spec["label"],
                        "num_pairs": len(batch_pairs),
                        "restore_rate": sum(restore_flags) / len(restore_flags),
                        "mean_margin_vs_clean_target": float(sum(margins) / len(margins)),
                    }
                )

            batch_path_summary_df = pd.DataFrame(batch_rows).sort_values("mean_margin_vs_clean_target", ascending=False).reset_index(drop=True)
            display(batch_path_summary_df)

            baseline_test_df = score_rows_with_optional_ablation(
                analysis_model,
                bundle,
                bundle.raw_splits["test"][:BATCH_LIMIT],
                DEVICE,
            )
            baseline_correct_mask = baseline_test_df["correct"].tolist()
            baseline_correct_rows = [
                row for row, is_correct in zip(bundle.raw_splits["test"][:BATCH_LIMIT], baseline_correct_mask) if is_correct
            ]
            baseline_correct_df = baseline_test_df.loc[baseline_test_df["correct"]].reset_index(drop=True)
            if baseline_correct_df.empty:
                raise RuntimeError("Baseline model has no correct prompts in the selected batch")

            batch_ablation_summary_rows = [
                {
                    "component": "baseline on baseline-correct subset",
                    "num_prompts": len(baseline_correct_df),
                    "accuracy": baseline_correct_df["correct"].mean(),
                    "mean_margin": baseline_correct_df["margin"].mean(),
                    "margin_delta_vs_baseline": 0.0,
                    "prediction_change_rate": 0.0,
                    "flip_rate_from_correct": 0.0,
                }
            ]

            for layer_index in range(checkpoint["config"]["n_layers"]):
                for head_index in range(checkpoint["config"]["n_heads"]):
                    ablated_df = score_rows_with_optional_ablation(
                        analysis_model,
                        bundle,
                        baseline_correct_rows,
                        DEVICE,
                        ablation={"layer_index": layer_index, "head_index": head_index},
                    )
                    batch_ablation_summary_rows.append(
                        {
                            "component": f"L{layer_index + 1}H{head_index} final-position ablated",
                            "num_prompts": len(ablated_df),
                            "accuracy": ablated_df["correct"].mean(),
                            "mean_margin": ablated_df["margin"].mean(),
                            "margin_delta_vs_baseline": ablated_df["margin"].mean() - baseline_correct_df["margin"].mean(),
                            "prediction_change_rate": (ablated_df["predicted_token"] != baseline_correct_df["predicted_token"]).mean(),
                            "flip_rate_from_correct": (~ablated_df["correct"]).mean(),
                        }
                    )

            batch_ablation_summary_df = pd.DataFrame(batch_ablation_summary_rows).sort_values("margin_delta_vs_baseline").reset_index(drop=True)
            display(batch_ablation_summary_df)
            """
        ),
        markdown_cell(
            """
            ## QK / OV Analysis

            This cell exposes the two most direct attention-head objects:

            - QK ranking: where a head wants to look
            - OV write: what a head writes when it reads a source

            Why this helps:

            - it separates routing from content
            - it makes the query-like head and value-like head legible in the model's own math
            """
        ),
        code_cell(
            """
            query_layer_index = int(query_head_row["layer_index"])
            query_head_index = int(query_head_row["head_index"])
            value_layer_index = int(value_head_row["layer_index"])
            value_head_index = int(value_head_row["head_index"])

            print(f"QK analysis head: L{query_layer_index + 1}H{query_head_index}")
            clean_qk_df = build_qk_table(clean_prompt, clean_cache, query_layer_index, query_head_index)
            corrupt_qk_df = build_qk_table(corrupt_prompt, corrupt_cache, query_layer_index, query_head_index)
            display(clean_qk_df.sort_values("qk_score", ascending=False).reset_index(drop=True))
            display(corrupt_qk_df.sort_values("qk_score", ascending=False).reset_index(drop=True))

            print(f"OV analysis head: L{value_layer_index + 1}H{value_head_index}")
            clean_target_id = bundle.token_to_id[clean_target]
            foil_token_id = bundle.token_to_id[foil_token]
            value_positions = [index for index, token in enumerate(prompt_tokens) if token in bundle.value_tokens]

            ov_summary_rows = []
            for source_position in value_positions:
                source_logits = ov_source_logits(
                    analysis_model,
                    clean_cache,
                    value_layer_index,
                    value_head_index,
                    source_position,
                )
                top_token_id = int(source_logits.argmax().item())
                ov_summary_rows.append(
                    {
                        "source_position": source_position,
                        "source_token": prompt_tokens[source_position],
                        "top_written_token": bundle.id_to_token[top_token_id],
                        "top_written_logit": float(source_logits[top_token_id].item()),
                        "target_logit": float(source_logits[clean_target_id].item()),
                        "foil_logit": float(source_logits[foil_token_id].item()),
                        "target_minus_foil": float((source_logits[clean_target_id] - source_logits[foil_token_id]).item()),
                    }
                )

            ov_summary_df = pd.DataFrame(ov_summary_rows).sort_values("target_minus_foil", ascending=False).reset_index(drop=True)
            display(ov_summary_df)
            display(build_ov_topk_table(analysis_model, bundle, clean_cache, value_layer_index, value_head_index, clean_value_position))
            """
        ),
        markdown_cell(
            """
            ## Circuit Trace: `L1H0 -> Destination.Q -> Destination Write`

            This is the compact computational trace section.

            Why this helps:

            - it checks whether the upstream write predicts the downstream query shift
            - it compares a path patch against a direct Q patch
            - it decomposes the destination head's final write by source position
            """
        ),
        code_cell(
            """
            l1h0_final_source_patch = {
                "kind": "head_resid",
                "layer_index": 0,
                "head_index": 0,
                "source_positions": [final_position_index],
            }

            clean_destination_qk_df = build_qk_table(clean_prompt, clean_cache, layer_index=destination["layer_index"], head_index=destination["head_index"]).rename(
                columns={"position": "source_position", "token": "source_token"}
            )
            corrupt_destination_qk_df = build_qk_table(corrupt_prompt, corrupt_cache, layer_index=destination["layer_index"], head_index=destination["head_index"]).rename(
                columns={"position": "source_position", "token": "source_token"}
            )
            path_destination_qk_df = build_path_patched_attention_table(
                analysis_model,
                corrupt_prompt,
                clean_cache,
                corrupt_cache,
                source_patch=l1h0_final_source_patch,
                destination=destination,
            )
            q_patch_destination_qk_df = build_qkv_patched_attention_table(
                analysis_model,
                corrupt_prompt,
                clean_cache,
                corrupt_cache,
                destination=destination,
                components=["q"],
            )

            score_retarget_df = clean_destination_qk_df.rename(
                columns={"source_token": "clean_source_token", "qk_score": "clean_qk_score", "attention_weight": "clean_attention"}
            )
            score_retarget_df = score_retarget_df.merge(
                corrupt_destination_qk_df.rename(
                    columns={"source_token": "corrupt_source_token", "qk_score": "corrupt_qk_score", "attention_weight": "corrupt_attention"}
                ),
                on=["source_position"],
            )
            score_retarget_df = score_retarget_df.merge(
                path_destination_qk_df.rename(
                    columns={"source_token": "path_source_token", "qk_score": "path_qk_score", "attention_weight": "path_attention"}
                ),
                on=["source_position"],
            )
            score_retarget_df = score_retarget_df.merge(
                q_patch_destination_qk_df.rename(
                    columns={"source_token": "q_patch_source_token", "qk_score": "q_patch_qk_score", "attention_weight": "q_patch_attention"}
                ),
                on=["source_position"],
            )
            score_retarget_df["path_qk_delta_vs_corrupt"] = score_retarget_df["path_qk_score"] - score_retarget_df["corrupt_qk_score"]
            score_retarget_df["q_patch_qk_delta_vs_corrupt"] = score_retarget_df["q_patch_qk_score"] - score_retarget_df["corrupt_qk_score"]
            score_retarget_df["path_attention_delta_vs_corrupt"] = score_retarget_df["path_attention"] - score_retarget_df["corrupt_attention"]
            score_retarget_df["q_patch_attention_delta_vs_corrupt"] = score_retarget_df["q_patch_attention"] - score_retarget_df["corrupt_attention"]

            path_details = compute_path_patched_head_details(
                analysis_model,
                clean_cache,
                corrupt_cache,
                source_patch=l1h0_final_source_patch,
                destination=destination,
            )

            clean_l1h0_write = head_residual_contribution(analysis_model, clean_cache, layer_index=0, head_index=0)[0, final_position_index].detach().cpu()
            corrupt_l1h0_write = head_residual_contribution(analysis_model, corrupt_cache, layer_index=0, head_index=0)[0, final_position_index].detach().cpu()
            delta_h = clean_l1h0_write - corrupt_l1h0_write

            destination_attn = analysis_model.blocks[destination["layer_index"]].attn
            head_dim = destination_attn.head_dim
            head_start = destination["head_index"] * head_dim
            head_stop = head_start + head_dim
            w_q_head = destination_attn.q_proj.weight[head_start:head_stop, :].detach().cpu()

            linear_delta_q = torch.matmul(w_q_head, delta_h)
            cos_final = destination_attn.rope_cos[final_position_index].detach().cpu().view(1, 1, 1, -1)
            sin_final = destination_attn.rope_sin[final_position_index].detach().cpu().view(1, 1, 1, -1)
            predicted_delta_q = apply_rope(linear_delta_q.view(1, 1, 1, -1), cos_final, sin_final).view(-1)

            corrupt_q = corrupt_cache["blocks"][destination["layer_index"]]["attention"]["q"][0, destination["head_index"], final_position_index, :].detach().cpu()
            path_q = path_details["q"][0, destination["head_index"], final_position_index, :].detach().cpu()
            actual_delta_q = path_q - corrupt_q
            corrupt_k = corrupt_cache["blocks"][destination["layer_index"]]["attention"]["k"][0, destination["head_index"], :, :].detach().cpu()

            query_write_summary_df = pd.DataFrame(
                [
                    {
                        "clean_L1H0_final_norm": float(clean_l1h0_write.norm().item()),
                        "corrupt_L1H0_final_norm": float(corrupt_l1h0_write.norm().item()),
                        "delta_write_norm": float(delta_h.norm().item()),
                        "predicted_delta_q_norm": float(predicted_delta_q.norm().item()),
                        "actual_delta_q_norm": float(actual_delta_q.norm().item()),
                        "delta_q_cosine_similarity": float(F.cosine_similarity(predicted_delta_q.unsqueeze(0), actual_delta_q.unsqueeze(0)).item()),
                    }
                ]
            )

            query_effect_rows = []
            for source_position, source_token in enumerate(clean_prompt.split()):
                approx_score_delta = float(torch.dot(predicted_delta_q, corrupt_k[source_position]) / math.sqrt(head_dim))
                actual_score_delta = float(
                    score_retarget_df.loc[
                        score_retarget_df["source_position"] == source_position,
                        "path_qk_delta_vs_corrupt",
                    ].iloc[0]
                )
                query_effect_rows.append(
                    {
                        "source_position": source_position,
                        "source_token": source_token,
                        "approx_score_delta_from_L1H0_write": approx_score_delta,
                        "actual_score_delta_from_path_patch": actual_score_delta,
                    }
                )
            query_effect_df = pd.DataFrame(query_effect_rows).sort_values("actual_score_delta_from_path_patch", ascending=False).reset_index(drop=True)

            destination_source_write_df = build_head_source_write_table(
                model=analysis_model,
                bundle=bundle,
                prompt=clean_prompt,
                attention_pattern=clean_cache["blocks"][destination["layer_index"]]["attention"]["pattern"][0, destination["head_index"], final_position_index, :].detach().cpu(),
                value_vectors=clean_cache["blocks"][destination["layer_index"]]["attention"]["v"][0, destination["head_index"], :, :].detach().cpu(),
                layer_index=destination["layer_index"],
                head_index=destination["head_index"],
                target_token=clean_target,
                foil_token=foil_token,
            )

            display(query_write_summary_df)
            display(score_retarget_df.sort_values("path_qk_delta_vs_corrupt", ascending=False).reset_index(drop=True))
            display(query_effect_df)
            display(destination_source_write_df.sort_values("target_minus_foil", ascending=False).reset_index(drop=True))
            """
        ),
        markdown_cell(
            """
            ## Sparse Feature Setup And Site Comparison

            This cell brings the SAE artifacts into the same notebook.

            Why this helps:

            - it compares the broad residual site against the narrowed `L1H0` site
            - it also compares the upstream `L1H0` site against the downstream `L2H0.Q` site
            - it selects one support/control feature panel dynamically instead of hardcoding feature ids
            """
        ),
        code_cell(
            """
            site_compare_df = pd.DataFrame(
                [
                    {
                        "site": resid_data["site"],
                        "site_description": resid_data["site_description"],
                        **resid_data["history_tail"][-1],
                    },
                    {
                        "site": l1h0_data["site"],
                        "site_description": l1h0_data["site_description"],
                        **l1h0_data["history_tail"][-1],
                    },
                ]
            )
            l2q_compare_df = pd.DataFrame(
                [
                    {
                        "site": l1h0_data["site"],
                        "site_description": l1h0_data["site_description"],
                        **l1h0_data["history_tail"][-1],
                    },
                    {
                        "site": l2q_data["site"],
                        "site_description": l2q_data["site_description"],
                        **l2q_data["history_tail"][-1],
                    },
                ]
            )

            display(site_compare_df[["site", "site_description", "val_recon_loss", "val_mean_active_features", "train_recon_loss", "train_mean_active_features"]])
            display(l2q_compare_df[["site", "site_description", "val_recon_loss", "val_mean_active_features", "train_recon_loss", "train_mean_active_features"]])
            display(feature_df[["feature_index", "mean_clean_activation", "mean_corrupt_activation", "mean_delta", "query_alignment", "query_cosine", "mechanism_score", "top_logit_tokens"]])
            display(l2q_feature_df[["feature_index", "mean_clean_activation", "mean_corrupt_activation", "mean_delta", "query_alignment", "query_cosine", "mechanism_score", "projection_space"]])
            display(pd.DataFrame([{
                "support_features": support_features,
                "control_features": control_features,
                "downstream_support_features": downstream_feature_set,
            }]))
            """
        ),
        markdown_cell(
            """
            ## Feature Dashboard And Label Semantics

            This section turns the saved SAE features into prompt-level objects.

            Why this helps:

            - top examples show what a feature tends to fire on
            - grouped label summaries test whether a feature is query-like, value-like, or position-like
            - this is the feature-level analogue of the head focus tables above
            """
        ),
        code_cell(
            """
            def feature_examples_table(data: dict, feature_index: int) -> pd.DataFrame:
                key = str(feature_index)
                if key not in data["top_examples"]:
                    raise KeyError(f"Feature {feature_index} not present in saved top_examples")
                return pd.DataFrame(data["top_examples"][key])

            selected_feature_summary_df = (
                feature_df[feature_df["feature_index"].isin(selected_features)]
                .sort_values("mechanism_score", ascending=False)
                .reset_index(drop=True)
            )
            display(selected_feature_summary_df)

            for feature_index in selected_features:
                display(Markdown(f"### Feature `{feature_index}` ({selected_feature_roles[feature_index]}) top examples"))
                display(feature_examples_table(l1h0_data, feature_index))

            test_activations, test_records = collect_split_activations(
                analysis_model,
                bundle,
                split="test",
                site="block1_final_l1h0",
            )
            feature_activation_df = build_feature_activation_table(
                feature_sae,
                test_activations,
                test_records,
                bundle,
                selected_features,
            )

            preview_columns = [
                "split",
                "index",
                "query_key",
                "target",
                "query_pair_index",
                "correct_value_position",
                "corrupt_target",
            ] + [f"feature_{feature_index}_activation" for feature_index in selected_features]
            display(feature_activation_df[preview_columns].head(8))

            for group_column in ["query_key", "target", "correct_value_position", "query_pair_index", "corrupt_target"]:
                print(f"Grouped by {group_column}")
                display(build_feature_group_summary_table(feature_activation_df, selected_features, group_column))
            """
        ),
        markdown_cell(
            """
            ## Feature Local Causes And Single-Feature Interventions

            This cell answers two different questions.

            1. local encoder cause:
               which parts of the `L1H0` vector push a feature on or off?
            2. causal behavior:
               what happens if that feature is ablated on the clean prompt or patched from clean into the corrupt prompt?

            Why this helps:

            - the encoder contribution table is the feature-level analogue of neuron read decomposition
            - the intervention table shows whether a feature actually changes `L2H0.Q`, attention, and margin
            """
        ),
        code_cell(
            """
            def build_modified_source(base_source_tensor: torch.Tensor, modified_vector: torch.Tensor, position_index: int) -> torch.Tensor:
                updated = base_source_tensor.clone()
                updated[0, position_index, :] = modified_vector.to(updated.dtype)
                return updated

            source_patch = {
                "kind": "head_resid",
                "layer_index": 0,
                "head_index": 0,
                "source_positions": [final_position_index],
            }

            encoder_summary_rows = []
            for feature_index in selected_features:
                clean_encoder_df = build_feature_encoder_contribution_table(feature_sae, clean_l1h0_vector, feature_index)
                corrupt_encoder_df = build_feature_encoder_contribution_table(feature_sae, corrupt_l1h0_vector, feature_index)
                contribution_delta_df = clean_encoder_df[["dimension", "contribution"]].merge(
                    corrupt_encoder_df[["dimension", "contribution"]],
                    on="dimension",
                    suffixes=("_clean", "_corrupt"),
                )
                contribution_delta_df["contribution_delta"] = (
                    contribution_delta_df["contribution_clean"] - contribution_delta_df["contribution_corrupt"]
                )
                contribution_delta_df["abs_contribution_delta"] = contribution_delta_df["contribution_delta"].abs()

                clean_activation = float(feature_sae.encode(clean_l1h0_vector.unsqueeze(0))[0, feature_index].item())
                corrupt_activation = float(feature_sae.encode(corrupt_l1h0_vector.unsqueeze(0))[0, feature_index].item())
                encoder_summary_rows.append(
                    {
                        "feature_index": feature_index,
                        "role": selected_feature_roles[feature_index],
                        "clean_activation": clean_activation,
                        "corrupt_activation": corrupt_activation,
                        "activation_delta": clean_activation - corrupt_activation,
                    }
                )
                display(Markdown(f"### Feature `{feature_index}` local encoder causes"))
                display(
                    contribution_delta_df.sort_values("abs_contribution_delta", ascending=False).head(10).reset_index(drop=True)
                )

            encoder_feature_summary_df = pd.DataFrame(encoder_summary_rows)
            display(encoder_feature_summary_df)

            single_rows = []
            for feature_index in selected_features:
                clean_ablation = intervene_on_sae_features(
                    feature_sae,
                    clean_l1h0_vector,
                    feature_indices=[feature_index],
                    mode="ablate",
                )
                clean_ablation_result = score_feature_intervention(
                    model=analysis_model,
                    bundle=bundle,
                    prompt=clean_prompt,
                    target_token=clean_target,
                    base_cache=clean_cache,
                    source_patch=source_patch,
                    modified_source_tensor=build_modified_source(clean_l1h0_source, clean_ablation["reconstructed"], final_position_index),
                    destination_layer_index=1,
                    device=DEVICE,
                )

                corrupt_patch = intervene_on_sae_features(
                    feature_sae,
                    corrupt_l1h0_vector,
                    feature_indices=[feature_index],
                    mode="patch",
                    source_vector=clean_l1h0_vector,
                )
                corrupt_patch_result = score_feature_intervention(
                    model=analysis_model,
                    bundle=bundle,
                    prompt=corrupt_prompt,
                    target_token=clean_target,
                    base_cache=corrupt_cache,
                    source_patch=source_patch,
                    modified_source_tensor=build_modified_source(corrupt_l1h0_source, corrupt_patch["reconstructed"], final_position_index),
                    destination_layer_index=1,
                    device=DEVICE,
                )

                for label, result, base_q, base_pattern, payload in [
                    ("clean ablate", clean_ablation_result, clean_l2h0_q, clean_l2h0_pattern, clean_ablation),
                    ("corrupt patch", corrupt_patch_result, corrupt_l2h0_q, corrupt_l2h0_pattern, corrupt_patch),
                ]:
                    intervened_q = result["details"]["destination_cache"]["attention"]["q"][0, 0, final_position_index, :].detach().cpu()
                    intervened_pattern = result["details"]["destination_cache"]["attention"]["pattern"][0, 0, final_position_index, :].detach().cpu()
                    single_rows.append(
                        {
                            "feature_index": feature_index,
                            "role": selected_feature_roles[feature_index],
                            "intervention": label,
                            "predicted_token": result["predicted_token"],
                            "foil_token": result["foil_token"],
                            "margin_vs_clean_target": result["margin"],
                            "feature_base_activation": float(payload["base_features"][feature_index].item()),
                            "feature_modified_activation": float(payload["modified_features"][feature_index].item()),
                            "l2h0_q_delta_norm": float((intervened_q - base_q).norm().item()),
                            "attention_on_clean_value": float(intervened_pattern[clean_value_position].item()),
                            "attention_on_corrupt_value": float(intervened_pattern[corrupt_value_position].item()),
                            "attention_shift_to_clean_value": float((intervened_pattern[clean_value_position] - base_pattern[clean_value_position]).item()),
                            "attention_shift_from_corrupt_value": float((intervened_pattern[corrupt_value_position] - base_pattern[corrupt_value_position]).item()),
                        }
                    )

            single_feature_df = pd.DataFrame(single_rows)
            display(single_feature_df)
            """
        ),
        markdown_cell(
            """
            ## Feature Batch Check, Lens, And Cross-Layer Feature Sets

            This is the compact feature synthesis cell.

            Why this helps:

            - the batch check asks whether single features matter beyond one prompt
            - the lens tables show what decoder directions point toward in token space
            - the upstream/downstream set comparison connects `L1H0` features to `L2H0.Q` features
            """
        ),
        code_cell(
            """
            batch_rows = []
            for feature_index in selected_features:
                clean_correct_count = 0
                clean_ablate_correct_count = 0
                corrupt_restore_count = 0
                clean_margin_values = []
                clean_ablate_margin_values = []
                corrupt_patch_margin_values = []

                for row in bundle.raw_splits["test"][:BATCH_LIMIT]:
                    row_clean_prompt = row["prompt"]
                    row_clean_target = row["target"]
                    row_corrupt_prompt, _ = make_query_swap_prompt(row)
                    row_final_position_index = len(row_clean_prompt.split()) - 1
                    row_source_patch = {
                        "kind": "head_resid",
                        "layer_index": 0,
                        "head_index": 0,
                        "source_positions": [row_final_position_index],
                    }

                    row_clean_result, row_clean_cache = run_prompt(
                        analysis_model,
                        bundle,
                        row_clean_prompt,
                        DEVICE,
                        expected_target=row_clean_target,
                        return_cache=True,
                    )
                    row_corrupt_result, row_corrupt_cache = run_prompt(
                        analysis_model,
                        bundle,
                        row_corrupt_prompt,
                        DEVICE,
                        expected_target=row_clean_target,
                        return_cache=True,
                    )
                    if row_clean_cache is None or row_corrupt_cache is None:
                        raise ValueError("Expected caches during feature batch check")

                    row_clean_source = head_residual_contribution(analysis_model, row_clean_cache, layer_index=0, head_index=0)
                    row_corrupt_source = head_residual_contribution(analysis_model, row_corrupt_cache, layer_index=0, head_index=0)
                    row_clean_vector = row_clean_source[0, row_final_position_index].detach().cpu()
                    row_corrupt_vector = row_corrupt_source[0, row_final_position_index].detach().cpu()

                    row_clean_ablation = intervene_on_sae_features(feature_sae, row_clean_vector, [feature_index], mode="ablate")
                    row_clean_ablation_result = score_feature_intervention(
                        model=analysis_model,
                        bundle=bundle,
                        prompt=row_clean_prompt,
                        target_token=row_clean_target,
                        base_cache=row_clean_cache,
                        source_patch=row_source_patch,
                        modified_source_tensor=build_modified_source(row_clean_source, row_clean_ablation["reconstructed"], row_final_position_index),
                        destination_layer_index=1,
                        device=DEVICE,
                    )

                    row_corrupt_patch = intervene_on_sae_features(feature_sae, row_corrupt_vector, [feature_index], mode="patch", source_vector=row_clean_vector)
                    row_corrupt_patch_result = score_feature_intervention(
                        model=analysis_model,
                        bundle=bundle,
                        prompt=row_corrupt_prompt,
                        target_token=row_clean_target,
                        base_cache=row_corrupt_cache,
                        source_patch=row_source_patch,
                        modified_source_tensor=build_modified_source(row_corrupt_source, row_corrupt_patch["reconstructed"], row_final_position_index),
                        destination_layer_index=1,
                        device=DEVICE,
                    )

                    clean_correct_count += int(row_clean_result["predicted_token"] == row_clean_target)
                    clean_ablate_correct_count += int(row_clean_ablation_result["predicted_token"] == row_clean_target)
                    corrupt_restore_count += int(row_corrupt_patch_result["predicted_token"] == row_clean_target)
                    clean_margin_values.append(row_clean_result["margin"])
                    clean_ablate_margin_values.append(row_clean_ablation_result["margin"])
                    corrupt_patch_margin_values.append(row_corrupt_patch_result["margin"])

                batch_rows.append(
                    {
                        "feature_index": feature_index,
                        "role": selected_feature_roles[feature_index],
                        "num_examples": BATCH_LIMIT,
                        "clean_accuracy": clean_correct_count / BATCH_LIMIT,
                        "clean_ablate_accuracy": clean_ablate_correct_count / BATCH_LIMIT,
                        "corrupt_patch_restore_rate": corrupt_restore_count / BATCH_LIMIT,
                        "mean_clean_margin": float(sum(clean_margin_values) / BATCH_LIMIT),
                        "mean_clean_ablate_margin": float(sum(clean_ablate_margin_values) / BATCH_LIMIT),
                        "mean_corrupt_patch_margin": float(sum(corrupt_patch_margin_values) / BATCH_LIMIT),
                    }
                )

            batch_feature_df = pd.DataFrame(batch_rows)
            display(batch_feature_df)

            def top_token_rows(logits: torch.Tensor, id_to_token: dict[int, str], top_k: int = 5) -> list[dict[str, object]]:
                top_logits, top_indices = torch.topk(logits, k=min(top_k, logits.shape[0]))
                return [
                    {"token": id_to_token[int(index.item())], "logit": float(value.item())}
                    for value, index in zip(top_logits, top_indices)
                ]

            lens_rows = []
            for label, vector in [
                ("clean L1H0 write", clean_l1h0_vector),
                ("corrupt L1H0 write", corrupt_l1h0_vector),
                ("clean-minus-corrupt L1H0 delta", clean_l1h0_vector - corrupt_l1h0_vector),
            ]:
                logits = residual_vector_to_logits(analysis_model, vector)
                lens_rows.append(
                    {
                        "object": label,
                        "clean_target_logit": float(logits[bundle.token_to_id[clean_target]].item()),
                        "corrupt_target_logit": float(logits[bundle.token_to_id[corrupt_target]].item()),
                        "clean_minus_corrupt_target": float(
                            (logits[bundle.token_to_id[clean_target]] - logits[bundle.token_to_id[corrupt_target]]).item()
                        ),
                        "top_tokens": top_token_rows(logits, bundle.id_to_token),
                    }
                )

            for feature_index in selected_features:
                decoder_vector = feature_sae.decoder.weight[:, feature_index].detach().cpu()
                decoder_logits = residual_vector_to_logits(analysis_model, decoder_vector)
                clean_activation = float(feature_sae.encode(clean_l1h0_vector.unsqueeze(0))[0, feature_index].item())
                corrupt_activation = float(feature_sae.encode(corrupt_l1h0_vector.unsqueeze(0))[0, feature_index].item())
                clean_feature_vector = decoder_vector * clean_activation
                corrupt_feature_vector = decoder_vector * corrupt_activation
                clean_feature_logits = residual_vector_to_logits(analysis_model, clean_feature_vector)
                corrupt_feature_logits = residual_vector_to_logits(analysis_model, corrupt_feature_vector)

                lens_rows.extend(
                    [
                        {
                            "object": f"feature {feature_index} decoder direction",
                            "clean_target_logit": float(decoder_logits[bundle.token_to_id[clean_target]].item()),
                            "corrupt_target_logit": float(decoder_logits[bundle.token_to_id[corrupt_target]].item()),
                            "clean_minus_corrupt_target": float(
                                (decoder_logits[bundle.token_to_id[clean_target]] - decoder_logits[bundle.token_to_id[corrupt_target]]).item()
                            ),
                            "top_tokens": top_token_rows(decoder_logits, bundle.id_to_token),
                        },
                        {
                            "object": f"feature {feature_index} clean activated contribution",
                            "clean_target_logit": float(clean_feature_logits[bundle.token_to_id[clean_target]].item()),
                            "corrupt_target_logit": float(clean_feature_logits[bundle.token_to_id[corrupt_target]].item()),
                            "clean_minus_corrupt_target": float(
                                (clean_feature_logits[bundle.token_to_id[clean_target]] - clean_feature_logits[bundle.token_to_id[corrupt_target]]).item()
                            ),
                            "top_tokens": top_token_rows(clean_feature_logits, bundle.id_to_token),
                        },
                        {
                            "object": f"feature {feature_index} corrupt activated contribution",
                            "clean_target_logit": float(corrupt_feature_logits[bundle.token_to_id[clean_target]].item()),
                            "corrupt_target_logit": float(corrupt_feature_logits[bundle.token_to_id[corrupt_target]].item()),
                            "clean_minus_corrupt_target": float(
                                (corrupt_feature_logits[bundle.token_to_id[clean_target]] - corrupt_feature_logits[bundle.token_to_id[corrupt_target]]).item()
                            ),
                            "top_tokens": top_token_rows(corrupt_feature_logits, bundle.id_to_token),
                        },
                    ]
                )

            lens_df = pd.DataFrame(lens_rows)
            display(lens_df)

            upstream_feature_set = support_features[:2]
            set_rows = []
            for label, site_kind, feature_set in [
                ("L1H0 upstream set", "upstream", upstream_feature_set),
                ("L2H0.Q downstream set", "downstream", downstream_feature_set),
            ]:
                if site_kind == "upstream":
                    clean_intervention = intervene_on_sae_features(feature_sae, clean_l1h0_vector, feature_set, mode="ablate")
                    clean_result_set = score_feature_intervention(
                        model=analysis_model,
                        bundle=bundle,
                        prompt=clean_prompt,
                        target_token=clean_target,
                        base_cache=clean_cache,
                        source_patch=source_patch,
                        modified_source_tensor=build_modified_source(clean_l1h0_source, clean_intervention["reconstructed"], final_position_index),
                        destination_layer_index=1,
                        device=DEVICE,
                    )
                    corrupt_intervention = intervene_on_sae_features(feature_sae, corrupt_l1h0_vector, feature_set, mode="patch", source_vector=clean_l1h0_vector)
                    corrupt_result_set = score_feature_intervention(
                        model=analysis_model,
                        bundle=bundle,
                        prompt=corrupt_prompt,
                        target_token=clean_target,
                        base_cache=corrupt_cache,
                        source_patch=source_patch,
                        modified_source_tensor=build_modified_source(corrupt_l1h0_source, corrupt_intervention["reconstructed"], final_position_index),
                        destination_layer_index=1,
                        device=DEVICE,
                    )
                    clean_query = clean_result_set["details"]["destination_cache"]["attention"]["q"][0, 0, final_position_index, :].detach().cpu()
                    corrupt_query = corrupt_result_set["details"]["destination_cache"]["attention"]["q"][0, 0, final_position_index, :].detach().cpu()
                    clean_attn = clean_result_set["details"]["destination_cache"]["attention"]["pattern"][0, 0, final_position_index, :].detach().cpu()
                    corrupt_attn = corrupt_result_set["details"]["destination_cache"]["attention"]["pattern"][0, 0, final_position_index, :].detach().cpu()
                else:
                    clean_intervention = intervene_on_sae_features(l2q_sae, clean_l2h0_q, feature_set, mode="ablate")
                    clean_result_set = score_query_feature_intervention(
                        model=analysis_model,
                        bundle=bundle,
                        prompt=clean_prompt,
                        target_token=clean_target,
                        base_cache=clean_cache,
                        layer_index=1,
                        head_index=0,
                        position_index=final_position_index,
                        modified_query_vector=clean_intervention["reconstructed"],
                        device=DEVICE,
                    )
                    corrupt_intervention = intervene_on_sae_features(l2q_sae, corrupt_l2h0_q, feature_set, mode="patch", source_vector=clean_l2h0_q)
                    corrupt_result_set = score_query_feature_intervention(
                        model=analysis_model,
                        bundle=bundle,
                        prompt=corrupt_prompt,
                        target_token=clean_target,
                        base_cache=corrupt_cache,
                        layer_index=1,
                        head_index=0,
                        position_index=final_position_index,
                        modified_query_vector=corrupt_intervention["reconstructed"],
                        device=DEVICE,
                    )
                    clean_query = clean_result_set["details"]["modified_q"][0, 0, final_position_index, :].detach().cpu()
                    corrupt_query = corrupt_result_set["details"]["modified_q"][0, 0, final_position_index, :].detach().cpu()
                    clean_attn = clean_result_set["details"]["pattern"][0, 0, final_position_index, :].detach().cpu()
                    corrupt_attn = corrupt_result_set["details"]["pattern"][0, 0, final_position_index, :].detach().cpu()

                set_rows.append(
                    {
                        "site_label": label,
                        "feature_set": feature_set,
                        "clean_ablate_predicted_token": clean_result_set["predicted_token"],
                        "clean_ablate_margin": clean_result_set["margin"],
                        "clean_ablate_q_delta_norm": float((clean_query - clean_l2h0_q).norm().item()),
                        "clean_ablate_attention_on_clean_value": float(clean_attn[clean_value_position].item()),
                        "corrupt_patch_predicted_token": corrupt_result_set["predicted_token"],
                        "corrupt_patch_margin_vs_clean_target": corrupt_result_set["margin"],
                        "corrupt_patch_q_delta_norm": float((corrupt_query - corrupt_l2h0_q).norm().item()),
                        "corrupt_patch_attention_on_clean_value": float(corrupt_attn[clean_value_position].item()),
                        "corrupt_patch_attention_on_corrupt_value": float(corrupt_attn[corrupt_value_position].item()),
                    }
                )

            feature_set_df = pd.DataFrame(set_rows)
            display(feature_set_df)
            """
        ),
        markdown_cell(
            """
            ## Neuron Dashboard And Upstream Causes

            This cell starts from the strongest `L1` MLP neurons on the anchor prompt and traces what drives them.

            Why this helps:

            - it identifies neurons that matter by exact single-neuron ablation, not by raw activation size
            - it compares clean vs corrupt activation
            - it asks which upstream heads and source tokens inside those heads move the neuron back toward clean
            """
        ),
        code_cell(
            """
            single_prompt_l1_df = build_mlp_neuron_contribution_table(
                model=analysis_model,
                bundle=bundle,
                prompt=clean_prompt,
                cache=clean_cache,
                layer_index=0,
                target_token=clean_target,
                foil_token=foil_token,
                device=DEVICE,
                position=final_position_index,
                top_k=5,
                include_exact_ablation=True,
            )
            selected_neurons = (
                single_prompt_l1_df
                .sort_values(["exact_ablation_margin_drop", "activated"], ascending=[False, False])
                .head(5)["neuron_index"]
                .astype(int)
                .tolist()
            )
            display(
                single_prompt_l1_df
                .sort_values(["exact_ablation_margin_drop", "activated"], ascending=[False, False])
                .head(5)
                .reset_index(drop=True)
            )

            clean_activated = clean_cache["blocks"][0]["mlp"]["activated"][0, -1, :].detach().cpu()
            corrupt_activated = corrupt_cache["blocks"][0]["mlp"]["activated"][0, -1, :].detach().cpu()
            shift_rows = []
            for neuron_index in selected_neurons:
                clean_value = float(clean_activated[neuron_index].item())
                corrupt_value = float(corrupt_activated[neuron_index].item())
                shift_rows.append(
                    {
                        "neuron_index": neuron_index,
                        "clean_activation": clean_value,
                        "corrupt_activation": corrupt_value,
                        "clean_minus_corrupt": clean_value - corrupt_value,
                    }
                )
            shift_df = pd.DataFrame(shift_rows).sort_values("clean_minus_corrupt", ascending=False).reset_index(drop=True)
            display(shift_df)

            head_effect_df = build_mlp_neuron_upstream_head_effect_table(
                model=analysis_model,
                prompt=clean_prompt,
                cache=clean_cache,
                layer_index=0,
                neuron_indices=selected_neurons,
                position=-1,
            )
            head_effect_summary_df = (
                head_effect_df
                .groupby(["head_index", "component"], as_index=False)
                .agg(
                    mean_abs_activation_delta=("abs_activation_delta", "mean"),
                    max_abs_activation_delta=("abs_activation_delta", "max"),
                    mean_signed_activation_delta=("activation_delta", "mean"),
                )
                .sort_values(["mean_abs_activation_delta", "max_abs_activation_delta"], ascending=[False, False])
                .reset_index(drop=True)
            )
            display(head_effect_summary_df)

            head_patch_df = build_mlp_neuron_clean_corrupt_head_patch_table(
                model=analysis_model,
                clean_prompt=clean_prompt,
                clean_cache=clean_cache,
                corrupt_prompt=corrupt_prompt,
                corrupt_cache=corrupt_cache,
                layer_index=0,
                neuron_indices=selected_neurons,
                position=-1,
            )
            head_patch_summary_df = (
                head_patch_df
                .groupby(["head_index", "component"], as_index=False)
                .agg(
                    mean_recovery_toward_clean=("recovery_toward_clean", "mean"),
                    max_recovery_toward_clean=("recovery_toward_clean", "max"),
                )
                .sort_values(["mean_recovery_toward_clean", "max_recovery_toward_clean"], ascending=[False, False])
                .reset_index(drop=True)
            )
            dominant_upstream_head = int(head_patch_summary_df.iloc[0]["head_index"])
            display(head_patch_summary_df)

            source_patch_df = build_mlp_neuron_clean_corrupt_source_patch_table(
                model=analysis_model,
                clean_prompt=clean_prompt,
                clean_cache=clean_cache,
                corrupt_prompt=corrupt_prompt,
                corrupt_cache=corrupt_cache,
                layer_index=0,
                head_index=dominant_upstream_head,
                neuron_indices=selected_neurons,
                position=-1,
            )
            source_patch_summary_df = (
                source_patch_df
                .groupby(["source_position", "clean_source_token", "corrupt_source_token"], as_index=False)
                .agg(
                    mean_recovery_toward_clean=("recovery_toward_clean", "mean"),
                    max_recovery_toward_clean=("recovery_toward_clean", "max"),
                )
                .sort_values(["mean_recovery_toward_clean", "max_recovery_toward_clean"], ascending=[False, False])
                .reset_index(drop=True)
            )
            display(source_patch_summary_df)
            """
        ),
        markdown_cell(
            """
            ## Neuron Read Decomposition And Batch Neuron Checks

            This is the compact neuron synthesis cell.

            Why this helps:

            - the read decomposition shows which input dimensions move `gate` and `up`
            - the grouped label tables test whether the selected neurons are query-like, value-like, or positional
            - the batch ablation table checks whether these neurons stay important beyond one prompt
            """
        ),
        code_cell(
            """
            for neuron_index in selected_neurons[:3]:
                print(f"Neuron {neuron_index}")
                display(
                    build_mlp_neuron_read_comparison_table(
                        model=analysis_model,
                        clean_cache=clean_cache,
                        corrupt_cache=corrupt_cache,
                        layer_index=0,
                        neuron_index=neuron_index,
                        position=-1,
                    ).head(12)
                )

            neuron_activation_df = collect_mlp_neuron_activation_table(
                model=analysis_model,
                bundle=bundle,
                split="test",
                layer_index=0,
                device=DEVICE,
                neuron_indices=selected_neurons,
                position=-1,
            )

            for group_column in ["query_key", "target", "correct_value_position", "query_pair_index", "corrupt_target"]:
                print(f"Grouped by {group_column}")
                display(build_mlp_neuron_group_summary_table(neuron_activation_df, selected_neurons, group_column))

            top_examples = build_top_mlp_neuron_examples(
                neuron_activation_df,
                neuron_indices=selected_neurons,
                top_k=5,
            )
            for neuron_index in selected_neurons:
                display(Markdown(f"### Neuron `{neuron_index}` top prompts"))
                display(pd.DataFrame(top_examples[neuron_index]))

            batch_ablation_df = build_mlp_neuron_batch_ablation_table(
                model=analysis_model,
                bundle=bundle,
                split="test",
                layer_index=0,
                device=DEVICE,
                neuron_indices=selected_neurons,
                limit=NEURON_BATCH_LIMIT,
                position=-1,
            )
            display(batch_ablation_df.sort_values("mean_margin_drop", ascending=False).reset_index(drop=True))
            """
        ),
        markdown_cell(
            """
            ## Register And Stage Summary

            This final section is the synthesis layer.

            It does **not** magically solve the internal algorithm. It simply compresses the previous tracking objects into the best current register candidates:

            - where query-key identity is most readable
            - which heads behave most like selected-slot trackers
            - where selected-value identity becomes strongly readable
            - which head writes the strongest selected-value evidence
            """
        ),
        code_cell(
            """
            stage_margin_df = build_layer_feature_readout_table(
                model=analysis_model,
                bundle=bundle,
                cache=clean_cache,
                target_token=clean_target,
                foil_token=foil_token,
            )
            stage_variable_df = build_stage_variable_readout_table(
                model=analysis_model,
                bundle=bundle,
                cache=clean_cache,
                row=example,
            )
            display(stage_margin_df[["stage", "top_token", "target_minus_foil", "top_k_tokens"]])
            display(stage_variable_df)

            single_prompt_head_df = build_single_prompt_head_role_table(
                model=analysis_model,
                bundle=bundle,
                row=example,
                cache=clean_cache,
                target_token=clean_target,
                foil_token=foil_token,
            )
            single_prompt_slot_df = build_single_prompt_slot_routing_table(
                model=analysis_model,
                row=example,
                cache=clean_cache,
            )
            query_swap_head_df = build_query_swap_head_role_comparison_table(
                model=analysis_model,
                clean_row=example,
                clean_cache=clean_cache,
                corrupt_row=corrupt_row,
                corrupt_cache=corrupt_cache,
            )
            query_swap_slot_df = build_query_swap_slot_routing_comparison_table(
                model=analysis_model,
                clean_row=example,
                clean_cache=clean_cache,
                corrupt_row=corrupt_row,
                corrupt_cache=corrupt_cache,
            )
            query_swap_slot_df["clean_and_corrupt_top_slot_match_selected"] = (
                query_swap_slot_df["clean_top_slot_matches_selected"] & query_swap_slot_df["corrupt_top_slot_matches_selected"]
            )

            batch_stage_variable_df = collect_stage_variable_readout_table(
                model=analysis_model,
                bundle=bundle,
                split="test",
                device=DEVICE,
                limit=BATCH_LIMIT,
            )
            stage_summary_df = build_stage_variable_summary_table(batch_stage_variable_df)
            stage_order = stage_variable_df["stage"].tolist()
            stage_summary_df["stage"] = pd.Categorical(stage_summary_df["stage"], categories=stage_order, ordered=True)
            stage_summary_df = stage_summary_df.sort_values("stage").reset_index(drop=True)

            batch_head_role_df = collect_head_role_attention_table(
                model=analysis_model,
                bundle=bundle,
                split="test",
                device=DEVICE,
                limit=BATCH_LIMIT,
            )
            head_role_summary_df = build_head_role_summary_table(batch_head_role_df)

            batch_slot_routing_df = collect_slot_routing_table(
                model=analysis_model,
                bundle=bundle,
                split="test",
                device=DEVICE,
                limit=BATCH_LIMIT,
            )
            slot_routing_summary_df = build_slot_routing_summary_table(batch_slot_routing_df)

            display(stage_summary_df)
            display(slot_routing_summary_df.sort_values(["top_selected_slot_rate", "mean_selected_slot_attention"], ascending=[False, False]).reset_index(drop=True))
            display(head_role_summary_df.sort_values("mean_query_key_attention", ascending=False).reset_index(drop=True))

            best_query_stage = stage_summary_df.sort_values(
                ["query_top_key_rate", "mean_query_minus_best_distractor_key"],
                ascending=[False, False],
            ).iloc[0]
            best_selected_value_stage = stage_summary_df.sort_values(
                ["target_top_value_rate", "mean_target_minus_best_distractor_value"],
                ascending=[False, False],
            ).iloc[0]
            best_query_head = head_role_summary_df.sort_values(
                ["mean_query_key_attention", "top_query_key_rate"],
                ascending=[False, False],
            ).iloc[0]
            best_slot_head = slot_routing_summary_df.sort_values(
                ["top_selected_slot_rate", "mean_selected_slot_attention"],
                ascending=[False, False],
            ).iloc[0]
            best_value_head = single_prompt_head_df.sort_values(
                ["selected_value_weighted_target_minus_foil", "selected_value_attention"],
                ascending=[False, False],
            ).iloc[0]

            register_map_df = pd.DataFrame(
                [
                    {
                        "register": "query_key_reader",
                        "best_candidate_site": best_query_head["component"],
                        "evidence": f"mean_query_key_attention={best_query_head['mean_query_key_attention']:.3f}, top_query_key_rate={best_query_head['top_query_key_rate']:.3f}",
                    },
                    {
                        "register": "selected_slot_tracker",
                        "best_candidate_site": best_slot_head["component"],
                        "evidence": f"top_selected_slot_rate={best_slot_head['top_selected_slot_rate']:.3f}, mean_selected_slot_attention={best_slot_head['mean_selected_slot_attention']:.3f}",
                    },
                    {
                        "register": "selected_value_readout",
                        "best_candidate_site": best_selected_value_stage["stage"],
                        "evidence": f"target_top_value_rate={best_selected_value_stage['target_top_value_rate']:.3f}, mean_target_minus_best_distractor_value={best_selected_value_stage['mean_target_minus_best_distractor_value']:.3f}",
                    },
                    {
                        "register": "selected_value_writer",
                        "best_candidate_site": best_value_head["component"],
                        "evidence": f"selected_value_weighted_target_minus_foil={best_value_head['selected_value_weighted_target_minus_foil']:.3f}",
                    },
                ]
            )
            display(register_map_df)
            """
        ),
        markdown_cell(
            """
            ## What This Notebook Now Gives You

            This unified notebook does not claim the internal algorithm is solved. What it *does* give you is one canonical analysis artifact with no split across four notebook styles.

            It now contains:

            - activation patching
            - path patching
            - causal ablations
            - QK / OV analysis
            - circuit tracing
            - sparse autoencoder feature dashboards
            - feature logit lens and cross-layer feature-set checks
            - neuron-level upstream-cause and read-decomposition tables
            - a register-style summary at the end

            The remaining gap is no longer "where is the data?" but "how do we turn these tracked objects into a minimal symbolic program?"
            """
        ),
    ]

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.11",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(build_notebook(), indent=1) + "\n")
    print(f"Wrote notebook to {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
