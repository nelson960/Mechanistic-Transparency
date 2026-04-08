from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import torch

from scripts.device_utils import resolve_training_device
from scripts.kv_benchmark import instantiate_kv_model
from scripts.kv_algorithm_oracle import annotate_row
from scripts.kv_algorithm_record import build_final_position_site_list
from scripts.kv_algorithm_sweeps import generate_controlled_sweeps
from scripts.kv_retrieve_analysis import (
    load_checkpoint_model,
    load_dataset_bundle,
    ov_source_logits,
    run_prompt,
)
from scripts.tiny_transformer_core import TinyDecoderTransformer
from scripts.training_dynamics import build_checkpoint_epoch_schedule, build_run_manifest


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "bin" / "python"


def run_command(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(PYTHON), *args],
        cwd=str(ROOT),
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        check=True,
        capture_output=True,
        text=True,
    )


def make_manifest_dict(dataset_dir: Path, output_dir: Path, *, seed: int, epochs: int = 3) -> dict:
    dense_through_epoch = min(2, epochs)
    log_spaced_epoch_count = 1 if epochs > dense_through_epoch else 0
    return {
        "benchmark": {"name": "kv_retrieval"},
        "dataset": {
            "dataset_dir": str(dataset_dir),
            "train_split_by_pairs": {"2": "train_2_pairs", "3": "train_3_pairs"},
            "eval_splits": {"val": "val", "test": "test", "ood": "test_ood_4_pairs"},
            "sweep_base_split": "test",
        },
        "model": {
            "d_model": 16,
            "n_heads": 2,
            "d_ff": 32,
            "n_layers": 2,
            "max_seq_len": 16,
        },
        "training": {
            "epochs": epochs,
            "batch_size": 8,
            "learning_rate": 0.02,
            "weight_decay": 0.0,
            "seed": seed,
            "curriculum": "off",
            "device": "cpu",
        },
        "checkpoint_schedule": {
            "dense_through_epoch": dense_through_epoch,
            "log_spaced_epoch_count": log_spaced_epoch_count,
            "save_epoch_zero": True,
            "save_final": True,
            "best_metric": "val_accuracy",
        },
        "battery": {
            "train_probe_limit": 8,
            "sweep_base_limit": 4,
            "eval_batch_size": 8,
            "role_top_k": 3,
        },
        "sae_tracking": {
            "enabled": True,
            "sites": ["block1_final_resid_after_mlp", "block2_head0_final_q"],
            "train_limit": 8,
            "val_limit": 4,
            "hidden_multiplier": 2,
            "l1_coeff": 0.001,
            "learning_rate": 0.01,
            "batch_size": 4,
            "epochs": 2,
            "seed": seed,
            "top_features_per_site": 3,
            "superposition_cosine_threshold": 0.2,
        },
        "training_interventions": [],
        "summary_thresholds": {
            "behavior_birth_val_accuracy": 0.9,
            "operator_birth_score": 0.85,
            "operator_family_min_score": 0.75,
            "variable_birth_score": 0.85,
            "variable_family_min_score": 0.75,
            "faithfulness_birth_score": 0.85,
            "faithfulness_family_min_score": 0.75,
        },
        "output_dir": str(output_dir),
    }


class TrainingDynamicsTest(unittest.TestCase):
    def test_resolve_training_device_auto_prefers_mps_when_available(self) -> None:
        from unittest.mock import patch

        with patch("scripts.device_utils._mps_is_available", return_value=True), patch(
            "scripts.device_utils._cuda_is_available", return_value=False
        ):
            self.assertEqual(resolve_training_device("auto"), "mps")

    def test_resolve_training_device_rejects_unavailable_mps(self) -> None:
        from unittest.mock import patch

        with patch("scripts.device_utils._mps_is_available", return_value=False):
            with self.assertRaises(ValueError):
                resolve_training_device("mps")

    def test_manifest_validation_rejects_non_divisible_heads(self) -> None:
        manifest_dict = make_manifest_dict(ROOT / "dataset" / "kv_retrieve_3", ROOT / "temp" / "invalid", seed=0)
        manifest_dict["model"]["d_model"] = 15
        with self.assertRaises(ValueError):
            build_run_manifest(manifest_dict)

    def test_checkpoint_schedule_matches_pilot_structure(self) -> None:
        manifest_dict = make_manifest_dict(ROOT / "dataset" / "kv_retrieve_3", ROOT / "temp" / "pilot", seed=0, epochs=150)
        manifest_dict["model"]["d_model"] = 32
        manifest_dict["model"]["d_ff"] = 64
        manifest_dict["checkpoint_schedule"]["dense_through_epoch"] = 20
        manifest_dict["checkpoint_schedule"]["log_spaced_epoch_count"] = 24
        manifest = build_run_manifest(manifest_dict)
        schedule = build_checkpoint_epoch_schedule(manifest)
        self.assertEqual(schedule[0], 0)
        self.assertEqual(schedule[20], 20)
        self.assertIn(150, schedule)
        self.assertEqual(len(schedule), 45)
        self.assertEqual(schedule, sorted(set(schedule)))

    def test_dynamic_site_registry_matches_model_shape(self) -> None:
        model = TinyDecoderTransformer(
            vocab_size=20,
            d_model=16,
            n_heads=2,
            d_ff=32,
            n_layers=3,
            max_seq_len=16,
        )
        site_list = build_final_position_site_list(model)
        expected_count = (3 * (2 + (2 * 5))) + 1
        self.assertEqual(len(site_list), expected_count)
        self.assertIn("block3_head1_final_v", site_list)
        self.assertIn("final_hidden", site_list)

    def test_sweep_families_preserve_intended_invariants(self) -> None:
        base_row = load_dataset_bundle(ROOT / "dataset" / "kv_retrieve_3").raw_splits["test"][0]
        ood_row = load_dataset_bundle(ROOT / "dataset" / "kv_retrieve_3").raw_splits["test_ood_4_pairs"][0]
        sweep_rows = generate_controlled_sweeps([base_row], longer_context_rows=[ood_row])
        family_names = {str(row["family_name"]) for row in sweep_rows}
        self.assertEqual(
            family_names,
            {
                "query_key_sweep",
                "slot_permutation",
                "value_permutation",
                "same_answer_different_slot",
                "same_slot_different_answer",
                "longer_context_ood",
            },
        )

        base_annotation = annotate_row(base_row)
        same_answer_row = next(row for row in sweep_rows if row["family_name"] == "same_answer_different_slot")
        same_answer_annotation = annotate_row(same_answer_row)
        self.assertEqual(same_answer_annotation.selected_value, base_annotation.selected_value)
        self.assertNotEqual(same_answer_annotation.matching_slot, base_annotation.matching_slot)

        same_slot_row = next(row for row in sweep_rows if row["family_name"] == "same_slot_different_answer")
        same_slot_annotation = annotate_row(same_slot_row)
        self.assertEqual(same_slot_annotation.matching_slot, base_annotation.matching_slot)
        self.assertNotEqual(same_slot_annotation.selected_value, base_annotation.selected_value)

    def test_dataset_generator_writes_multi_train_splits(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir) / "dataset"
            run_command(
                "scripts/generate_kv_retrieval_dataset.py",
                "--outdir",
                str(dataset_dir),
                "--train-size",
                "12",
                "--val-size",
                "4",
                "--test-size",
                "4",
                "--ood-size",
                "4",
                "--train-context-pairs",
                "2,3",
                "--seed",
                "11",
            )
            self.assertTrue((dataset_dir / "train_2_pairs.jsonl").exists())
            self.assertTrue((dataset_dir / "train_3_pairs.jsonl").exists())
            metadata = json.loads((dataset_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["training_splits"], {"2": "train_2_pairs", "3": "train_3_pairs"})

    def test_dataset_generator_supports_fixed_query_slot_policy(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir) / "dataset"
            run_command(
                "scripts/generate_kv_retrieval_dataset.py",
                "--outdir",
                str(dataset_dir),
                "--train-size",
                "8",
                "--val-size",
                "4",
                "--test-size",
                "4",
                "--ood-size",
                "4",
                "--context-pairs",
                "2",
                "--train-context-pairs",
                "2",
                "--ood-context-pairs",
                "3",
                "--query-slot-policy",
                "fixed_first",
                "--seed",
                "13",
            )
            first_train_row = json.loads((dataset_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(first_train_row["query_key"], first_train_row["context_pairs"][0]["key"])
            metadata = json.loads((dataset_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["generation_rules"]["query_slot_policy"], "fixed_first")

    def test_manifest_accepts_initialization_scale_and_model_rescales_weights(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir) / "dataset"
            run_command(
                "scripts/generate_kv_retrieval_dataset.py",
                "--outdir",
                str(dataset_dir),
                "--train-size",
                "8",
                "--val-size",
                "4",
                "--test-size",
                "4",
                "--ood-size",
                "4",
                "--train-context-pairs",
                "2",
                "--seed",
                "5",
            )
            bundle = load_dataset_bundle(dataset_dir)
            manifest_dict = make_manifest_dict(dataset_dir, Path(temp_dir) / "run_seed0", seed=0)
            manifest_base = build_run_manifest(manifest_dict)
            manifest_scaled = build_run_manifest({**manifest_dict, "initialization": {"scale": 2.0}})
            self.assertEqual(manifest_base.initialization.scale, 1.0)
            self.assertEqual(manifest_scaled.initialization.scale, 2.0)

            torch.manual_seed(0)
            base_model = instantiate_kv_model(manifest_base, bundle, device=torch.device("cpu"))
            torch.manual_seed(0)
            scaled_model = instantiate_kv_model(manifest_scaled, bundle, device=torch.device("cpu"))
            base_norm = float(base_model.blocks[0].attn.q_proj.weight.norm().item())
            scaled_norm = float(scaled_model.blocks[0].attn.q_proj.weight.norm().item())
            self.assertAlmostEqual(scaled_norm / base_norm, 2.0, places=5)

    def test_factor_screen_builder_writes_python_specific_scripts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            manifest_dir = temp_root / "manifests"
            dataset_root = temp_root / "datasets"
            run_root = temp_root / "runs"
            run_command(
                "-m",
                "research.phase3.scripts.build_kv_factor_screen",
                "--manifest-outdir",
                str(manifest_dir),
                "--dataset-root",
                str(dataset_root),
                "--run-root",
                str(run_root),
                "--python-bin",
                "/opt/miniconda3/envs/ml/bin/python",
                "--seeds",
                "0,1",
                "--init-scales",
                "1.0",
                "--weight-decays",
                "0.0",
                "--d-models",
                "64",
                "--query-slot-policies",
                "balanced",
                "--epochs",
                "1",
            )
            dataset_script = (manifest_dir / "dataset_build_commands.sh").read_text(encoding="utf-8")
            run_script = (manifest_dir / "run_manifests_sequential.sh").read_text(encoding="utf-8")
            manifest_matrix = (manifest_dir / "manifest_matrix.csv").read_text(encoding="utf-8")
            manifest_payload = json.loads((manifest_dir / "seed0_slot_balanced_d64_wd0_init1.json").read_text(encoding="utf-8"))
            self.assertIn("/opt/miniconda3/envs/ml/bin/python -m scripts.generate_kv_retrieval_dataset", dataset_script)
            self.assertIn("/opt/miniconda3/envs/ml/bin/python -m research.phase3.scripts.run_kv_factor_screen", run_script)
            self.assertIn(str(run_root), manifest_matrix)
            self.assertEqual(manifest_payload["checkpoint_schedule"]["dense_through_epoch"], 1)
            self.assertEqual(manifest_payload["checkpoint_schedule"]["log_spaced_epoch_count"], 0)

    def test_ov_source_logits_respects_batch_index(self) -> None:
        model = TinyDecoderTransformer(
            vocab_size=2,
            d_model=2,
            n_heads=1,
            d_ff=4,
            n_layers=1,
            max_seq_len=2,
        )
        with torch.no_grad():
            model.blocks[0].attn.v_proj.weight.copy_(torch.eye(2))
            model.blocks[0].attn.o_proj.weight.copy_(torch.eye(2))
            model.token_embed.weight.copy_(torch.eye(2))

        cache = {
            "blocks": [
                {
                    "attn_in": torch.tensor(
                        [
                            [[1.0, 0.0], [0.0, 1.0]],
                            [[0.0, 1.0], [1.0, 0.0]],
                        ],
                        dtype=torch.float32,
                    )
                }
            ]
        }
        logits_batch0 = ov_source_logits(model, cache, layer_index=0, head_index=0, source_position=0, batch_index=0)
        logits_batch1 = ov_source_logits(model, cache, layer_index=0, head_index=0, source_position=0, batch_index=1)
        self.assertEqual(logits_batch0.argmax().item(), 0)
        self.assertEqual(logits_batch1.argmax().item(), 1)

    def test_legacy_checkpoint_compatibility(self) -> None:
        bundle = load_dataset_bundle(ROOT / "dataset" / "kv_retrieve_3")
        checkpoint_path = ROOT / "models" / "kv_retrieve_3" / "selected_checkpoint.pt"
        checkpoint, model = load_checkpoint_model(checkpoint_path, device="cpu")
        self.assertIn("selected_metrics", checkpoint)
        row = bundle.raw_splits["test"][0]
        result, _cache = run_prompt(model, bundle, row["prompt"], device="cpu", expected_target=row["target"])
        self.assertIn("predicted_token", result)
        self.assertTrue(result["correct"])

    def test_smoke_cli_pipeline_outputs_expected_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            dataset_dir = temp_root / "dataset"
            run_dir = temp_root / "run_seed0"
            manifest_path = temp_root / "manifest.json"

            run_command(
                "scripts/generate_kv_retrieval_dataset.py",
                "--outdir",
                str(dataset_dir),
                "--train-size",
                "16",
                "--val-size",
                "8",
                "--test-size",
                "8",
                "--ood-size",
                "8",
                "--train-context-pairs",
                "2,3",
                "--seed",
                "7",
            )
            manifest_path.write_text(
                json.dumps(make_manifest_dict(dataset_dir, run_dir, seed=0), indent=2),
                encoding="utf-8",
            )

            run_command("scripts/train_run.py", "--manifest", str(manifest_path))
            history_rows = [
                json.loads(line)
                for line in (run_dir / "train_history.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertGreater(len(history_rows), 0)
            self.assertIn("total_grad_norm", history_rows[0])
            self.assertIn("total_update_norm", history_rows[0])
            self.assertIn("parameter_metrics", history_rows[0])
            self.assertGreater(len(history_rows[0]["parameter_metrics"]), 0)
            run_command("scripts/run_checkpoint_battery.py", "--run-dir", str(run_dir))
            run_command("scripts/summarize_training_dynamics.py", "--target-dir", str(run_dir))

            checkpoint_files = sorted((run_dir / "checkpoints").glob("*.pt"))
            self.assertGreaterEqual(len(checkpoint_files), 3)
            for checkpoint_path in checkpoint_files:
                checkpoint_id = checkpoint_path.stem
                battery_dir = run_dir / "battery" / checkpoint_id
                self.assertTrue((battery_dir / "behavior.json").exists())
                self.assertTrue((battery_dir / "variable_scores.csv").exists())
                self.assertTrue((battery_dir / "variable_faithfulness.csv").exists())
                self.assertTrue((battery_dir / "operator_scores.csv").exists())
                self.assertTrue((battery_dir / "localization.csv").exists())
                self.assertTrue((battery_dir / "weight_metrics.json").exists())
                self.assertTrue((battery_dir / "neuron_scores.csv").exists())
                self.assertTrue((battery_dir / "feature_scores.csv").exists())
                self.assertTrue((battery_dir / "superposition_metrics.json").exists())
                self.assertTrue((battery_dir / "canonical_site_vectors.pt").exists())
                self.assertTrue((battery_dir / "sae").is_dir())
                self.assertTrue((battery_dir / "tensors.pt").exists())

            summary_dir = run_dir / "summaries"
            self.assertTrue((summary_dir / "checkpoint_index.csv").exists())
            self.assertTrue((summary_dir / "emergence.csv").exists())
            self.assertTrue((summary_dir / "seed_stability.csv").exists())
            self.assertTrue((summary_dir / "neuron_dynamics.csv").exists())
            self.assertTrue((summary_dir / "feature_dynamics.csv").exists())
            self.assertTrue((summary_dir / "superposition_dynamics.csv").exists())
            self.assertTrue((summary_dir / "representation_drift.csv").exists())
            self.assertTrue((summary_dir / "operator_handoffs.csv").exists())
            self.assertTrue((summary_dir / "role_matching.csv").exists())
            self.assertTrue((summary_dir / "clamp_responsiveness.csv").exists())
            self.assertTrue((summary_dir / "run_summary.json").exists())

            notebook_path = temp_root / "artifact_notebook.ipynb"
            analysis_notebook_path = temp_root / "artifact_analysis_notebook.ipynb"
            run_command(
                "scripts/generate_kv_retrieve_algorithm_discovery_notebook.py",
                "--run-dir",
                str(run_dir),
                "--checkpoint-id",
                checkpoint_files[0].stem,
                "--out",
                str(notebook_path),
            )
            run_command(
                "scripts/generate_kv_retrieve_algorithm_notebook.py",
                "--run-dir",
                str(run_dir),
                "--checkpoint-id",
                checkpoint_files[0].stem,
                "--out",
                str(analysis_notebook_path),
            )
            self.assertTrue(notebook_path.exists())
            self.assertTrue(analysis_notebook_path.exists())

    def test_dataset_rows_and_checkpoint_epochs_are_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            dataset_dir_a = temp_root / "dataset_a"
            dataset_dir_b = temp_root / "dataset_b"
            for dataset_dir in [dataset_dir_a, dataset_dir_b]:
                run_command(
                    "scripts/generate_kv_retrieval_dataset.py",
                    "--outdir",
                    str(dataset_dir),
                    "--train-size",
                    "12",
                    "--val-size",
                    "4",
                    "--test-size",
                    "4",
                    "--ood-size",
                    "4",
                    "--train-context-pairs",
                    "2,3",
                    "--seed",
                    "19",
                )

            self.assertEqual(
                (dataset_dir_a / "train_2_pairs.jsonl").read_text(encoding="utf-8"),
                (dataset_dir_b / "train_2_pairs.jsonl").read_text(encoding="utf-8"),
            )
            self.assertEqual(
                (dataset_dir_a / "train_3_pairs.jsonl").read_text(encoding="utf-8"),
                (dataset_dir_b / "train_3_pairs.jsonl").read_text(encoding="utf-8"),
            )

            run_dir_a = temp_root / "run_a"
            run_dir_b = temp_root / "run_b"
            manifest_a = temp_root / "manifest_a.json"
            manifest_b = temp_root / "manifest_b.json"
            manifest_a.write_text(json.dumps(make_manifest_dict(dataset_dir_a, run_dir_a, seed=3, epochs=2), indent=2), encoding="utf-8")
            manifest_b.write_text(json.dumps(make_manifest_dict(dataset_dir_a, run_dir_b, seed=3, epochs=2), indent=2), encoding="utf-8")

            run_command("scripts/train_run.py", "--manifest", str(manifest_a))
            run_command("scripts/train_run.py", "--manifest", str(manifest_b))

            checkpoint_names_a = sorted(path.name for path in (run_dir_a / "checkpoints").glob("*.pt"))
            checkpoint_names_b = sorted(path.name for path in (run_dir_b / "checkpoints").glob("*.pt"))
            self.assertEqual(checkpoint_names_a, checkpoint_names_b)

    def test_training_interventions_are_logged_in_history(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            dataset_dir = temp_root / "dataset"
            run_dir = temp_root / "run_intervention"
            manifest_path = temp_root / "manifest.json"

            run_command(
                "scripts/generate_kv_retrieval_dataset.py",
                "--outdir",
                str(dataset_dir),
                "--train-size",
                "12",
                "--val-size",
                "4",
                "--test-size",
                "4",
                "--ood-size",
                "4",
                "--train-context-pairs",
                "2,3",
                "--seed",
                "13",
            )
            manifest_dict = make_manifest_dict(dataset_dir, run_dir, seed=0, epochs=2)
            manifest_dict["training_interventions"] = [
                {
                    "name": "zero_head0_epoch1",
                    "kind": "head_resid_final_scale",
                    "layer_index": 0,
                    "head_index": 0,
                    "epoch_start": 1,
                    "epoch_end": 1,
                    "scale": 0.0,
                    "position": "final",
                }
            ]
            manifest_path.write_text(json.dumps(manifest_dict, indent=2), encoding="utf-8")

            run_command("scripts/train_run.py", "--manifest", str(manifest_path))
            history_rows = [
                json.loads(line)
                for line in (run_dir / "train_history.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertGreater(len(history_rows), 0)
            first_epoch_rows = [row for row in history_rows if int(row["epoch"]) == 1]
            self.assertGreater(len(first_epoch_rows), 0)
            self.assertEqual(len(first_epoch_rows[0]["active_interventions"]), 1)
            self.assertEqual(first_epoch_rows[0]["active_interventions"][0]["name"], "zero_head0_epoch1")


if __name__ == "__main__":
    unittest.main()
