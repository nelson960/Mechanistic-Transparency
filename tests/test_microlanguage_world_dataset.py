from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import torch

from scripts.generate_microlanguage_world_dataset import parse_prompt, resolve_prompt_target
from scripts.kv_retrieve_analysis import load_dataset_bundle
from scripts.microlanguage_world_benchmark import _encode_microlanguage_dense_sequence_batch
from scripts.tiny_transformer_core import TinyQueryGroupEncoderTransformer
from scripts.training_dynamics import build_run_manifest, evaluate_next_token_rows


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


class MicrolanguageWorldDatasetTest(unittest.TestCase):
    def _make_manifest_dict(
        self,
        dataset_dir: Path,
        run_dir: Path,
        *,
        seed: int,
        epochs: int = 2,
        early_stopping: dict | None = None,
        supervision_mode: str = "next_token_vocab",
    ) -> dict:
        dense_through_epoch = min(1, epochs)
        log_spaced_epoch_count = 0 if epochs <= dense_through_epoch else 1
        manifest = {
            "benchmark": {"name": "microlanguage_world_next_token"},
            "dataset": {
                "dataset_dir": str(dataset_dir),
                "train_split_by_pairs": {"default": "train"},
                "eval_splits": {
                    "val": "val",
                    "test": "test",
                    "ood": "test_ood_longer_context",
                },
                "sweep_base_split": "test",
                "supervision_mode": supervision_mode,
            },
            "model": {
                "d_model": 16,
                "n_heads": 2,
                "d_ff": 32,
                "n_layers": 2,
                "max_seq_len": 256,
            },
            "training": {
                "epochs": epochs,
                "batch_size": 4,
                "learning_rate": 0.01,
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
                "train_probe_limit": 1,
                "sweep_base_limit": 1,
                "eval_batch_size": 8,
                "role_top_k": 1,
            },
            "sae_tracking": {
                "enabled": False,
                "sites": [],
                "train_limit": 1,
                "val_limit": 1,
                "hidden_multiplier": 2,
                "l1_coeff": 0.001,
                "learning_rate": 0.01,
                "batch_size": 4,
                "epochs": 1,
                "seed": seed,
                "top_features_per_site": 1,
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
            "output_dir": str(run_dir),
        }
        if early_stopping is not None:
            manifest["early_stopping"] = early_stopping
        return manifest

    def test_generator_writes_bundle_and_targets_match_prompt_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir) / "dataset"
            run_command(
                "scripts/generate_microlanguage_world_dataset.py",
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
                "--seed",
                "11",
            )
            metadata = json.loads((dataset_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(
                metadata["splits"],
                {
                    "train": 8,
                    "val": 4,
                    "test": 4,
                    "test_ood_longer_context": 4,
                },
            )
            self.assertIn("latent_state", metadata)
            self.assertIn("query_families", metadata["latent_state"])
            bundle = load_dataset_bundle(dataset_dir)
            self.assertEqual(sorted(bundle.raw_splits), sorted(metadata["splits"]))
            self.assertGreater(len(bundle.vocab), 0)

            for split_name in metadata["splits"]:
                rows = [
                    json.loads(line)
                    for line in (dataset_dir / f"{split_name}.jsonl").read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                self.assertEqual(len(rows), metadata["splits"][split_name])
                self.assertGreater(len(rows), 0)
                for row in rows[:2]:
                    target, trace = resolve_prompt_target(str(row["prompt"]))
                    self.assertEqual(target, row["target"])
                    self.assertEqual(trace, row["trace"])
                    self.assertEqual(len(trace), int(row["query_depth"]))

    def test_balanced_query_family_policy_cycles_requested_subset(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir) / "dataset"
            run_command(
                "scripts/generate_microlanguage_world_dataset.py",
                "--outdir",
                str(dataset_dir),
                "--train-size",
                "6",
                "--val-size",
                "2",
                "--test-size",
                "2",
                "--ood-size",
                "2",
                "--query-families",
                "person_room,item_zone,item_owner_role",
                "--query-family-policy",
                "balanced",
                "--seed",
                "3",
            )
            train_rows = [
                json.loads(line)
                for line in (dataset_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            observed = [str(row["query_family"]) for row in train_rows]
            self.assertEqual(
                observed,
                [
                    "person_room",
                    "item_zone",
                    "item_owner_role",
                    "person_room",
                    "item_zone",
                    "item_owner_role",
                ],
            )

    def test_v2_small_preset_restricts_relations_verbs_and_query_families(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir) / "dataset"
            run_command(
                "scripts/generate_microlanguage_world_dataset.py",
                "--preset",
                "v2_small",
                "--outdir",
                str(dataset_dir),
                "--train-size",
                "10",
                "--val-size",
                "4",
                "--test-size",
                "4",
                "--ood-size",
                "4",
                "--seed",
                "13",
            )
            metadata = json.loads((dataset_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["preset"], "v2_small")
            self.assertEqual(
                metadata["generation_rules"]["query_families"],
                ["item_owner", "person_room", "room_zone", "item_room", "item_zone"],
            )
            self.assertEqual(
                metadata["generation_rules"]["active_relations"],
                ["item_owner", "person_room", "room_zone"],
            )
            self.assertEqual(metadata["generation_rules"]["max_surface_verbs_per_relation"], 1)
            self.assertEqual(sorted(metadata["latent_state"]["direct_relations"]), ["item_owner", "person_room", "room_zone"])
            self.assertEqual(sorted(metadata["vocabulary_groups"]["verbs"]), ["assign", "move", "zone"])
            self.assertEqual(
                sorted(metadata["vocabulary_groups"]["query_families"]),
                ["item_owner", "item_room", "item_zone", "person_room", "room_zone"],
            )
            train_rows = [
                json.loads(line)
                for line in (dataset_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            allowed_relations = {"item_owner", "person_room", "room_zone"}
            allowed_verbs = {"assign", "move", "zone"}
            for row in train_rows:
                parsed_target, trace = resolve_prompt_target(str(row["prompt"]))
                self.assertEqual(parsed_target, row["target"])
                self.assertEqual(trace, row["trace"])
                parsed_prompt = parse_prompt(str(row["prompt"]))
                event_verbs = [str(event["surface_verb"]) for event in parsed_prompt["events"]]
                self.assertTrue(set(event_verbs) <= allowed_verbs)
                self.assertTrue(set(row["relation_update_counts"]) <= allowed_relations)

    def test_v3_core_preset_has_lighter_ranges_and_same_core_relations(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir) / "dataset"
            run_command(
                "scripts/generate_microlanguage_world_dataset.py",
                "--preset",
                "v3_core",
                "--outdir",
                str(dataset_dir),
                "--train-size",
                "10",
                "--val-size",
                "4",
                "--test-size",
                "4",
                "--ood-size",
                "4",
                "--seed",
                "23",
            )
            metadata = json.loads((dataset_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["preset"], "v3_core")
            self.assertEqual(metadata["generation_rules"]["id_person_range"], [3, 3])
            self.assertEqual(metadata["generation_rules"]["id_item_range"], [4, 4])
            self.assertEqual(metadata["generation_rules"]["id_room_range"], [2, 2])
            self.assertEqual(metadata["generation_rules"]["id_extra_update_range"], [1, 1])
            self.assertEqual(metadata["generation_rules"]["ood_extra_update_range"], [2, 2])
            self.assertEqual(sorted(metadata["latent_state"]["direct_relations"]), ["item_owner", "person_room", "room_zone"])
            self.assertEqual(sorted(metadata["vocabulary_groups"]["verbs"]), ["assign", "move", "zone"])

    def test_v4_twohop_core_preset_removes_three_hop_family_and_prunes_unused_vocab(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir) / "dataset"
            run_command(
                "scripts/generate_microlanguage_world_dataset.py",
                "--preset",
                "v4_twohop_core",
                "--outdir",
                str(dataset_dir),
                "--train-size",
                "10",
                "--val-size",
                "4",
                "--test-size",
                "4",
                "--ood-size",
                "4",
                "--seed",
                "29",
            )
            metadata = json.loads((dataset_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["preset"], "v4_twohop_core")
            self.assertEqual(
                metadata["generation_rules"]["query_families"],
                ["item_owner", "person_room", "item_room"],
            )
            self.assertEqual(
                metadata["generation_rules"]["active_relations"],
                ["item_owner", "person_room"],
            )
            self.assertEqual(metadata["generation_rules"]["id_extra_update_range"], [0, 0])
            self.assertEqual(metadata["generation_rules"]["ood_extra_update_range"], [1, 1])
            self.assertEqual(sorted(metadata["latent_state"]["direct_relations"]), ["item_owner", "person_room"])
            self.assertEqual(sorted(metadata["vocabulary_groups"]["verbs"]), ["assign", "move"])
            self.assertEqual(
                sorted(metadata["vocabulary_groups"]["query_families"]),
                ["item_owner", "item_room", "person_room"],
            )
            self.assertNotIn("zones", metadata["vocabulary_groups"])
            self.assertNotIn("moods", metadata["vocabulary_groups"])
            self.assertNotIn("roles", metadata["vocabulary_groups"])
            self.assertNotIn("badges", metadata["vocabulary_groups"])
            self.assertNotIn("colors", metadata["vocabulary_groups"])
            self.assertNotIn("Z0", metadata["vocabulary"]["values"])
            self.assertNotIn("M0", metadata["vocabulary"]["values"])
            self.assertNotIn("Role0", metadata["vocabulary"]["values"])
            self.assertNotIn("B0", metadata["vocabulary"]["values"])
            self.assertNotIn("C0", metadata["vocabulary"]["values"])

    def test_v5_same_target_core_uses_only_room_targets_and_mixed_train_ranges(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir) / "dataset"
            run_command(
                "scripts/generate_microlanguage_world_dataset.py",
                "--preset",
                "v5_same_target_core",
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
                "--seed",
                "31",
            )
            metadata = json.loads((dataset_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["preset"], "v5_same_target_core")
            self.assertEqual(
                metadata["generation_rules"]["query_families"],
                ["person_room", "item_room"],
            )
            self.assertEqual(
                metadata["generation_rules"]["active_relations"],
                ["person_room", "item_owner"],
            )
            self.assertEqual(metadata["generation_rules"]["id_person_range"], [3, 4])
            self.assertEqual(metadata["generation_rules"]["id_item_range"], [4, 5])
            self.assertEqual(metadata["generation_rules"]["id_room_range"], [2, 3])
            self.assertEqual(metadata["generation_rules"]["id_extra_update_range"], [0, 1])
            self.assertEqual(metadata["generation_rules"]["ood_extra_update_range"], [1, 1])
            self.assertEqual(sorted(metadata["vocabulary_groups"]["verbs"]), ["assign", "move"])
            self.assertEqual(
                sorted(metadata["vocabulary_groups"]["query_families"]),
                ["item_room", "person_room"],
            )
            self.assertEqual(sorted(metadata["latent_state"]["direct_relations"]), ["item_owner", "person_room"])
            for query_name, query_spec in metadata["latent_state"]["query_families"].items():
                self.assertEqual(query_spec["target_group"], "rooms", query_name)
            self.assertEqual(metadata["prompt_length_summary"]["test_ood_longer_context"]["mean_rounded"] > metadata["prompt_length_summary"]["train"]["mean_rounded"], True)

    def test_v6_same_target_clean_has_no_overwrite_and_fixed_counts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir) / "dataset"
            run_command(
                "scripts/generate_microlanguage_world_dataset.py",
                "--preset",
                "v6_same_target_clean",
                "--outdir",
                str(dataset_dir),
                "--train-size",
                "10",
                "--val-size",
                "4",
                "--test-size",
                "4",
                "--ood-size",
                "4",
                "--seed",
                "37",
            )
            metadata = json.loads((dataset_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["preset"], "v6_same_target_clean")
            self.assertEqual(metadata["generation_rules"]["query_families"], ["person_room", "item_room"])
            self.assertEqual(metadata["generation_rules"]["id_extra_update_range"], [0, 0])
            self.assertEqual(metadata["generation_rules"]["ood_extra_update_range"], [0, 0])
            self.assertEqual(metadata["generation_rules"]["id_person_range"], [3, 3])
            self.assertEqual(metadata["generation_rules"]["id_item_range"], [4, 4])
            self.assertEqual(metadata["generation_rules"]["id_room_range"], [2, 2])
            self.assertEqual(metadata["generation_rules"]["ood_person_range"], [4, 4])
            self.assertEqual(metadata["generation_rules"]["ood_item_range"], [5, 5])
            self.assertEqual(metadata["generation_rules"]["ood_room_range"], [3, 3])
            for query_name, query_spec in metadata["latent_state"]["query_families"].items():
                self.assertEqual(query_spec["target_group"], "rooms", query_name)

    def test_v7_counterbalanced_rooms_blocks_majority_room_shortcut(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir) / "dataset"
            run_command(
                "scripts/generate_microlanguage_world_dataset.py",
                "--preset",
                "v7_counterbalanced_rooms",
                "--outdir",
                str(dataset_dir),
                "--train-size",
                "200",
                "--val-size",
                "200",
                "--test-size",
                "200",
                "--ood-size",
                "200",
                "--seed",
                "41",
            )
            metadata = json.loads((dataset_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["preset"], "v7_counterbalanced_rooms")
            self.assertEqual(metadata["generation_rules"]["balanced_relations"], ["person_room", "item_owner"])
            self.assertEqual(metadata["generation_rules"]["id_person_range"], [4, 4])
            self.assertEqual(metadata["generation_rules"]["id_item_range"], [4, 4])
            self.assertEqual(metadata["generation_rules"]["id_room_range"], [2, 2])
            self.assertEqual(metadata["generation_rules"]["ood_person_range"], [6, 6])
            self.assertEqual(metadata["generation_rules"]["ood_item_range"], [6, 6])
            self.assertEqual(metadata["generation_rules"]["ood_room_range"], [3, 3])

            def majority_room(prompt: str) -> str:
                room_counts = Counter(token for token in prompt.split() if token.startswith("R"))
                if not room_counts:
                    raise ValueError(f"Prompt contains no room tokens: {prompt}")
                return max(room_counts.items(), key=lambda item: (item[1], item[0]))[0]

            for split_name, max_accuracy in {
                "val": 0.60,
                "test": 0.60,
                "test_ood_longer_context": 0.45,
            }.items():
                rows = [
                    json.loads(line)
                    for line in (dataset_dir / f"{split_name}.jsonl").read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                heuristic_accuracy = sum(
                    1 for row in rows if majority_room(str(row["prompt"])) == row["target"]
                ) / len(rows)
                self.assertLessEqual(
                    heuristic_accuracy,
                    max_accuracy,
                    f"majority-room heuristic remained too strong on {split_name}: {heuristic_accuracy}",
                )

            sample_rows = [
                json.loads(line)
                for line in (dataset_dir / "test.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            for row in sample_rows[:10]:
                room_counts = Counter(token for token in str(row["prompt"]).split() if token.startswith("R"))
                self.assertEqual(sorted(room_counts.values()), [2, 2])

    def test_v8_scaffolded_room_chain_uses_split_specific_query_families(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir) / "dataset"
            run_command(
                "scripts/generate_microlanguage_world_dataset.py",
                "--preset",
                "v8_scaffolded_room_chain",
                "--outdir",
                str(dataset_dir),
                "--train-size",
                "12",
                "--val-size",
                "6",
                "--test-size",
                "6",
                "--ood-size",
                "6",
                "--seed",
                "43",
            )
            metadata = json.loads((dataset_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["preset"], "v8_scaffolded_room_chain")
            self.assertEqual(
                metadata["generation_rules"]["split_query_families"],
                {
                    "train": ["person_room", "item_owner", "item_room"],
                    "val": ["person_room", "item_room"],
                    "test": ["person_room", "item_room"],
                    "test_ood_longer_context": ["person_room", "item_room"],
                },
            )
            train_rows = [
                json.loads(line)
                for line in (dataset_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            val_rows = [
                json.loads(line)
                for line in (dataset_dir / "val.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertIn("item_owner", {str(row["query_family"]) for row in train_rows})
            self.assertNotIn("item_owner", {str(row["query_family"]) for row in val_rows})

    def test_v9_canonical_room_chain_uses_relation_names_in_events(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir) / "dataset"
            run_command(
                "scripts/generate_microlanguage_world_dataset.py",
                "--preset",
                "v9_canonical_room_chain",
                "--outdir",
                str(dataset_dir),
                "--train-size",
                "12",
                "--val-size",
                "6",
                "--test-size",
                "6",
                "--ood-size",
                "6",
                "--seed",
                "53",
            )
            metadata = json.loads((dataset_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["preset"], "v9_canonical_room_chain")
            self.assertEqual(metadata["generation_rules"]["event_relation_token_mode"], "relation_name")
            self.assertEqual(metadata["sequence_format"], "<bos> relation_token subject value ; relation_token subject value ; ... ; Q query_family subject ->")
            self.assertEqual(metadata["vocabulary_groups"]["verbs"], [])
            self.assertEqual(
                metadata["latent_state"]["direct_relations"]["person_room"]["event_tokens"],
                ["person_room"],
            )
            self.assertEqual(
                metadata["latent_state"]["direct_relations"]["item_owner"]["event_tokens"],
                ["item_owner"],
            )

            train_rows = [
                json.loads(line)
                for line in (dataset_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            parsed_prompt = parse_prompt(str(train_rows[0]["prompt"]))
            event_tokens = [str(event["surface_verb"]) for event in parsed_prompt["events"]]
            self.assertTrue(set(event_tokens) <= {"person_room", "item_owner"})

    def test_v10_person_room_direct_is_direct_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir) / "dataset"
            run_command(
                "scripts/generate_microlanguage_world_dataset.py",
                "--preset",
                "v10_person_room_direct",
                "--outdir",
                str(dataset_dir),
                "--train-size",
                "24",
                "--val-size",
                "8",
                "--test-size",
                "8",
                "--ood-size",
                "8",
                "--seed",
                "61",
            )
            metadata = json.loads((dataset_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["preset"], "v10_person_room_direct")
            self.assertEqual(metadata["generation_rules"]["query_families"], ["person_room"])
            self.assertEqual(metadata["generation_rules"]["active_relations"], ["person_room"])
            self.assertEqual(metadata["generation_rules"]["balanced_relations"], ["person_room"])
            self.assertEqual(metadata["generation_rules"]["event_relation_token_mode"], "relation_name")
            self.assertEqual(sorted(metadata["latent_state"]["direct_relations"]), ["person_room"])
            self.assertEqual(sorted(metadata["latent_state"]["query_families"]), ["person_room"])
            self.assertEqual(metadata["vocabulary_groups"]["verbs"], [])

            for split_name in ("train", "val", "test", "test_ood_longer_context"):
                rows = [
                    json.loads(line)
                    for line in (dataset_dir / f"{split_name}.jsonl").read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                self.assertTrue(rows)
                self.assertEqual({str(row["query_family"]) for row in rows}, {"person_room"})

    def test_v11_person_room_story_produces_long_parseable_story_prompts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir) / "dataset"
            run_command(
                "scripts/generate_microlanguage_world_dataset.py",
                "--preset",
                "v11_person_room_story",
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
                "--seed",
                "67",
            )
            metadata = json.loads((dataset_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["preset"], "v11_person_room_story")
            self.assertEqual(metadata["generation_rules"]["prompt_style"], "story")
            self.assertEqual(metadata["generation_rules"]["id_story_word_range"], [110, 160])
            self.assertEqual(metadata["generation_rules"]["ood_story_word_range"], [180, 260])
            self.assertIn("story_words", metadata["vocabulary_groups"])
            vocab_tokens = set(metadata["vocabulary"]["special"]) | set(metadata["vocabulary"]["keys"]) | set(metadata["vocabulary"]["values"])

            train_rows = [
                json.loads(line)
                for line in (dataset_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertTrue(train_rows)
            prompt_lengths = [len(str(row["prompt"]).split()) for row in train_rows]
            self.assertGreaterEqual(min(prompt_lengths), 110)
            sample_prompt = str(train_rows[0]["prompt"])
            self.assertIn("relation", sample_prompt.split())
            self.assertIn("subject", sample_prompt.split())
            self.assertIn("value", sample_prompt.split())
            self.assertTrue(set(sample_prompt.split()) <= vocab_tokens)
            parsed_target, trace = resolve_prompt_target(sample_prompt)
            self.assertEqual(parsed_target, train_rows[0]["target"])
            self.assertEqual(trace, train_rows[0]["trace"])
            parsed_prompt = parse_prompt(sample_prompt)
            self.assertEqual(len(parsed_prompt["events"]), 4)
            self.assertEqual(parsed_prompt["query_family"], "person_room")

    def test_dense_value_answer_vocab_marks_event_values_and_final_answer(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir) / "dataset"
            run_command(
                "scripts/generate_microlanguage_world_dataset.py",
                "--preset",
                "v10_person_room_direct",
                "--outdir",
                str(dataset_dir),
                "--train-size",
                "2",
                "--val-size",
                "1",
                "--test-size",
                "1",
                "--ood-size",
                "1",
                "--seed",
                "71",
            )
            bundle = load_dataset_bundle(dataset_dir)
            row = bundle.raw_splits["train"][0]
            input_ids, role_ids, target_ids, loss_mask = _encode_microlanguage_dense_sequence_batch(
                [row],
                bundle.token_to_id,
                torch.device("cpu"),
            )
            self.assertEqual(input_ids.shape, role_ids.shape)
            self.assertEqual(input_ids.shape, target_ids.shape)
            self.assertEqual(loss_mask.shape, target_ids.shape)
            self.assertTrue(bool(loss_mask[0, -1].item()))
            expected_supervised_positions = int(row["relation_update_counts"]["person_room"]) + 1
            self.assertEqual(int(loss_mask[0].sum().item()), expected_supervised_positions)

    def test_query_group_encoder_transformer_selects_query_subject_role(self) -> None:
        model = TinyQueryGroupEncoderTransformer(
            vocab_size=16,
            d_model=8,
            n_heads=2,
            d_ff=16,
            n_layers=1,
            max_seq_len=8,
            group_head_output_sizes={"rooms": 2},
            classifier_role_id=8,
            num_role_ids=11,
        )
        input_ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=torch.long)
        role_ids = torch.tensor([[0, 8, 1, 3], [0, 1, 8, 3]], dtype=torch.long)
        logits = model.forward_group_logits(input_ids, group_name="rooms", role_ids=role_ids)
        self.assertEqual(tuple(logits.shape), (2, 2))
        bad_role_ids = torch.tensor([[0, 1, 1, 3], [0, 1, 8, 8]], dtype=torch.long)
        with self.assertRaisesRegex(ValueError, "exactly one classifier role position"):
            model.forward_group_logits(input_ids, group_name="rooms", role_ids=bad_role_ids)

    def test_query_target_group_answer_mask_removes_invalid_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir) / "dataset"
            run_dir = Path(temp_dir) / "run"
            run_command(
                "scripts/generate_microlanguage_world_dataset.py",
                "--preset",
                "v5_same_target_core",
                "--outdir",
                str(dataset_dir),
                "--train-size",
                "6",
                "--val-size",
                "2",
                "--test-size",
                "2",
                "--ood-size",
                "2",
                "--seed",
                "41",
            )
            bundle = load_dataset_bundle(dataset_dir)
            row = bundle.raw_splits["val"][0]
            target_id = bundle.token_to_id[str(row["target"])]
            invalid_person_id = bundle.token_to_id[bundle.metadata["vocabulary_groups"]["persons"][0]]

            class FixedLogitModel(torch.nn.Module):
                def __init__(self, vocab_size: int, invalid_id: int, valid_id: int) -> None:
                    super().__init__()
                    self.vocab_size = vocab_size
                    self.invalid_id = invalid_id
                    self.valid_id = valid_id

                def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
                    batch_size, seq_len = input_ids.shape
                    logits = torch.full((batch_size, seq_len, self.vocab_size), -50.0, dtype=torch.float32)
                    logits[:, -1, self.invalid_id] = 10.0
                    logits[:, -1, self.valid_id] = 5.0
                    return logits

            model = FixedLogitModel(len(bundle.vocab), invalid_person_id, target_id)
            unmasked = evaluate_next_token_rows(
                model,
                [row],
                token_to_id=bundle.token_to_id,
                id_to_token=bundle.id_to_token,
                device=torch.device("cpu"),
                batch_size=1,
            )
            masked = evaluate_next_token_rows(
                model,
                [row],
                token_to_id=bundle.token_to_id,
                id_to_token=bundle.id_to_token,
                device=torch.device("cpu"),
                batch_size=1,
                dataset_metadata=bundle.metadata,
                answer_space_mode="query_target_group",
            )
            self.assertEqual(unmasked["accuracy"], 0.0)
            self.assertEqual(masked["accuracy"], 1.0)

    def test_active_query_target_group_answer_mask_removes_inactive_valid_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir) / "dataset"
            run_command(
                "scripts/generate_microlanguage_world_dataset.py",
                "--preset",
                "v7_counterbalanced_rooms",
                "--outdir",
                str(dataset_dir),
                "--train-size",
                "6",
                "--val-size",
                "2",
                "--test-size",
                "2",
                "--ood-size",
                "2",
                "--seed",
                "47",
            )
            bundle = load_dataset_bundle(dataset_dir)
            row = bundle.raw_splits["val"][0]
            active_rooms = [token for token in str(row["prompt"]).split() if token.startswith("R")]
            inactive_room = next(
                token for token in bundle.metadata["vocabulary_groups"]["rooms"] if token not in active_rooms
            )
            target_id = bundle.token_to_id[str(row["target"])]
            inactive_room_id = bundle.token_to_id[inactive_room]

            class FixedLogitModel(torch.nn.Module):
                def __init__(self, vocab_size: int, invalid_id: int, valid_id: int) -> None:
                    super().__init__()
                    self.vocab_size = vocab_size
                    self.invalid_id = invalid_id
                    self.valid_id = valid_id

                def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
                    batch_size, seq_len = input_ids.shape
                    logits = torch.full((batch_size, seq_len, self.vocab_size), -50.0, dtype=torch.float32)
                    logits[:, -1, self.invalid_id] = 10.0
                    logits[:, -1, self.valid_id] = 5.0
                    return logits

            model = FixedLogitModel(len(bundle.vocab), inactive_room_id, target_id)
            group_masked = evaluate_next_token_rows(
                model,
                [row],
                token_to_id=bundle.token_to_id,
                id_to_token=bundle.id_to_token,
                device=torch.device("cpu"),
                batch_size=1,
                dataset_metadata=bundle.metadata,
                answer_space_mode="query_target_group",
            )
            active_masked = evaluate_next_token_rows(
                model,
                [row],
                token_to_id=bundle.token_to_id,
                id_to_token=bundle.id_to_token,
                device=torch.device("cpu"),
                batch_size=1,
                dataset_metadata=bundle.metadata,
                answer_space_mode="active_query_target_group",
            )
            self.assertEqual(group_masked["accuracy"], 0.0)
            self.assertEqual(active_masked["accuracy"], 1.0)

    def test_ood_split_has_longer_prompts_and_generation_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_a = Path(temp_dir) / "dataset_a"
            dataset_b = Path(temp_dir) / "dataset_b"
            args = [
                "scripts/generate_microlanguage_world_dataset.py",
                "--train-size",
                "6",
                "--val-size",
                "2",
                "--test-size",
                "2",
                "--ood-size",
                "2",
                "--seed",
                "19",
            ]
            run_command(*args, "--outdir", str(dataset_a))
            run_command(*args, "--outdir", str(dataset_b))
            self.assertEqual(
                (dataset_a / "train.jsonl").read_text(encoding="utf-8"),
                (dataset_b / "train.jsonl").read_text(encoding="utf-8"),
            )
            metadata = json.loads((dataset_a / "metadata.json").read_text(encoding="utf-8"))
            self.assertGreater(
                metadata["prompt_length_summary"]["test_ood_longer_context"]["mean_rounded"],
                metadata["prompt_length_summary"]["train"]["mean_rounded"],
            )

    def test_manifest_validation_accepts_microlanguage_benchmark(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir) / "dataset"
            run_dir = Path(temp_dir) / "run_seed0"
            run_command(
                "scripts/generate_microlanguage_world_dataset.py",
                "--outdir",
                str(dataset_dir),
                "--train-size",
                "6",
                "--val-size",
                "2",
                "--test-size",
                "2",
                "--ood-size",
                "2",
                "--seed",
                "5",
            )
            manifest = build_run_manifest(self._make_manifest_dict(dataset_dir, run_dir, seed=0))
            self.assertEqual(manifest.benchmark.name, "microlanguage_world_next_token")
            self.assertEqual(manifest.dataset.train_split_by_pairs, {"default": "train"})

    def test_training_smoke_with_early_stopping_writes_early_stop_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            dataset_dir = temp_root / "dataset"
            run_dir = temp_root / "run_seed0"
            run_command(
                "scripts/generate_microlanguage_world_dataset.py",
                "--preset",
                "v2_small",
                "--outdir",
                str(dataset_dir),
                "--train-size",
                "48",
                "--val-size",
                "16",
                "--test-size",
                "16",
                "--ood-size",
                "16",
                "--seed",
                "17",
            )
            manifest_payload = self._make_manifest_dict(
                dataset_dir,
                run_dir,
                seed=0,
                epochs=4,
                early_stopping={
                    "enabled": True,
                    "patience": 1,
                    "min_delta": 1.0,
                    "warmup_epochs": 1,
                    "restore_best_state": True,
                },
            )
            manifest_path = temp_root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")
            run_command("-m", "scripts.train_microlanguage_world_run", "--manifest", str(manifest_path))
            early_stop_checkpoints = sorted((run_dir / "checkpoints").glob("early_stop_epoch_*.pt"))
            self.assertEqual(len(early_stop_checkpoints), 1)

    def test_v9_group_head_training_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            dataset_dir = temp_root / "dataset"
            run_dir = temp_root / "run_seed0"
            run_command(
                "scripts/generate_microlanguage_world_dataset.py",
                "--preset",
                "v9_canonical_room_chain",
                "--outdir",
                str(dataset_dir),
                "--train-size",
                "48",
                "--val-size",
                "16",
                "--test-size",
                "16",
                "--ood-size",
                "16",
                "--seed",
                "59",
            )
            manifest_payload = self._make_manifest_dict(
                dataset_dir,
                run_dir,
                seed=0,
                epochs=2,
                supervision_mode="query_target_group_head",
            )
            manifest_payload["dataset"]["answer_space_mode"] = "active_query_target_group"
            manifest_path = temp_root / "manifest_group_head.json"
            manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")
            run_command("-m", "scripts.train_microlanguage_world_run", "--manifest", str(manifest_path))
            checkpoints = sorted((run_dir / "checkpoints").glob("*.pt"))
            self.assertTrue(checkpoints)

    def test_sanity_ladder_builder_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            dataset_root = temp_root / "datasets"
            manifest_dir = temp_root / "manifests"
            run_root = temp_root / "runs"
            run_command(
                "-m",
                "research.phase3.scripts.build_microlanguage_sanity_ladder",
                "--dataset-root",
                str(dataset_root),
                "--manifest-outdir",
                str(manifest_dir),
                "--run-root",
                str(run_root),
                "--device",
                "cpu",
            )
            self.assertTrue((manifest_dir / "dataset_build_commands.sh").exists())
            self.assertTrue((manifest_dir / "run_full_pipeline.sh").exists())
            manifest_paths = sorted(manifest_dir.glob("*/*.json"))
            self.assertEqual(len(manifest_paths), 3)

    def test_baseline_builder_and_training_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            dataset_dir = temp_root / "dataset"
            manifest_dir = temp_root / "manifests"
            run_root = temp_root / "runs"
            run_command(
                "scripts/generate_microlanguage_world_dataset.py",
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
                "--num-persons",
                "6",
                "--num-items",
                "8",
                "--num-rooms",
                "5",
                "--num-zones",
                "4",
                "--num-moods",
                "5",
                "--num-roles",
                "5",
                "--num-badges",
                "5",
                "--num-colors",
                "5",
                "--id-person-range",
                "3,3",
                "--id-item-range",
                "4,4",
                "--id-room-range",
                "2,2",
                "--id-extra-update-range",
                "2,2",
                "--ood-person-range",
                "4,4",
                "--ood-item-range",
                "5,5",
                "--ood-room-range",
                "3,3",
                "--ood-extra-update-range",
                "4,4",
                "--query-families",
                "person_room,item_zone",
                "--seed",
                "7",
            )
            run_command(
                "-m",
                "research.phase3.scripts.build_microlanguage_world_baseline",
                "--dataset-dir",
                str(dataset_dir),
                "--manifest-outdir",
                str(manifest_dir),
                "--run-root",
                str(run_root),
                "--seeds",
                "0",
                "--device",
                "cpu",
                "--epochs",
                "2",
                "--batch-size",
                "4",
                "--eval-batch-size",
                "8",
                "--d-model",
                "16",
                "--n-heads",
                "2",
                "--d-ff",
                "32",
                "--n-layers",
                "2",
                "--max-seq-len",
                "256",
                "--dense-through-epoch",
                "1",
                "--log-spaced-epoch-count",
                "0",
            )
            run_script = (manifest_dir / "run_manifests_sequential.sh").read_text(encoding="utf-8")
            self.assertIn("python -m scripts.train_microlanguage_world_run", run_script)

            manifest_path = next(manifest_dir.glob("*.json"))
            run_command("-m", "scripts.train_microlanguage_world_run", "--manifest", str(manifest_path))

            run_dir = next(run_root.glob("*"))
            checkpoint_files = sorted((run_dir / "checkpoints").glob("*.pt"))
            self.assertGreaterEqual(len(checkpoint_files), 3)
            history_rows = [
                json.loads(line)
                for line in (run_dir / "train_history.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertGreater(len(history_rows), 0)
            self.assertIn("total_grad_norm", history_rows[0])
            self.assertTrue((run_dir / "manifest.json").exists())

            run_command("-m", "scripts.run_microlanguage_checkpoint_eval", "--run-dir", str(run_dir))
            first_battery_dir = run_dir / "battery" / checkpoint_files[0].stem
            self.assertTrue((first_battery_dir / "behavior.json").exists())
            self.assertTrue((first_battery_dir / "slice_metrics.csv").exists())
            self.assertTrue((first_battery_dir / "scored_rows.csv").exists())

            run_command("-m", "scripts.summarize_microlanguage_world_training", "--target-dir", str(run_root))
            summary_dir = run_dir / "summaries"
            self.assertTrue((summary_dir / "checkpoint_index.csv").exists())
            self.assertTrue((summary_dir / "slice_dynamics.csv").exists())
            self.assertTrue((summary_dir / "seed_stability.csv").exists())
            self.assertTrue((summary_dir / "slice_seed_stability.csv").exists())
            self.assertTrue((summary_dir / "stability_diagnostics.csv").exists())
            self.assertTrue((summary_dir / "run_summary.json").exists())

    def test_calibration_builder_writes_train_eval_and_summary_scripts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            dataset_dir = temp_root / "dataset"
            manifest_dir = temp_root / "manifests"
            run_root = temp_root / "runs"
            run_command(
                "scripts/generate_microlanguage_world_dataset.py",
                "--outdir",
                str(dataset_dir),
                "--train-size",
                "6",
                "--val-size",
                "2",
                "--test-size",
                "2",
                "--ood-size",
                "2",
                "--num-persons",
                "6",
                "--num-items",
                "8",
                "--num-rooms",
                "5",
                "--num-zones",
                "4",
                "--num-moods",
                "5",
                "--num-roles",
                "5",
                "--num-badges",
                "5",
                "--num-colors",
                "5",
                "--id-person-range",
                "3,3",
                "--id-item-range",
                "4,4",
                "--id-room-range",
                "2,2",
                "--id-extra-update-range",
                "2,2",
                "--ood-person-range",
                "4,4",
                "--ood-item-range",
                "5,5",
                "--ood-room-range",
                "3,3",
                "--ood-extra-update-range",
                "4,4",
                "--query-families",
                "person_room,item_zone",
                "--seed",
                "9",
            )
            run_command(
                "-m",
                "research.phase3.scripts.build_microlanguage_calibration_wave",
                "--dataset-dir",
                str(dataset_dir),
                "--manifest-outdir",
                str(manifest_dir),
                "--run-root",
                str(run_root),
                "--seeds",
                "0",
                "--learning-rates",
                "3e-4",
                "--n-layers",
                "2",
                "--epochs-list",
                "2",
                "--max-seq-lens",
                "256",
                "--device",
                "cpu",
                "--batch-size",
                "4",
                "--eval-batch-size",
                "8",
                "--d-model",
                "16",
                "--n-heads",
                "2",
                "--d-ff",
                "32",
                "--dense-through-epoch",
                "1",
                "--log-spaced-epoch-count",
                "0",
            )
            self.assertTrue((manifest_dir / "manifest_matrix.csv").exists())
            train_script = (manifest_dir / "run_manifests_sequential.sh").read_text(encoding="utf-8")
            eval_script = (manifest_dir / "run_evals_sequential.sh").read_text(encoding="utf-8")
            summary_script = (manifest_dir / "summarize_condition_blocks.sh").read_text(encoding="utf-8")
            pipeline_script = (manifest_dir / "run_full_pipeline.sh").read_text(encoding="utf-8")
            self.assertIn("python -m scripts.train_microlanguage_world_run", train_script)
            self.assertIn("python -m scripts.run_microlanguage_checkpoint_eval", eval_script)
            self.assertIn("python -m scripts.summarize_microlanguage_world_training", summary_script)
            self.assertIn("python -m scripts.train_microlanguage_world_run", pipeline_script)
            self.assertIn("python -m scripts.run_microlanguage_checkpoint_eval", pipeline_script)
            self.assertIn("python -m scripts.summarize_microlanguage_world_training", pipeline_script)


if __name__ == "__main__":
    unittest.main()
