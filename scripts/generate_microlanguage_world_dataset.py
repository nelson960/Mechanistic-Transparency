#!/usr/bin/env python3
"""Generate a synthetic next-token microlanguage world-model dataset."""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


RESERVED_PROMPT_TOKENS = {"<bos>", ";", "Q", "->"}


@dataclass(frozen=True)
class RelationSpec:
    name: str
    subject_group: str
    value_group: str
    surface_verbs: tuple[str, ...]


@dataclass(frozen=True)
class QuerySpec:
    name: str
    subject_group: str
    target_group: str
    trace: tuple[str, ...]


@dataclass(frozen=True)
class DatasetPreset:
    dataset_name: str
    train_size: int
    val_size: int
    test_size: int
    ood_size: int
    query_families: tuple[str, ...]
    id_person_range: str
    id_item_range: str
    id_room_range: str
    id_extra_update_range: str
    ood_person_range: str
    ood_item_range: str
    ood_room_range: str
    ood_extra_update_range: str
    max_surface_verbs_per_relation: int
    event_relation_token_mode: str = "surface_verb"
    balanced_relations: tuple[str, ...] = ()
    train_query_families: tuple[str, ...] | None = None
    val_query_families: tuple[str, ...] | None = None
    test_query_families: tuple[str, ...] | None = None
    ood_query_families: tuple[str, ...] | None = None
    prompt_style: str = "symbolic"
    id_story_word_range: str | None = None
    ood_story_word_range: str | None = None


RELATION_SPECS: dict[str, RelationSpec] = {
    "person_room": RelationSpec(
        name="person_room",
        subject_group="persons",
        value_group="rooms",
        surface_verbs=("move", "goto", "enter"),
    ),
    "person_mood": RelationSpec(
        name="person_mood",
        subject_group="persons",
        value_group="moods",
        surface_verbs=("mood", "feel", "status"),
    ),
    "person_role": RelationSpec(
        name="person_role",
        subject_group="persons",
        value_group="roles",
        surface_verbs=("role", "job", "duty"),
    ),
    "person_badge": RelationSpec(
        name="person_badge",
        subject_group="persons",
        value_group="badges",
        surface_verbs=("badge", "wear", "tag"),
    ),
    "item_owner": RelationSpec(
        name="item_owner",
        subject_group="items",
        value_group="persons",
        surface_verbs=("assign", "give", "hold"),
    ),
    "item_color": RelationSpec(
        name="item_color",
        subject_group="items",
        value_group="colors",
        surface_verbs=("paint", "color", "coat"),
    ),
    "room_zone": RelationSpec(
        name="room_zone",
        subject_group="rooms",
        value_group="zones",
        surface_verbs=("zone", "area", "sector"),
    ),
}

QUERY_SPECS: dict[str, QuerySpec] = {
    "person_room": QuerySpec(
        name="person_room",
        subject_group="persons",
        target_group="rooms",
        trace=("person_room",),
    ),
    "person_mood": QuerySpec(
        name="person_mood",
        subject_group="persons",
        target_group="moods",
        trace=("person_mood",),
    ),
    "person_role": QuerySpec(
        name="person_role",
        subject_group="persons",
        target_group="roles",
        trace=("person_role",),
    ),
    "person_badge": QuerySpec(
        name="person_badge",
        subject_group="persons",
        target_group="badges",
        trace=("person_badge",),
    ),
    "person_zone": QuerySpec(
        name="person_zone",
        subject_group="persons",
        target_group="zones",
        trace=("person_room", "room_zone"),
    ),
    "room_zone": QuerySpec(
        name="room_zone",
        subject_group="rooms",
        target_group="zones",
        trace=("room_zone",),
    ),
    "item_owner": QuerySpec(
        name="item_owner",
        subject_group="items",
        target_group="persons",
        trace=("item_owner",),
    ),
    "item_color": QuerySpec(
        name="item_color",
        subject_group="items",
        target_group="colors",
        trace=("item_color",),
    ),
    "item_room": QuerySpec(
        name="item_room",
        subject_group="items",
        target_group="rooms",
        trace=("item_owner", "person_room"),
    ),
    "item_zone": QuerySpec(
        name="item_zone",
        subject_group="items",
        target_group="zones",
        trace=("item_owner", "person_room", "room_zone"),
    ),
    "item_owner_mood": QuerySpec(
        name="item_owner_mood",
        subject_group="items",
        target_group="moods",
        trace=("item_owner", "person_mood"),
    ),
    "item_owner_role": QuerySpec(
        name="item_owner_role",
        subject_group="items",
        target_group="roles",
        trace=("item_owner", "person_role"),
    ),
    "item_owner_badge": QuerySpec(
        name="item_owner_badge",
        subject_group="items",
        target_group="badges",
        trace=("item_owner", "person_badge"),
    ),
}

SURFACE_VERB_TO_RELATION: dict[str, str] = {
    verb: relation_name
    for relation_name, spec in RELATION_SPECS.items()
    for verb in spec.surface_verbs
}


STORY_MARKER_TOKENS = ("relation", "subject", "value")
STORY_RELATION_PLACEHOLDER = "{relation_token}"
STORY_SUBJECT_PLACEHOLDER = "{subject_token}"
STORY_VALUE_PLACEHOLDER = "{value_token}"
STORY_EVENT_TEMPLATES: tuple[tuple[str, ...], ...] = (
    (
        "during",
        "the",
        "briefing",
        "the",
        "record",
        "noted",
        "relation",
        STORY_RELATION_PLACEHOLDER,
        "subject",
        STORY_SUBJECT_PLACEHOLDER,
        "value",
        STORY_VALUE_PLACEHOLDER,
        "while",
        "the",
        "staff",
        "kept",
        "careful",
        "notes",
    ),
    (
        "later",
        "the",
        "official",
        "report",
        "repeated",
        "relation",
        STORY_RELATION_PLACEHOLDER,
        "subject",
        STORY_SUBJECT_PLACEHOLDER,
        "value",
        STORY_VALUE_PLACEHOLDER,
        "inside",
        "the",
        "daily",
        "summary",
        "for",
        "review",
    ),
    (
        "throughout",
        "the",
        "shift",
        "the",
        "journal",
        "said",
        "relation",
        STORY_RELATION_PLACEHOLDER,
        "subject",
        STORY_SUBJECT_PLACEHOLDER,
        "value",
        STORY_VALUE_PLACEHOLDER,
        "and",
        "the",
        "watchers",
        "kept",
        "written",
        "details",
    ),
)
STORY_TEMPLATE_WORDS = tuple(
    sorted(
        {
            token
            for template in STORY_EVENT_TEMPLATES
            for token in template
            if token not in {STORY_RELATION_PLACEHOLDER, STORY_SUBJECT_PLACEHOLDER, STORY_VALUE_PLACEHOLDER}
        }
    )
)
STORY_FILLER_VOCAB = (
    "after",
    "again",
    "along",
    "around",
    "before",
    "briefing",
    "careful",
    "clerk",
    "corridor",
    "daily",
    "details",
    "during",
    "earlier",
    "entry",
    "evening",
    "everyone",
    "final",
    "hallway",
    "incident",
    "inside",
    "journal",
    "later",
    "logbook",
    "long",
    "morning",
    "nearby",
    "noted",
    "notes",
    "observers",
    "office",
    "official",
    "paper",
    "quiet",
    "record",
    "report",
    "review",
    "shift",
    "shortly",
    "staff",
    "summary",
    "throughout",
    "timeline",
    "watchers",
    "witness",
    "written",
)


DATASET_PRESETS: dict[str, DatasetPreset] = {
    "default": DatasetPreset(
        dataset_name="Microlanguage-World-v1",
        train_size=12000,
        val_size=1500,
        test_size=1500,
        ood_size=1500,
        query_families=tuple(QUERY_SPECS),
        id_person_range="4,5",
        id_item_range="6,7",
        id_room_range="3,4",
        id_extra_update_range="6,9",
        ood_person_range="6,7",
        ood_item_range="8,9",
        ood_room_range="5,6",
        ood_extra_update_range="12,16",
        max_surface_verbs_per_relation=3,
    ),
    "v2_small": DatasetPreset(
        dataset_name="Microlanguage-World-v2-small",
        train_size=30000,
        val_size=3000,
        test_size=3000,
        ood_size=3000,
        query_families=(
            "item_owner",
            "person_room",
            "room_zone",
            "item_room",
            "item_zone",
        ),
        id_person_range="3,4",
        id_item_range="4,5",
        id_room_range="3,3",
        id_extra_update_range="2,4",
        ood_person_range="5,6",
        ood_item_range="6,7",
        ood_room_range="4,5",
        ood_extra_update_range="5,7",
        max_surface_verbs_per_relation=1,
    ),
    "v3_core": DatasetPreset(
        dataset_name="Microlanguage-World-v3-core",
        train_size=20000,
        val_size=2000,
        test_size=2000,
        ood_size=2000,
        query_families=(
            "item_owner",
            "person_room",
            "room_zone",
            "item_room",
            "item_zone",
        ),
        id_person_range="3,3",
        id_item_range="4,4",
        id_room_range="2,2",
        id_extra_update_range="1,1",
        ood_person_range="4,4",
        ood_item_range="5,5",
        ood_room_range="3,3",
        ood_extra_update_range="2,2",
        max_surface_verbs_per_relation=1,
    ),
    "v4_twohop_core": DatasetPreset(
        dataset_name="Microlanguage-World-v4-twohop-core",
        train_size=20000,
        val_size=2000,
        test_size=2000,
        ood_size=2000,
        query_families=(
            "item_owner",
            "person_room",
            "item_room",
        ),
        id_person_range="3,3",
        id_item_range="4,4",
        id_room_range="2,2",
        id_extra_update_range="0,0",
        ood_person_range="4,4",
        ood_item_range="5,5",
        ood_room_range="3,3",
        ood_extra_update_range="1,1",
        max_surface_verbs_per_relation=1,
    ),
    "v5_same_target_core": DatasetPreset(
        dataset_name="Microlanguage-World-v5-same-target-core",
        train_size=20000,
        val_size=2000,
        test_size=2000,
        ood_size=2000,
        query_families=(
            "person_room",
            "item_room",
        ),
        id_person_range="3,4",
        id_item_range="4,5",
        id_room_range="2,3",
        id_extra_update_range="0,1",
        ood_person_range="4,5",
        ood_item_range="5,6",
        ood_room_range="3,4",
        ood_extra_update_range="1,1",
        max_surface_verbs_per_relation=1,
    ),
    "v6_same_target_clean": DatasetPreset(
        dataset_name="Microlanguage-World-v6-same-target-clean",
        train_size=20000,
        val_size=2000,
        test_size=2000,
        ood_size=2000,
        query_families=(
            "person_room",
            "item_room",
        ),
        id_person_range="3,3",
        id_item_range="4,4",
        id_room_range="2,2",
        id_extra_update_range="0,0",
        ood_person_range="4,4",
        ood_item_range="5,5",
        ood_room_range="3,3",
        ood_extra_update_range="0,0",
        max_surface_verbs_per_relation=1,
    ),
    "v7_counterbalanced_rooms": DatasetPreset(
        dataset_name="Microlanguage-World-v7-counterbalanced-rooms",
        train_size=20000,
        val_size=2000,
        test_size=2000,
        ood_size=2000,
        query_families=(
            "person_room",
            "item_room",
        ),
        id_person_range="4,4",
        id_item_range="4,4",
        id_room_range="2,2",
        id_extra_update_range="0,0",
        ood_person_range="6,6",
        ood_item_range="6,6",
        ood_room_range="3,3",
        ood_extra_update_range="0,0",
        max_surface_verbs_per_relation=1,
        balanced_relations=("person_room", "item_owner"),
    ),
    "v8_scaffolded_room_chain": DatasetPreset(
        dataset_name="Microlanguage-World-v8-scaffolded-room-chain",
        train_size=20000,
        val_size=2000,
        test_size=2000,
        ood_size=2000,
        query_families=(
            "person_room",
            "item_owner",
            "item_room",
        ),
        id_person_range="4,4",
        id_item_range="4,4",
        id_room_range="2,2",
        id_extra_update_range="0,0",
        ood_person_range="4,4",
        ood_item_range="4,4",
        ood_room_range="2,2",
        ood_extra_update_range="0,0",
        max_surface_verbs_per_relation=1,
        balanced_relations=("person_room", "item_owner"),
        train_query_families=("person_room", "item_owner", "item_room"),
        val_query_families=("person_room", "item_room"),
        test_query_families=("person_room", "item_room"),
        ood_query_families=("person_room", "item_room"),
    ),
    "v9_canonical_room_chain": DatasetPreset(
        dataset_name="Microlanguage-World-v9-canonical-room-chain",
        train_size=20000,
        val_size=2000,
        test_size=2000,
        ood_size=2000,
        query_families=(
            "person_room",
            "item_owner",
            "item_room",
        ),
        id_person_range="4,4",
        id_item_range="4,4",
        id_room_range="2,2",
        id_extra_update_range="0,0",
        ood_person_range="4,4",
        ood_item_range="4,4",
        ood_room_range="2,2",
        ood_extra_update_range="0,0",
        max_surface_verbs_per_relation=1,
        event_relation_token_mode="relation_name",
        balanced_relations=("person_room", "item_owner"),
        train_query_families=("person_room", "item_owner", "item_room"),
        val_query_families=("person_room", "item_room"),
        test_query_families=("person_room", "item_room"),
        ood_query_families=("person_room", "item_room"),
    ),
    "v10_person_room_direct": DatasetPreset(
        dataset_name="Microlanguage-World-v10-person-room-direct",
        train_size=20000,
        val_size=2000,
        test_size=2000,
        ood_size=2000,
        query_families=("person_room",),
        id_person_range="4,4",
        id_item_range="1,1",
        id_room_range="2,2",
        id_extra_update_range="0,0",
        ood_person_range="4,4",
        ood_item_range="1,1",
        ood_room_range="2,2",
        ood_extra_update_range="0,0",
        max_surface_verbs_per_relation=1,
        event_relation_token_mode="relation_name",
        balanced_relations=("person_room",),
        train_query_families=("person_room",),
        val_query_families=("person_room",),
        test_query_families=("person_room",),
        ood_query_families=("person_room",),
    ),
    "v11_person_room_story": DatasetPreset(
        dataset_name="Microlanguage-World-v11-person-room-story",
        train_size=4096,
        val_size=512,
        test_size=512,
        ood_size=512,
        query_families=("person_room",),
        id_person_range="4,4",
        id_item_range="1,1",
        id_room_range="2,2",
        id_extra_update_range="0,0",
        ood_person_range="4,4",
        ood_item_range="1,1",
        ood_room_range="2,2",
        ood_extra_update_range="0,0",
        max_surface_verbs_per_relation=1,
        event_relation_token_mode="relation_name",
        balanced_relations=("person_room",),
        train_query_families=("person_room",),
        val_query_families=("person_room",),
        test_query_families=("person_room",),
        ood_query_families=("person_room",),
        prompt_style="story",
        id_story_word_range="110,160",
        ood_story_word_range="180,260",
    ),
}


def _build_token_group(prefix: str, count: int) -> list[str]:
    if count <= 0:
        raise ValueError(f"Expected positive count for prefix {prefix!r}, got {count}")
    return [f"{prefix}{index}" for index in range(count)]


def _validate_disjoint_tokens(token_groups: dict[str, list[str]]) -> None:
    seen: dict[str, str] = {}
    for group_name, tokens in token_groups.items():
        for token in tokens:
            if not token or token.strip() != token:
                raise ValueError(f"Token {token!r} in group {group_name!r} must be a non-empty stripped string")
            if any(character.isspace() for character in token):
                raise ValueError(f"Token {token!r} in group {group_name!r} contains whitespace")
            if token in RESERVED_PROMPT_TOKENS:
                raise ValueError(f"Token {token!r} in group {group_name!r} collides with reserved prompt syntax")
            other_group = seen.get(token)
            if other_group is not None:
                raise ValueError(f"Token {token!r} appears in both {other_group!r} and {group_name!r}")
            seen[token] = group_name


def active_relation_names_for_query_families(query_families: Sequence[str]) -> list[str]:
    active_relations: list[str] = []
    seen: set[str] = set()
    for query_family in query_families:
        for relation_name in QUERY_SPECS[query_family].trace:
            if relation_name not in seen:
                active_relations.append(relation_name)
                seen.add(relation_name)
    if not active_relations:
        raise ValueError("Expected at least one active relation")
    return active_relations


def active_optional_group_names_for_task(
    *,
    query_families: Sequence[str],
    active_relation_names: Sequence[str],
) -> list[str]:
    optional_groups = {"zones", "moods", "roles", "badges", "colors"}
    active_groups: list[str] = []
    seen: set[str] = set()
    for relation_name in active_relation_names:
        relation = RELATION_SPECS[relation_name]
        for group_name in (relation.subject_group, relation.value_group):
            if group_name in optional_groups and group_name not in seen:
                active_groups.append(group_name)
                seen.add(group_name)
    for query_family in query_families:
        query_spec = QUERY_SPECS[query_family]
        for group_name in (query_spec.subject_group, query_spec.target_group):
            if group_name in optional_groups and group_name not in seen:
                active_groups.append(group_name)
                seen.add(group_name)
    return active_groups


def build_relation_surface_verbs(
    active_relation_names: Sequence[str],
    *,
    max_surface_verbs_per_relation: int,
    event_relation_token_mode: str,
) -> dict[str, tuple[str, ...]]:
    if event_relation_token_mode not in {"surface_verb", "relation_name"}:
        raise ValueError(
            "event_relation_token_mode must be 'surface_verb' or 'relation_name', "
            f"got {event_relation_token_mode!r}"
        )
    if max_surface_verbs_per_relation <= 0:
        raise ValueError(
            f"max_surface_verbs_per_relation must be positive, got {max_surface_verbs_per_relation}"
        )
    selected: dict[str, tuple[str, ...]] = {}
    for relation_name in active_relation_names:
        if event_relation_token_mode == "relation_name":
            selected[relation_name] = (relation_name,)
            continue
        relation = RELATION_SPECS[relation_name]
        surface_verbs = relation.surface_verbs[:max_surface_verbs_per_relation]
        if not surface_verbs:
            raise ValueError(f"Relation {relation_name!r} has no available surface verbs")
        selected[relation_name] = tuple(surface_verbs)
    return selected


def parse_relation_name_list(text: str) -> list[str]:
    names = [item.strip() for item in text.split(",") if item.strip()]
    unknown = [name for name in names if name not in RELATION_SPECS]
    if unknown:
        raise ValueError(f"Unknown relation name(s): {unknown}")
    return names


def build_token_groups(
    args: argparse.Namespace,
    *,
    query_families: Sequence[str],
    active_relation_names: Sequence[str],
    relation_surface_verbs: dict[str, tuple[str, ...]],
) -> dict[str, list[str]]:
    query_family_token_set = set(query_families)
    event_relation_tokens = sorted(
        {
            token
            for tokens in relation_surface_verbs.values()
            for token in tokens
            if token not in query_family_token_set
        }
    )
    token_groups = {
        "persons": _build_token_group("P", args.num_persons),
        "items": _build_token_group("I", args.num_items),
        "rooms": _build_token_group("R", args.num_rooms),
        "verbs": event_relation_tokens,
        "query_families": list(query_families),
    }
    if args.prompt_style == "story":
        token_groups["story_words"] = list(
            dict.fromkeys(list(STORY_TEMPLATE_WORDS) + list(STORY_FILLER_VOCAB))
        )
    for group_name in active_optional_group_names_for_task(
        query_families=query_families,
        active_relation_names=active_relation_names,
    ):
        if group_name == "zones":
            token_groups[group_name] = _build_token_group("Z", args.num_zones)
        elif group_name == "moods":
            token_groups[group_name] = _build_token_group("M", args.num_moods)
        elif group_name == "roles":
            token_groups[group_name] = _build_token_group("Role", args.num_roles)
        elif group_name == "badges":
            token_groups[group_name] = _build_token_group("B", args.num_badges)
        elif group_name == "colors":
            token_groups[group_name] = _build_token_group("C", args.num_colors)
        else:
            raise ValueError(f"Unsupported optional token group {group_name!r}")
    _validate_disjoint_tokens(token_groups)
    return token_groups


def build_vocab(token_groups: dict[str, list[str]]) -> dict[str, list[str]]:
    key_tokens = (
        token_groups.get("verbs", [])
        + token_groups.get("story_words", [])
        + token_groups["query_families"]
        + token_groups.get("items", [])
    )
    value_tokens = (
        token_groups.get("persons", [])
        + token_groups.get("rooms", [])
        + token_groups.get("zones", [])
        + token_groups.get("moods", [])
        + token_groups.get("roles", [])
        + token_groups.get("badges", [])
        + token_groups.get("colors", [])
    )
    overlap = sorted(set(key_tokens) & set(value_tokens))
    if overlap:
        raise ValueError(f"Microlanguage keys/values must be disjoint, found overlap: {overlap}")
    return {
        "special": ["<bos>", ";", "Q", "->"],
        "keys": key_tokens,
        "values": value_tokens,
    }


def parse_range(text: str, *, minimum: int = 1) -> tuple[int, int]:
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 2 or not all(parts):
        raise ValueError(f"Expected range in 'low,high' form, got {text!r}")
    low = int(parts[0])
    high = int(parts[1])
    if low < minimum or high < minimum:
        qualifier = "non-negative" if minimum == 0 else "positive"
        raise ValueError(f"Range bounds must be {qualifier}, got {text!r}")
    if low > high:
        raise ValueError(f"Range lower bound exceeds upper bound in {text!r}")
    return low, high


def parse_query_family_list(text: str) -> list[str]:
    families = [item.strip() for item in text.split(",") if item.strip()]
    if not families:
        raise ValueError("Expected at least one query family")
    unknown = [family for family in families if family not in QUERY_SPECS]
    if unknown:
        raise ValueError(f"Unknown query families: {unknown}")
    deduped: list[str] = []
    seen: set[str] = set()
    for family in families:
        if family not in seen:
            deduped.append(family)
            seen.add(family)
    return deduped


def parse_optional_query_family_list(text: str | None) -> list[str] | None:
    if text is None:
        return None
    if not text.strip():
        return []
    return parse_query_family_list(text)


def sample_count(rng: random.Random, bounds: tuple[int, int]) -> int:
    low, high = bounds
    return rng.randint(low, high)


def choose_distinct_tokens(rng: random.Random, pool: Sequence[str], count: int, *, label: str) -> list[str]:
    if count > len(pool):
        raise ValueError(f"Requested {count} {label}, but only {len(pool)} tokens exist")
    return rng.sample(list(pool), count)


def choose_value(
    rng: random.Random,
    pool: Sequence[str],
    *,
    disallow: str | None = None,
) -> str:
    candidates = [token for token in pool if token != disallow]
    if not candidates:
        raise ValueError(f"No available value choices remain after excluding {disallow!r}")
    return rng.choice(candidates)


def _resolve_group_pool(
    *,
    token_groups: dict[str, list[str]],
    active_groups: dict[str, list[str]],
    group_name: str,
) -> list[str]:
    if group_name in active_groups:
        return active_groups[group_name]
    if group_name not in token_groups:
        raise ValueError(f"Unknown token group {group_name!r}")
    return token_groups[group_name]


def build_event(
    *,
    rng: random.Random,
    relation_name: str,
    subject: str,
    value: str,
    relation_surface_verbs: dict[str, tuple[str, ...]],
) -> dict[str, str]:
    surface_verbs = relation_surface_verbs.get(relation_name)
    if surface_verbs is None:
        raise ValueError(f"Missing surface verbs for active relation {relation_name!r}")
    return {
        "relation": relation_name,
        "surface_verb": rng.choice(surface_verbs),
        "subject": subject,
        "value": value,
    }


def build_world_events(
    *,
    rng: random.Random,
    token_groups: dict[str, list[str]],
    active_groups: dict[str, list[str]],
    extra_update_count: int,
    active_relation_names: Sequence[str],
    relation_surface_verbs: dict[str, tuple[str, ...]],
    balanced_relations: Sequence[str],
) -> list[dict[str, str]]:
    events: list[dict[str, str]] = []
    latest_sampled_values: dict[tuple[str, str], str] = {}

    for relation_name in active_relation_names:
        relation = RELATION_SPECS[relation_name]
        subjects = active_groups[relation.subject_group]
        values = _resolve_group_pool(
            token_groups=token_groups,
            active_groups=active_groups,
            group_name=relation.value_group,
        )
        shuffled_subjects = list(subjects)
        rng.shuffle(shuffled_subjects)
        assigned_values: list[str]
        if relation_name in balanced_relations:
            if len(values) > len(shuffled_subjects):
                raise ValueError(
                    f"Balanced relation {relation_name!r} requires at least as many subjects as values, "
                    f"got {len(shuffled_subjects)} subjects and {len(values)} values"
                )
            shuffled_values = list(values)
            rng.shuffle(shuffled_values)
            full_cycles, remainder = divmod(len(shuffled_subjects), len(shuffled_values))
            assigned_values = shuffled_values * full_cycles + shuffled_values[:remainder]
            rng.shuffle(assigned_values)
        else:
            assigned_values = [choose_value(rng, values) for _ in shuffled_subjects]
        for subject, value in zip(shuffled_subjects, assigned_values, strict=True):
            events.append(
                build_event(
                    rng=rng,
                    relation_name=relation_name,
                    subject=subject,
                    value=value,
                    relation_surface_verbs=relation_surface_verbs,
                )
            )
            latest_sampled_values[(relation_name, subject)] = value

    for _ in range(extra_update_count):
        relation_name = rng.choice(list(active_relation_names))
        relation = RELATION_SPECS[relation_name]
        subject = rng.choice(active_groups[relation.subject_group])
        values = _resolve_group_pool(
            token_groups=token_groups,
            active_groups=active_groups,
            group_name=relation.value_group,
        )
        previous_value = latest_sampled_values[(relation_name, subject)]
        value = choose_value(rng, values, disallow=previous_value)
        events.append(
            build_event(
                rng=rng,
                relation_name=relation_name,
                subject=subject,
                value=value,
                relation_surface_verbs=relation_surface_verbs,
            )
        )
        latest_sampled_values[(relation_name, subject)] = value

    rng.shuffle(events)
    return events


def replay_world_state(events: Sequence[dict[str, str]]) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, int]]]:
    state = {relation_name: {} for relation_name in RELATION_SPECS}
    last_update_index = {relation_name: {} for relation_name in RELATION_SPECS}
    for index, event in enumerate(events):
        relation_name = event["relation"]
        subject = event["subject"]
        value = event["value"]
        state[relation_name][subject] = value
        last_update_index[relation_name][subject] = index
    return state, last_update_index


def resolve_query(
    *,
    query_family: str,
    subject: str,
    final_state: dict[str, dict[str, str]],
    last_update_index: dict[str, dict[str, int]],
) -> tuple[str, list[dict[str, Any]]]:
    if query_family not in QUERY_SPECS:
        raise ValueError(f"Unknown query family {query_family!r}")
    query_spec = QUERY_SPECS[query_family]
    trace_rows: list[dict[str, Any]] = []
    current_subject = subject
    for relation_name in query_spec.trace:
        relation_state = final_state.get(relation_name)
        relation_positions = last_update_index.get(relation_name)
        if relation_state is None or relation_positions is None:
            raise ValueError(f"Missing relation state for {relation_name!r}")
        if current_subject not in relation_state or current_subject not in relation_positions:
            raise ValueError(
                f"Query {query_family!r} is undefined for subject {current_subject!r}; "
                f"missing relation {relation_name!r}"
            )
        value = relation_state[current_subject]
        trace_rows.append(
            {
                "relation": relation_name,
                "subject": current_subject,
                "value": value,
                "event_index": relation_positions[current_subject],
            }
        )
        current_subject = value
    return current_subject, trace_rows


def _random_partition(rng: random.Random, total: int, parts: int) -> list[int]:
    if parts <= 0:
        raise ValueError(f"Expected positive parts, got {parts}")
    buckets = [0 for _ in range(parts)]
    for _ in range(total):
        buckets[rng.randrange(parts)] += 1
    return buckets


def build_story_event_segment(
    *,
    rng: random.Random,
    event: dict[str, str],
    filler_count: int,
) -> list[str]:
    template = rng.choice(STORY_EVENT_TEMPLATES)
    replacement_map = {
        STORY_RELATION_PLACEHOLDER: event["surface_verb"],
        STORY_SUBJECT_PLACEHOLDER: event["subject"],
        STORY_VALUE_PLACEHOLDER: event["value"],
    }
    base_tokens = [replacement_map.get(token, token) for token in template]
    filler_tokens = [rng.choice(STORY_FILLER_VOCAB) for _ in range(filler_count)]
    return base_tokens + filler_tokens


def render_story_prompt(
    *,
    rng: random.Random,
    events: Sequence[dict[str, str]],
    query_family: str,
    query_subject: str,
    story_word_range: tuple[int, int],
) -> str:
    target_length = rng.randint(story_word_range[0], story_word_range[1])
    base_segments = [build_story_event_segment(rng=rng, event=event, filler_count=0) for event in events]
    current_length = 1 + len(events) + len("Q".split()) + 3 + sum(len(segment) for segment in base_segments)
    filler_needed = max(0, target_length - current_length)
    filler_allocation = _random_partition(rng, filler_needed, len(events))
    event_segments = [
        build_story_event_segment(rng=rng, event=event, filler_count=filler_count)
        for event, filler_count in zip(events, filler_allocation, strict=True)
    ]
    event_chunks = [" ".join(segment) for segment in event_segments]
    return "<bos> " + " ; ".join(event_chunks) + f" ; Q {query_family} {query_subject} ->"


def render_prompt(
    *,
    rng: random.Random,
    events: Sequence[dict[str, str]],
    query_family: str,
    query_subject: str,
    prompt_style: str,
    story_word_range: tuple[int, int] | None,
) -> str:
    if prompt_style == "story":
        if story_word_range is None:
            raise ValueError("Story prompt rendering requires a story_word_range")
        return render_story_prompt(
            rng=rng,
            events=events,
            query_family=query_family,
            query_subject=query_subject,
            story_word_range=story_word_range,
        )
    if prompt_style != "symbolic":
        raise ValueError(f"Unsupported prompt_style {prompt_style!r}")
    event_chunks = [f"{event['surface_verb']} {event['subject']} {event['value']}" for event in events]
    return "<bos> " + " ; ".join(event_chunks) + f" ; Q {query_family} {query_subject} ->"


def parse_prompt(prompt: str) -> dict[str, Any]:
    tokens = prompt.split()
    if not tokens or tokens[0] != "<bos>":
        raise ValueError(f"Prompt must start with '<bos>', got {prompt!r}")
    segments: list[list[str]] = []
    current: list[str] = []
    for token in tokens[1:]:
        if token == ";":
            if not current:
                raise ValueError(f"Empty prompt segment in {prompt!r}")
            segments.append(current)
            current = []
            continue
        current.append(token)
    if current:
        segments.append(current)
    if len(segments) < 2:
        raise ValueError(f"Prompt must contain at least one event and one query segment, got {prompt!r}")

    query_segment = segments[-1]
    if len(query_segment) != 4 or query_segment[0] != "Q" or query_segment[-1] != "->":
        raise ValueError(f"Malformed query segment in prompt {prompt!r}")
    query_family = query_segment[1]
    query_subject = query_segment[2]

    events: list[dict[str, str]] = []
    for segment_index, event_tokens in enumerate(segments[:-1]):
        if len(event_tokens) == 3:
            surface_verb, subject, value = event_tokens
        else:
            try:
                relation_marker = event_tokens.index("relation")
                subject_marker = event_tokens.index("subject")
                value_marker = event_tokens.index("value")
            except ValueError as exc:
                raise ValueError(f"Malformed event segment {segment_index} in prompt {prompt!r}") from exc
            if relation_marker + 1 >= len(event_tokens) or subject_marker + 1 >= len(event_tokens) or value_marker + 1 >= len(event_tokens):
                raise ValueError(f"Malformed story event segment {segment_index} in prompt {prompt!r}")
            surface_verb = event_tokens[relation_marker + 1]
            subject = event_tokens[subject_marker + 1]
            value = event_tokens[value_marker + 1]
        relation_name = SURFACE_VERB_TO_RELATION.get(surface_verb)
        if relation_name is None and surface_verb in RELATION_SPECS:
            relation_name = surface_verb
        if relation_name is None:
            raise ValueError(f"Unknown event verb {surface_verb!r} in prompt {prompt!r}")
        events.append(
            {
                "relation": relation_name,
                "surface_verb": surface_verb,
                "subject": subject,
                "value": value,
            }
        )

    return {
        "events": events,
        "query_family": query_family,
        "query_subject": query_subject,
    }


def resolve_prompt_target(prompt: str) -> tuple[str, list[dict[str, Any]]]:
    parsed = parse_prompt(prompt)
    final_state, last_update_index = replay_world_state(parsed["events"])
    return resolve_query(
        query_family=parsed["query_family"],
        subject=parsed["query_subject"],
        final_state=final_state,
        last_update_index=last_update_index,
    )


def choose_query_family(query_families: Sequence[str], *, example_index: int, rng: random.Random, policy: str) -> str:
    if policy == "balanced":
        return query_families[example_index % len(query_families)]
    if policy == "random":
        return rng.choice(list(query_families))
    raise ValueError(f"Unsupported query family policy {policy!r}")


def generate_example(
    *,
    example_id: str,
    split: str,
    example_index: int,
    rng: random.Random,
    token_groups: dict[str, list[str]],
    person_range: tuple[int, int],
    item_range: tuple[int, int],
    room_range: tuple[int, int],
    extra_update_range: tuple[int, int],
    query_families: Sequence[str],
    query_family_policy: str,
    active_relation_names: Sequence[str],
    relation_surface_verbs: dict[str, tuple[str, ...]],
    balanced_relations: Sequence[str],
    prompt_style: str,
    story_word_range: tuple[int, int] | None,
) -> dict[str, Any]:
    active_groups = {
        "persons": choose_distinct_tokens(
            rng,
            token_groups["persons"],
            sample_count(rng, person_range),
            label="persons",
        ),
        "items": choose_distinct_tokens(
            rng,
            token_groups["items"],
            sample_count(rng, item_range),
            label="items",
        ),
        "rooms": choose_distinct_tokens(
            rng,
            token_groups["rooms"],
            sample_count(rng, room_range),
            label="rooms",
        ),
    }
    extra_update_count = sample_count(rng, extra_update_range)
    events = build_world_events(
        rng=rng,
        token_groups=token_groups,
        active_groups=active_groups,
        extra_update_count=extra_update_count,
        active_relation_names=active_relation_names,
        relation_surface_verbs=relation_surface_verbs,
        balanced_relations=balanced_relations,
    )
    final_state, last_update_index = replay_world_state(events)
    query_family = choose_query_family(
        query_families,
        example_index=example_index,
        rng=rng,
        policy=query_family_policy,
    )
    query_spec = QUERY_SPECS[query_family]
    query_subject = rng.choice(active_groups[query_spec.subject_group])
    target, trace = resolve_query(
        query_family=query_family,
        subject=query_subject,
        final_state=final_state,
        last_update_index=last_update_index,
    )
    prompt = render_prompt(
        rng=rng,
        events=events,
        query_family=query_family,
        query_subject=query_subject,
        prompt_style=prompt_style,
        story_word_range=story_word_range,
    )
    relation_update_counts: dict[str, int] = {relation_name: 0 for relation_name in active_relation_names}
    for event in events:
        relation_update_counts[event["relation"]] += 1
    return {
        "id": example_id,
        "task": "microlanguage_world_next_token",
        "split": split,
        "prompt": prompt,
        "target": target,
        "query_family": query_family,
        "query_subject": query_subject,
        "query_depth": len(query_spec.trace),
        "active_counts": {group_name: len(values) for group_name, values in active_groups.items()},
        "update_count": len(events),
        "relation_update_counts": relation_update_counts,
        "trace": trace,
    }


def generate_split(
    *,
    rng: random.Random,
    split: str,
    size: int,
    token_groups: dict[str, list[str]],
    person_range: tuple[int, int],
    item_range: tuple[int, int],
    room_range: tuple[int, int],
    extra_update_range: tuple[int, int],
    query_families: Sequence[str],
    query_family_policy: str,
    active_relation_names: Sequence[str],
    relation_surface_verbs: dict[str, tuple[str, ...]],
    balanced_relations: Sequence[str],
    prompt_style: str,
    story_word_range: tuple[int, int] | None,
    allow_duplicate_prompts: bool,
    global_seen_prompts: set[str],
    max_attempt_multiplier: int,
) -> list[dict[str, Any]]:
    if size < 0:
        raise ValueError(f"Split size must be non-negative, got {size} for split {split!r}")
    if max_attempt_multiplier <= 0:
        raise ValueError(f"Expected positive max_attempt_multiplier, got {max_attempt_multiplier}")
    rows: list[dict[str, Any]] = []
    local_seen_prompts: set[str] = set()
    max_attempts = max(1, size) * max_attempt_multiplier
    attempts = 0
    while len(rows) < size:
        if attempts >= max_attempts:
            raise ValueError(
                f"Could not generate {size} unique prompts for split {split!r} after {attempts} attempts. "
                "Reduce split size, enlarge the vocabulary, or allow duplicate prompts."
            )
        attempts += 1
        row = generate_example(
            example_id=f"{split}_{len(rows):06d}",
            split=split,
            example_index=len(rows),
            rng=rng,
            token_groups=token_groups,
            person_range=person_range,
            item_range=item_range,
            room_range=room_range,
            extra_update_range=extra_update_range,
            query_families=query_families,
            query_family_policy=query_family_policy,
            active_relation_names=active_relation_names,
            relation_surface_verbs=relation_surface_verbs,
            balanced_relations=balanced_relations,
            prompt_style=prompt_style,
            story_word_range=story_word_range,
        )
        prompt = str(row["prompt"])
        if not allow_duplicate_prompts:
            if prompt in local_seen_prompts or prompt in global_seen_prompts:
                continue
            local_seen_prompts.add(prompt)
            global_seen_prompts.add(prompt)
        rows.append(row)
    return rows


def validate_split_ranges(
    *,
    token_groups: dict[str, list[str]],
    person_range: tuple[int, int],
    item_range: tuple[int, int],
    room_range: tuple[int, int],
) -> None:
    if person_range[1] > len(token_groups["persons"]):
        raise ValueError("Person range exceeds available person vocabulary")
    if item_range[1] > len(token_groups["items"]):
        raise ValueError("Item range exceeds available item vocabulary")
    if room_range[1] > len(token_groups["rooms"]):
        raise ValueError("Room range exceeds available room vocabulary")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def build_metadata(
    *,
    args: argparse.Namespace,
    token_groups: dict[str, list[str]],
    vocab: dict[str, list[str]],
    split_sizes: dict[str, int],
    prompt_length_summary: dict[str, dict[str, int]],
    query_families: Sequence[str],
    split_query_families: dict[str, Sequence[str]],
    active_relation_names: Sequence[str],
    relation_surface_verbs: dict[str, tuple[str, ...]],
    balanced_relations: Sequence[str],
) -> dict[str, Any]:
    id_person_range = list(parse_range(args.id_person_range))
    id_item_range = list(parse_range(args.id_item_range))
    id_room_range = list(parse_range(args.id_room_range))
    id_extra_update_range = list(parse_range(args.id_extra_update_range, minimum=0))
    ood_person_range = list(parse_range(args.ood_person_range))
    ood_item_range = list(parse_range(args.ood_item_range))
    ood_room_range = list(parse_range(args.ood_room_range))
    ood_extra_update_range = list(parse_range(args.ood_extra_update_range, minimum=0))
    id_story_word_range = list(parse_range(args.id_story_word_range)) if args.prompt_style == "story" else None
    ood_story_word_range = list(parse_range(args.ood_story_word_range)) if args.prompt_style == "story" else None
    ood_differences: list[str] = []
    if id_person_range != ood_person_range:
        ood_differences.append("more active persons")
    if id_item_range != ood_item_range:
        ood_differences.append("more active items")
    if id_room_range != ood_room_range:
        ood_differences.append("more active rooms")
    if id_extra_update_range != ood_extra_update_range:
        ood_differences.append("more updates")
    if prompt_length_summary["test_ood_longer_context"]["mean_rounded"] > prompt_length_summary["train"]["mean_rounded"]:
        ood_differences.insert(0, "longer prompts")
    if not ood_differences:
        ood_description = "Matched-structure held-out split with new prompt combinations."
    else:
        ood_description = "Held-out split with " + ", ".join(ood_differences) + "."
    event_token_label = "verb" if args.event_relation_token_mode == "surface_verb" else "relation_token"
    if args.prompt_style == "story":
        sequence_format = (
            "<bos> story_words relation relation_token subject entity value entity story_words ; "
            "... ; Q query_family subject ->"
        )
    else:
        sequence_format = (
            f"<bos> {event_token_label} subject value ; {event_token_label} subject value ; "
            "... ; Q query_family subject ->"
        )
    return {
        "name": args.dataset_name,
        "preset": args.preset,
        "seed": args.seed,
        "task": "microlanguage_world_next_token",
        "vocabulary": vocab,
        "vocabulary_groups": token_groups,
        "splits": split_sizes,
        "training_splits": {"default": "train"},
        "sequence_format": sequence_format,
        "target": "Single next-token prediction of the queried latent-state value.",
        "latent_state": {
            "direct_relations": {
                relation_name: {
                    "subject_group": relation.subject_group,
                    "value_group": relation.value_group,
                    "event_tokens": list(relation_surface_verbs[relation_name]),
                }
                for relation_name, relation in RELATION_SPECS.items()
                if relation_name in active_relation_names
            },
            "query_families": {
                query_name: {
                    "subject_group": query.subject_group,
                    "target_group": query.target_group,
                    "trace": list(query.trace),
                }
                for query_name, query in QUERY_SPECS.items()
                if query_name in query_families
            },
        },
        "generation_rules": {
            "active_relations": list(active_relation_names),
            "query_families": list(query_families),
            "split_query_families": {
                split_name: list(split_families)
                for split_name, split_families in split_query_families.items()
            },
            "query_family_policy": args.query_family_policy,
            "max_surface_verbs_per_relation": args.max_surface_verbs_per_relation,
            "event_relation_token_mode": args.event_relation_token_mode,
            "balanced_relations": list(balanced_relations),
            "prompt_style": args.prompt_style,
            "id_person_range": id_person_range,
            "id_item_range": id_item_range,
            "id_room_range": id_room_range,
            "id_extra_update_range": id_extra_update_range,
            "ood_person_range": ood_person_range,
            "ood_item_range": ood_item_range,
            "ood_room_range": ood_room_range,
            "ood_extra_update_range": ood_extra_update_range,
            "id_story_word_range": id_story_word_range,
            "ood_story_word_range": ood_story_word_range,
            "allow_duplicate_prompts": args.allow_duplicate_prompts,
            "max_attempt_multiplier": args.max_attempt_multiplier,
        },
        "ood_rule": {
            "split": "test_ood_longer_context",
            "description": ood_description,
        },
        "prompt_length_summary": prompt_length_summary,
    }


def summarize_prompt_lengths(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    lengths = [len(str(row["prompt"]).split()) for row in rows]
    return {
        "min": min(lengths),
        "max": max(lengths),
        "mean_rounded": round(sum(lengths) / len(lengths)),
    }


def apply_dataset_preset(args: argparse.Namespace) -> None:
    preset = DATASET_PRESETS[args.preset]
    explicit_options = set(getattr(args, "_explicit_options", set()))
    preset_fields = {
        "--dataset-name": preset.dataset_name,
        "--train-size": preset.train_size,
        "--val-size": preset.val_size,
        "--test-size": preset.test_size,
        "--ood-size": preset.ood_size,
        "--query-families": ",".join(preset.query_families),
        "--id-person-range": preset.id_person_range,
        "--id-item-range": preset.id_item_range,
        "--id-room-range": preset.id_room_range,
        "--id-extra-update-range": preset.id_extra_update_range,
        "--ood-person-range": preset.ood_person_range,
        "--ood-item-range": preset.ood_item_range,
        "--ood-room-range": preset.ood_room_range,
        "--ood-extra-update-range": preset.ood_extra_update_range,
        "--max-surface-verbs-per-relation": preset.max_surface_verbs_per_relation,
        "--event-relation-token-mode": preset.event_relation_token_mode,
        "--balanced-relations": ",".join(preset.balanced_relations),
        "--train-query-families": (
            ",".join(preset.train_query_families) if preset.train_query_families is not None else None
        ),
        "--val-query-families": (
            ",".join(preset.val_query_families) if preset.val_query_families is not None else None
        ),
        "--test-query-families": (
            ",".join(preset.test_query_families) if preset.test_query_families is not None else None
        ),
        "--ood-query-families": (
            ",".join(preset.ood_query_families) if preset.ood_query_families is not None else None
        ),
        "--prompt-style": preset.prompt_style,
        "--id-story-word-range": preset.id_story_word_range,
        "--ood-story-word-range": preset.ood_story_word_range,
    }
    field_names = {
        "--dataset-name": "dataset_name",
        "--train-size": "train_size",
        "--val-size": "val_size",
        "--test-size": "test_size",
        "--ood-size": "ood_size",
        "--query-families": "query_families",
        "--id-person-range": "id_person_range",
        "--id-item-range": "id_item_range",
        "--id-room-range": "id_room_range",
        "--id-extra-update-range": "id_extra_update_range",
        "--ood-person-range": "ood_person_range",
        "--ood-item-range": "ood_item_range",
        "--ood-room-range": "ood_room_range",
        "--ood-extra-update-range": "ood_extra_update_range",
        "--max-surface-verbs-per-relation": "max_surface_verbs_per_relation",
        "--event-relation-token-mode": "event_relation_token_mode",
        "--balanced-relations": "balanced_relations",
        "--train-query-families": "train_query_families",
        "--val-query-families": "val_query_families",
        "--test-query-families": "test_query_families",
        "--ood-query-families": "ood_query_families",
        "--prompt-style": "prompt_style",
        "--id-story-word-range": "id_story_word_range",
        "--ood-story-word-range": "ood_story_word_range",
    }
    for option_name, value in preset_fields.items():
        if option_name not in explicit_options and value is not None:
            setattr(args, field_names[option_name], value)
    if "--outdir" not in explicit_options and args.outdir == "dataset/phase3/microlanguage_world_v1" and args.preset != "default":
        args.outdir = f"dataset/phase3/{args.preset}"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description="Generate a synthetic microlanguage next-token world-model dataset.")
    parser.add_argument(
        "--preset",
        choices=sorted(DATASET_PRESETS),
        default="default",
        help="Named dataset preset. Non-default presets replace the standard generation defaults.",
    )
    parser.add_argument(
        "--outdir",
        default="dataset/phase3/microlanguage_world_v1",
        help="Output directory for dataset files.",
    )
    parser.add_argument(
        "--dataset-name",
        default="Microlanguage-World-v1",
        help="Dataset metadata name to write into metadata.json.",
    )
    parser.add_argument("--train-size", type=int, default=12000)
    parser.add_argument("--val-size", type=int, default=1500)
    parser.add_argument("--test-size", type=int, default=1500)
    parser.add_argument("--ood-size", type=int, default=1500)
    parser.add_argument("--num-persons", type=int, default=10)
    parser.add_argument("--num-items", type=int, default=14)
    parser.add_argument("--num-rooms", type=int, default=7)
    parser.add_argument("--num-zones", type=int, default=5)
    parser.add_argument("--num-moods", type=int, default=8)
    parser.add_argument("--num-roles", type=int, default=8)
    parser.add_argument("--num-badges", type=int, default=8)
    parser.add_argument("--num-colors", type=int, default=8)
    parser.add_argument(
        "--id-person-range",
        type=str,
        default="4,5",
        help="Inclusive low,high range for active persons in train/val/test splits.",
    )
    parser.add_argument(
        "--id-item-range",
        type=str,
        default="6,7",
        help="Inclusive low,high range for active items in train/val/test splits.",
    )
    parser.add_argument(
        "--id-room-range",
        type=str,
        default="3,4",
        help="Inclusive low,high range for active rooms in train/val/test splits.",
    )
    parser.add_argument(
        "--id-extra-update-range",
        type=str,
        default="6,9",
        help="Inclusive low,high range for extra updates in train/val/test splits.",
    )
    parser.add_argument(
        "--ood-person-range",
        type=str,
        default="6,7",
        help="Inclusive low,high range for active persons in the OOD split.",
    )
    parser.add_argument(
        "--ood-item-range",
        type=str,
        default="8,9",
        help="Inclusive low,high range for active items in the OOD split.",
    )
    parser.add_argument(
        "--ood-room-range",
        type=str,
        default="5,6",
        help="Inclusive low,high range for active rooms in the OOD split.",
    )
    parser.add_argument(
        "--ood-extra-update-range",
        type=str,
        default="12,16",
        help="Inclusive low,high range for extra updates in the OOD split.",
    )
    parser.add_argument(
        "--query-families",
        type=str,
        default=",".join(QUERY_SPECS),
        help="Comma-separated query families to sample from.",
    )
    parser.add_argument(
        "--query-family-policy",
        choices=["balanced", "random"],
        default="balanced",
        help="How to schedule query families across each split.",
    )
    parser.add_argument(
        "--train-query-families",
        type=str,
        default=None,
        help="Optional comma-separated query families for the train split only.",
    )
    parser.add_argument(
        "--val-query-families",
        type=str,
        default=None,
        help="Optional comma-separated query families for the val split only.",
    )
    parser.add_argument(
        "--test-query-families",
        type=str,
        default=None,
        help="Optional comma-separated query families for the test split only.",
    )
    parser.add_argument(
        "--ood-query-families",
        type=str,
        default=None,
        help="Optional comma-separated query families for the OOD split only.",
    )
    parser.add_argument(
        "--max-surface-verbs-per-relation",
        type=int,
        default=3,
        help="Maximum number of surface verbs to expose per active relation.",
    )
    parser.add_argument(
        "--event-relation-token-mode",
        choices=["surface_verb", "relation_name"],
        default="surface_verb",
        help="Whether fact events use surface verbs or canonical relation names.",
    )
    parser.add_argument(
        "--prompt-style",
        choices=["symbolic", "story"],
        default="symbolic",
        help="Whether to render prompts as compact symbolic sequences or longer story-like segments.",
    )
    parser.add_argument(
        "--id-story-word-range",
        type=str,
        default="110,160",
        help="Inclusive low,high total prompt-length target for train/val/test when prompt-style=story.",
    )
    parser.add_argument(
        "--ood-story-word-range",
        type=str,
        default="180,260",
        help="Inclusive low,high total prompt-length target for OOD when prompt-style=story.",
    )
    parser.add_argument(
        "--balanced-relations",
        type=str,
        default="",
        help="Comma-separated active relations whose subject assignments should be balanced across available values.",
    )
    parser.add_argument(
        "--allow-duplicate-prompts",
        action="store_true",
        help="Allow duplicate prompt strings across and within splits.",
    )
    parser.add_argument(
        "--max-attempt-multiplier",
        type=int,
        default=200,
        help="Maximum sampling attempts per requested row when duplicate prompts are disallowed.",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    args._explicit_options = {
        token.split("=", 1)[0]
        for token in argv
        if token.startswith("--")
    }
    return args


def main() -> None:
    args = parse_args()
    apply_dataset_preset(args)
    default_query_families = parse_query_family_list(args.query_families)
    split_query_families = {
        "train": parse_optional_query_family_list(args.train_query_families) or default_query_families,
        "val": parse_optional_query_family_list(args.val_query_families) or default_query_families,
        "test": parse_optional_query_family_list(args.test_query_families) or default_query_families,
        "test_ood_longer_context": parse_optional_query_family_list(args.ood_query_families) or default_query_families,
    }
    query_families: list[str] = []
    seen_query_families: set[str] = set()
    for split_families in split_query_families.values():
        for family in split_families:
            if family not in seen_query_families:
                query_families.append(family)
                seen_query_families.add(family)
    active_relation_names = active_relation_names_for_query_families(query_families)
    balanced_relations = parse_relation_name_list(args.balanced_relations)
    inactive_balanced_relations = [name for name in balanced_relations if name not in active_relation_names]
    if inactive_balanced_relations:
        raise ValueError(
            "Balanced relations must be active in the current task, got inactive relation(s): "
            f"{inactive_balanced_relations}"
        )
    relation_surface_verbs = build_relation_surface_verbs(
        active_relation_names,
        max_surface_verbs_per_relation=args.max_surface_verbs_per_relation,
        event_relation_token_mode=args.event_relation_token_mode,
    )
    token_groups = build_token_groups(
        args,
        query_families=query_families,
        active_relation_names=active_relation_names,
        relation_surface_verbs=relation_surface_verbs,
    )
    vocab = build_vocab(token_groups)

    id_person_range = parse_range(args.id_person_range)
    id_item_range = parse_range(args.id_item_range)
    id_room_range = parse_range(args.id_room_range)
    id_extra_update_range = parse_range(args.id_extra_update_range, minimum=0)
    ood_person_range = parse_range(args.ood_person_range)
    ood_item_range = parse_range(args.ood_item_range)
    ood_room_range = parse_range(args.ood_room_range)
    ood_extra_update_range = parse_range(args.ood_extra_update_range, minimum=0)
    id_story_word_range = parse_range(args.id_story_word_range) if args.prompt_style == "story" else None
    ood_story_word_range = parse_range(args.ood_story_word_range) if args.prompt_style == "story" else None
    validate_split_ranges(
        token_groups=token_groups,
        person_range=id_person_range,
        item_range=id_item_range,
        room_range=id_room_range,
    )
    validate_split_ranges(
        token_groups=token_groups,
        person_range=ood_person_range,
        item_range=ood_item_range,
        room_range=ood_room_range,
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    global_seen_prompts: set[str] = set()

    split_rows = {
        "train": generate_split(
            rng=rng,
            split="train",
            size=args.train_size,
            token_groups=token_groups,
            person_range=id_person_range,
            item_range=id_item_range,
            room_range=id_room_range,
            extra_update_range=id_extra_update_range,
            query_families=split_query_families["train"],
            query_family_policy=args.query_family_policy,
            active_relation_names=active_relation_names,
            relation_surface_verbs=relation_surface_verbs,
            balanced_relations=balanced_relations,
            prompt_style=args.prompt_style,
            story_word_range=id_story_word_range,
            allow_duplicate_prompts=args.allow_duplicate_prompts,
            global_seen_prompts=global_seen_prompts,
            max_attempt_multiplier=args.max_attempt_multiplier,
        ),
        "val": generate_split(
            rng=rng,
            split="val",
            size=args.val_size,
            token_groups=token_groups,
            person_range=id_person_range,
            item_range=id_item_range,
            room_range=id_room_range,
            extra_update_range=id_extra_update_range,
            query_families=split_query_families["val"],
            query_family_policy=args.query_family_policy,
            active_relation_names=active_relation_names,
            relation_surface_verbs=relation_surface_verbs,
            balanced_relations=balanced_relations,
            prompt_style=args.prompt_style,
            story_word_range=id_story_word_range,
            allow_duplicate_prompts=args.allow_duplicate_prompts,
            global_seen_prompts=global_seen_prompts,
            max_attempt_multiplier=args.max_attempt_multiplier,
        ),
        "test": generate_split(
            rng=rng,
            split="test",
            size=args.test_size,
            token_groups=token_groups,
            person_range=id_person_range,
            item_range=id_item_range,
            room_range=id_room_range,
            extra_update_range=id_extra_update_range,
            query_families=split_query_families["test"],
            query_family_policy=args.query_family_policy,
            active_relation_names=active_relation_names,
            relation_surface_verbs=relation_surface_verbs,
            balanced_relations=balanced_relations,
            prompt_style=args.prompt_style,
            story_word_range=id_story_word_range,
            allow_duplicate_prompts=args.allow_duplicate_prompts,
            global_seen_prompts=global_seen_prompts,
            max_attempt_multiplier=args.max_attempt_multiplier,
        ),
        "test_ood_longer_context": generate_split(
            rng=rng,
            split="test_ood_longer_context",
            size=args.ood_size,
            token_groups=token_groups,
            person_range=ood_person_range,
            item_range=ood_item_range,
            room_range=ood_room_range,
            extra_update_range=ood_extra_update_range,
            query_families=split_query_families["test_ood_longer_context"],
            query_family_policy=args.query_family_policy,
            active_relation_names=active_relation_names,
            relation_surface_verbs=relation_surface_verbs,
            balanced_relations=balanced_relations,
            prompt_style=args.prompt_style,
            story_word_range=ood_story_word_range,
            allow_duplicate_prompts=args.allow_duplicate_prompts,
            global_seen_prompts=global_seen_prompts,
            max_attempt_multiplier=args.max_attempt_multiplier,
        ),
    }

    for split_name, rows in split_rows.items():
        write_jsonl(outdir / f"{split_name}.jsonl", rows)

    prompt_length_summary = {
        split_name: summarize_prompt_lengths(rows)
        for split_name, rows in split_rows.items()
    }
    metadata = build_metadata(
        args=args,
        token_groups=token_groups,
        vocab=vocab,
        split_sizes={split_name: len(rows) for split_name, rows in split_rows.items()},
        prompt_length_summary=prompt_length_summary,
        query_families=query_families,
        split_query_families=split_query_families,
        active_relation_names=active_relation_names,
        relation_surface_verbs=relation_surface_verbs,
        balanced_relations=balanced_relations,
    )
    (outdir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
