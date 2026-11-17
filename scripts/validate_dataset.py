"""Offline, standard-library audit of immutable official CSVs; never prints Test text."""
import ast
import csv
import hashlib
import itertools
import json
import math
import platform
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "sources/upstream/data/correct_guess_task"
SPLITS = ("train", "val", "test")
INPUTS = ("base_text", "personality_only")
EXPECTED_COLUMNS = ["", "base_text", "leaning_only", "event_only", "demo_only",
                    "personality_only", "all_text", "output"]
EXPECTED_ROWS = dict(train=8497, val=544, test=478)  # User-provided Stage 0 expectations only.
TRAITS = ("conscientiousness", "extraversion", "neuroticism", "openness", "agreeableness")
BASE = re.compile(r"remaining: (\[.*?\]), rationale: (.*), target: (.*), hint: (.*)", re.DOTALL)
PERSONALITY = re.compile(r"GIVER: \[(.*?)\], GUESSER: \[(.*?)\], (.*)", re.DOTALL)
SENTINELS = re.compile(r"(?<!\w)(None|NaN|null|NA|N/A)(?!\w)", re.IGNORECASE)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(split):
    with (DATA / f"{split}.csv").open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return reader.fieldnames, rows


def parse_base(text):
    match = BASE.fullmatch(text)
    if not match:
        raise ValueError("base grammar mismatch")
    words, rationale, target, hint = match.groups()
    words = ast.literal_eval(words)
    if not isinstance(words, list) or not all(isinstance(w, str) for w in words):
        raise ValueError("remaining is not a list of strings")
    # Ambiguous separator occurrences are reported, rather than silently certified.
    if any(text.count(marker) != 1 for marker in ("remaining: ", ", rationale: ", ", target: ", ", hint: ")):
        raise ValueError("ambiguous or repeated base separator")
    return dict(remaining=words, rationale=rationale, target=target, hint=hint)


def parse_profile(text):
    if text == "None":
        return None
    parts = text.split(", ")
    keys, values = [], []
    for part in parts:
        key, value = part.split(": ")
        keys.append(key)
        if not re.fullmatch(r"-?\d+(?:\.\d+)?", value):
            raise ValueError("unexpected trait value")
        values.append(float(value))
    if tuple(keys) != TRAITS:
        raise ValueError("unexpected trait names/order")
    return dict(zip(keys, values))


def distribution(values):
    values = sorted(values)
    if not values:
        return None
    def q(p):
        index = (len(values) - 1) * p
        lo, hi = math.floor(index), math.ceil(index)
        return round(values[lo] + (values[hi] - values[lo]) * (index - lo), 3)
    return dict(min=values[0], p25=q(.25), median=q(.5), p75=q(.75), p95=q(.95),
                p99=q(.99), max=values[-1], mean=round(sum(values) / len(values), 3))


def duplicates(keys):
    counts = Counter(keys)
    repeated = [n for n in counts.values() if n > 1]
    return dict(unique_keys=len(counts), duplicate_groups=len(repeated),
                rows_in_duplicate_groups=sum(repeated), excess_rows=sum(n - 1 for n in repeated))


def grouping(rows, field):
    groups = defaultdict(Counter)
    for row in rows:
        groups[row[field]][row["output"]] += 1
    return groups


def conflicts(groups):
    bad = [labels for labels in groups.values() if len(labels) > 1]
    return dict(conflicting_input_groups=len(bad), rows_in_conflicting_groups=sum(sum(v.values()) for v in bad))


def conflict_locations(rows, field):
    groups = defaultdict(list)
    for i, row in enumerate(rows, 1):
        groups[row[field]].append(dict(csv_data_row_1based=i, output=row["output"]))
    return [dict(input_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(), rows=items)
            for text, items in groups.items() if len({r["output"] for r in items}) > 1]


def column_audit(rows, column, examples=False):
    values = [r.get(column) for r in rows]
    strings = [v for v in values if isinstance(v, str)]
    exact = Counter(v for v in strings if v.strip().casefold() in {"none", "nan", "null", "na", "n/a"})
    embedded = Counter()
    for value in strings:
        embedded.update(set(m.group(0) for m in SENTINELS.finditer(value)))
    info = dict(python_types=dict(Counter(type(v).__name__ for v in values)),
                unique_values=len(set(strings)), structurally_absent_fields=sum(v is None for v in values),
                empty_strings=sum(v == "" for v in strings),
                whitespace_only_strings=sum(bool(v) and not v.strip() for v in strings),
                exact_sentinel_values=dict(exact), rows_with_embedded_sentinels=dict(embedded))
    if examples:
        # Only Train column representatives; sensitive/background columns omitted from examples file.
        info["train_first_value_preview"] = strings[0][:240] if strings else None
    return info


def audit_pair(rows):
    counts = Counter()
    errors = defaultdict(list)
    missing_labels = defaultdict(Counter)
    values = {role: {t: Counter() for t in TRAITS} for role in ("giver", "guesser")}
    remaining_lengths = []
    multi_groups = defaultdict(list)
    for i, row in enumerate(rows, 1):
        try:
            base = parse_base(row["base_text"])
            counts["base_grammar_valid"] += 1
            remaining_lengths.append(len(base["remaining"]))
            if base["target"] not in base["remaining"]:
                counts["target_not_in_remaining"] += 1
            for field in ("rationale", "target", "hint"):
                if not base[field].strip():
                    counts[f"empty_{field}"] += 1
            if len(base["hint"].split()) > 1:
                counts["hint_with_multiple_whitespace_words"] += 1
        except (ValueError, SyntaxError, TypeError) as exc:
            errors[str(exc)].append(i)
            continue
        match = PERSONALITY.fullmatch(row["personality_only"])
        if not match:
            errors["personality grammar mismatch"].append(i)
            continue
        giver, guesser, suffix = match.groups()
        try:
            profiles = dict(giver=parse_profile(giver), guesser=parse_profile(guesser))
            counts["personality_profile_grammar_valid"] += 1
            personality_base = parse_base(suffix)
            counts["personality_base_grammar_valid"] += 1
        except (ValueError, SyntaxError, TypeError) as exc:
            errors[str(exc)].append(i)
            continue
        if suffix == row["base_text"]:
            counts["base_suffix_exact_match"] += 1
        else:
            errors["base suffix differs"].append(i)
        for field in ("remaining", "rationale", "target", "hint"):
            if base[field] == personality_base[field]:
                counts[f"parsed_{field}_equal"] += 1
            else:
                errors[f"parsed {field} differs"].append(i)
        missing = {role: profile is None for role, profile in profiles.items()}
        pattern = "both" if all(missing.values()) else "giver_only" if missing["giver"] else "guesser_only" if missing["guesser"] else "neither"
        counts[f"missing_profile_{pattern}"] += 1
        missing_labels[pattern][row["output"]] += 1
        for role, profile in profiles.items():
            if profile is not None:
                for trait, value in profile.items():
                    values[role][trait][str(value)] += 1
        # A candidate shared turn context, NOT an actual turn identifier.
        key = (tuple(base["remaining"]), base["hint"], giver, guesser)
        multi_groups[key].append((i, base["target"], row["output"]))
    multi = [items for items in multi_groups.values() if len({x[1] for x in items}) > 1]
    return dict(counts=dict(counts), exceptions={k: dict(count=len(v), csv_data_rows_1based=v) for k, v in errors.items()},
                missing_profile_label_counts={k: dict(v) for k, v in missing_labels.items()},
                trait_value_counts={role: {t: dict(v) for t, v in traits.items()} for role, traits in values.items()},
                remaining_word_count=distribution(remaining_lengths),
                candidate_shared_context_groups_with_distinct_targets=len(multi),
                candidate_shared_context_groups_with_distinct_targets_and_mixed_labels=sum(len({x[2] for x in g}) > 1 for g in multi))


def main():
    manifest = json.loads((ROOT / "sources/manifest.json").read_text(encoding="utf-8"))
    integrity = []
    for entry in manifest["files"]:
        path = ROOT / entry["local_path"]
        actual = digest(path) if path.exists() else None
        integrity.append(dict(path=entry["local_path"], expected_sha256=entry["sha256"],
                              actual_sha256=actual, matches=actual == entry["sha256"]))
    if not all(x["matches"] for x in integrity):
        raise SystemExit("Source integrity check failed; originals were not modified by this script.")
    result = dict(schema_version=1, repository_commit=manifest["commit"],
                  generated_at_utc=datetime.now(timezone.utc).isoformat(),
                  python_version=platform.python_version(), python_implementation=platform.python_implementation(),
                  command="python scripts/validate_dataset.py", actual_argv=sys.argv,
                  script_sha256=digest(Path(__file__)), manifest_sha256=digest(ROOT / "sources/manifest.json"),
                  quantiles="Linear interpolation at (n-1)*p; characters=len(str); whitespace words=len(str.split()). Not model tokens.",
                  source_integrity=integrity, splits={}, between_splits={})
    all_rows = {}
    headers = {}
    for split in (*SPLITS, "all"):
        columns, rows = load(split)
        headers[split], all_rows[split] = columns, rows
        if columns != EXPECTED_COLUMNS:
            raise SystemExit(f"Unexpected schema in {split}: {columns}")
        if any(None in row or any(v is None for v in row.values()) for row in rows):
            raise SystemExit(f"Ragged CSV in {split}")
        if split == "all":
            continue
        info = dict(rows=len(rows), columns=columns, stage0_expected_rows=EXPECTED_ROWS[split],
                    matches_stage0_row_expectation=len(rows) == EXPECTED_ROWS[split],
                    label_counts=dict(Counter(r["output"] for r in rows)),
                    column_audit={c: column_audit(rows, c, split == "train") for c in columns},
                    index=dict(unique=len({r[""] for r in rows}),
                               consecutive_zero_based=[r[""] for r in rows] == [str(i) for i in range(len(rows))]),
                    duplicate_checks={}, input_label_conflicts={}, length_distributions={},
                    condition_comparison=audit_pair(rows))
        for name, cols in [("full_row_including_export_index", columns),
                           ("full_row_excluding_export_index", columns[1:]),
                           ("base_text", ["base_text"]), ("personality_only", ["personality_only"])]:
            info["duplicate_checks"][name] = duplicates(tuple(r[c] for c in cols) for r in rows)
        for field in INPUTS:
            info["input_label_conflicts"][field] = conflicts(grouping(rows, field))
            info["input_label_conflicts"][field]["locations_without_input_text"] = conflict_locations(rows, field)
            info["length_distributions"][field] = dict(characters=distribution([len(r[field]) for r in rows]),
                                                       whitespace_words=distribution([len(r[field].split()) for r in rows]))
        info["length_distributions"]["added_personality_characters"] = distribution([len(r["personality_only"]) - len(r["base_text"]) for r in rows])
        result["splits"][split] = info
    for a, b in itertools.combinations(SPLITS, 2):
        checks = {}
        for name, cols in [("full_row_including_export_index", EXPECTED_COLUMNS),
                           ("full_row_excluding_export_index", EXPECTED_COLUMNS[1:]),
                           ("base_text", ["base_text"]), ("personality_only", ["personality_only"])]:
            ca, cb = (Counter(tuple(r[c] for c in cols) for r in all_rows[s]) for s in (a, b))
            shared = ca.keys() & cb.keys()
            checks[name] = dict(shared_unique_keys=len(shared), rows_in_first=sum(ca[k] for k in shared),
                                rows_in_second=sum(cb[k] for k in shared), cross_split_row_pairs=sum(ca[k] * cb[k] for k in shared))
            if name in INPUTS:
                ga, gb = grouping(all_rows[a], name), grouping(all_rows[b], name)
                shared_text = ga.keys() & gb.keys()
                checks[name]["shared_inputs_with_differing_labels_across_splits"] = sum(any(x != y for x in ga[k] for y in gb[k]) for k in shared_text)
        result["between_splits"][f"{a}__{b}"] = checks
    combined = sum((all_rows[s] for s in SPLITS), [])
    result["combined_input_label_conflicts"] = {field: conflicts(grouping(combined, field)) for field in INPUTS}
    cols = EXPECTED_COLUMNS[1:]
    pool = Counter(tuple(r[c] for c in cols) for r in all_rows["all"])
    union = Counter(tuple(r[c] for c in cols) for r in combined)
    result["all_csv_consistency"] = dict(rows=len(all_rows["all"]), split_rows=len(combined),
                                         multiset_equal_excluding_export_index=pool == union,
                                         extra_in_all=sum((pool-union).values()), extra_in_splits=sum((union-pool).values()),
                                         note="all.csv is a consistency reference, never an additional split.")
    selected = [1]
    missing_row = next((i for i, r in enumerate(all_rows["train"], 1) if "[None]" in r["personality_only"]), None)
    if missing_row and missing_row not in selected:
        selected.append(missing_row)
    examples = [dict(split="train", csv_data_row_1based=i, export_index=all_rows["train"][i-1][""],
                     **{k: all_rows["train"][i-1][k] for k in (*INPUTS, "output")}) for i in selected]
    result["checks"] = dict(source_hashes_match=all(x["matches"] for x in integrity),
                          expected_labels_only=all(set(result["splits"][s]["label_counts"]) <= {"True", "False"} for s in SPLITS),
                          all_condition_pairs_valid=all(result["splits"][s]["condition_comparison"]["counts"].get("base_suffix_exact_match", 0) == len(all_rows[s]) and not result["splits"][s]["condition_comparison"]["exceptions"] for s in SPLITS),
                          all_csv_matches_split_union=pool == union)
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    (reports / "dataset_validation.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (reports / "train_examples.json").write_text(json.dumps(examples, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    table = ["# Train-only paired input examples", "", "Generated by `python scripts/validate_dataset.py`. Row numbers exclude the CSV header.", ""]
    for e in examples:
        table += [f"## Train data row {e['csv_data_row_1based']} (export index {e['export_index']}, output {e['output']})", "", "| Condition A: base_text | Condition B: personality_only |", "| --- | --- |", "| " + " | ".join(e[k].replace("|", "\\|").replace("\n", "<br>") for k in INPUTS) + " |", ""]
    (reports / "train_examples.md").write_text("\n".join(table), encoding="utf-8")
    print(json.dumps(dict(rows={s: result["splits"][s]["rows"] for s in SPLITS}, checks=result["checks"],
                          labels={s: result["splits"][s]["label_counts"] for s in SPLITS}), indent=2))
    if not all(result["checks"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
