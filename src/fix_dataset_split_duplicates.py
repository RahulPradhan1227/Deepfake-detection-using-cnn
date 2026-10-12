import argparse
import hashlib
import json
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPLIT_FILE = ROOT / "results" / "data_split.json"
SPLIT_NAMES = ("train", "validation", "test")
PRIORITY = {"train": 0, "validation": 1, "test": 2}


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_path(value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Back up and write the corrected split file",
    )
    args = parser.parse_args()

    with SPLIT_FILE.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)

    groups = defaultdict(list)
    missing = []
    invalid = []

    for split in SPLIT_NAMES:
        samples = data.get(split)
        if not isinstance(samples, list):
            raise ValueError(f"Missing or invalid split: {split}")

        for index, sample in enumerate(samples):
            if not isinstance(sample, dict) or "path" not in sample or "label" not in sample:
                invalid.append(f"{split}[{index}]")
                continue

            path = resolve_path(str(sample["path"]))
            if not path.is_file():
                missing.append(str(path))
                continue

            try:
                label = int(sample["label"])
            except (TypeError, ValueError):
                invalid.append(f"{split}[{index}] invalid label")
                continue

            if label not in (0, 1):
                invalid.append(f"{split}[{index}] invalid label {label}")
                continue

            groups[sha256_file(path)].append({
                "split": split,
                "sample": sample,
                "label": label,
                "path": str(path),
            })

    if missing or invalid:
        print("STOP: Missing files or invalid entries were found.")
        print("Missing:", len(missing), "Invalid:", len(invalid))
        print("Examples:", (missing + invalid)[:10])
        print("No changes were made.")
        return

    cross_split_groups = 0
    same_split_groups = 0
    conflicting_label_groups = []
    removals = {split: 0 for split in SPLIT_NAMES}
    cleaned = {split: [] for split in SPLIT_NAMES}

    for digest, records in groups.items():
        labels = {r["label"] for r in records}
        splits = {r["split"] for r in records}

        if len(labels) > 1:
            conflicting_label_groups.append({
                "sha256": digest,
                "files": [
                    {"split": r["split"], "label": r["label"], "path": r["path"]}
                    for r in records
                ],
            })
            continue

        if len(splits) > 1:
            cross_split_groups += 1
        elif len(records) > 1:
            same_split_groups += 1

        # Keep one representative. Prefer train, then validation, then test.
        records_sorted = sorted(
            records,
            key=lambda r: (PRIORITY[r["split"]], r["path"].casefold()),
        )
        keeper = records_sorted[0]
        cleaned[keeper["split"]].append(keeper["sample"])

        for duplicate in records_sorted[1:]:
            removals[duplicate["split"]] += 1

    print("=" * 65)
    print("EXACT-DUPLICATE SPLIT CLEANUP — DRY RUN" if not args.apply
          else "EXACT-DUPLICATE SPLIT CLEANUP — APPLY")
    print("=" * 65)
    print("Unique content groups:", len(groups))
    print("Cross-split duplicate groups:", cross_split_groups)
    print("Same-split duplicate groups:", same_split_groups)
    print("Conflicting-label groups:", len(conflicting_label_groups))

    for split in SPLIT_NAMES:
        old_count = len(data[split])
        new_count = len(cleaned[split])
        print(f"{split}: {old_count} -> {new_count} (remove {removals[split]})")

    if conflicting_label_groups:
        report_path = ROOT / "results" / "duplicate_label_conflicts.json"
        report_path.write_text(
            json.dumps(conflicting_label_groups, indent=2),
            encoding="utf-8",
        )
        print("Conflicting labels need manual review:", report_path)
        print("No changes made. Resolve conflicts before applying.")
        return

    if not args.apply:
        print("\nDry run only. No files were changed.")
        print("If the counts look reasonable, run again with --apply.")
        return

    backup = SPLIT_FILE.with_name(
        f"data_split.backup_{datetime.now():%Y%m%d_%H%M%S}.json"
    )
    shutil.copy2(SPLIT_FILE, backup)

    # Keep each split's entries in a deterministic path order.
    for split in SPLIT_NAMES:
        cleaned[split].sort(key=lambda sample: str(sample["path"]).casefold())

    # Preserve other top-level metadata, if present.
    output = dict(data)
    output.update(cleaned)

    temp_file = SPLIT_FILE.with_suffix(".json.tmp")
    temp_file.write_text(json.dumps(output, indent=2), encoding="utf-8")
    temp_file.replace(SPLIT_FILE)

    print("\nCorrected split file saved:", SPLIT_FILE)
    print("Original backup saved:", backup)
    print("Image files and model checkpoints were not changed.")
    print("Next: rerun the dataset audit, then retrain and evaluate.")


if __name__ == "__main__":
    main()
