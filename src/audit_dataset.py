import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPLIT_FILE = ROOT / "results" / "data_split.json"
REPORT_FILE = ROOT / "results" / "dataset_audit.json"

EXPECTED_LABELS = {"real": 0, "fake": 1}
SPLIT_NAMES = ("train", "validation", "test")


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_path(value):
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def folder_class(path):
    matches = [
        part.lower()
        for part in path.parts
        if part.lower() in EXPECTED_LABELS
    ]
    return matches[-1] if matches else None


def main():
    if not SPLIT_FILE.exists():
        raise FileNotFoundError(
            f"Split file not found: {SPLIT_FILE}"
        )

    with SPLIT_FILE.open("r", encoding="utf-8") as file:
        data = json.load(file)

    report = {
        "project_root": str(ROOT),
        "split_file": str(SPLIT_FILE),
        "expected_mapping": {
            "0": "REAL",
            "1": "FAKE",
        },
        "splits": {},
        "errors": [],
        "warnings": [],
        "cross_split_path_duplicates": [],
        "cross_split_exact_content_duplicates": [],
        "same_split_exact_content_duplicates": [],
    }

    all_paths = defaultdict(list)
    all_hashes = defaultdict(list)
    total_images = 0
    total_bytes = 0

    for split in SPLIT_NAMES:
        samples = data.get(split)

        if not isinstance(samples, list):
            report["errors"].append(
                f"Missing or invalid split: {split}"
            )
            continue

        label_counts = Counter()
        folder_counts = Counter()
        missing_files = []
        invalid_labels = []
        folder_label_mismatches = []
        repeated_paths_in_split = Counter()
        split_records = []

        for index, item in enumerate(samples):
            if not isinstance(item, dict):
                report["errors"].append(
                    f"{split}[{index}] is not an object"
                )
                continue

            if "path" not in item or "label" not in item:
                report["errors"].append(
                    f"{split}[{index}] lacks path or label"
                )
                continue

            raw_path = item["path"]

            try:
                label = int(item["label"])
            except (TypeError, ValueError):
                invalid_labels.append({
                    "index": index,
                    "label": repr(item["label"]),
                    "path": str(raw_path),
                })
                continue

            label_counts[str(label)] += 1

            if label not in (0, 1):
                invalid_labels.append({
                    "index": index,
                    "label": label,
                    "path": str(raw_path),
                })

            path = resolve_path(str(raw_path))
            path_key = str(path).casefold()
            repeated_paths_in_split[path_key] += 1
            all_paths[path_key].append({
                "split": split,
                "path": str(path),
                "label": label,
            })

            detected_class = folder_class(path)

            if detected_class:
                folder_counts[detected_class] += 1
                expected_label = EXPECTED_LABELS[detected_class]

                if label != expected_label:
                    folder_label_mismatches.append({
                        "path": str(path),
                        "label": label,
                        "folder_class": detected_class,
                        "expected_label": expected_label,
                    })

            if not path.is_file():
                missing_files.append(str(path))
                continue

            try:
                size = path.stat().st_size
                digest = sha256_file(path)
            except OSError as exc:
                report["errors"].append(
                    f"Cannot read {path}: {exc}"
                )
                continue

            total_images += 1
            total_bytes += size

            record = {
                "split": split,
                "path": str(path),
                "label": label,
                "sha256": digest,
            }

            split_records.append(record)
            all_hashes[digest].append(record)

        report["splits"][split] = {
            "sample_count": len(samples),
            "label_counts": dict(label_counts),
            "folder_counts": dict(folder_counts),
            "missing_file_count": len(missing_files),
            "missing_file_examples": missing_files[:20],
            "invalid_label_count": len(invalid_labels),
            "invalid_label_examples": invalid_labels[:20],
            "folder_label_mismatch_count": len(
                folder_label_mismatches
            ),
            "folder_label_mismatch_examples":
                folder_label_mismatches[:20],
            "repeated_path_count": sum(
                count - 1
                for count in repeated_paths_in_split.values()
                if count > 1
            ),
            "exact_hashes_checked": len(split_records),
        }

    # Detect paths reused across multiple splits.
    for path_key, records in all_paths.items():
        split_set = {record["split"] for record in records}
        if len(split_set) > 1:
            report["cross_split_path_duplicates"].append({
                "path": records[0]["path"],
                "splits": sorted(split_set),
            })

    # Detect identical file contents across splits.
    for digest, records in all_hashes.items():
        split_set = {record["split"] for record in records}

        if len(split_set) > 1:
            report["cross_split_exact_content_duplicates"].append({
                "sha256": digest,
                "splits": sorted(split_set),
                "files": [
                    {
                        "split": record["split"],
                        "path": record["path"],
                        "label": record["label"],
                    }
                    for record in records[:10]
                ],
                "total_matching_files": len(records),
            })
        elif len(records) > 1:
            report["same_split_exact_content_duplicates"].append({
                "sha256": digest,
                "split": records[0]["split"],
                "files": [
                    record["path"] for record in records[:10]
                ],
                "total_matching_files": len(records),
            })

    report["summary"] = {
        "total_existing_images_hashed": total_images,
        "total_bytes_hashed": total_bytes,
        "cross_split_path_duplicate_groups": len(
            report["cross_split_path_duplicates"]
        ),
        "cross_split_exact_duplicate_groups": len(
            report["cross_split_exact_content_duplicates"]
        ),
        "same_split_exact_duplicate_groups": len(
            report["same_split_exact_content_duplicates"]
        ),
    }

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("=" * 65)
    print("DATASET SPLIT AND EXACT-DUPLICATE AUDIT")
    print("=" * 65)

    for split, result in report["splits"].items():
        print(f"\n{split.upper()}")
        print("Samples:", result["sample_count"])
        print("Labels:", result["label_counts"])
        print("Folders:", result["folder_counts"])
        print("Missing files:", result["missing_file_count"])
        print("Invalid labels:", result["invalid_label_count"])
        print("Folder/label mismatches:",
              result["folder_label_mismatch_count"])
        print("Repeated paths:", result["repeated_path_count"])

    print("\nSUMMARY")
    print(json.dumps(report["summary"], indent=2))
    print("\nErrors:", len(report["errors"]))

    if report["cross_split_path_duplicates"]:
        print("\nCross-split path duplicate examples:")
        for item in report["cross_split_path_duplicates"][:10]:
            print(item)

    if report["cross_split_exact_content_duplicates"]:
        print("\nCross-split exact-content duplicate examples:")
        for item in report[
            "cross_split_exact_content_duplicates"
        ][:10]:
            print(item)

    print("\nFull report:", REPORT_FILE)


if __name__ == "__main__":
    main()
