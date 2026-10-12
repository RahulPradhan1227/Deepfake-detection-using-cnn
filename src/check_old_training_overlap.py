import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OLD_FILE = ROOT / "results" / "data_split.backup_20261012_001253.json"
NEW_FILE = ROOT / "results" / "data_split.json"

def load_split(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)

def resolve_path(value):
    p = Path(value)
    return p if p.is_absolute() else ROOT / p

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

old = load_split(OLD_FILE)
new = load_split(NEW_FILE)

old_train_paths = {
    resolve_path(item["path"]).resolve()
    for item in old["train"]
}
old_train_paths = {p for p in old_train_paths if p.is_file()}

print("Original training paths:", len(old_train_paths))
print("Checking original training images against corrected validation/test...")
print("This may take a few minutes while image hashes are calculated.")

old_hashes = {}
for i, path in enumerate(old_train_paths, 1):
    digest = sha256(path)
    old_hashes.setdefault(digest, []).append(str(path))
    if i % 10000 == 0:
        print(f"Hashed original training images: {i:,}")

old_path_set = {str(p).casefold() for p in old_train_paths}
hash_cache = {}

for split_name in ("validation", "test"):
    samples = new[split_name]
    path_overlaps = []
    content_overlaps = []
    missing = 0

    for item in samples:
        path = resolve_path(item["path"]).resolve()

        if not path.is_file():
            missing += 1
            continue

        if str(path).casefold() in old_path_set:
            path_overlaps.append(str(path))

        digest = hash_cache.get(path)
        if digest is None:
            digest = sha256(path)
            hash_cache[path] = digest

        if digest in old_hashes:
            content_overlaps.append({
                "current_path": str(path),
                "old_training_match": old_hashes[digest][0]
            })

    print(f"\n{split_name.upper()}")
    print("Samples:", len(samples))
    print("Missing files:", missing)
    print("Same-path overlap with old training:", len(path_overlaps))
    print("Exact-content overlap with old training:", len(content_overlaps))

    if content_overlaps:
        print("Examples of exact-content overlaps:")
        for match in content_overlaps[:5]:
            print("Current:", match["current_path"])
            print("Old train:", match["old_training_match"])

print("\nOverlap audit complete.")
