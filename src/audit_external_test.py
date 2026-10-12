import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OLD_SPLIT = ROOT / "results" / "data_split.backup_20261012_001253.json"
EXTERNAL_TEST = Path(r"C:\Users\am842\Downloads\archive (1)\ddata\test")

def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

if not OLD_SPLIT.is_file():
    raise FileNotFoundError(OLD_SPLIT)
if not EXTERNAL_TEST.is_dir():
    raise FileNotFoundError(EXTERNAL_TEST)

with open(OLD_SPLIT, "r", encoding="utf-8") as f:
    old = json.load(f)

training_paths = {
    Path(item["path"]).resolve()
    for item in old["train"]
}
training_paths = {p for p in training_paths if p.is_file()}

print("Hashing original training images:", len(training_paths))
training_hashes = set()
for i, path in enumerate(training_paths, 1):
    training_hashes.add(digest(path))
    if i % 10000 == 0:
        print(f"Training images hashed: {i:,}")

external_files = [
    p for p in EXTERNAL_TEST.rglob("*")
    if p.is_file() and p.suffix.lower() in {
        ".jpg", ".jpeg", ".png", ".webp", ".bmp"
    }
]

print("\nExternal test images:", len(external_files))
overlaps = []
unreadable = []

for i, path in enumerate(external_files, 1):
    try:
        if digest(path) in training_hashes:
            overlaps.append(str(path))
    except OSError:
        unreadable.append(str(path))

    if i % 2000 == 0:
        print(f"External test images checked: {i:,}")

print("\n========== OVERLAP AUDIT ==========")
print("External test images:", len(external_files))
print("Exact-content matches with original training:", len(overlaps))
print("Unreadable files:", len(unreadable))

if overlaps:
    print("\nExamples of matching files:")
    for path in overlaps[:10]:
        print(path)

print("\nAudit complete. No files were modified.")
