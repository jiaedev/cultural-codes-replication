"""Restore omitted snapshot files from immutable manifests; never replace different bytes."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ("sources/manifest.json", "reports/tokenizer_manifest.json")


def restore_manifest(root, manifest_path, check_only=False):
    root = Path(root).resolve()
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    counts = {"verified": 0, "restored": 0}
    for entry in manifest["files"]:
        relative = entry.get("local_path", entry.get("path"))
        if not isinstance(relative, str):
            raise ValueError("Manifest entry has no relative file path")
        path = (root / relative).resolve()
        if path == root or not path.is_relative_to(root):
            raise ValueError(f"Manifest path leaves the project: {relative}")
        expected = entry["sha256"]
        if path.exists():
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError(f"Existing file differs from recorded bytes: {relative}")
            counts["verified"] += 1
            continue
        if check_only:
            raise FileNotFoundError(f"Missing recorded file: {relative}")
        url = entry["url"]
        if not url.startswith("https://"):
            raise ValueError("Recorded downloads must use HTTPS")
        request = urllib.request.Request(url, headers={"User-Agent": "Cultural-Codes-Restore"})
        with urllib.request.urlopen(request, timeout=60) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"Downloaded bytes fail recorded SHA-256: {relative}")
        path.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation also prevents overwriting a file created during the download.
        with path.open("xb") as stream:
            stream.write(data)
        counts["restored"] += 1
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="Verify locally; do not download")
    args = parser.parse_args()
    for name in MANIFESTS:
        print(name, restore_manifest(ROOT, ROOT / name, args.check_only))


if __name__ == "__main__":
    main()
