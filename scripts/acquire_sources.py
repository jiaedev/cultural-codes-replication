"""Acquire a small, pinned official source snapshot without installing packages."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "9bf4550e681f7a42ac406439b00b0c717f59f13c"
REPO = "https://github.com/SALT-NLP/codenames"
FILES = ["README.md", "models/baseline_classification.py", "select_best_model.py",
         "train_classification.sh", "train_all_best.sh", "duet/src/Main.elm",
         "duet/src/User.elm", "duet/src/Game.elm", "duet/gameapi/handler.go"]
FILES += [f"data/correct_guess_task/{s}.csv" for s in ("train", "val", "test", "all")]


def main():
    manifest_path = ROOT / "sources/manifest.json"
    if manifest_path.exists():
        raise SystemExit("Snapshot already recorded; validate it instead of overwriting originals.")
    entries = []

    def fetch(url, local, blob=None):
        path = ROOT / local
        if path.exists():
            raise SystemExit(f"Refusing to overwrite {local}")
        request = urllib.request.Request(url, headers={"User-Agent": "Cultural-Codes-Stage2"})
        with urllib.request.urlopen(request, timeout=60) as response:
            data = response.read()
            final_url = response.url
        if blob:
            git_sha = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
            if git_sha != blob:
                raise ValueError(f"Git blob mismatch: {local}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        entries.append(dict(url=url, resolved_url=final_url, local_path=local,
                            acquired_at_utc=datetime.now(timezone.utc).isoformat(),
                            bytes=len(data), sha256=hashlib.sha256(data).hexdigest(), git_blob_sha1=blob))
        print(f"Saved {local}: {len(data)} bytes", flush=True)
        return data

    tree_url = f"https://api.github.com/repos/SALT-NLP/codenames/git/trees/{COMMIT}?recursive=1"
    tree = json.loads(fetch(tree_url, "sources/github_tree.json"))
    if tree["sha"] != COMMIT or tree["truncated"]:
        raise ValueError("Incomplete or wrong repository tree")
    blobs = {entry["path"]: entry["sha"] for entry in tree["tree"] if entry["type"] == "blob"}
    for name in FILES:
        fetch(f"https://raw.githubusercontent.com/SALT-NLP/codenames/{COMMIT}/{name}",
              f"sources/upstream/{name}", blobs[name])
    fetch("https://aclanthology.org/2023.findings-acl.410/", "sources/paper/acl_page.html")
    fetch("https://aclanthology.org/2023.findings-acl.410.pdf", "sources/paper/2023.findings-acl.410.pdf")
    manifest = dict(repository=REPO, commit=COMMIT, paper_version="ACL published version, Findings 2023",
                    acquisition_script="scripts/acquire_sources.py", files=entries)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
