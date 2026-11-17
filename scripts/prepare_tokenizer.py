"""Fetch only pinned tokenizer/config files; never pretrained weights."""
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone
import urllib.request

ROOT = Path(__file__).resolve().parents[1]

def main():
    spec = json.loads((ROOT / "configs/model_candidate.json").read_text(encoding="utf-8"))
    folder = ROOT / spec["local_path"]
    folder.mkdir(parents=True, exist_ok=True)
    manifest_path = ROOT / "reports/tokenizer_manifest.json"
    if manifest_path.exists():
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        for entry in prior["files"]:
            assert hashlib.sha256((ROOT / entry["path"]).read_bytes()).hexdigest() == entry["sha256"]
        print("Existing pinned tokenizer/config hashes verified.")
        return
    entries = []
    for name in ("config.json", "tokenizer_config.json", "tokenizer.json", "vocab.json", "merges.txt"):
        url = f"https://huggingface.co/{spec['repository']}/resolve/{spec['revision']}/{name}"
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read()
        path = folder / name
        if path.exists() and path.read_bytes() != data:
            raise RuntimeError(f"Refusing to replace different bytes: {name}")
        path.write_bytes(data)
        entries.append(dict(path=path.relative_to(ROOT).as_posix(), url=url, bytes=len(data),
                            sha256=hashlib.sha256(data).hexdigest(),
                            acquired_at_utc=datetime.now(timezone.utc).isoformat()))
    manifest_path.write_text(json.dumps(dict(repository=spec["repository"], revision=spec["revision"],
        pretrained_weights_downloaded=False, files=entries), indent=2)+"\n", encoding="utf-8")
    print("Pinned tokenizer/config saved; no pretrained weights downloaded.")

if __name__ == "__main__":
    main()
