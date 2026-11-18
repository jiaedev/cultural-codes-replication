"""Restore the protocol's exact pretrained weight file and verify its bytes."""
import hashlib
import json
import os
from pathlib import Path
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def main():
    spec = json.loads((ROOT / "configs/experimental_protocol.v1.json").read_text(encoding="utf-8"))
    model = spec["model"]
    target_dir = ROOT / model["local_config"]
    target = target_dir / model["pretrained_file"]
    if not target.exists():
        os.environ["HF_HOME"] = str(ROOT / ".cache/huggingface")
        hf_hub_download(repo_id=model["repository"], revision=model["revision"],
                        filename=model["pretrained_file"], local_dir=target_dir)
    if target.stat().st_size != model["pretrained_bytes"] or sha256(target) != model["pretrained_sha256"]:
        raise ValueError("Pretrained source file does not match the pinned revision metadata; preserve it for diagnosis")
    print(json.dumps({"repository": model["repository"], "revision": model["revision"],
                      "file": str(target.relative_to(ROOT)).replace("\\", "/"),
                      "bytes": target.stat().st_size, "sha256": sha256(target)}, indent=2))

if __name__ == "__main__":
    main()
