"""Label-only integrity audit of all splits; no Test tokenization or predictions."""
import csv
import json
import sys
from pathlib import Path
from datetime import datetime, timezone
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cultural_codes.data import label_summary

def main():
    result = dict(generated_at_utc=datetime.now(timezone.utc).isoformat(),
                  mapping={"False": 0, "True": 1}, test_predictions_generated=False, splits={})
    for split in ("train", "val", "test"):
        with (ROOT / f"sources/upstream/data/correct_guess_task/{split}.csv").open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        result["splits"][split] = dict(rows=len(rows), **label_summary(rows))
    (ROOT / "reports/label_conversion.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
