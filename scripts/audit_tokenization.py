"""Train/Validation only; investigate the candidate's architectural capacity."""
import json
import os
os.environ['USE_TORCH'] = '0'  # Tokenizer/config audit does not need a tensor backend.
import sys
from pathlib import Path
from datetime import datetime, timezone
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from cultural_codes.data import development_rows, label_summary, CONDITIONS
from validate_dataset import distribution, BASE
from transformers import AutoTokenizer, AutoConfig

def main():
    spec = json.loads((ROOT / "configs/model_candidate.json").read_text(encoding="utf-8"))
    local = ROOT / spec["local_path"]
    tokenizer = AutoTokenizer.from_pretrained(local, local_files_only=True, use_fast=True)
    config = AutoConfig.from_pretrained(local, local_files_only=True)
    # RoBERTa nonpadding positions begin at pad_token_id+1.
    position_capacity = config.max_position_embeddings - config.pad_token_id - 1
    capacity = min(tokenizer.model_max_length, position_capacity)
    result = dict(generated_at_utc=datetime.now(timezone.utc).isoformat(), model=spec,
                  tokenizer_class=type(tokenizer).__name__, tokenizer_is_fast=tokenizer.is_fast,
                  tokenizer_model_max_length=tokenizer.model_max_length,
                  max_position_embeddings=config.max_position_embeddings,
                  usable_total_sequence_capacity=capacity,
                  capacity_note="Architecture/tokenizer bound, not a Stage 4 chosen maximum.",
                  special_tokens_per_single_input=tokenizer.num_special_tokens_to_add(pair=False),
                  truncation_side=tokenizer.truncation_side, padding_side=tokenizer.padding_side,
                  special_tokens_map=tokenizer.special_tokens_map, splits={}, test_tokenized=False)
    for split, rows in development_rows(ROOT).items():
        section = dict(rows=len(rows), labels=label_summary(rows), conditions={})
        lengths = {}
        for condition in CONDITIONS:
            texts = [r[condition] for r in rows]
            full = tokenizer(texts, truncation=False, padding=False, add_special_tokens=True)
            bounded = tokenizer(texts, truncation=True, max_length=capacity, padding=False,
                                add_special_tokens=True, return_offsets_mapping=True)
            sizes = [len(ids) for ids in full["input_ids"]]
            lengths[condition] = sizes
            lost = {field: 0 for field in ("remaining", "rationale", "target", "hint")}
            changed, roundtrip = 0, 0
            for row, text, ids, limited, offsets in zip(rows, texts, full["input_ids"], bounded["input_ids"], bounded["offset_mapping"]):
                changed += ids != limited
                roundtrip += tokenizer.decode(limited, skip_special_tokens=True, clean_up_tokenization_spaces=False) == text
                start = 0 if condition == "base_text" else len(text) - len(row["base_text"])
                match = BASE.fullmatch(row["base_text"])
                if not match:
                    raise ValueError("Unexpected Stage 2 base grammar")
                retained_end = max((end for begin, end in offsets), default=0)
                for group, field in enumerate(lost, 1):
                    lost[field] += start + match.end(group) > retained_end
            section["conditions"][condition] = dict(tokens_including_special_tokens=distribution(sizes),
                sequences_changed_at_capacity=changed, full_string_roundtrip_matches=roundtrip,
                rows_with_base_field_content_cut_at_capacity=lost)
        section["added_personality_tokens"] = distribution([b-a for a,b in zip(lengths["base_text"], lengths["personality_only"])])
        section["both_conditions_fit_capacity"] = all(max(lengths[c]) <= capacity for c in CONDITIONS)
        result["splits"][split] = section
    (ROOT / "reports/tokenization_audit.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
