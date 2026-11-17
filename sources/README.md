# Source provenance

## Acquisition

Repository: [SALT-NLP/codenames](https://github.com/SALT-NLP/codenames)

Pinned revision: `9bf4550e681f7a42ac406439b00b0c717f59f13c`. All repository artifacts use this fixed revision; current main was not treated as equivalent. The recursive GitHub tree is preserved in `github_tree.json`; it is not truncated. No stand-alone release-time CSV-construction, raw-rationale normalization, or survey-scoring pipeline was identified in that tree. The inspected classification script consumes prepared CSVs and performs tokenizer/label preprocessing; that is different from constructing the release from game logs. This is a statement about the inspected revision/artifacts, not every possible external artifact or later branch.

Acquisition command: `python scripts/acquire_sources.py`, CPython 3.13.16. Executed 2026-11-17, 04:59:22-04:59:33 UTC (13:59:22-13:59:33 Asia/Seoul). `manifest.json` records exact per-file timestamps, requested/resolved URLs, byte counts, SHA-256 hashes, and Git blob hashes for repository files. Repository file bytes were checked against the pinned tree's Git blob identifiers before saving.

| Official source path | Local path |
| --- | --- |
| `README.md` | `upstream/README.md` |
| `models/baseline_classification.py` | `upstream/models/baseline_classification.py` |
| `train_classification.sh`, `train_all_best.sh`, `select_best_model.py` | Same paths below `upstream/` |
| `duet/src/Main.elm`, `duet/src/User.elm`, `duet/src/Game.elm`, `duet/gameapi/handler.go` | Same paths below `upstream/` |
| `data/correct_guess_task/{train,val,test,all}.csv` | Same paths below `upstream/` |
| `https://aclanthology.org/2023.findings-acl.410/` | `paper/acl_page.html` |
| `https://aclanthology.org/2023.findings-acl.410.pdf` | `paper/2023.findings-acl.410.pdf` |

Each repository file's raw URL has the form `https://raw.githubusercontent.com/SALT-NLP/codenames/<pinned-revision>/<official-source-path>`; the exact URLs are in the manifest. This is a selected source snapshot, not a full checkout. The UI code was inspected only as supporting evidence for target/rationale collection, not executed.

## Dataset SHA-256

| CSV | Bytes | SHA-256 |
| --- | ---: | --- |
| train.csv | 26523895 | `365353ea03de2d98e9aad3c1f1b906fdbcbe72aa292f77db41b587b0f466f48c` |
| val.csv | 1759804 | `a5bcf752d81b9ff10c033d112a85b32b9cee6426eaff57fa3ab95f5d13106b9f` |
| test.csv | 1565947 | `a596a3ca1763c6bf5fe96400030847a4d704118a97c29ba7bd42016a944f411c` |
| all.csv | 29850732 | `3e359d6753e9b7bae1494d533caf6340f3bbf7467b0182a50a85c2a75ab74a03` |

Originals are unchanged. CSVs are omitted from Git; their exact URLs and hashes remain in the manifest. From the repository root, restore missing recorded files before validation:

```powershell
python scripts/restore_recorded_files.py
python scripts/validate_dataset.py
```

The restoration helper checks existing bytes and verifies downloads against the original manifest without replacing it. `acquire_sources.py` is the original collection utility for an empty snapshot and refuses existing manifests/files; normal fresh checkouts should use restoration. The same helper restores the separately pinned tokenizer/config files listed in `reports/tokenizer_manifest.json`. See [environment setup](../environment/README.md) for the complete sequence.

## Published paper inspection

Only the ACL published version was used, not arXiv. The upstream README links arXiv, but that does not establish version equivalence. The ACL PDF contains 20 pages including appendices/checklist. Relevant evidence: §§3.2-3.4, Table 1, §4.3, §4.4, §5, Table 5, Appendices B, D.2.2 and E.

`pypdf` was used for a text derivative; Poppler rendered PDF pages 7 and 16 to confirm Table 5/§4.4/§5 and the questionnaire visually. The PDF was not edited. These optional inspection tools are separate from the CSV audit dependencies. With the tools available, the project-relative inspection commands are:

```python
from pathlib import Path
from pypdf import PdfReader
reader = PdfReader('sources/paper/2023.findings-acl.410.pdf')
Path('reports/acl_paper_text.txt').write_text(
    '\n\n'.join('=== PDF page '+str(i+1)+' ===\n'+p.extract_text()
                 for i, p in enumerate(reader.pages)), encoding='utf-8')
```

```text
pdftoppm -f 7 -l 7 -scale-to 1500 -png -singlefile sources/paper/2023.findings-acl.410.pdf reports/paper_page7
pdftoppm -f 16 -l 16 -scale-to 1600 -png -singlefile sources/paper/2023.findings-acl.410.pdf reports/paper_page16
```
