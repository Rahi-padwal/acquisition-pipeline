# Acquired-company detection pipeline

Detects whether a company has been acquired, using several independent signals
that are combined into one confidence score.

Accquired company:
<img width="1156" height="435" alt="image" src="https://github.com/user-attachments/assets/a09dfc0e-00aa-4c28-944f-7704bcd036e1" />

Not acquired company:
<img width="1106" height="492" alt="image" src="https://github.com/user-attachments/assets/b6140388-cc7a-42fb-b88e-d467c023185d" />


## Layout

| Folder | Stage | Purpose |
|---|---|---|
| `name_matching/` | 1 | Acquirer signals from the company name (`check_name_signals.py`, `data_source.py`) |
| `description_matching/` | 2 | Acquirer signals from the company description |
| `logo_matching/` | 3 | Logo OCR signals (`detection.py`, `ocr.py`, `fetch_logo.py`, `es_client.py`) |
| `combined_matching/` | all | Combines the stages: `combined_check.py`, S3 / logo detection, batch runners, DB migration |
| `data/` | Local-only inputs and samples (git-ignored, not in the repo) |

Each stage folder is self-contained; `combined_matching` adds the sibling folders to `sys.path`.

## Running (from `combined_matching/`)

```
python run_pipeline_on_file.py "test data.csv" [db_path] [output.xlsx]   # stages 1-2 on a file
python s3_detection.py [input.csv] [db_path]                              # S3 detection
python logo_detection.py [input.csv] [db_path]                            # logo OCR stage
python run_full_batch.py                                                  # full run -> combined_acquisitions_full.db
```

## Setup

```
uv sync
```
Copy each `.env.example` to `.env` and fill in credentials (`.env` files are git-ignored).
Generated outputs (`*.db`, `*.xlsx`, `*.log`) are git-ignored.
