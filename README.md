# EXO MlxRing Loader

`recreate_mlxring_instance.py` recreates a local EXO model instance as `MlxRing` when the default RDMA or `MlxJaccl` path is unstable.

## What It Does

- Reloads a chosen EXO model using the safer `MlxRing` transport
- Prefers `Pipeline` sharding when a usable preview exists
- Waits for runners to become ready
- Optionally sends a test request to the OpenAI-compatible EXO API

## Model Selection

- If `--model` is provided, it uses that exact model ID
- If there is only one active model in `/state`, it auto-selects it
- If there is only one model in `/v1/models`, it auto-selects it
- If multiple models are available and the script is run in an interactive terminal, it prompts you to choose
- Interactive selection accepts search text and fuzzy model names, not just exact IDs
- If multiple models are available and the script is run non-interactively, it exits with a clear error telling you to pass `--model`

## Usage

```bash
python3 recreate_mlxring_instance.py
```

Common examples:

```bash
python3 recreate_mlxring_instance.py --model mlx-community/Qwen3.5-35B-A3B-4bit --test
python3 recreate_mlxring_instance.py --keep-existing --dry-run
python3 recreate_mlxring_instance.py --delete-all-instances --wait-seconds 240
```

## Requirements

- Python 3
- A running EXO instance, by default at `http://localhost:52415`

## Notes

- Unless `--keep-existing` is used, the script deletes existing instances for the selected model before creating a new one
- If no usable `MlxRing` preview exists, EXO usually does not have enough free memory for that model placement
- The script waits on the exact new instance it creates, so stale failed runners from older launches do not trigger false failures
