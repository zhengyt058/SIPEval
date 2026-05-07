# SIPEval

SIPEval is an evaluation and data-generation toolkit for the SIP project.
It provides scripts to:
- build modified text/image data,
- generate model outputs through APIs,
- format judge inputs, and
- run automatic judging pipelines.

## Repository Structure

- `1_classification/`: classification-related preprocessing and scripts.
- `2_modify_image/`: image-side transformation and augmentation scripts.
- `2_modify_text/`: text-side transformation scripts.
- `3_inference/`: QA construction, API generation, and judge-result formatting.
- `generate_api.sh`: one-click generation entry for API-based inference.
- `modify_text.sh`: one-click entry for text modification tasks.
- `judge.sh`: one-click entry to prepare judge data and run evaluation.

## Quick Start

1. Clone the repository:

```bash
git clone https://github.com/zhengyt058/SIPEval.git
cd SIPEval
```

2. Prepare a Python environment (recommended: Python 3.10+):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
```

3. Install required packages used by your selected pipeline.
   (Dependencies are script-specific and should be installed based on actual usage.)

## Typical Workflow

1. **Modify data** (text/image):
   - run scripts in `2_modify_text/` and/or `2_modify_image/`.
2. **Build QA files and generate answers**:
   - use scripts in `3_inference/` or `generate_api.sh`.
3. **Format evaluation inputs and run judge**:
   - use `3_inference/form_custom_data.py`,
   - then run `judge.sh` for batch evaluation.

## Run Examples

Generate API results:

```bash
bash generate_api.sh
```

Run text modification (default mode: `t2`):

```bash
bash modify_text.sh t2
```

Run judge for one or more models:

```bash
bash judge.sh gpt-4o-mini
bash judge.sh gpt-4o-mini qwen2.5-vl-72b
```

## Notes

- Some shell scripts currently contain absolute local paths (for example `/mnt/shared-storage-user/...`).
  Update these paths to match your own environment before running.
- API-based generation requires valid API credentials (for example `API_BASE_URL` and `API_KEY`).

## License

No license file is currently included in this repository.
