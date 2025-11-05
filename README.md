# ECG-k-Fold

Utilities for creating reproducible k-fold splits for open-access ECG datasets (FAIR compliant), packaged as a Python library with a CLI and CI/CD. Includes an option to push datasets to the Hugging Face Hub.

## Install

```bash
pip install ecg-k-fold
```

For development:

```bash
git clone <this-repo>
cd ECG-k-Fold
make init
```

Or manually:

```bash
git clone <this-repo>
cd ECG-k-Fold
# Install uv if not already installed: curl -LsSf https://astral.sh/uv/install.sh | sh
uv pip install -e .[dev]
uv run pre-commit install
```

## CLI

Push a local dataset directory to the Hugging Face Hub. Requires an access token with write permissions.

```bash
export HF_TOKEN=hf_************************
ecg-k-fold push-dataset ./path/to/dataset \
  --repo-id org_or_user/dataset-name \
  --private \
  --commit-message "Add v1"
```

Flags:
- `--repo-id`: target dataset repo, e.g. `your-org/ecg-dataset`
- `--private/--public`: repository visibility (default private)
- `--commit-message`: commit title for the upload
- `--token`: override `HF_TOKEN` env var

Dataset contents are uploaded as-is; common temp files are ignored.

## Python API

```python
from pathlib import Path
from ecg_k_fold.datasets.push_hf import push_local_dataset_to_hub

push_local_dataset_to_hub(
    local_dir=Path("./path/to/dataset"),
    repo_id="your-org/ecg-dataset",
    private=True,
    commit_message="Initial upload",
    hf_token="hf_...",
)
```

## CI/CD

- Lint, type-check, and tests run on PRs and pushes to `develop` and `main` (`.github/workflows/ci.yml`).
- Publishing to PyPI happens on pushing a tag like `vX.Y.Z` (`.github/workflows/publish.yml`). Uses PyPI Trusted Publishers, no PyPI token needed. See https://docs.pypi.org/trusted-publishers/
- A manual workflow can push datasets to Hugging Face (`.github/workflows/hf-dataset-push.yml`). Set `HF_TOKEN` in repo secrets.

## Gitflow

- Default branches: `main` (stable), `develop` (integration).
- Create feature branches off `develop` (`feature/<name>`), open PRs into `develop`.
- Release by creating a tag `vX.Y.Z` on `main` to trigger PyPI publish.

See `CONTRIBUTING.md` for details.

## License

Apache-2.0. See `LICENSE`.
