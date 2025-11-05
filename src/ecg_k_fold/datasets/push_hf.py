from __future__ import annotations

from pathlib import Path
from typing import Iterable

from huggingface_hub import HfApi, create_repo


def _iter_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file():
            yield path


def push_local_dataset_to_hub(
    local_dir: Path,
    repo_id: str,
    private: bool = True,
    commit_message: str = "Add/Update dataset",
    hf_token: str | None = None,
) -> None:
    local_dir = Path(local_dir)
    if not local_dir.exists() or not local_dir.is_dir():
        raise ValueError(f"Local dataset directory does not exist: {local_dir}")

    create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True, token=hf_token)
    api = HfApi(token=hf_token)

    api.upload_folder(
        folder_path=str(local_dir),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=commit_message,
        allow_patterns=None,
        ignore_patterns=["*.tmp", "*.log", "__pycache__/**", ".DS_Store"],
    )

