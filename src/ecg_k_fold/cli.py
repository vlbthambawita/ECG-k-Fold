import os
from pathlib import Path
import click

from ecg_k_fold.datasets.push_hf import push_local_dataset_to_hub


@click.group(help="ECG k-Fold CLI utilities")
def app() -> None:
    pass


@app.command("push-dataset", help="Push a local dataset directory to the Hugging Face Hub")
@click.argument("local_path", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--repo-id", required=True, help="Target dataset repo id, e.g. org/name")
@click.option(
    "--private/--public",
    "privacy",
    default=True,
    help="Create as private (default) or public dataset",
)
@click.option(
    "--commit-message",
    default="Add/Update dataset",
    show_default=True,
    help="Commit message for the push",
)
@click.option(
    "--token",
    envvar="HF_TOKEN",
    default=None,
    help="Hugging Face token (uses HF_TOKEN from env if not provided)",
)
def push_dataset(
    local_path: Path, repo_id: str, privacy: bool, commit_message: str, token: str | None
) -> None:
    hf_token = token or os.getenv("HF_TOKEN")
    if not hf_token:
        raise click.ClickException("HF token not provided. Set --token or HF_TOKEN env var.")

    push_local_dataset_to_hub(
        local_dir=local_path,
        repo_id=repo_id,
        private=privacy,
        commit_message=commit_message,
        hf_token=hf_token,
    )
    click.echo(f"Pushed dataset from {local_path} to {repo_id}")

