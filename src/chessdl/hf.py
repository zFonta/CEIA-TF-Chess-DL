"""Hugging Face Hub storage for the dataset shards.

Shards are pushed to the Hub as soon as they are finished. That is what makes
them durable: the Colab disk is temporary, so a shard that exists only under
``/content`` is one disconnect away from being lost.

The namespace comes from configuration (``HF_NAMESPACE``, defaulting to the
project owner) and the token from the environment or Colab's secret store --
never from a file in the repository.
"""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

from chessdl.colab import get_secret

TOKEN_ENV_VAR = "HF_TOKEN"


class MissingTokenError(RuntimeError):
    """Raised when an upload is attempted without a Hugging Face token."""


def get_token(required: bool = False) -> str | None:
    """Fetch the Hugging Face token from Colab secrets or the environment."""
    token = get_secret(TOKEN_ENV_VAR)
    if token is None and required:
        raise MissingTokenError(
            f"No Hugging Face token found. Set the {TOKEN_ENV_VAR} environment "
            "variable, or add it to Colab's secrets panel and grant this "
            "notebook access."
        )
    return token


def ensure_repo(repo_id: str, token: str | None = None, private: bool = False) -> str:
    """Create the dataset repository if it does not exist yet."""
    api = HfApi(token=token or get_token(required=True))
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    return repo_id


def upload_shard(
    path: str | Path,
    repo_id: str,
    path_in_repo: str | None = None,
    token: str | None = None,
) -> str:
    """Upload one shard, returning the path it took inside the repository."""
    local = Path(path)
    target = path_in_repo or f"data/{local.name}"
    api = HfApi(token=token or get_token(required=True))
    api.upload_file(
        path_or_fileobj=str(local),
        path_in_repo=target,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"Add {local.name}",
    )
    return target


def upload_file(
    path: str | Path,
    repo_id: str,
    path_in_repo: str,
    token: str | None = None,
    commit_message: str | None = None,
) -> str:
    """Upload an arbitrary file (a dataset card, a config snapshot) to the repo."""
    api = HfApi(token=token or get_token(required=True))
    api.upload_file(
        path_or_fileobj=str(Path(path)),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=commit_message or f"Update {path_in_repo}",
    )
    return path_in_repo


def download_dataset(
    repo_id: str,
    local_dir: str | Path,
    token: str | None = None,
) -> Path:
    """Download every shard of the dataset (requirement 1.2: reloadable data)."""
    path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(local_dir),
        token=token or get_token(),
    )
    return Path(path)
