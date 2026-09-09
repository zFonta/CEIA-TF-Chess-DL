"""Hugging Face Hub storage for everything the pipeline needs to keep.

The Hub is the only durable location: the Colab disk is temporary, so anything
that exists only under ``/content`` is one disconnect away from being lost. Two
repositories are used -- one for the labelled shards, which are the deliverable,
and one for the pipeline's working data (the filtered extract and the resume
state), which keeps the dataset repository free of anything but the data.

The namespace comes from configuration (``HF_NAMESPACE``, defaulting to the
project owner) and the token from the environment or Colab's secret store --
never from a file in the repository.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from huggingface_hub.errors import (
    EntryNotFoundError,
    HfHubHTTPError,
    RepositoryNotFoundError,
)

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


def file_exists(repo_id: str, path_in_repo: str, token: str | None = None) -> bool:
    """Whether a file is present in the repository.

    A missing repository counts as a missing file: on a first run neither exists
    yet, and that is not an error.
    """
    api = HfApi(token=token or get_token())
    try:
        return api.file_exists(
            repo_id=repo_id, filename=path_in_repo, repo_type="dataset"
        )
    except (RepositoryNotFoundError, HfHubHTTPError):
        return False


def download_file(
    repo_id: str,
    path_in_repo: str,
    local_path: str | Path,
    token: str | None = None,
) -> Path | None:
    """Fetch one file from the repository, or return None if it is not there.

    The file is copied to ``local_path`` rather than left in the Hub cache, so
    the rest of the pipeline works with plain paths and does not have to know
    where the download came from.
    """
    target = Path(local_path)
    try:
        cached = hf_hub_download(
            repo_id=repo_id,
            filename=path_in_repo,
            repo_type="dataset",
            token=token or get_token(),
        )
    except (RepositoryNotFoundError, EntryNotFoundError, HfHubHTTPError):
        return None

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(cached, target)
    return target
