"""Rebuilding a trained engine from a checkpoint (WBS 5).

A checkpoint stores weights, and weights alone do not say what to load them
into: ``load_state_dict`` on the wrong shape raises, and on a *compatible* wrong
shape it succeeds and gives a model that quietly is not the one that was
trained. So the shape travels with the weights -- every checkpoint written by
this project carries its ``model_config`` -- and this module turns that record
back into a network.

The architecture is recognised by the fields present rather than by a name:
``channels`` belongs to the ResNet, ``d_model`` to the transformer. That way a
checkpoint written before any naming convention existed still loads.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ..models.resnet import ChessResNet, ResNetConfig
from ..models.transformer import ChessTransformer, TransformerConfig
from ..training.checkpoint import BEST_NAME, HubCheckpoints, load_checkpoint


class UnknownArchitectureError(ValueError):
    """The checkpoint does not describe an architecture this project knows."""


def build_model(model_config: dict[str, Any]) -> torch.nn.Module:
    """Construct the network a checkpoint's ``model_config`` describes."""
    if "channels" in model_config:
        return ChessResNet(ResNetConfig(**{
            k: v for k, v in model_config.items()
            if k in ResNetConfig.__dataclass_fields__
        }))
    if "d_model" in model_config:
        return ChessTransformer(TransformerConfig(**{
            k: v for k, v in model_config.items()
            if k in TransformerConfig.__dataclass_fields__
        }))
    raise UnknownArchitectureError(
        f"no reconozco la arquitectura: {sorted(model_config)}. "
        "Se esperaba 'channels' (ResNet) o 'd_model' (transformer)."
    )


def load_local(path: str | Path, device: str = "cpu") -> torch.nn.Module:
    """Rebuild and load the model a checkpoint file holds."""
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    model = build_model(payload.get("model_config") or {})
    load_checkpoint(path, model, map_location=device)
    return model


def load_from_hub(
    repo_id: str,
    run_name: str,
    local_dir: str | Path = "./checkpoints",
    token: str | None = None,
    name: str = BEST_NAME,
    device: str = "cpu",
) -> torch.nn.Module:
    """Fetch a run's checkpoint from the Hub and rebuild its model.

    Defaults to the *best* checkpoint rather than the last one, because the last
    epoch is routinely not the best: both architectures reached their minimum
    around two thirds of the way through and got worse afterwards.
    """
    checkpoints = HubCheckpoints(
        repo_id=repo_id, run_name=run_name, local_dir=local_dir, token=token
    )
    path = checkpoints.fetch(name)
    if path is None:
        raise FileNotFoundError(f"no encontre {name} en {repo_id}/{run_name}")
    return load_local(path, device=device)
