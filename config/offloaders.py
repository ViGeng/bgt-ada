"""Compatibility view over offloaders used by the default approach catalog."""

from __future__ import annotations

from .schema import OffloaderConfig


def default_offloaders() -> dict[str, OffloaderConfig]:
    """Return unique offloader configs referenced by default approaches."""
    from .approaches import default_approaches

    offloaders: dict[str, OffloaderConfig] = {}
    for approach in default_approaches():
        offloader = approach.offloader
        if offloader.name in offloaders:
            continue
        offloaders[offloader.name] = OffloaderConfig(
            name=offloader.name,
            policy_id=offloader.policy_id,
            params=dict(offloader.params),
        )
    return offloaders

