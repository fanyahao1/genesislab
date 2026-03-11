"""Depth camera sensor wrapper over ``gs.sensors.DepthCamera``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

from genesislab.utils.configclass import configclass
from .sensor_base import GenesisSensorBase, GenesisSensorBaseCfg
from .genesis_sensor_utils import to_tensor


class GenesisDepthCameraSensor(GenesisSensorBase):
    """Depth camera sensor backed by ``gs.sensors.DepthCamera``.

    Exposes depth images from the underlying Genesis sensor.
    """

    @dataclass
    class Data:
        """Exposed buffers.

        Attributes:
            depth: Tensor of shape (num_envs, H, W).
        """

        depth: torch.Tensor

    def __init__(
        self,
        cfg: "GenesisDepthCameraSensorCfg",
        num_envs: int,
        device: str = "cuda",
        genesis_sensor: Any | None = None,
    ) -> None:
        self._gs_sensor = genesis_sensor
        super().__init__(cfg=cfg, num_envs=num_envs, device=device)

    def set_genesis_sensor(self, genesis_sensor: Any) -> None:
        """Attach the underlying Genesis ``DepthCamera`` sensor."""
        self._gs_sensor = genesis_sensor

    def _initialize_impl(self) -> None:
        super()._initialize_impl()
        self._data = self.Data(depth=torch.empty(0, device=self.device))

    @property
    def data(self) -> "GenesisDepthCameraSensor.Data":
        self._update_outdated_buffers()
        return self._data

    def _extract_depth_tensor(self) -> torch.Tensor:
        """Read depth image(s) from Genesis."""
        if self._gs_sensor is None:
            raise RuntimeError(
                f"GenesisDepthCameraSensor '{self.cfg.name or 'unnamed'}' has no Genesis sensor attached. "
                f"Call set_genesis_sensor() or pass genesis_sensor in the constructor."
            )

        if not hasattr(self._gs_sensor, "read_image"):
            raise AttributeError(
                f"Underlying Genesis depth camera sensor has no 'read_image' method: {type(self._gs_sensor)}"
            )

        depth = self._gs_sensor.read_image()
        depth_t = to_tensor(depth, device=self.device, dtype=torch.float32)

        # Normalize shapes: (H, W) or (num_envs, H, W)
        if depth_t.dim() == 2:
            depth_t = depth_t.unsqueeze(0)
        if depth_t.dim() != 3:
            raise ValueError(
                f"Depth tensor must have shape (num_envs, H, W) or (H, W), got {depth_t.shape}"
            )

        if depth_t.shape[0] != self.num_envs:
            if self.num_envs == 1 and depth_t.shape[0] == 1:
                return depth_t
            raise ValueError(
                f"Depth camera batch size mismatch: expected num_envs={self.num_envs}, got {depth_t.shape[0]}"
            )
        return depth_t

    def _update_buffers_impl(self, env_ids: Sequence[int] | torch.Tensor) -> None:
        depth = self._extract_depth_tensor()
        self._data.depth = depth


@configclass
class GenesisDepthCameraSensorCfg(GenesisSensorBaseCfg):
    """Configuration for :class:`GenesisDepthCameraSensor`."""

    class_type: type = GenesisDepthCameraSensor
    name: str = None

