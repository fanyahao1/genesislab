"""Simple contact sensor implementations for GenesisLab.

These are lightweight, Python-side sensors that mimic the IsaacLab contact
sensor interfaces used in the locomotion velocity tasks. They do **not**
currently read real contact forces from the Genesis engine, but they expose
the same attributes so that MDP terms can be wired up without runtime errors.

Once Genesis exposes contact-force APIs, these sensors can be extended to
populate real data instead of zeros.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from genesislab.utils.configclass import configclass


@configclass
class ContactSensorCfg:
    """Configuration for a contact sensor.

    Attributes:
        name: Logical name of the sensor. If None, the key in SceneCfg.sensors
            is used.
        entity_name: Name of the articulated entity to which this sensor is
            conceptually attached (typically ``"robot"``).
        history_length: Number of past steps of contact forces to keep.
        track_air_time: Whether to track a simple notion of link "air time".
            Currently this is a no-op placeholder kept for API compatibility.
    """

    name: str | None = None
    entity_name: str = "robot"
    history_length: int = 3
    track_air_time: bool = True


@dataclass
class _ContactSensorData:
    """Data buffers exposed by :class:`ContactSensor`.

    For now, these are zero tensors with the right shapes. The fields mirror
    those used in the MDP reward/termination functions:

    - ``net_forces_w_history``: Tensor of shape
      ``(history_length, num_envs, num_channels, 3)``.
    - ``last_air_time``: Tensor of shape ``(num_envs, num_channels)``.
    """

    net_forces_w_history: torch.Tensor
    last_air_time: torch.Tensor


class ContactSensor:
    """Lightweight contact sensor stub for GenesisLab.

    The sensor keeps per-environment buffers of contact-force history and
    "air time". At the moment, these buffers are **not** populated from the
    engine and remain zeros, but the object shape and attributes are aligned
    with IsaacLab's contact sensor so that MDP code can be wired without
    additional guards.
    """

    def __init__(self, cfg: ContactSensorCfg, num_envs: int, device: str = "cuda"):
        self.cfg = cfg
        self.num_envs = int(num_envs)
        self.device = device

        # We keep a single "channel" per sensor for now. Once Genesis exposes
        # per-link contact information we can expand this dimension.
        history_len = max(int(cfg.history_length), 1)
        num_channels = 1

        net_forces = torch.zeros(
            history_len,
            self.num_envs,
            num_channels,
            3,
            device=device,
            dtype=torch.float32,
        )
        last_air_time = torch.zeros(
            self.num_envs,
            num_channels,
            device=device,
            dtype=torch.float32,
        )
        self.data = _ContactSensorData(
            net_forces_w_history=net_forces,
            last_air_time=last_air_time,
        )

    def update(self, dt: float) -> None:
        """Update the sensor data for a simulation step.

        Args:
            dt: Physics time-step in seconds.

        Note:
            This method currently only advances "time" for air-time tracking
            and maintains zero contact forces. It exists so that future
            implementations can hook into Genesis contact APIs without
            touching the rest of the code.
        """
        # Roll history along the time dimension and keep zeros.
        self.data.net_forces_w_history = torch.roll(
            self.data.net_forces_w_history, shifts=-1, dims=0
        )
        self.data.net_forces_w_history[-1, ...] = 0.0

        # Very simple placeholder: just accumulate air time everywhere.
        # Downstream MDP terms already guard against using these buffers
        # in real training by returning zeros.
        self.data.last_air_time += dt

    # IsaacLab-style helper used in feet_air_time (kept for future use).
    def compute_first_contact(self, step_dt: float) -> torch.Tensor:
        """Placeholder for first-contact computation.

        Returns:
            A boolean tensor of shape ``(num_envs, num_channels)`` indicating
            whether a first contact was detected at the current step.
        """
        return torch.zeros(
            self.num_envs,
            self.data.last_air_time.shape[1],
            dtype=torch.bool,
            device=self.device,
        )

