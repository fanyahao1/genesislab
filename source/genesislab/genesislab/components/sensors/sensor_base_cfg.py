"""Base configuration class for sensors in GenesisLab."""

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

from genesislab.utils.configclass import configclass

if TYPE_CHECKING:
    from .sensor_base import SensorBase


@configclass
class SensorBaseCfg:
    """Base configuration parameters for a sensor.

    All sensor configuration classes should inherit from this base class.
    """

    class_type: type["SensorBase"] = MISSING
    """The associated sensor class.

    The class should inherit from :class:`SensorBase`.
    """

    name: str = None
    """Logical name of the sensor. If None, the key in SceneCfg.sensors is used."""

    update_period: float = 0.0
    """Update period of the sensor buffers (in seconds). Defaults to 0.0 (update every step)."""

    history_length: int = 0
    """Number of past frames to store in the sensor buffers. Defaults to 0, which means that only
    the current data is stored (no history)."""

    debug_vis: bool = False
    """Whether to visualize the sensor. Defaults to False."""
