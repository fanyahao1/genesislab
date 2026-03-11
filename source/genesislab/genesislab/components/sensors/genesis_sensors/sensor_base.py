"""Base class for sensors in GenesisLab.

This class defines an interface for sensors similar to how the AssetBase class works.
Each sensor class should inherit from this class and implement the abstract methods.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import torch

from genesislab.utils.configclass import configclass

from ..sensor_base import SensorBase, SensorBaseCfg

class GenesisSensorBase(SensorBase):
    cfg: "GenesisSensorBaseCfg"
    pass

@configclass
class GenesisSensorBaseCfg(SensorBaseCfg):
    pass