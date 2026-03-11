"""Compatibility shim that re-exports Genesis-native sensor wrappers.

Historically, all Genesis-native sensors lived in this module. They have now
been split into dedicated submodules, but imports from ``genesis_sensors``
continue to work.
"""

from .genesis_contact_bool_sensor import GenesisContactBoolSensor, GenesisContactBoolSensorCfg
from .genesis_imu_sensor import GenesisImuSensor, GenesisImuSensorCfg
from .genesis_camera_sensor import GenesisCameraSensor, GenesisCameraSensorCfg
from .genesis_lidar_sensor import GenesisLidarSensor, GenesisLidarSensorCfg
from .genesis_depth_camera_sensor import (
    GenesisDepthCameraSensor,
    GenesisDepthCameraSensorCfg,
)

__all__ = [
    "GenesisContactBoolSensor",
    "GenesisContactBoolSensorCfg",
    "GenesisImuSensor",
    "GenesisImuSensorCfg",
    "GenesisCameraSensor",
    "GenesisCameraSensorCfg",
    "GenesisLidarSensor",
    "GenesisLidarSensorCfg",
    "GenesisDepthCameraSensor",
    "GenesisDepthCameraSensorCfg",
]

