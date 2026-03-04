"""Robot configuration for GenesisLab."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from genesislab.utils.configclass import configclass


@configclass
class RobotCfg:
    """Configuration for a robot entity to be added to a Genesis scene.

    This config is deliberately minimal and focused on rigid-body robots loaded
    from URDF/MJCF/USD. More advanced options (materials, surfaces, soft bodies)
    can be added incrementally as needed.
    """

    morph_type: Literal['URDF', 'MJCF', 'USD']
    """Type of morph to use: 'URDF', 'MJCF', 'USD', etc."""

    morph_path: str
    """Path to the robot asset file (URDF, MJCF, USD, etc.)."""

    initial_pose: dict[str, Any] = field(
        default_factory=lambda: {"pos": [0.0, 0.0, 0.0], "quat": [0.0, 0.0, 0.0, 1.0]}
    )
    """Initial pose of the robot. Dict with 'pos' (list[float]) and 'quat' (list[float])."""

    material: dict[str, Any] | None = None
    """Material configuration. If None, uses a default rigid material."""

    surface: dict[str, Any] | None = None
    """Surface configuration for contact. If None, uses a default surface."""

    fixed_base: bool = False
    """Whether the robot base is fixed (non-floating)."""

    # Control configuration
    control_dofs: list[str] | None = None
    """List of joint names to control. If None, all actuated joints are controlled."""

    pd_gains: dict[str, tuple[float, float]] | None = None
    """Optional PD gains per joint name as (kp, kd). If None, env/task may set defaults."""

    # Additional morph options
    morph_options: dict[str, Any] = field(default_factory=dict)
    """Additional options passed to the Genesis morph constructor."""

