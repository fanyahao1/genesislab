from __future__ import annotations

import torch


def _normalize_quat(quat: torch.Tensor) -> torch.Tensor:
    """Normalize quaternion assumed in [x, y, z, w] format."""
    return quat / torch.norm(quat, dim=-1, keepdim=True).clamp_min(1e-8)


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Quaternion multiplication for [x, y, z, w] format."""
    q1 = _normalize_quat(q1)
    q2 = _normalize_quat(q2)
    x1, y1, z1, w1 = q1.unbind(-1)
    x2, y2, z2, w2 = q2.unbind(-1)

    # Convert to (w, x, y, z)
    w1_, x1_, y1_, z1_ = w1, x1, y1, z1
    w2_, x2_, y2_, z2_ = w2, x2, y2, z2

    w = w1_ * w2_ - x1_ * x2_ - y1_ * y2_ - z1_ * z2_
    x = w1_ * x2_ + x1_ * w2_ + y1_ * z2_ - z1_ * y2_
    y = w1_ * y2_ - x1_ * z2_ + y1_ * w2_ + z1_ * x2_
    z = w1_ * z2_ + x1_ * y2_ - y1_ * x2_ + z1_ * w2_

    # Back to [x, y, z, w]
    return torch.stack([x, y, z, w], dim=-1)


def quat_inv(quat: torch.Tensor) -> torch.Tensor:
    """Quaternion inverse for [x, y, z, w] format."""
    quat = _normalize_quat(quat)
    x, y, z, w = quat.unbind(-1)
    return torch.stack([-x, -y, -z, w], dim=-1)


def quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Rotate vector(s) by quaternion(s) (both broadcastable).

    quat: (..., 4) in [x, y, z, w]
    vec:  (..., 3)
    """
    quat = _normalize_quat(quat)
    x, y, z, w = quat.unbind(-1)
    xyz = torch.stack([x, y, z], dim=-1)
    v = vec
    # v' = v + 2*w*cross(xyz, v) + 2*cross(xyz, cross(xyz, v))
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + w.unsqueeze(-1) * t + torch.cross(xyz, t, dim=-1)


def quat_apply_inverse(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Rotate vector(s) by inverse quaternion(s)."""
    quat = _normalize_quat(quat)
    x, y, z, w = quat.unbind(-1)
    xyz = torch.stack([x, y, z], dim=-1)
    v = vec
    # quat_apply_inverse: v' = v - 2*w*cross(xyz, v) + 2*cross(xyz, cross(xyz, v))
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v - w.unsqueeze(-1) * t + torch.cross(xyz, t, dim=-1)


def quat_from_euler_xyz(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """Create quaternion [x, y, z, w] from XYZ Euler angles (radians)."""
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return torch.stack([x, y, z, w], dim=-1)


def yaw_quat(yaw: torch.Tensor) -> torch.Tensor:
    """Quaternion [x, y, z, w] for pure yaw rotation about +Z."""
    zero = torch.zeros_like(yaw)
    return quat_from_euler_xyz(zero, zero, yaw)


def quat_error_magnitude(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Angular distance between two quaternions (radians)."""
    dq = quat_mul(quat_inv(q1), q2)
    dq = _normalize_quat(dq)
    # dq = [x, y, z, w]
    w = dq[..., 3].clamp(-1.0, 1.0)
    return 2.0 * torch.acos(torch.abs(w))


def sample_uniform(
    low: float | torch.Tensor,
    high: float | torch.Tensor,
    shape: tuple[int, ...],
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """Uniform sampler mirroring IsaacLab's ``sample_uniform`` helper."""
    low_t = torch.as_tensor(low, dtype=torch.float32, device=device)
    high_t = torch.as_tensor(high, dtype=torch.float32, device=device)
    return low_t + (high_t - low_t) * torch.rand(shape, device=device)


def matrix_from_quat(quat: torch.Tensor) -> torch.Tensor:
    """Convert quaternion [x, y, z, w] to rotation matrix (..., 3, 3)."""
    quat = _normalize_quat(quat)
    x, y, z, w = quat.unbind(-1)

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    m00 = 1.0 - 2.0 * (yy + zz)
    m01 = 2.0 * (xy - wz)
    m02 = 2.0 * (xz + wy)

    m10 = 2.0 * (xy + wz)
    m11 = 1.0 - 2.0 * (xx + zz)
    m12 = 2.0 * (yz - wx)

    m20 = 2.0 * (xz - wy)
    m21 = 2.0 * (yz + wx)
    m22 = 1.0 - 2.0 * (xx + yy)

    return torch.stack(
        [
            torch.stack([m00, m01, m02], dim=-1),
            torch.stack([m10, m11, m12], dim=-1),
            torch.stack([m20, m21, m22], dim=-1),
        ],
        dim=-2,
    )


def subtract_frame_transforms(
    pos0: torch.Tensor,
    quat0: torch.Tensor,
    pos1: torch.Tensor,
    quat1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Relative transform from frame-0 to frame-1.

    Returns (pos_rel, quat_rel) such that:
        T_rel = inv(T0) * T1
    """
    quat0_inv = quat_inv(quat0)
    pos_rel = quat_apply(quat0_inv, pos1 - pos0)
    quat_rel = quat_mul(quat0_inv, quat1)
    return pos_rel, quat_rel


__all__ = [
    "quat_mul",
    "quat_inv",
    "quat_apply",
    "quat_apply_inverse",
    "quat_from_euler_xyz",
    "yaw_quat",
    "quat_error_magnitude",
    "sample_uniform",
    "matrix_from_quat",
    "subtract_frame_transforms",
]

