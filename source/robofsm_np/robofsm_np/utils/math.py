import numpy as np


def vec_dot(vec1: np.ndarray, vec2: np.ndarray, keepdim: bool = False) -> np.ndarray:
    """Dot product between two vectors.

    Args:
        vec1 (np.ndarray): vector
        vec2 (np.ndarray): vector
        keepdim (bool, optional):
            Whether to keep the dimension of resulting vector.
            Defaults to False.

    Returns:
        np.ndarray: Resulting vector.
    """
    return (vec1 * vec2).sum(axis=-1, keepdims=keepdim)


def vec_norm_pow(vec: np.ndarray, p: float = 2.0, keepdim: bool = False) -> np.ndarray:
    """P-powered p-norm of vector.

    Args:
        vec (np.ndarray): vector
        p (float): p
        keepdim (bool, optional):
            Whether to keep the dimension of resulting vector.
            Defaults to False.

    Returns:
        np.ndarray: Resulting vector.
    """
    return (np.abs(vec) ** p).sum(axis=-1, keepdims=keepdim)


def vec_sq_norm(vec: np.ndarray, keepdim: bool = False) -> np.ndarray:
    """Squared norm of vector.

    Note: Deprecated. Use `vec_norm_pow` instead.

    Args:
        vec (np.ndarray): vector
        keepdim (bool, optional):
            Whether to keep the dimension of resulting vector.
            Defaults to False.

    Returns:
        np.ndarray: Resulting vector.
    """
    return vec_norm_pow(vec, p=2.0, keepdim=keepdim)


def vec_norm(vec: np.ndarray, p: float = 2.0, keepdim: bool = False) -> np.ndarray:
    """Norm of vector.

    Args:
        vec (np.ndarray): vector
        p (float): p
        keepdim (bool, optional):
            Whether to keep the dimension of resulting vector.
            Defaults to False.

    Returns:
        np.ndarray: Resulting vector.
    """
    return vec_norm_pow(vec, p=p, keepdim=keepdim) ** (1.0 / p)


def quat_apply(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Apply quaternion to vector.

    Note:
        - Shape of quaternion is (..., 4) and order of elements is (w, x, y, z).
        - Shape of vector is (..., 3) and order of elements is (x, y, z).

    Args:
        quat (np.ndarray): quaternion to apply
        vec (np.ndarray): vector

    Returns:
        np.ndarray: vector rotated by given quaternion
    """
    quat_v = quat[..., 1:]
    quat_w = quat[..., :1]

    uvec = np.cross(quat_v, vec, axis=-1)
    uuvec = np.cross(quat_v, uvec, axis=-1)

    return vec + 2.0 * (quat_w * uvec + uuvec)


def quat_mul(quat1: np.ndarray, quat2: np.ndarray) -> np.ndarray:
    """Multiply two quaternion.

    Note:
        - Shape of quaternion is (..., 4) and order of elements is (w, x, y, z).
        - `quat1` and `quat2` must be broadcastable.

    Args:
        quat1 (np.ndarray): pre-multiplier quaternion
        quat2 (np.ndarray): post-multiplier quaternion

    Returns:
        np.ndarray: Resulting quaternion.
    """
    w1, x1, y1, z1 = quat1[..., 0], quat1[..., 1], quat1[..., 2], quat1[..., 3]
    w2, x2, y2, z2 = quat2[..., 0], quat2[..., 1], quat2[..., 2], quat2[..., 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return np.stack((w, x, y, z), axis=-1)


def quat_conj(quat: np.ndarray) -> np.ndarray:
    """Conjugate of quaternion

    Note:
        - Shape of quaternion is (..., 4) and order of elements is (w, x, y, z).

    Args:
        quat (np.ndarray): quaternion

    Returns:
        np.ndarray: Resulting quaternion.
    """
    return np.concatenate((quat[..., :1], -quat[..., 1:]), axis=-1)


def quat_twist(quat: np.ndarray, twist_axis: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Get twist components from given quaternion and twist axis.

    This twist quaternion is same with twist quaternion in swing-twist
    (or twist-swing) decomposition.

    Note:
        - Shape of quaternion is (..., 4) and order of elements is (w, x, y, z).
        - Shape of vector is (..., 3) and order of elements is (x, y, z).
        - `quat[..., 1:]` and `twist_axis` must be broadcastable.

    Args:
        quat (np.ndarray): original quaternion
        twist_axis (np.ndarray): unit vector of twist axis.
        eps (float, optional):
            Epsilon to handle singular point.
            Defaults to 1e-6.

    Returns:
        np.ndarray: Resulting twist quaternion.
    """
    proj = (quat[..., 1:] * twist_axis).sum(axis=-1, keepdims=True) * twist_axis

    twist_un = np.concatenate([quat[..., :1], proj], axis=-1)
    twist_un_norm = vec_norm(twist_un, keepdim=True)

    identity_quat = np.zeros_like(quat)
    identity_quat[..., 0] = 1.0

    twist = np.where(
        twist_un_norm > eps,
        twist_un / np.clip(twist_un_norm, eps, None),
        identity_quat,
    )
    return twist
