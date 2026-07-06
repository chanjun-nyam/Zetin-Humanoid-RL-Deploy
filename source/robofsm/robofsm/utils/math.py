import torch as th



def vec_dot(vec1: th.Tensor, vec2: th.Tensor, keepdim: bool = False) -> th.Tensor:
    """Dot product between two vectors.

    Args:
        vec1 (th.Tensor): vector
        vec2 (th.Tensor): vector
        keepdim (bool, optional):
            Whether to keep the dimension of resulting vector.
            Defaults to False.

    Returns:
        th.Tensor: Resulting vector.
    """
    return (vec1 * vec2).sum(dim=-1,keepdim=keepdim)



def vec_norm_pow(vec: th.Tensor, p: float = 2.0, keepdim: bool = False) -> th.Tensor:
    """P-powered p-norm of vector.

    Args:
        vec (th.Tensor): vector
        p (float): p
        keepdim (bool, optional):
            Whether to keep the dimension of resulting vector.
            Defaults to False.

    Returns:
        th.Tensor: Resulting vector.
    """
    return vec.abs().pow(p).sum(dim=-1, keepdim=keepdim)



def vec_sq_norm(vec: th.Tensor, keepdim: bool = False) -> th.Tensor:
    """Squared norm of vector.

    Note: Deprecated. Use `vec_norm_pow` instead.

    Args:
        vec (th.Tensor): vector
        keepdim (bool, optional):
            Whether to keep the dimension of resulting vector.
            Defaults to False.

    Returns:
        th.Tensor: Resulting vector.
    """
    return vec_norm_pow(vec, p=2.0, keepdim=keepdim)



def vec_norm(vec: th.Tensor, p: float = 2.0, keepdim: bool = False) -> th.Tensor:
    """Norm of vector.

    Args:
        vec (th.Tensor): vector
        p (float): p
        keepdim (bool, optional):
            Whether to keep the dimension of resulting vector.
            Defaults to False.

    Returns:
        th.Tensor: Resulting vector.
    """
    return vec_norm_pow(vec, p=p, keepdim=keepdim).pow(1 / p)



def quat_apply(quat: th.Tensor, vec: th.Tensor) -> th.Tensor:
    """Apply quaternion to vector.

    Note:
        - Shape of quaternion is (..., 4) and order of elements is (w, x, y, z).
        - Shape of vector is (..., 3) and order of elements is (x, y, z).

    Args:
        quat (th.Tensor): quaternion to apply
        vec (th.Tensor): vector

    Returns:
        th.Tensor: vector rotated by given quaternion
    """
    quat_v = quat[...,1:]
    quat_w = quat[...,:1]

    uvec = th.cross(quat_v, vec, dim=-1)
    uuvec = th.cross(quat_v, uvec, dim=-1)

    return vec + 2.0 * (quat_w * uvec + uuvec)



def quat_mul(quat1: th.Tensor, quat2: th.Tensor) -> th.Tensor:
    """Multiply two quaternion.

    Note:
        - Shape of quaternion is (..., 4) and order of elements is (w, x, y, z).
        - `quat1` and `quat2` must be broadcastable.

    Args:
        quat1 (th.Tensor): pre-multiplier quaternion
        quat2 (th.Tensor): post-multiplier quaternion

    Returns:
        th.Tensor: Resulting quaternion.
    """
    w1, x1, y1, z1 = th.unbind(quat1, dim=-1)
    w2, x2, y2, z2 = th.unbind(quat2, dim=-1)

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return th.stack((w, x, y, z), dim=-1)



def quat_conj(quat: th.Tensor) -> th.Tensor:
    """Conjugate of quaternion

    Note:
        - Shape of quaternion is (..., 4) and order of elements is (w, x, y, z).

    Args:
        quat (th.Tensor): quaternion

    Returns:
        th.Tensor: Resulting quaternion.
    """
    return th.cat((quat[...,:1], -quat[...,1:]), dim=-1)



def quat_twist(quat: th.Tensor, twist_axis: th.Tensor, eps: float = 1e-6) -> th.Tensor:
    """Get twist components from given quaternion and twist axis.

    This twist quaternion is same with twist quaternion is swing-twist(or twist-swing) decomposition.

    Note:
        - Shape of quaternion is (..., 4) and order of elements is (w, x, y, z).
        - Shape of vector is (..., 3) and order of elements is (x, y, z).
        - `quat[...,1:]` and `twist_axis` must be broadcastable.

    Args:
        quat (th.Tensor): original quaternion
        twist_axis (th.Tensor): unit vector of twist axis.
        eps (float, optional):
            Epsilon to handle singular point.
            Defaults to 1e-6.

    Returns:
        th.Tensor: Resulting twist quaternion.
    """
    proj = (quat[...,1:] * twist_axis).sum(dim=-1,keepdim=True) * twist_axis

    twist_un = th.cat([quat[...,:1], proj], dim=-1)
    twist_un_norm = vec_norm(twist_un, keepdim=True)

    identity_quat = th.zeros_like(quat)
    identity_quat[...,0] = 1.0

    twist = th.where(
        condition=twist_un_norm>eps,
        input=twist_un/twist_un_norm.clamp(min=eps),
        other=identity_quat,
    )
    return twist
