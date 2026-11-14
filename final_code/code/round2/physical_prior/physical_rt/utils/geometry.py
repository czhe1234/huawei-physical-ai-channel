"""Geometry utilities."""

import drjit as dr
import mitsuba as mi
from typing import Tuple
from round2.physical_prior.physical_rt.utils.misc import safe_atan2


def theta_hat(theta: mi.Float, phi: mi.Float) -> mi.Vector3f:
    # pylint: disable=line-too-long
    r"""
    Computes the spherical unit vector :math:`\hat{\boldsymbol{\theta}}(\theta, \varphi)`
    as defined in :eq:`spherical_vecs`

    :param theta: Zenith angle :math:`\theta` [rad]
    :param phi: Azimuth angle :math:`\varphi` [rad]
    """
    sin_theta, cos_theta = dr.sincos(theta)
    sin_phi, cos_phi = dr.sincos(phi)
    v = mi.Vector3f(cos_theta*cos_phi,
                    cos_theta*sin_phi,
                     -sin_theta)
    return v

def theta_phi_from_unit_vec(v: mi.Vector3f) -> Tuple[mi.Float, mi.Float]:
    # pylint: disable=line-too-long
    r"""
    Computes zenith and azimuth angles (:math:`\theta,\varphi`)
    from unit-norm vectors as described in :eq:`theta_phi`

    :param v: Unit vector

    :return: Zenith angle :math:`\theta` [rad] and azimuth angle :math:`\varphi` [rad]
    """

    # Clip z for numerical stability
    z = dr.clip(v.z, -1, 1)
    theta = dr.safe_acos(z)
    phi = safe_atan2(v.y, v.x)
    return theta, phi

def rotation_matrix(angles: mi.Point3f) -> mi.Matrix3f:
    # pylint: disable=line-too-long
    r"""
    Computes the rotation matrix as defined in :eq:`rotation`

    The closed-form expression in (7.1-4) :cite:p:`TR38901_RT` is used.

    :param angles: Angles for the rotations :math:`(\alpha,\beta,\gamma)`
        [rad] that define rotations about the axes :math:`(z, y, x)`,
        respectively
    """

    a = angles.x
    b = angles.y
    c = angles.z
    sin_a, cos_a = dr.sincos(a)
    sin_b, cos_b = dr.sincos(b)
    sin_c, cos_c = dr.sincos(c)

    r_11 = cos_a*cos_b
    r_12 = cos_a*sin_b*sin_c - sin_a*cos_c
    r_13 = cos_a*sin_b*cos_c + sin_a*sin_c

    r_21 = sin_a*cos_b
    r_22 = sin_a*sin_b*sin_c + cos_a*cos_c
    r_23 = sin_a*sin_b*cos_c - cos_a*sin_c

    r_31 = -sin_b
    r_32 = cos_b*sin_c
    r_33 = cos_b*cos_c

    rot_mat = mi.Matrix3f([[r_11, r_12, r_13],
                           [r_21, r_22, r_23],
                           [r_31, r_32, r_33]])

    return rot_mat

def point_plane_reflection(p: mi.Point3f,
                           n: mi.Vector3f | mi.Normal3f,
                           v: mi.Point3f,
                           active: mi.Bool) -> mi.Point3f:
    # pylint: disable=line-too-long
    r"""
    Computes the mirror image of a point with respect to a plane

    Given a point :math:`\mathbf{p}`, a plane with normal :math:`\mathbf{n}`,
    and a point :math:`\mathbf{v}` on the plane, this function computes the
    mirror image of the point using:

    .. math::
        \mathbf{p}' = \mathbf{p} - 2((\mathbf{p} - \mathbf{v}) \cdot \mathbf{n})\mathbf{n}

    :param p: Point to mirror :math:`\mathbf{p}`
    :param n: Unit normal vector of the plane :math:`\mathbf{n}`
    :param v: Point on the plane :math:`\mathbf{v}`
    :param active: Boolean mask indicating which elements to process

    :return: Mirrored point :math:`\mathbf{p}'`
    """

    p_prime = dr.select(active,
                        p - 2 * dr.dot(p - v, n) * n,
                        p)
    return p_prime
