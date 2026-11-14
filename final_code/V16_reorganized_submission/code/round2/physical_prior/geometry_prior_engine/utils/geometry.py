import drjit as dr
import mitsuba as mi
from typing import Tuple
from round2.physical_prior.geometry_prior_engine.utils.misc import safe_atan2


def theta_hat(theta: mi.Float, phi: mi.Float) -> mi.Vector3f:

    sin_theta, cos_theta = dr.sincos(theta)
    sin_phi, cos_phi = dr.sincos(phi)
    v = mi.Vector3f(cos_theta * cos_phi, cos_theta * sin_phi, -sin_theta)
    return v


def theta_phi_from_unit_vec(v: mi.Vector3f) -> Tuple[mi.Float, mi.Float]:


    z = dr.clip(v.z, -1, 1)
    theta = dr.safe_acos(z)
    phi = safe_atan2(v.y, v.x)
    return theta, phi


def rotation_matrix(angles: mi.Point3f) -> mi.Matrix3f:


    a = angles.x
    b = angles.y
    c = angles.z
    sin_a, cos_a = dr.sincos(a)
    sin_b, cos_b = dr.sincos(b)
    sin_c, cos_c = dr.sincos(c)

    r_11 = cos_a * cos_b
    r_12 = cos_a * sin_b * sin_c - sin_a * cos_c
    r_13 = cos_a * sin_b * cos_c + sin_a * sin_c

    r_21 = sin_a * cos_b
    r_22 = sin_a * sin_b * sin_c + cos_a * cos_c
    r_23 = sin_a * sin_b * cos_c - cos_a * sin_c

    r_31 = -sin_b
    r_32 = cos_b * sin_c
    r_33 = cos_b * cos_c

    rot_mat = mi.Matrix3f([[r_11, r_12, r_13], [r_21, r_22, r_23], [r_31, r_32, r_33]])

    return rot_mat


def mirror_point_across_plane(
    p: mi.Point3f, n: mi.Vector3f | mi.Normal3f, v: mi.Point3f, active: mi.Bool
) -> mi.Point3f:


    p_prime = dr.select(active, p - 2 * dr.dot(p - v, n) * n, p)
    return p_prime
