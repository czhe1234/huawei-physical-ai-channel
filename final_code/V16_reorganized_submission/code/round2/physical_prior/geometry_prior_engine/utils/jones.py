import drjit as dr
import mitsuba as mi

from .geometry import theta_phi_from_unit_vec, theta_hat
from .misc import isclose


def implicit_basis_vector(k: mi.Vector3f) -> mi.Vector3f:


    theta, phi = theta_phi_from_unit_vec(k)
    v = theta_hat(theta, phi)
    return v


def jones_matrix_rotator(
    k: mi.Vector3f, s_current: mi.Vector3f, s_target: mi.Vector3f
) -> mi.Matrix2f:

    c = dr.dot(s_current, s_target)
    s = dr.dot(k, dr.cross(s_current, s_target))

    rotator = mi.Matrix2f([[c, s], [-s, c]])
    return rotator


def jones_matrix_rotator_flip_forward(k: mi.Vector3f) -> mi.Matrix2f:


    s_current = implicit_basis_vector(k)
    s_target = implicit_basis_vector(-k)
    a = dr.dot(s_current, s_target)
    rotator = mi.Matrix2f(a, 0.0, 0.0, -a)
    return rotator


def to_world_jones_rotator(to_world: mi.Matrix3f, k_local: mi.Vector3f) -> mi.Matrix2f:


    k_world = to_world @ k_local

    s_current = implicit_basis_vector(k_local)
    s_current = to_world @ s_current

    s_target = implicit_basis_vector(k_world)

    rotator = jones_matrix_rotator(k_world, s_current, s_target)

    return rotator


def jones_matrix_to_world_implicit(
    c1: mi.Complex2f,
    c2: mi.Complex2f,
    to_world: mi.Matrix3f,
    k_in_local: mi.Vector3f,
    k_out_local: mi.Vector3f,
) -> mi.Matrix4f:


    normal_incidence = isclose(k_in_local.z, mi.Float(-1.0))
    si_target_local = mi.Vector3f(k_in_local.y, -k_in_local.x, 0.0)
    si_target_local = dr.normalize(si_target_local)
    si_target_local = dr.select(
        normal_incidence, mi.Vector3f(1.0, 0.0, 0.0), si_target_local
    )
    so_current_local = si_target_local

    k_in_world = to_world @ k_in_local
    k_out_world = to_world @ k_out_local

    to_local = to_world.T
    si_current_world = implicit_basis_vector(k_in_world)
    si_current_local = to_local @ si_current_world
    in_rotator = jones_matrix_rotator(k_in_local, si_current_local, si_target_local)

    so_current_world = to_world @ so_current_local
    so_target_world = implicit_basis_vector(k_out_world)
    out_rotator = jones_matrix_rotator(k_out_world, so_current_world, so_target_world)

    c1_real = c1.real
    c1_imag = c1.imag

    c2_real = c2.real
    c2_imag = c2.imag

    real = mi.Matrix2f(
        c1_real * out_rotator[0, 0],
        c2_real * out_rotator[0, 1],
        c1_real * out_rotator[1, 0],
        c2_real * out_rotator[1, 1],
    )
    real @= in_rotator

    imag = mi.Matrix2f(
        c1_imag * out_rotator[0, 0],
        c2_imag * out_rotator[0, 1],
        c1_imag * out_rotator[1, 0],
        c2_imag * out_rotator[1, 1],
    )
    imag @= in_rotator

    return mi.Matrix4f(
        real[0, 0],
        real[0, 1],
        -imag[0, 0],
        -imag[0, 1],
        real[1, 0],
        real[1, 1],
        -imag[1, 0],
        -imag[1, 1],
        imag[0, 0],
        imag[0, 1],
        real[0, 0],
        real[0, 1],
        imag[1, 0],
        imag[1, 1],
        real[1, 0],
        real[1, 1],
    )


def jones_vec_dot(u: mi.Vector4f, v: mi.Vector4f) -> mi.Complex2f:


    u_real = mi.Vector2f(u.x, u.y)
    u_imag = mi.Vector2f(u.z, u.w)

    v_real = mi.Vector2f(v.x, v.y)
    v_imag = mi.Vector2f(v.z, v.w)

    a_real = dr.dot(u_real, v_real) + dr.dot(u_imag, v_imag)
    a_imag = dr.dot(u_real, v_imag) - dr.dot(u_imag, v_real)
    a = mi.Complex2f(a_real, a_imag)
    return a
