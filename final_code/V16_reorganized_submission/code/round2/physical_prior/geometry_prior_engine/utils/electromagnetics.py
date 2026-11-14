import drjit as dr
import mitsuba as mi
from typing import Tuple
from scipy.constants import epsilon_0

from .misc import complex_sqrt


def complex_dielectric_response(
    eta_r: mi.Float, sigma: mi.Float, omega: mi.Float
) -> mi.Complex2f:

    eta_i = sigma * dr.rcp(omega * epsilon_0)
    eta = mi.Complex2f(eta_r, -eta_i)
    return eta


def interface_reflection_coefficients(
    cos_theta: mi.Float, eta: mi.Complex2f
) -> Tuple[mi.Complex2f, mi.Complex2f]:


    sin_theta_sqr = 1.0 - cos_theta * cos_theta
    a = complex_sqrt(eta - sin_theta_sqr)

    r_te = (cos_theta - a) * dr.rcp(cos_theta + a)

    r_tm = (eta * cos_theta - a) * dr.rcp(eta * cos_theta + a)

    return r_te, r_tm


def finite_layer_coefficients(
    cos_theta: mi.Float, eta: mi.Complex2f, d: mi.Float, wavelength: mi.Float
) -> Tuple[mi.Complex2f, mi.Complex2f, mi.Complex2f, mi.Complex2f]:


    sin_theta_sqr = 1.0 - dr.square(cos_theta)

    q = dr.two_pi * d * dr.rcp(wavelength) * complex_sqrt(eta - sin_theta_sqr)

    r_te_p, r_tm_p = interface_reflection_coefficients(cos_theta, eta)

    r_te_p_sqr = dr.square(r_te_p)
    r_tm_p_sqr = dr.square(r_tm_p)

    exp_j_q = dr.exp(mi.Complex2f(0.0, -1.0) * q)
    exp_j_2q = dr.exp(mi.Complex2f(0.0, -2.0) * q)

    denom_te = 1.0 - r_te_p_sqr * exp_j_2q
    inv_denom_te = dr.rcp(denom_te)
    denom_tm = 1.0 - r_tm_p_sqr * exp_j_2q
    inv_denom_tm = dr.rcp(denom_tm)

    r_te = r_te_p * (1.0 - exp_j_2q) * inv_denom_te
    r_tm = r_tm_p * (1.0 - exp_j_2q) * inv_denom_tm

    t_te = (1.0 - r_te_p_sqr) * exp_j_q * inv_denom_te
    t_tm = (1.0 - r_tm_p_sqr) * exp_j_q * inv_denom_tm

    return r_te, r_tm, t_te, t_tm
