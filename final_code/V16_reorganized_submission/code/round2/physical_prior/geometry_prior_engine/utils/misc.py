import drjit as dr
import mitsuba as mi

from .complex import cpx_sqrt


def complex_sqrt(x: mi.Complex2f) -> mi.Complex2f:
    return mi.Complex2f(*cpx_sqrt((dr.real(x), dr.imag(x))))


def isclose(
    a: mi.Float,
    b: mi.Float,
    rtol: mi.Float = 1e-5,
    atol: mi.Float = 1e-8,
) -> mi.Bool:
    close = dr.abs(a - b) < atol + dr.abs(b) * rtol
    close &= ~dr.isinf(b)
    close &= ~(dr.isnan(a) | dr.isnan(b))
    return close


def spectrum_to_matrix_4f(s: mi.Spectrum) -> mi.Matrix4f:
    return mi.Matrix4f(
        s.array.x.x.x,
        s.array.x.y.x,
        s.array.x.z.x,
        s.array.x.w.x,
        s.array.y.x.x,
        s.array.y.y.x,
        s.array.y.z.x,
        s.array.y.w.x,
        s.array.z.x.x,
        s.array.z.y.x,
        s.array.z.z.x,
        s.array.z.w.x,
        s.array.w.x.x,
        s.array.w.y.x,
        s.array.w.z.x,
        s.array.w.w.x,
    )


def safe_atan2(y: mi.Float, x: mi.Float) -> mi.Float:
    both_zero = (x == 0.0) & (y == 0.0)
    return dr.select(both_zero, 0.0, dr.atan2(y, x))
