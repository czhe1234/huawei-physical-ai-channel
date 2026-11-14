from typing import Tuple

import drjit as dr
import mitsuba as mi


def cpx_sqrt(
    x: Tuple[mi.TensorXf, mi.TensorXf],
) -> Tuple[mi.TensorXf, mi.TensorXf]:

    x_r, x_i = x
    magnitude = dr.safe_sqrt(dr.square(x_r) + dr.square(x_i))
    y_r = dr.safe_sqrt(0.5 * (magnitude + x_r))
    y_i = dr.sign(x_i) * dr.safe_sqrt(0.5 * (magnitude - x_r))
    return y_r, y_i
