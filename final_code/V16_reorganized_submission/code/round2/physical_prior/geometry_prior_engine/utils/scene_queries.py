import drjit as dr
import mitsuba as mi
from typing import Callable

from round2.physical_prior.geometry_prior_engine.constants import EPSILON_FLOAT


def fibonacci_lattice(num_points: int) -> mi.Point2f:

    golden_ratio = (1.0 + dr.sqrt(mi.Float64(5.0))) / 2.0
    ns = dr.arange(mi.Float64, 0, num_points)

    x = ns / golden_ratio
    x = x - dr.floor(x)
    y = ns / (num_points - 1)

    return mi.Point2f(x, y)


def build_directional_scene_queries(
    lattice: Callable[[int], mi.Point2f],
    samples_per_src: int,
    src_positions: mi.Point3f,
) -> mi.Ray3f:

    num_sources = dr.shape(src_positions)[1]

    samples_on_square = lattice(samples_per_src)
    k_world = mi.warp.square_to_uniform_sphere(samples_on_square)

    k_world = dr.tile(k_world, num_sources)

    origins = dr.repeat(src_positions, samples_per_src)

    query = mi.Ray3f(o=origins, d=k_world)

    query.time = dr.zeros(mi.Float, dr.width(k_world))

    return query


def offset_p(p: mi.Point3f, d: mi.Vector3f, n: mi.Vector3f) -> mi.Point3f:


    a = (1.0 + dr.max(dr.abs(p), axis=0)) * EPSILON_FLOAT

    a = dr.detach(dr.mulsign(a, dr.dot(d, n)))
    po = dr.fma(a, n, p)
    return po


def build_segment_scene_query(
    p: mi.Point3f, t: mi.Point3f, n: mi.Vector3f | None = None
) -> mi.Ray3f:

    if n is None:
        po = p
    else:
        po = offset_p(p, t - p, n)

    d = t - po
    maxt = dr.norm(d)
    d /= maxt
    maxt *= 1.0 - EPSILON_FLOAT

    query = mi.Ray3f(po, d, maxt=maxt, time=0.0, wavelengths=mi.Color0f())

    query.time = dr.zeros(mi.Float, dr.width(query.d))
    return query
