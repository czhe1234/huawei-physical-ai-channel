"""Warp kernels for the self-built shooting-and-bouncing path sampler."""

from __future__ import annotations

import warp as wp


MAX_TRACE_DEPTH = 16


@wp.func
def oriented_normal(direction: wp.vec3, normal: wp.vec3) -> wp.vec3:
    if wp.dot(direction, normal) > 0.0:
        return -normal
    return normal


@wp.func
def local_to_world(local: wp.vec3, normal: wp.vec3) -> wp.vec3:
    helper = wp.vec3(0.0, 0.0, 1.0)
    if wp.abs(normal[2]) > 0.9:
        helper = wp.vec3(1.0, 0.0, 0.0)
    tangent = wp.normalize(wp.cross(helper, normal))
    bitangent = wp.cross(normal, tangent)
    return wp.normalize(
        tangent * local[0] + bitangent * local[1] + normal * wp.abs(local[2])
    )


@wp.kernel
def trace_paths(
    mesh: wp.uint64,
    face_family: wp.array(dtype=wp.int32),
    source: wp.vec3,
    seed: wp.int32,
    ray_offset: wp.int32,
    max_depth: wp.int32,
    diffuse_probability: wp.float32,
    ground_family: wp.int32,
    epsilon: wp.float32,
    maximum_path_length: wp.float32,
    minimum_energy: wp.float32,
    counts: wp.array(dtype=wp.uint8),
    families: wp.array(dtype=wp.int32),
    interactions: wp.array(dtype=wp.uint8),
    vertices: wp.array(dtype=wp.vec3),
    segment_length: wp.array(dtype=wp.float32),
    incidence: wp.array(dtype=wp.float32),
    energy_after: wp.array(dtype=wp.float32),
    initial_direction: wp.array(dtype=wp.vec3),
    terminal_energy: wp.array(dtype=wp.float32),
    total_length: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    state = wp.rand_init(seed, tid + ray_offset)
    direction = wp.sample_unit_sphere_surface(state)
    initial_direction[tid] = direction
    origin = source
    energy = float(1.0)
    path_length = float(0.0)
    count = int(0)

    for depth in range(MAX_TRACE_DEPTH):
        if depth >= max_depth:
            break
        remaining = maximum_path_length - path_length
        if remaining <= 0.0 or energy < minimum_energy:
            break
        query = wp.mesh_query_ray(mesh, origin, direction, remaining)
        if not query.result:
            break
        hit = origin + direction * query.t
        normal = oriented_normal(direction, wp.normalize(query.normal))
        cosine = wp.clamp(-wp.dot(direction, normal), 0.0, 1.0)
        family = face_family[query.face]
        interaction = wp.uint8(1)
        next_direction = direction - normal * (2.0 * wp.dot(direction, normal))
        mechanism_gain = 0.58 + 0.32 * (1.0 - cosine)

        random_value = wp.randf(state)
        if diffuse_probability > 0.0 and random_value < diffuse_probability:
            interaction = wp.uint8(2)
            next_direction = local_to_world(wp.sample_unit_hemisphere(state), normal)
            mechanism_gain = diffuse_probability * (0.35 + 0.65 * cosine)
        else:
            reflect_probability = wp.clamp(0.48 + 0.42 * (1.0 - cosine), 0.35, 0.92)
            if family != ground_family and wp.randf(state) > reflect_probability:
                interaction = wp.uint8(4)
                next_direction = direction
                mechanism_gain = (1.0 - reflect_probability) * (0.70 + 0.30 * cosine)

        path_length = path_length + query.t
        spreading = 1.0 / (1.0 + 0.00035 * query.t * query.t)
        energy = energy * mechanism_gain * spreading
        index = tid * MAX_TRACE_DEPTH + depth
        families[index] = family
        interactions[index] = interaction
        vertices[index] = hit
        segment_length[index] = query.t
        incidence[index] = cosine
        energy_after[index] = energy
        count = depth + 1
        direction = wp.normalize(next_direction)
        origin = hit + direction * epsilon

    counts[tid] = wp.uint8(count)
    terminal_energy[tid] = energy
    total_length[tid] = path_length


@wp.kernel
def collect_segment_hits(
    mesh: wp.uint64,
    face_family: wp.array(dtype=wp.int32),
    starts: wp.array(dtype=wp.vec3),
    ends: wp.array(dtype=wp.vec3),
    epsilon: wp.float32,
    maximum_hits: wp.int32,
    counts: wp.array(dtype=wp.uint8),
    families: wp.array(dtype=wp.int32),
    vertices: wp.array(dtype=wp.vec3),
    incidence: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    start = starts[tid]
    end = ends[tid]
    vector = end - start
    length = wp.length(vector)
    count = int(0)
    if length > 2.0 * epsilon:
        direction = vector / length
        origin = start + direction * epsilon
        remaining = length - 2.0 * epsilon
        for hit_index in range(MAX_TRACE_DEPTH):
            if hit_index >= maximum_hits or remaining <= epsilon:
                break
            query = wp.mesh_query_ray(mesh, origin, direction, remaining)
            if not query.result:
                break
            hit = origin + direction * query.t
            normal = oriented_normal(direction, wp.normalize(query.normal))
            output = tid * MAX_TRACE_DEPTH + hit_index
            families[output] = face_family[query.face]
            vertices[output] = hit
            incidence[output] = wp.clamp(-wp.dot(direction, normal), 0.0, 1.0)
            count = hit_index + 1
            step = query.t + epsilon
            origin = hit + direction * epsilon
            remaining = remaining - step
    counts[tid] = wp.uint8(count)
