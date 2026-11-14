"""Warp kernels for the source-aligned physical propagation solver."""

from __future__ import annotations

import warp as wp


MAX_TRACE_DEPTH = 16
INTERACTION_SPECULAR = wp.uint8(1)
INTERACTION_DIFFUSE = wp.uint8(2)
INTERACTION_REFRACTION = wp.uint8(4)
MIN_SEGMENT_LENGTH = 0.01
PI = 3.141592653589793


@wp.func
def table_value(table: wp.array(dtype=wp.float32), cosine: float) -> float:
    count = table.shape[0]
    coordinate = wp.clamp(cosine, 0.0, 1.0) * float(count - 1)
    lower = int(wp.floor(coordinate))
    upper = wp.min(lower + 1, count - 1)
    fraction = coordinate - float(lower)
    return table[lower] * (1.0 - fraction) + table[upper] * fraction


@wp.func
def fibonacci_direction(index: int, count: int) -> wp.vec3:
    golden_ratio = (1.0 + wp.sqrt(5.0)) * 0.5
    x = float(index) / golden_ratio
    x = x - wp.floor(x)
    y = float(index) / float(wp.max(count - 1, 1))
    z = 1.0 - 2.0 * y
    radius = wp.sqrt(wp.max(1.0 - z * z, 0.0))
    angle = 2.0 * PI * x
    return wp.vec3(radius * wp.cos(angle), radius * wp.sin(angle), z)


@wp.func
def oriented_normal(direction: wp.vec3, normal: wp.vec3) -> wp.vec3:
    result = wp.normalize(normal)
    if wp.dot(direction, result) > 0.0:
        result = -result
    return result


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


@wp.func
def local_coordinates(direction: wp.vec3, look: wp.vec3) -> wp.vec3:
    forward = wp.normalize(look)
    alpha = wp.atan2(forward[1], forward[0])
    horizontal = wp.sqrt(
        wp.max(forward[0] * forward[0] + forward[1] * forward[1], 0.0)
    )
    theta = wp.atan2(horizontal, forward[2])
    beta = theta - 0.5 * PI
    sin_alpha = wp.sin(alpha)
    cos_alpha = wp.cos(alpha)
    sin_beta = wp.sin(beta)
    cos_beta = wp.cos(beta)
    axis_x = wp.vec3(cos_alpha * cos_beta, sin_alpha * cos_beta, -sin_beta)
    axis_y = wp.vec3(-sin_alpha, cos_alpha, 0.0)
    axis_z = wp.vec3(cos_alpha * sin_beta, sin_alpha * sin_beta, cos_beta)
    return wp.vec3(
        wp.dot(direction, axis_x),
        wp.dot(direction, axis_y),
        wp.dot(direction, axis_z),
    )


@wp.func
def tr38901_power(direction: wp.vec3, look: wp.vec3) -> float:
    local = local_coordinates(wp.normalize(direction), look)
    theta = wp.acos(wp.clamp(local[2], -1.0, 1.0))
    phi = wp.atan2(local[1], local[0])
    theta_3db = 65.0 / 180.0 * PI
    phi_3db = theta_3db
    vertical = -wp.min(12.0 * wp.pow((theta - 0.5 * PI) / theta_3db, 2.0), 30.0)
    horizontal_loss = -wp.min(12.0 * wp.pow(phi / phi_3db, 2.0), 30.0)
    gain_db = -wp.min(-(vertical + horizontal_loss), 30.0) + 8.0
    return wp.pow(10.0, gain_db / 10.0)


@wp.func
def dipole_power(direction: wp.vec3, look: wp.vec3) -> float:
    local = local_coordinates(wp.normalize(direction), look)
    sine_squared = wp.max(1.0 - local[2] * local[2], 0.0)
    return 1.5 * sine_squared


@wp.kernel
def trace_source_paths(
    mesh: wp.uint64,
    face_plane: wp.array(dtype=wp.int32),
    source: wp.vec3,
    seed: wp.int32,
    ray_offset: wp.int32,
    samples_per_source: wp.int32,
    max_depth: wp.int32,
    scattering_coefficient: wp.float32,
    ray_epsilon: wp.float32,
    specular_probability: wp.array(dtype=wp.float32),
    diffuse_probability: wp.array(dtype=wp.float32),
    transmission_probability: wp.array(dtype=wp.float32),
    reflection_power: wp.array(dtype=wp.float32),
    transmission_power: wp.array(dtype=wp.float32),
    counts: wp.array(dtype=wp.uint8),
    planes: wp.array(dtype=wp.int32),
    faces: wp.array(dtype=wp.int32),
    interactions: wp.array(dtype=wp.uint8),
    vertices: wp.array(dtype=wp.vec3),
    incidence: wp.array(dtype=wp.float32),
    event_pdf: wp.array(dtype=wp.float32),
    prefix_power: wp.array(dtype=wp.float32),
    initial_direction: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    global_index = tid + ray_offset
    state = wp.rand_init(seed, global_index)
    direction = fibonacci_direction(global_index, samples_per_source)
    initial_direction[tid] = direction
    origin = source
    power = float(1.0)
    solid_angle = 4.0 * PI / float(samples_per_source)
    count = int(0)
    scatter_squared = scattering_coefficient * scattering_coefficient

    for depth in range(MAX_TRACE_DEPTH):
        if depth >= max_depth:
            break
        query = wp.mesh_query_ray(mesh, origin, direction, 1.0e6)
        if not query.result or query.t <= MIN_SEGMENT_LENGTH:
            break
        hit = origin + direction * query.t
        normal = oriented_normal(direction, query.normal)
        cosine = wp.clamp(-wp.dot(direction, normal), 0.0, 1.0)
        probability_specular = table_value(specular_probability, cosine)
        probability_diffuse = table_value(diffuse_probability, cosine)
        probability_transmission = table_value(transmission_probability, cosine)
        probability_total = (
            probability_specular
            + probability_diffuse
            + probability_transmission
        )
        if probability_total <= 0.0:
            break

        sample = wp.randf(state)
        interaction = INTERACTION_REFRACTION
        selected_probability = probability_transmission
        next_direction = direction
        gain = table_value(transmission_power, cosine)
        if sample < probability_specular:
            interaction = INTERACTION_SPECULAR
            selected_probability = probability_specular
            next_direction = direction - 2.0 * wp.dot(direction, normal) * normal
            gain = table_value(reflection_power, cosine) * (1.0 - scatter_squared)
        elif sample < probability_specular + probability_diffuse:
            interaction = INTERACTION_DIFFUSE
            selected_probability = probability_diffuse
            next_direction = local_to_world(wp.sample_unit_hemisphere(state), normal)
            outgoing_cosine = wp.clamp(wp.dot(next_direction, normal), 0.0, 1.0)
        solid_angle = solid_angle / wp.max(selected_probability, 1.0e-12)
        if interaction == INTERACTION_DIFFUSE:
            gain = (
                scatter_squared
                * table_value(reflection_power, cosine)
                * outgoing_cosine
                / PI
                * solid_angle
            )
        power = power * wp.max(gain, 1.0e-30)
        output = tid * MAX_TRACE_DEPTH + depth
        planes[output] = face_plane[query.face]
        faces[output] = query.face
        interactions[output] = interaction
        vertices[output] = hit
        incidence[output] = cosine
        event_pdf[output] = selected_probability
        prefix_power[output] = power
        count = depth + 1
        if interaction == INTERACTION_DIFFUSE:
            solid_angle = 2.0 * PI
        direction = wp.normalize(next_direction)
        origin = hit + direction * ray_epsilon

    counts[tid] = wp.uint8(count)


@wp.kernel
def solve_image_routes(
    mesh: wp.uint64,
    face_plane: wp.array(dtype=wp.int32),
    face_family: wp.array(dtype=wp.int32),
    face_normal: wp.array(dtype=wp.vec3),
    source: wp.vec3,
    tx_look: wp.vec3,
    targets: wp.array(dtype=wp.vec3),
    route_count: wp.int32,
    route_depth: wp.array(dtype=wp.uint8),
    route_interaction: wp.array(dtype=wp.uint8),
    route_plane: wp.array(dtype=wp.int32),
    route_images: wp.array(dtype=wp.vec3),
    scattering_coefficient: wp.float32,
    ray_epsilon: wp.float32,
    specular_probability: wp.array(dtype=wp.float32),
    diffuse_probability: wp.array(dtype=wp.float32),
    transmission_probability: wp.array(dtype=wp.float32),
    reflection_power: wp.array(dtype=wp.float32),
    transmission_power: wp.array(dtype=wp.float32),
    valid: wp.array(dtype=wp.uint8),
    path_length: wp.array(dtype=wp.float32),
    departure: wp.array(dtype=wp.vec3),
    arrival: wp.array(dtype=wp.vec3),
    energy: wp.array(dtype=wp.float32),
    output_interaction: wp.array(dtype=wp.uint8),
    output_family: wp.array(dtype=wp.int32),
    output_face: wp.array(dtype=wp.int32),
    output_vertices: wp.array(dtype=wp.vec3),
    output_incidence: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    route = tid % route_count
    target_index = tid // route_count
    target = targets[target_index]
    depth = int(route_depth[route])
    current = target
    total_length = float(0.0)
    is_valid = bool(True)
    last_vertex = source

    for reverse in range(MAX_TRACE_DEPTH):
        if reverse >= depth:
            break
        step = depth - reverse - 1
        image = route_images[route * MAX_TRACE_DEPTH + step]
        vector = image - current
        distance = wp.length(vector)
        if distance <= MIN_SEGMENT_LENGTH:
            is_valid = False
            break
        direction = vector / distance
        origin = current + direction * ray_epsilon
        query = wp.mesh_query_ray(
            mesh, origin, direction, wp.max(distance - ray_epsilon, 0.0)
        )
        if not query.result:
            is_valid = False
            break
        hit = origin + direction * query.t
        segment = wp.length(current - hit)
        if segment <= MIN_SEGMENT_LENGTH:
            is_valid = False
            break
        interaction = route_interaction[route * MAX_TRACE_DEPTH + step]
        if interaction == INTERACTION_SPECULAR:
            expected_plane = route_plane[route * MAX_TRACE_DEPTH + step]
            if face_plane[query.face] != expected_plane:
                is_valid = False
                break
        output_index = tid * MAX_TRACE_DEPTH + step
        output_interaction[output_index] = interaction
        output_family[output_index] = face_family[query.face]
        output_face[output_index] = query.face
        output_vertices[output_index] = hit
        normal = oriented_normal(-direction, face_normal[query.face])
        output_incidence[output_index] = wp.clamp(
            wp.abs(wp.dot(direction, normal)), 0.0, 1.0
        )
        total_length = total_length + segment
        current = hit
        if step == depth - 1:
            last_vertex = hit

    if is_valid:
        vector = source - current
        distance = wp.length(vector)
        if distance <= MIN_SEGMENT_LENGTH:
            is_valid = False
        else:
            direction = vector / distance
            origin = current + direction * ray_epsilon
            query = wp.mesh_query_ray(
                mesh, origin, direction, wp.max(distance - 2.0 * ray_epsilon, 0.0)
            )
            if query.result:
                is_valid = False
            else:
                total_length = total_length + distance

    if is_valid:
        first_vertex = target
        if depth > 0:
            first_vertex = output_vertices[tid * MAX_TRACE_DEPTH]
        departure_vector = first_vertex - source
        arrival_vector = last_vertex - target
        if depth == 0:
            arrival_vector = source - target
        departure[tid] = departure_vector
        arrival[tid] = arrival_vector
        path_length[tid] = total_length

        material_power = float(1.0)
        scatter_squared = scattering_coefficient * scattering_coefficient
        for step in range(MAX_TRACE_DEPTH):
            if step >= depth:
                break
            cosine = output_incidence[tid * MAX_TRACE_DEPTH + step]
            interaction = output_interaction[tid * MAX_TRACE_DEPTH + step]
            if interaction == INTERACTION_SPECULAR:
                material_power = material_power * table_value(
                    reflection_power, cosine
                ) * (1.0 - scatter_squared)
            elif interaction == INTERACTION_REFRACTION:
                material_power = material_power * table_value(
                    transmission_power, cosine
                )
        tx_gain = tr38901_power(departure_vector, tx_look)
        rx_look = source - target
        rx_gain = dipole_power(arrival_vector, rx_look)
        energy[tid] = (
            material_power
            * tx_gain
            * rx_gain
            / wp.max(total_length * total_length, 1.0e-12)
        )
        valid[tid] = wp.uint8(1)
    else:
        valid[tid] = wp.uint8(0)


@wp.kernel
def solve_diffuse_routes(
    mesh: wp.uint64,
    face_plane: wp.array(dtype=wp.int32),
    face_family: wp.array(dtype=wp.int32),
    face_normal: wp.array(dtype=wp.vec3),
    source: wp.vec3,
    tx_look: wp.vec3,
    targets: wp.array(dtype=wp.vec3),
    records_per_target: wp.int32,
    selected_record: wp.array(dtype=wp.int32),
    record_prefix_depth: wp.array(dtype=wp.uint8),
    record_total_depth: wp.array(dtype=wp.uint8),
    record_interaction: wp.array(dtype=wp.uint8),
    record_plane: wp.array(dtype=wp.int32),
    record_face: wp.array(dtype=wp.int32),
    record_vertices: wp.array(dtype=wp.vec3),
    record_suffix_images: wp.array(dtype=wp.vec3),
    scattering_coefficient: wp.float32,
    samples_per_source: wp.int32,
    ray_epsilon: wp.float32,
    specular_probability: wp.array(dtype=wp.float32),
    diffuse_probability: wp.array(dtype=wp.float32),
    transmission_probability: wp.array(dtype=wp.float32),
    reflection_power: wp.array(dtype=wp.float32),
    transmission_power: wp.array(dtype=wp.float32),
    valid: wp.array(dtype=wp.uint8),
    path_length: wp.array(dtype=wp.float32),
    departure: wp.array(dtype=wp.vec3),
    arrival: wp.array(dtype=wp.vec3),
    energy: wp.array(dtype=wp.float32),
    output_interaction: wp.array(dtype=wp.uint8),
    output_family: wp.array(dtype=wp.int32),
    output_face: wp.array(dtype=wp.int32),
    output_vertices: wp.array(dtype=wp.vec3),
    output_incidence: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    target_index = tid // records_per_target
    record = selected_record[tid]
    target = targets[target_index]
    prefix_depth = int(record_prefix_depth[record])
    total_depth = int(record_total_depth[record])
    suffix_depth = total_depth - prefix_depth
    record_base = record * MAX_TRACE_DEPTH
    output_base = tid * MAX_TRACE_DEPTH
    is_valid = bool(True)

    for step in range(MAX_TRACE_DEPTH):
        if step >= prefix_depth:
            break
        face = record_face[record_base + step]
        output_interaction[output_base + step] = record_interaction[
            record_base + step
        ]
        output_family[output_base + step] = face_family[face]
        output_face[output_base + step] = face
        output_vertices[output_base + step] = record_vertices[record_base + step]

    diffuse_source = record_vertices[record_base + prefix_depth - 1]
    current = target
    for reverse in range(MAX_TRACE_DEPTH):
        if reverse >= suffix_depth:
            break
        suffix_step = suffix_depth - reverse - 1
        full_step = prefix_depth + suffix_step
        image = record_suffix_images[record_base + suffix_step]
        vector = image - current
        distance = wp.length(vector)
        if distance <= MIN_SEGMENT_LENGTH:
            is_valid = False
            break
        direction = vector / distance
        origin = current + direction * ray_epsilon
        query = wp.mesh_query_ray(
            mesh, origin, direction, wp.max(distance - ray_epsilon, 0.0)
        )
        if not query.result:
            is_valid = False
            break
        hit = origin + direction * query.t
        if wp.length(current - hit) <= MIN_SEGMENT_LENGTH:
            is_valid = False
            break
        interaction = record_interaction[record_base + full_step]
        if interaction == INTERACTION_SPECULAR:
            expected_plane = record_plane[record_base + full_step]
            if face_plane[query.face] != expected_plane:
                is_valid = False
                break
        output_interaction[output_base + full_step] = interaction
        output_family[output_base + full_step] = face_family[query.face]
        output_face[output_base + full_step] = query.face
        output_vertices[output_base + full_step] = hit
        current = hit

    if is_valid:
        vector = diffuse_source - current
        distance = wp.length(vector)
        if distance <= MIN_SEGMENT_LENGTH:
            is_valid = False
        else:
            direction = vector / distance
            origin = current + direction * ray_epsilon
            query = wp.mesh_query_ray(
                mesh, origin, direction, wp.max(distance - 2.0 * ray_epsilon, 0.0)
            )
            if query.result:
                is_valid = False

    if is_valid and suffix_depth == 0:
        diffuse_step = prefix_depth - 1
        previous = source
        if diffuse_step > 0:
            previous = output_vertices[output_base + diffuse_step - 1]
        incoming = wp.normalize(diffuse_source - previous)
        normal = oriented_normal(
            incoming, face_normal[output_face[output_base + diffuse_step]]
        )
        outgoing = wp.normalize(target - diffuse_source)
        if wp.dot(normal, outgoing) <= 0.0:
            is_valid = False

    if is_valid:
        total_length = float(0.0)
        ray_tube_length = float(0.0)
        material_power = float(1.0)
        solid_angle = 4.0 * PI / float(samples_per_source)
        scatter_squared = scattering_coefficient * scattering_coefficient
        previous = source
        for step in range(MAX_TRACE_DEPTH):
            if step >= total_depth:
                break
            vertex = output_vertices[output_base + step]
            segment = vertex - previous
            segment_length = wp.length(segment)
            if segment_length <= MIN_SEGMENT_LENGTH:
                is_valid = False
                break
            incoming = segment / segment_length
            next_vertex = target
            if step + 1 < total_depth:
                next_vertex = output_vertices[output_base + step + 1]
            outgoing_vector = next_vertex - vertex
            outgoing_length = wp.length(outgoing_vector)
            if outgoing_length <= MIN_SEGMENT_LENGTH:
                is_valid = False
                break
            outgoing = outgoing_vector / outgoing_length
            face = output_face[output_base + step]
            normal = oriented_normal(incoming, face_normal[face])
            cosine = wp.clamp(-wp.dot(incoming, normal), 0.0, 1.0)
            output_incidence[output_base + step] = cosine
            total_length = total_length + segment_length
            ray_tube_length = ray_tube_length + segment_length
            interaction = output_interaction[output_base + step]
            probability = float(0.0)
            gain = float(0.0)
            if interaction == INTERACTION_SPECULAR:
                probability = table_value(specular_probability, cosine)
                gain = table_value(reflection_power, cosine) * (
                    1.0 - scatter_squared
                )
            elif interaction == INTERACTION_REFRACTION:
                probability = table_value(transmission_probability, cosine)
                gain = table_value(transmission_power, cosine)
            elif interaction == INTERACTION_DIFFUSE:
                probability = table_value(diffuse_probability, cosine)
                outgoing_cosine = wp.clamp(wp.dot(outgoing, normal), 0.0, 1.0)
                gain = (
                    scatter_squared
                    * table_value(reflection_power, cosine)
                    * outgoing_cosine
                    / PI
                    * (solid_angle / wp.max(probability, 1.0e-12))
                )
            if probability <= 0.0 or gain <= 0.0:
                is_valid = False
                break
            solid_angle = solid_angle / wp.max(probability, 1.0e-12)
            material_power = material_power * gain
            if interaction == INTERACTION_DIFFUSE:
                solid_angle = 2.0 * PI
                ray_tube_length = 0.0
            previous = vertex

        if is_valid:
            final_segment = target - previous
            final_length = wp.length(final_segment)
            total_length = total_length + final_length
            ray_tube_length = ray_tube_length + final_length
            first_vertex = output_vertices[output_base]
            last_vertex = output_vertices[output_base + total_depth - 1]
            departure_vector = first_vertex - source
            arrival_vector = last_vertex - target
            tx_gain = tr38901_power(departure_vector, tx_look)
            rx_gain = dipole_power(arrival_vector, source - target)
            path_length[tid] = total_length
            departure[tid] = departure_vector
            arrival[tid] = arrival_vector
            energy[tid] = (
                material_power
                * tx_gain
                * rx_gain
                / wp.max(ray_tube_length * ray_tube_length, 1.0e-12)
            )
            valid[tid] = wp.uint8(1)

    if not is_valid:
        valid[tid] = wp.uint8(0)
