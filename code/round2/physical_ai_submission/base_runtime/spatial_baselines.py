"""Frozen runtime primitives for spatial baselines."""

from __future__ import annotations
from dataclasses import asdict, dataclass
import numpy as np
from scipy.spatial import cKDTree

TINY = 1e-30


@dataclass(frozen=True)
class BaselineConfig:
    name: str
    method: str
    k: int
    distance_power: float = 2.0
    beta: float = 140.3
    gamma: float = 0.1
    ridge: float = 0.3
    steering_alpha: float = 0.0
    horizontal_scale: float = 1.5 * np.pi
    vertical_scale: float = 1.5 * np.pi
    horizontal_axis_deg: float = 90.0
    power_mode: str = "native"
    zero_threshold: float | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def idw_weights(
    distances: np.ndarray, power: float, gate: np.ndarray | None = None
) -> np.ndarray:
    value = np.asarray(distances, dtype=np.float64)
    if value[0] <= 1e-12:
        weights = np.zeros_like(value)
        weights[0] = 1.0
    else:
        weights = np.power(np.maximum(value, 1e-06), -power)
        if gate is not None:
            local_gate = np.asarray(gate, dtype=np.float64)
            if local_gate.shape != weights.shape:
                raise ValueError(
                    f"gate shape mismatch: {local_gate.shape} != {weights.shape}"
                )
            weights *= np.maximum(local_gate, 1e-06)
        weights /= np.sum(weights)
    return weights.astype(np.float32)


def affine_coefficients(
    relative_xy: np.ndarray,
    distances: np.ndarray,
    distance_power: float,
    ridge: float,
    gate: np.ndarray | None = None,
) -> np.ndarray:
    weights = idw_weights(distances, distance_power, gate).astype(np.float64)
    design = np.column_stack(
        [
            np.ones(len(relative_xy), dtype=np.float64),
            np.asarray(relative_xy, dtype=np.float64),
        ]
    )
    weighted = design * weights[:, None]
    normal = design.T @ weighted
    scale = max(float(np.trace(normal)) / normal.shape[0], 1e-12)
    regularizer = np.eye(3, dtype=np.float64) * (ridge * scale)
    regularizer[0, 0] = 0.0
    try:
        inverse = np.linalg.inv(normal + regularizer)
    except np.linalg.LinAlgError:
        inverse = np.linalg.pinv(normal + regularizer)
    return (inverse[0] @ weighted.T).astype(np.float32)


def _unit_vectors(positions: np.ndarray, bs_position: np.ndarray) -> np.ndarray:
    delta = np.asarray(positions, dtype=np.float64) - np.asarray(bs_position)[None, :]
    return delta / np.maximum(np.linalg.norm(delta, axis=1, keepdims=True), TINY)


def _steering_phase(
    neighbor_positions: np.ndarray,
    query_position: np.ndarray,
    bs_position: np.ndarray,
    config: BaselineConfig,
) -> np.ndarray:
    positions = np.concatenate([neighbor_positions, query_position[None, :]], axis=0)
    unit = _unit_vectors(positions, bs_position)
    train_unit = unit[:-1]
    query_unit = unit[-1]
    theta = np.deg2rad(config.horizontal_axis_deg)
    horizontal_train = (
        np.cos(theta) * train_unit[:, 0] + np.sin(theta) * train_unit[:, 1]
    )
    horizontal_query = np.cos(theta) * query_unit[0] + np.sin(theta) * query_unit[1]
    vertical_train = -train_unit[:, 2]
    vertical_query = -query_unit[2]
    h_coord = np.arange(16, dtype=np.float32) - 7.5
    v_coord = np.arange(8, dtype=np.float32) - 3.5
    phase_h = (
        config.horizontal_scale
        * (horizontal_query - horizontal_train)[:, None]
        * h_coord[None, :]
    )
    phase_v = (
        config.vertical_scale
        * (vertical_query - vertical_train)[:, None]
        * v_coord[None, :]
    )
    phase = phase_h[:, None, :, None] + phase_v[:, None, None, :]
    phase = np.broadcast_to(phase, (len(neighbor_positions), 2, 16, 8))
    return np.exp(1j * phase.reshape(len(neighbor_positions), 256)).astype(np.complex64)


def _power_target(
    neighbor_channel: np.ndarray,
    distances: np.ndarray,
    config: BaselineConfig,
    gate: np.ndarray | None = None,
) -> float | None:
    if config.power_mode == "native":
        return None
    weights = idw_weights(distances, config.distance_power, gate).astype(np.float64)
    powers = np.sum(np.abs(neighbor_channel) ** 2, axis=(1, 2, 3), dtype=np.float64)
    if config.power_mode == "linear":
        return float(np.dot(weights, powers))
    if config.power_mode == "log_nonzero":
        valid = powers > 1e-30
        if not np.any(valid):
            return 0.0
        local_weights = weights[valid]
        local_weights /= np.sum(local_weights)
        return float(np.exp(np.dot(local_weights, np.log(powers[valid]))))
    raise ValueError(f"unknown power mode: {config.power_mode}")


def predict_one(
    neighbor_channel: np.ndarray,
    neighbor_positions: np.ndarray,
    query_position: np.ndarray,
    distances: np.ndarray,
    bs_position: np.ndarray,
    config: BaselineConfig,
    donor_gate: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float | bool]]:
    k = min(config.k, len(distances))
    channel = np.asarray(neighbor_channel[:k], dtype=np.complex64)
    positions = np.asarray(neighbor_positions[:k], dtype=np.float64)
    local_distances = np.asarray(distances[:k], dtype=np.float64)
    local_gate = None if donor_gate is None else np.asarray(donor_gate[:k])
    weights = idw_weights(local_distances, config.distance_power, local_gate)
    powers = np.sum(np.abs(channel) ** 2, axis=(1, 2, 3), dtype=np.float64)
    zero_probability = float(np.dot(weights, powers <= 1e-30))
    zero_gate = bool(
        config.zero_threshold is not None
        and powers[0] <= 1e-30
        and (zero_probability >= config.zero_threshold)
    )
    if zero_gate:
        return (
            np.zeros(channel.shape[1:], dtype=np.complex64),
            {"zero_probability": zero_probability, "zero_gate": True},
        )
    if config.method == "nearest":
        prediction = channel[0].copy()
    else:
        query_range = float(np.linalg.norm(query_position - bs_position))
        neighbor_range = np.linalg.norm(positions - bs_position[None, :], axis=1)
        subcarrier = np.linspace(-0.5, 0.5, channel.shape[-1], dtype=np.float32)
        beta = config.beta + config.gamma * subcarrier
        radial = np.exp(
            1j * (neighbor_range - query_range)[:, None] * beta[None, :]
        ).astype(np.complex64)
        aligned = channel * radial[:, None, None, :]
        if config.method == "raw_idw":
            aligned = channel
            coefficients = weights
        elif config.method == "radial_idw":
            coefficients = weights
        elif config.method == "radial_affine":
            coefficients = affine_coefficients(
                positions[:, :2] - query_position[None, :2],
                local_distances,
                config.distance_power,
                config.ridge,
                local_gate,
            )
        else:
            raise ValueError(f"unknown baseline method: {config.method}")
        radial_prediction = np.sum(coefficients[:, None, None, None] * aligned, axis=0)
        if config.steering_alpha > 0.0 and config.method != "raw_idw":
            steering = _steering_phase(positions, query_position, bs_position, config)
            steered = np.sum(
                coefficients[:, None, None, None]
                * steering[:, :, None, None]
                * aligned,
                axis=0,
            )
            prediction = (
                (1.0 - config.steering_alpha) * radial_prediction
                + config.steering_alpha * steered
            ).astype(np.complex64)
        else:
            prediction = radial_prediction.astype(np.complex64)
    target_power = _power_target(channel, local_distances, config, local_gate)
    if target_power is not None:
        current_power = float(np.sum(np.abs(prediction) ** 2, dtype=np.float64))
        if target_power <= 0.0:
            prediction.fill(0.0)
        elif current_power > TINY:
            prediction *= np.float32(np.sqrt(target_power / current_power))
    return (
        prediction.astype(np.complex64),
        {"zero_probability": zero_probability, "zero_gate": False},
    )


def predict_many(
    neighbor_channel: np.ndarray,
    neighbor_positions: np.ndarray,
    query_position: np.ndarray,
    distances: np.ndarray,
    bs_position: np.ndarray,
    configs: list[BaselineConfig],
    donor_gate: np.ndarray | None = None,
) -> tuple[np.ndarray, list[dict[str, float | bool]]]:
    """Evaluate several baselines while reusing their identical radial alignment."""
    if not configs:
        raise ValueError("predict_many requires at least one config")
    max_k = min(max((config.k for config in configs)), len(distances))
    all_channel = np.asarray(neighbor_channel[:max_k], dtype=np.complex64)
    all_positions = np.asarray(neighbor_positions[:max_k], dtype=np.float64)
    all_distances = np.asarray(distances[:max_k], dtype=np.float64)
    all_powers = np.sum(np.abs(all_channel) ** 2, axis=(1, 2, 3), dtype=np.float64)
    query_range = float(np.linalg.norm(query_position - bs_position))
    neighbor_range = np.linalg.norm(all_positions - bs_position[None, :], axis=1)
    subcarrier = np.linspace(-0.5, 0.5, all_channel.shape[-1], dtype=np.float32)
    aligned_cache: dict[tuple[float, float], np.ndarray] = {}
    steering_cache: dict[tuple[int, float, float, float], np.ndarray] = {}
    predictions: list[np.ndarray] = []
    diagnostics: list[dict[str, float | bool]] = []
    for config in configs:
        k = min(config.k, max_k)
        channel = all_channel[:k]
        positions = all_positions[:k]
        local_distances = all_distances[:k]
        local_gate = None if donor_gate is None else np.asarray(donor_gate[:k])
        weights = idw_weights(local_distances, config.distance_power, local_gate)
        powers = all_powers[:k]
        zero_probability = float(np.dot(weights, powers <= 1e-30))
        zero_gate = bool(
            config.zero_threshold is not None
            and powers[0] <= 1e-30
            and (zero_probability >= config.zero_threshold)
        )
        if zero_gate:
            predictions.append(np.zeros(channel.shape[1:], dtype=np.complex64))
            diagnostics.append(
                {"zero_probability": zero_probability, "zero_gate": True}
            )
            continue
        if config.method == "nearest":
            prediction = channel[0].copy()
        else:
            radial_key = (float(config.beta), float(config.gamma))
            if radial_key not in aligned_cache:
                beta = config.beta + config.gamma * subcarrier
                radial = np.exp(
                    1j * (neighbor_range - query_range)[:, None] * beta[None, :]
                ).astype(np.complex64)
                aligned_cache[radial_key] = all_channel * radial[:, None, None, :]
            aligned = aligned_cache[radial_key][:k]
            if config.method == "raw_idw":
                aligned = channel
                coefficients = weights
            elif config.method == "radial_idw":
                coefficients = weights
            elif config.method == "radial_affine":
                coefficients = affine_coefficients(
                    positions[:, :2] - query_position[None, :2],
                    local_distances,
                    config.distance_power,
                    config.ridge,
                    local_gate,
                )
            else:
                raise ValueError(f"unknown baseline method: {config.method}")
            if config.steering_alpha > 0.0 and config.method != "raw_idw":
                steering_key = (
                    k,
                    float(config.horizontal_scale),
                    float(config.vertical_scale),
                    float(config.horizontal_axis_deg),
                )
                if steering_key not in steering_cache:
                    steering_cache[steering_key] = _steering_phase(
                        positions, query_position, bs_position, config
                    )
                steered = np.sum(
                    coefficients[:, None, None, None]
                    * steering_cache[steering_key][:, :, None, None]
                    * aligned,
                    axis=0,
                )
                if config.steering_alpha == 1.0:
                    prediction = steered.astype(np.complex64)
                else:
                    radial_prediction = np.sum(
                        coefficients[:, None, None, None] * aligned, axis=0
                    )
                    prediction = (
                        (1.0 - config.steering_alpha) * radial_prediction
                        + config.steering_alpha * steered
                    ).astype(np.complex64)
            else:
                radial_prediction = np.sum(
                    coefficients[:, None, None, None] * aligned, axis=0
                )
                prediction = radial_prediction.astype(np.complex64)
        target_power = _power_target(channel, local_distances, config, local_gate)
        if target_power is not None:
            current_power = float(np.sum(np.abs(prediction) ** 2, dtype=np.float64))
            if target_power <= 0.0:
                prediction.fill(0.0)
            elif current_power > TINY:
                prediction *= np.float32(np.sqrt(target_power / current_power))
        predictions.append(prediction.astype(np.complex64))
        diagnostics.append({"zero_probability": zero_probability, "zero_gate": False})
    return (np.stack(predictions), diagnostics)


def batched_predict_many_torch(
    train_channel,
    neighbor_ids: np.ndarray,
    neighbor_positions: np.ndarray,
    query_positions: np.ndarray,
    distances: np.ndarray,
    bs_positions: np.ndarray,
    query_site: np.ndarray,
    configs_by_site: dict[int, list[BaselineConfig]],
    point_power: np.ndarray,
    donor_gate: np.ndarray | None = None,
):
    """Generate an arbitrary shared carrier family on one GPU."""
    import torch

    batch_size, donor_count = neighbor_ids.shape
    config_count = len(next(iter(configs_by_site.values())))
    raw_indices: list[int] = []
    radial_indices: list[int] = []
    steering_indices: list[int] = []
    templates = next(iter(configs_by_site.values()))
    for index, config in enumerate(templates):
        if config.method in ("nearest", "raw_idw"):
            raw_indices.append(index)
        elif config.steering_alpha > 0.0:
            if config.steering_alpha != 1.0:
                raise ValueError("batched GPU carrier requires steering_alpha=1")
            steering_indices.append(index)
        else:
            radial_indices.append(index)
    raw_weight = np.zeros(
        (batch_size, len(raw_indices), donor_count), dtype=np.complex64
    )
    radial_weight = np.zeros(
        (batch_size, len(radial_indices), donor_count), dtype=np.complex64
    )
    steering_weight = np.zeros(
        (batch_size, len(steering_indices), donor_count, train_channel.shape[1]),
        dtype=np.complex64,
    )
    radial_phase = np.empty(
        (batch_size, donor_count, train_channel.shape[-1]), dtype=np.complex64
    )
    target_power = np.full((batch_size, config_count), np.nan, dtype=np.float64)
    zero_probability = np.zeros((batch_size, config_count), dtype=np.float64)
    zero_gate = np.zeros((batch_size, config_count), dtype=bool)
    subcarrier = np.linspace(-0.5, 0.5, train_channel.shape[-1], dtype=np.float32)
    raw_slot = {index: slot for slot, index in enumerate(raw_indices)}
    radial_slot = {index: slot for slot, index in enumerate(radial_indices)}
    steering_slot = {index: slot for slot, index in enumerate(steering_indices)}
    for row in range(batch_size):
        site_value = int(query_site[row])
        configs = configs_by_site[site_value]
        donor_position = neighbor_positions[row]
        query_position = query_positions[row]
        query_range = float(np.linalg.norm(query_position - bs_positions[site_value]))
        donor_range = np.linalg.norm(
            donor_position - bs_positions[site_value][None], axis=1
        )
        beta_pairs = {(float(config.beta), float(config.gamma)) for config in configs}
        if len(beta_pairs) != 1:
            raise ValueError(
                "batched GPU carrier requires one beta/gamma pair per site"
            )
        beta_value, gamma_value = next(iter(beta_pairs))
        beta = beta_value + gamma_value * subcarrier
        radial_phase[row] = np.exp(
            1j * (donor_range - query_range)[:, None] * beta[None]
        ).astype(np.complex64)
        for index, config in enumerate(configs):
            k = min(int(config.k), donor_count)
            local_distance = distances[row, :k]
            local_gate = (
                None
                if donor_gate is None
                else np.asarray(donor_gate[row, :k], dtype=np.float32)
            )
            weights = idw_weights(local_distance, config.distance_power, local_gate)
            powers = np.asarray(point_power[neighbor_ids[row, :k]], dtype=np.float64)
            zero_probability[row, index] = float(np.dot(weights, powers <= 1e-30))
            is_zero = bool(
                config.zero_threshold is not None
                and powers[0] <= 1e-30
                and (zero_probability[row, index] >= config.zero_threshold)
            )
            zero_gate[row, index] = is_zero
            if config.power_mode == "linear":
                target_power[row, index] = float(np.dot(weights, powers))
            elif config.power_mode == "log_nonzero":
                valid = powers > 1e-30
                if not np.any(valid):
                    target_power[row, index] = 0.0
                else:
                    local_weights = weights[valid].astype(np.float64)
                    local_weights /= np.sum(local_weights)
                    target_power[row, index] = float(
                        np.exp(np.dot(local_weights, np.log(powers[valid])))
                    )
            elif config.power_mode != "native":
                raise ValueError(f"unknown power mode: {config.power_mode}")
            if is_zero:
                continue
            if config.method == "nearest":
                raw_weight[row, raw_slot[index], 0] = 1.0
                continue
            if config.method in ("raw_idw", "radial_idw"):
                coefficient = weights
            elif config.method == "radial_affine":
                coefficient = affine_coefficients(
                    donor_position[:k, :2] - query_position[None, :2],
                    local_distance,
                    config.distance_power,
                    config.ridge,
                    local_gate,
                )
            else:
                raise ValueError(config.method)
            if index in raw_slot:
                raw_weight[row, raw_slot[index], :k] = coefficient
            elif index in radial_slot:
                radial_weight[row, radial_slot[index], :k] = coefficient
            elif index in steering_slot:
                steering = _steering_phase(
                    donor_position[:k], query_position, bs_positions[site_value], config
                )
                steering_weight[row, steering_slot[index], :k] = (
                    coefficient[:, None] * steering
                )
            else:
                raise RuntimeError(f"unassigned carrier index {index}")
    device = train_channel.device
    neighbor_tensor = torch.as_tensor(neighbor_ids, dtype=torch.long, device=device)
    donor = train_channel[neighbor_tensor]
    phase = torch.as_tensor(radial_phase, device=device)
    aligned = donor * phase[:, :, None, None]
    output = torch.empty(
        (
            batch_size,
            config_count,
            train_channel.shape[1],
            train_channel.shape[2],
            train_channel.shape[3],
        ),
        dtype=torch.complex64,
        device=device,
    )
    if raw_indices:
        output[:, raw_indices] = torch.einsum(
            "bcd,bdaus->bcaus", torch.as_tensor(raw_weight, device=device), donor
        )
    if radial_indices:
        output[:, radial_indices] = torch.einsum(
            "bcd,bdaus->bcaus", torch.as_tensor(radial_weight, device=device), aligned
        )
    if steering_indices:
        output[:, steering_indices] = torch.einsum(
            "bcda,bdaus->bcaus",
            torch.as_tensor(steering_weight, device=device),
            aligned,
        )
    target = torch.as_tensor(target_power, dtype=torch.float64, device=device)
    scale_mask = torch.isfinite(target)
    if bool(scale_mask.any()):
        current = output.abs().square().sum(dim=(2, 3, 4), dtype=torch.float64)
        positive = scale_mask & (target > 0.0) & (current > TINY)
        scale = torch.ones_like(target, dtype=torch.float32)
        scale[positive] = torch.sqrt(target[positive] / current[positive]).to(
            torch.float32
        )
        output = output * scale[:, :, None, None, None]
        output[scale_mask & (target <= 0.0)] = 0.0
    return (output, zero_probability, zero_gate)


def query_site_neighbors(
    train_positions: np.ndarray,
    train_site: np.ndarray,
    fit_indices: np.ndarray,
    query_indices: np.ndarray,
    max_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    neighbor_indices = np.empty((len(query_indices), max_k), dtype=np.int64)
    distances = np.empty((len(query_indices), max_k), dtype=np.float64)
    for site in np.unique(train_site[query_indices]):
        fit_site = fit_indices[train_site[fit_indices] == site]
        query_rows = np.flatnonzero(train_site[query_indices] == site)
        tree = cKDTree(train_positions[fit_site, :2])
        site_distance, site_local = tree.query(
            train_positions[query_indices[query_rows], :2], k=max_k, workers=-1
        )
        neighbor_indices[query_rows] = fit_site[site_local]
        distances[query_rows] = site_distance
    return (distances, neighbor_indices)
