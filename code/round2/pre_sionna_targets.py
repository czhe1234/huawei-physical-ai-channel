"""Fit-only propagation states and spectral targets for Round 2 queries."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

from pre_sionna_spectral import (
    MH,
    MV,
    M,
    N,
    S,
    SpectralCache,
    channel_spectra_numpy,
)


@dataclass(frozen=True)
class QuerySpectralCache:
    ad: np.ndarray
    pas: np.ndarray
    pdp: np.ndarray
    global_pas: np.ndarray
    global_pdp: np.ndarray
    total_power: np.ndarray
    pas_entropy: np.ndarray
    pdp_entropy: np.ndarray


@dataclass(frozen=True)
class TargetBundle:
    train_ad: np.ndarray
    train_pas: np.ndarray
    train_pdp: np.ndarray
    consensus_ad: np.ndarray
    consensus_pas: np.ndarray
    consensus_pdp: np.ndarray
    controller_features: np.ndarray
    controller_feature_names: tuple[str, ...]
    predicted_log_power: np.ndarray
    zero_probability: np.ndarray
    metadata: dict[str, np.ndarray]


@dataclass
class SiteStateModel:
    labels: np.ndarray
    valid_fit: np.ndarray
    centers: np.ndarray
    center_temperature: float
    classifier: ExtraTreesClassifier
    power_model: ExtraTreesRegressor
    state_count: int
    zero_class: int


def _normalize(value: np.ndarray, axes: tuple[int, ...] | int) -> np.ndarray:
    array = np.maximum(np.asarray(value, dtype=np.float32), 0.0)
    total = array.sum(axis=axes, keepdims=True, dtype=np.float64)
    return np.divide(array, np.maximum(total, 1e-20), out=np.zeros_like(array), where=total > 0)


def _entropy(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    return -np.sum(array * np.log(np.maximum(array, 1e-30)), axis=1).astype(np.float32)


def _descriptor(global_pas: np.ndarray, global_pdp: np.ndarray) -> np.ndarray:
    pas = np.sqrt(np.maximum(global_pas, 0.0)).astype(np.float32)
    pdp = np.sqrt(np.maximum(global_pdp, 0.0)).astype(np.float32)
    return np.concatenate([pas, pdp], axis=1) / np.float32(np.sqrt(2.0))


def build_query_spectral_cache(
    channel: np.ndarray,
    directory: Path,
    prefix: str,
    batch_size: int = 8,
) -> QuerySpectralCache:
    directory.mkdir(parents=True, exist_ok=True)
    ad_path = directory / f"{prefix}_ad_support_f16.npy"
    pas_path = directory / f"{prefix}_pas_support_f16.npy"
    pdp_path = directory / f"{prefix}_pdp_support_f16.npy"
    descriptor_path = directory / f"{prefix}_spectral_descriptors.npz"
    expected = (len(channel), M, N, S)
    if ad_path.is_file() and pas_path.is_file() and pdp_path.is_file() and descriptor_path.is_file():
        ad = np.load(ad_path, mmap_mode="r")
        pas = np.load(pas_path, mmap_mode="r")
        pdp = np.load(pdp_path, mmap_mode="r")
        if ad.shape == (len(channel), MH, MV, S) and pas.shape == expected and pdp.shape == expected:
            values = np.load(descriptor_path)
            return QuerySpectralCache(
                ad=ad,
                pas=pas,
                pdp=pdp,
                global_pas=values["global_pas"].astype(np.float32),
                global_pdp=values["global_pdp"].astype(np.float32),
                total_power=values["total_power"].astype(np.float64),
                pas_entropy=values["pas_entropy"].astype(np.float32),
                pdp_entropy=values["pdp_entropy"].astype(np.float32),
            )

    ad_out = np.lib.format.open_memmap(
        ad_path, mode="w+", dtype=np.float16, shape=(len(channel), MH, MV, S)
    )
    pas_out = np.lib.format.open_memmap(pas_path, mode="w+", dtype=np.float16, shape=expected)
    pdp_out = np.lib.format.open_memmap(pdp_path, mode="w+", dtype=np.float16, shape=expected)
    global_pas = np.zeros((len(channel), M), dtype=np.float32)
    global_pdp = np.zeros((len(channel), S), dtype=np.float32)
    total_power = np.zeros(len(channel), dtype=np.float64)
    for start in range(0, len(channel), batch_size):
        end = min(start + batch_size, len(channel))
        block = np.asarray(channel[start:end], dtype=np.complex64)
        total_power[start:end] = np.sum(np.abs(block) ** 2, axis=(1, 2, 3), dtype=np.float64)
        ad, pas, pdp = channel_spectra_numpy(block)
        ad_out[start:end] = ad.astype(np.float16)
        pas_out[start:end] = pas.astype(np.float16)
        pdp_out[start:end] = pdp.astype(np.float16)
        global_pas[start:end] = pas.mean(axis=(2, 3), dtype=np.float64).astype(np.float32)
        global_pdp[start:end] = pdp.mean(axis=(1, 2), dtype=np.float64).astype(np.float32)
        print(f"{prefix} spectral cache {end}/{len(channel)}", flush=True)
    ad_out.flush()
    pas_out.flush()
    pdp_out.flush()
    pas_entropy = _entropy(global_pas)
    pdp_entropy = _entropy(global_pdp)
    np.savez_compressed(
        descriptor_path,
        global_pas=global_pas,
        global_pdp=global_pdp,
        total_power=total_power,
        pas_entropy=pas_entropy,
        pdp_entropy=pdp_entropy,
    )
    return QuerySpectralCache(
        ad=np.load(ad_path, mmap_mode="r"),
        pas=np.load(pas_path, mmap_mode="r"),
        pdp=np.load(pdp_path, mmap_mode="r"),
        global_pas=global_pas,
        global_pdp=global_pdp,
        total_power=total_power,
        pas_entropy=pas_entropy,
        pdp_entropy=pdp_entropy,
    )


def _fit_site_state_model(
    fit_indices: np.ndarray,
    train_geometry: np.ndarray,
    spectral: SpectralCache,
    seed: int,
) -> SiteStateModel:
    valid_fit = fit_indices[spectral.valid[fit_indices]]
    descriptors = _descriptor(
        spectral.global_pas[valid_fit], spectral.global_pdp[valid_fit]
    )
    state_count = min(12, max(4, len(valid_fit) // 80))
    cluster = KMeans(
        n_clusters=state_count,
        random_state=seed,
        n_init=12,
        max_iter=240,
    ).fit(descriptors)
    labels = np.full(len(spectral.valid), state_count, dtype=np.int16)
    labels[valid_fit] = cluster.labels_.astype(np.int16)
    fit_labels = labels[fit_indices]
    classifier = ExtraTreesClassifier(
        n_estimators=260,
        min_samples_leaf=6,
        max_features=0.82,
        class_weight="balanced",
        random_state=seed + 17,
        n_jobs=-1,
    )
    classifier.fit(train_geometry[fit_indices], fit_labels)
    power_model = ExtraTreesRegressor(
        n_estimators=300,
        min_samples_leaf=7,
        max_features=0.86,
        random_state=seed + 31,
        n_jobs=-1,
    )
    power_model.fit(
        train_geometry[valid_fit],
        np.log(np.maximum(spectral.total_power[valid_fit], 1e-30)),
    )
    assigned_distance = np.sum(
        (descriptors - cluster.cluster_centers_[cluster.labels_]) ** 2, axis=1
    )
    temperature = max(float(np.median(assigned_distance)), 1e-4)
    return SiteStateModel(
        labels=labels,
        valid_fit=valid_fit,
        centers=cluster.cluster_centers_.astype(np.float32),
        center_temperature=temperature,
        classifier=classifier,
        power_model=power_model,
        state_count=state_count,
        zero_class=state_count,
    )


def _predict_state(
    model: SiteStateModel,
    geometry: np.ndarray,
    base_descriptor: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = model.classifier.predict_proba(geometry)
    geometry_probability = np.zeros((len(geometry), model.state_count + 1), dtype=np.float32)
    geometry_probability[:, model.classifier.classes_.astype(np.int64)] = raw.astype(np.float32)
    distance = np.sum(
        (base_descriptor[:, None, :] - model.centers[None, :, :]) ** 2,
        axis=2,
    )
    logits = -distance / max(2.0 * model.center_temperature, 1e-6)
    logits -= np.max(logits, axis=1, keepdims=True)
    base_probability = np.exp(np.clip(logits, -60.0, 0.0)).astype(np.float32)
    base_probability /= np.maximum(base_probability.sum(axis=1, keepdims=True), 1e-20)
    zero_probability = geometry_probability[:, model.zero_class].copy()
    state = (
        0.58 * geometry_probability[:, : model.state_count]
        + 0.42 * base_probability * (1.0 - zero_probability[:, None])
    )
    state /= np.maximum(state.sum(axis=1, keepdims=True), 1e-20)
    return state.astype(np.float32), zero_probability.astype(np.float32), distance.astype(np.float32)


def _top_normalized(ids: np.ndarray, weights: np.ndarray, count: int = 24) -> tuple[np.ndarray, np.ndarray]:
    take = min(count, len(ids))
    order = np.argsort(weights)[::-1][:take]
    selected_ids = ids[order]
    selected_weights = np.maximum(weights[order], 0.0).astype(np.float64)
    if float(selected_weights.sum()) <= 0.0:
        selected_weights[:] = 1.0
    selected_weights /= selected_weights.sum()
    return selected_ids.astype(np.int64), selected_weights.astype(np.float32)


def _weighted_row(cache: np.ndarray, ids: np.ndarray, weights: np.ndarray) -> np.ndarray:
    values = np.asarray(cache[ids], dtype=np.float32)
    return np.tensordot(weights.astype(np.float32), values, axes=(0, 0)).astype(np.float32)


def _query_groups(positions: np.ndarray, site: np.ndarray) -> np.ndarray:
    groups = np.full(len(positions), -1, dtype=np.int32)
    offset = 0
    for site_value in np.unique(site):
        rows = np.flatnonzero(site == site_value)
        labels = DBSCAN(eps=15.0, min_samples=2).fit_predict(positions[rows, :2])
        for label in np.unique(labels):
            members = rows[labels == label]
            if label < 0:
                for member in members:
                    groups[member] = offset
                    offset += 1
            else:
                groups[members] = offset
                offset += 1
    return groups


def _consensus_neighbors(
    positions: np.ndarray,
    site: np.ndarray,
    state_probability: list[np.ndarray],
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray, np.ndarray]:
    groups = _query_groups(positions, site)
    neighbor_ids: list[np.ndarray] = []
    neighbor_weights: list[np.ndarray] = []
    effective_count = np.ones(len(positions), dtype=np.float32)
    nearest_distance = np.full(len(positions), np.inf, dtype=np.float32)
    for row in range(len(positions)):
        candidates = np.flatnonzero(groups == groups[row])
        distance = np.linalg.norm(
            positions[candidates, :2] - positions[row, None, :2], axis=1
        )
        order = np.argsort(distance)[: min(12, len(candidates))]
        ids = candidates[order]
        selected_distance = distance[order]
        similarity = np.ones(len(ids), dtype=np.float32)
        if len(state_probability[row]):
            similarity = np.asarray(
                [
                    float(np.dot(state_probability[row], state_probability[index]))
                    if len(state_probability[index]) == len(state_probability[row])
                    else 0.0
                    for index in ids
                ],
                dtype=np.float32,
            )
        weights = np.exp(-selected_distance / 8.0).astype(np.float32) * (0.35 + 0.65 * similarity)
        weights[selected_distance > 28.0] = 0.0
        weights[ids == row] += 0.70
        weights /= np.maximum(weights.sum(), 1e-20)
        neighbor_ids.append(ids.astype(np.int64))
        neighbor_weights.append(weights.astype(np.float32))
        effective_count[row] = np.float32(1.0 / max(float(np.sum(weights * weights)), 1e-20))
        nonself = selected_distance[ids != row]
        if len(nonself):
            nearest_distance[row] = np.float32(nonself[0])
    return neighbor_ids, neighbor_weights, effective_count, nearest_distance


def build_targets(
    train_positions: np.ndarray,
    query_positions: np.ndarray,
    train_site: np.ndarray,
    query_site: np.ndarray,
    train_geometry: np.ndarray,
    query_geometry: np.ndarray,
    spectral: SpectralCache,
    query_spectral: QuerySpectralCache,
    fit_groups: list[tuple[int, np.ndarray, np.ndarray]],
    directory: Path,
    prefix: str,
    seed: int = 20_260_805,
) -> TargetBundle:
    """Build targets; each tuple is (group id, fit train ids, query row ids)."""

    directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "train_ad": directory / f"{prefix}_target_train_ad_f16.npy",
        "train_pas": directory / f"{prefix}_target_train_pas_f16.npy",
        "train_pdp": directory / f"{prefix}_target_train_pdp_f16.npy",
        "consensus_ad": directory / f"{prefix}_target_consensus_ad_f16.npy",
        "consensus_pas": directory / f"{prefix}_target_consensus_pas_f16.npy",
        "consensus_pdp": directory / f"{prefix}_target_consensus_pdp_f16.npy",
        "metadata": directory / f"{prefix}_target_metadata.npz",
    }
    expected = (len(query_positions), M, N, S)
    ready = all(path.is_file() for path in paths.values())
    if ready:
        metadata = np.load(paths["metadata"], allow_pickle=False)
        train_pas = np.load(paths["train_pas"], mmap_mode="r")
        if train_pas.shape == expected:
            names = tuple(metadata["controller_feature_names"].astype(str).tolist())
            reserved = {
                "controller_features", "controller_feature_names", "predicted_log_power", "zero_probability"
            }
            return TargetBundle(
                train_ad=np.load(paths["train_ad"], mmap_mode="r"),
                train_pas=train_pas,
                train_pdp=np.load(paths["train_pdp"], mmap_mode="r"),
                consensus_ad=np.load(paths["consensus_ad"], mmap_mode="r"),
                consensus_pas=np.load(paths["consensus_pas"], mmap_mode="r"),
                consensus_pdp=np.load(paths["consensus_pdp"], mmap_mode="r"),
                controller_features=metadata["controller_features"].astype(np.float32),
                controller_feature_names=names,
                predicted_log_power=metadata["predicted_log_power"].astype(np.float32),
                zero_probability=metadata["zero_probability"].astype(np.float32),
                metadata={key: metadata[key] for key in metadata.files if key not in reserved},
            )

    train_ad = np.lib.format.open_memmap(
        paths["train_ad"], mode="w+", dtype=np.float16, shape=(len(query_positions), MH, MV, S)
    )
    train_pas = np.lib.format.open_memmap(paths["train_pas"], mode="w+", dtype=np.float16, shape=expected)
    train_pdp = np.lib.format.open_memmap(paths["train_pdp"], mode="w+", dtype=np.float16, shape=expected)
    consensus_ad = np.lib.format.open_memmap(
        paths["consensus_ad"], mode="w+", dtype=np.float16, shape=(len(query_positions), MH, MV, S)
    )
    consensus_pas = np.lib.format.open_memmap(paths["consensus_pas"], mode="w+", dtype=np.float16, shape=expected)
    consensus_pdp = np.lib.format.open_memmap(paths["consensus_pdp"], mode="w+", dtype=np.float16, shape=expected)

    base_descriptor = _descriptor(query_spectral.global_pas, query_spectral.global_pdp)
    train_descriptor = _descriptor(spectral.global_pas, spectral.global_pdp)
    predicted_log_power = np.zeros(len(query_positions), dtype=np.float32)
    zero_probability = np.zeros(len(query_positions), dtype=np.float32)
    nearest_fit = np.zeros(len(query_positions), dtype=np.float32)
    median_fit = np.zeros(len(query_positions), dtype=np.float32)
    effective_donors = np.zeros(len(query_positions), dtype=np.float32)
    state_entropy = np.zeros(len(query_positions), dtype=np.float32)
    state_max = np.zeros(len(query_positions), dtype=np.float32)
    donor_pas_similarity = np.zeros(len(query_positions), dtype=np.float32)
    donor_pdp_similarity = np.zeros(len(query_positions), dtype=np.float32)
    donor_ad_similarity = np.zeros(len(query_positions), dtype=np.float32)
    state_probabilities: list[np.ndarray] = [np.empty(0, dtype=np.float32) for _ in query_positions]

    for group_id, fit_indices, query_rows in fit_groups:
        for site_value in np.unique(query_site[query_rows]):
            local_query = query_rows[query_site[query_rows] == site_value]
            local_fit = fit_indices[train_site[fit_indices] == site_value]
            model = _fit_site_state_model(
                local_fit,
                train_geometry,
                spectral,
                seed + 1009 * int(group_id) + 97 * int(site_value),
            )
            probabilities, model_zero, _ = _predict_state(
                model,
                query_geometry[local_query],
                base_descriptor[local_query],
            )
            geometry_columns = np.r_[6:12, 14:train_geometry.shape[1]]
            scale = np.std(train_geometry[local_fit][:, geometry_columns], axis=0)
            scale = np.maximum(scale, 1e-3)
            tree = cKDTree(train_positions[local_fit, :2])
            candidate_k = min(96, len(local_fit))
            distance, local_candidate = tree.query(
                query_positions[local_query, :2], k=candidate_k, workers=-1
            )
            if candidate_k == 1:
                distance = distance[:, None]
                local_candidate = local_candidate[:, None]

            geometry_power = model.power_model.predict(query_geometry[local_query]).astype(np.float32)
            for local_row, query_row in enumerate(local_query):
                candidates = local_fit[local_candidate[local_row]].astype(np.int64)
                dxy = distance[local_row].astype(np.float32)
                geometry_delta = (
                    (train_geometry[candidates][:, geometry_columns] - query_geometry[query_row, geometry_columns])
                    / scale
                )
                geometry_distance = np.mean(geometry_delta * geometry_delta, axis=1).astype(np.float32)
                pas_similarity = np.sum(
                    np.sqrt(np.maximum(query_spectral.global_pas[query_row], 0.0))[None, :]
                    * np.sqrt(np.maximum(spectral.global_pas[candidates], 0.0)),
                    axis=1,
                ).astype(np.float32)
                pdp_similarity = np.sum(
                    np.sqrt(np.maximum(query_spectral.global_pdp[query_row], 0.0))[None, :]
                    * np.sqrt(np.maximum(spectral.global_pdp[candidates], 0.0)),
                    axis=1,
                ).astype(np.float32)
                candidate_label = model.labels[candidates].astype(np.int64)
                state_affinity = np.zeros(len(candidates), dtype=np.float32)
                valid_label = candidate_label < model.state_count
                state_affinity[valid_label] = probabilities[local_row, candidate_label[valid_label]]
                spatial = np.power(np.maximum(dxy + 1.5, 1e-3), -1.65).astype(np.float32)
                geometry_weight = np.exp(-0.28 * np.minimum(geometry_distance, 12.0)).astype(np.float32)
                state_weight = 0.20 + 0.80 * state_affinity
                valid_weight = spectral.valid[candidates].astype(np.float32)
                common = spatial * geometry_weight * state_weight * valid_weight
                pas_weight = common * np.exp(2.6 * (pas_similarity - 0.70)).astype(np.float32)
                pdp_weight = common * np.exp(2.9 * (pdp_similarity - 0.70)).astype(np.float32)
                ad_weight = common * np.exp(1.35 * (pas_similarity + pdp_similarity - 1.40)).astype(np.float32)
                pas_ids, pas_weights = _top_normalized(candidates, pas_weight)
                pdp_ids, pdp_weights = _top_normalized(candidates, pdp_weight)
                ad_ids, ad_weights = _top_normalized(candidates, ad_weight)
                train_pas[query_row] = _normalize(
                    _weighted_row(spectral.pas, pas_ids, pas_weights), axes=0
                ).astype(np.float16)
                train_pdp[query_row] = _normalize(
                    _weighted_row(spectral.pdp, pdp_ids, pdp_weights), axes=2
                ).astype(np.float16)
                train_ad[query_row] = _normalize(
                    _weighted_row(spectral.ad, ad_ids, ad_weights), axes=(0, 1, 2)
                ).astype(np.float16)

                all_weight = spatial * geometry_weight
                all_weight /= np.maximum(all_weight.sum(), 1e-20)
                knn_zero = float(np.sum(all_weight * (~spectral.valid[candidates])))
                zero_probability[query_row] = np.float32(0.58 * model_zero[local_row] + 0.42 * knn_zero)
                valid_candidates = candidates[spectral.valid[candidates]]
                valid_common = common[spectral.valid[candidates]]
                if len(valid_candidates) == 0:
                    predicted_log_power[query_row] = geometry_power[local_row]
                else:
                    valid_common /= np.maximum(valid_common.sum(), 1e-20)
                    knn_power = float(
                        np.sum(valid_common * np.log(np.maximum(spectral.total_power[valid_candidates], 1e-30)))
                    )
                    predicted_log_power[query_row] = np.float32(
                        0.68 * knn_power + 0.32 * geometry_power[local_row]
                    )
                nearest_fit[query_row] = dxy[0]
                median_fit[query_row] = np.median(dxy[: min(16, len(dxy))])
                effective_donors[query_row] = np.float32(1.0 / np.sum(ad_weights * ad_weights))
                state_entropy[query_row] = np.float32(
                    -np.sum(probabilities[local_row] * np.log(np.maximum(probabilities[local_row], 1e-20)))
                )
                state_max[query_row] = np.max(probabilities[local_row])
                candidate_lookup = {int(index): slot for slot, index in enumerate(candidates)}
                donor_pas_similarity[query_row] = (
                    np.sum(
                        pas_weights
                        * np.asarray(
                            [pas_similarity[candidate_lookup[int(index)]] for index in pas_ids],
                            dtype=np.float32,
                        )
                    )
                    if len(pas_ids)
                    else 0.0
                )
                donor_pdp_similarity[query_row] = (
                    np.sum(
                        pdp_weights
                        * np.asarray(
                            [pdp_similarity[candidate_lookup[int(index)]] for index in pdp_ids],
                            dtype=np.float32,
                        )
                    )
                    if len(pdp_ids)
                    else 0.0
                )
                donor_ad_similarity[query_row] = 0.5 * (
                    donor_pas_similarity[query_row] + donor_pdp_similarity[query_row]
                )
                state_probabilities[query_row] = probabilities[local_row].copy()
                print(
                    f"{prefix} targets group={group_id} site={int(site_value)} "
                    f"query={local_row + 1}/{len(local_query)}",
                    flush=True,
                )
        train_ad.flush()
        train_pas.flush()
        train_pdp.flush()

    consensus_ids, consensus_weights, consensus_effective, consensus_distance = _consensus_neighbors(
        query_positions, query_site, state_probabilities
    )
    for row in range(len(query_positions)):
        ids = consensus_ids[row]
        weights = consensus_weights[row]
        consensus_ad[row] = _normalize(
            _weighted_row(query_spectral.ad, ids, weights), axes=(0, 1, 2)
        ).astype(np.float16)
        consensus_pas[row] = _normalize(
            _weighted_row(query_spectral.pas, ids, weights), axes=0
        ).astype(np.float16)
        consensus_pdp[row] = _normalize(
            _weighted_row(query_spectral.pdp, ids, weights), axes=2
        ).astype(np.float16)
        if (row + 1) % 50 == 0 or row + 1 == len(query_positions):
            print(f"{prefix} consensus {row + 1}/{len(query_positions)}", flush=True)

    consensus_ad.flush()
    consensus_pas.flush()
    consensus_pdp.flush()

    base_log_power = np.log(np.maximum(query_spectral.total_power, 1e-30)).astype(np.float32)
    target_global_pas = np.asarray(train_pas, dtype=np.float32).mean(axis=(2, 3), dtype=np.float64).astype(np.float32)
    target_global_pdp = np.asarray(train_pdp, dtype=np.float32).mean(axis=(1, 2), dtype=np.float64).astype(np.float32)
    target_pas_cos = np.sum(
        np.sqrt(np.maximum(target_global_pas, 0.0))
        * np.sqrt(np.maximum(query_spectral.global_pas, 0.0)),
        axis=1,
    ).astype(np.float32)
    target_pdp_cos = np.sum(
        np.sqrt(np.maximum(target_global_pdp, 0.0))
        * np.sqrt(np.maximum(query_spectral.global_pdp, 0.0)),
        axis=1,
    ).astype(np.float32)
    extra = np.column_stack(
        [
            base_log_power,
            predicted_log_power,
            np.clip(predicted_log_power - base_log_power, -4.0, 4.0),
            zero_probability,
            nearest_fit,
            median_fit,
            effective_donors,
            state_entropy,
            state_max,
            donor_pas_similarity,
            donor_pdp_similarity,
            donor_ad_similarity,
            target_pas_cos,
            target_pdp_cos,
            query_spectral.pas_entropy,
            query_spectral.pdp_entropy,
            consensus_effective,
            consensus_distance,
        ]
    ).astype(np.float32)
    extra_names = (
        "base_log_power", "predicted_log_power", "predicted_log_power_delta", "zero_probability",
        "nearest_fit_distance", "median16_fit_distance", "effective_donors", "state_entropy",
        "state_max_probability", "donor_pas_similarity", "donor_pdp_similarity", "donor_ad_similarity",
        "target_pas_hellinger", "target_pdp_hellinger", "base_pas_entropy", "base_pdp_entropy",
        "consensus_effective_count", "consensus_nearest_distance",
    )
    controller_features = np.concatenate([query_geometry, extra], axis=1).astype(np.float32)
    geometry_names = tuple(f"geometry_{index}" for index in range(query_geometry.shape[1]))
    controller_names = geometry_names + extra_names
    metadata = {
        "nearest_fit": nearest_fit,
        "median_fit": median_fit,
        "effective_donors": effective_donors,
        "state_entropy": state_entropy,
        "state_max": state_max,
        "donor_pas_similarity": donor_pas_similarity,
        "donor_pdp_similarity": donor_pdp_similarity,
        "donor_ad_similarity": donor_ad_similarity,
        "target_pas_hellinger": target_pas_cos,
        "target_pdp_hellinger": target_pdp_cos,
        "consensus_effective": consensus_effective,
        "consensus_distance": consensus_distance,
    }
    np.savez_compressed(
        paths["metadata"],
        controller_features=controller_features,
        controller_feature_names=np.asarray(controller_names, dtype="U48"),
        predicted_log_power=predicted_log_power,
        zero_probability=zero_probability,
        **metadata,
    )
    return TargetBundle(
        train_ad=np.load(paths["train_ad"], mmap_mode="r"),
        train_pas=np.load(paths["train_pas"], mmap_mode="r"),
        train_pdp=np.load(paths["train_pdp"], mmap_mode="r"),
        consensus_ad=np.load(paths["consensus_ad"], mmap_mode="r"),
        consensus_pas=np.load(paths["consensus_pas"], mmap_mode="r"),
        consensus_pdp=np.load(paths["consensus_pdp"], mmap_mode="r"),
        controller_features=controller_features,
        controller_feature_names=controller_names,
        predicted_log_power=predicted_log_power,
        zero_probability=zero_probability,
        metadata=metadata,
    )
