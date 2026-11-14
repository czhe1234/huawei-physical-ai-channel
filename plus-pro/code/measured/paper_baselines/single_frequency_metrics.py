"""Single-frequency complex-channel metrics, with no artificial PDP score."""
from __future__ import annotations

import numpy as np


def channel_array(value):
    array = np.asarray(value)
    if array.ndim == 5:
        if array.shape[-1] != 1:
            raise ValueError("Only one frequency is allowed in this evaluator")
        array = array[..., 0]
    if array.ndim != 4 or not np.iscomplexobj(array):
        raise ValueError("Expected complex [position, panel, horizontal, vertical] channels")
    if not np.isfinite(array).all():
        raise ValueError("Non-finite complex channel")
    return array.astype(np.complex128, copy=False)


def evaluate(prediction, reference):
    """Aggregate raw error globally; phase alignment is diagnostic only.

    All phase corrections below use evaluation truth and MUST NOT be used by
    a model or be reported as deployable predictions. Power is relative to the
    common measured calibration rather than an unsupported absolute dBm scale.
    """
    pred, truth = channel_array(prediction), channel_array(reference)
    if pred.shape != truth.shape:
        raise ValueError("Prediction and reference shape mismatch")
    energy = np.sum(np.abs(truth) ** 2, axis=(2, 3))
    predicted_energy = np.sum(np.abs(pred) ** 2, axis=(2, 3))
    if np.any(energy <= 0):
        raise ValueError("Reference has a zero-energy position/panel")
    error = np.sum(np.abs(pred - truth) ** 2, axis=(2, 3))
    dot = np.sum(pred.conj() * truth, axis=(2, 3))
    phase = np.divide(dot, np.abs(dot), out=np.ones_like(dot), where=np.abs(dot) > 0)
    aligned_error = np.sum(np.abs(pred * phase[..., None, None] - truth) ** 2, axis=(2, 3))
    floor = np.finfo(np.float64).tiny
    power_error_db = 10 * np.log10(np.maximum(predicted_energy / energy, floor))
    point_nmse = error.sum(axis=1) / energy.sum(axis=1)
    raw_nmse = float(error.sum() / energy.sum())
    points = {"reference_energy": energy, "prediction_energy": predicted_energy,
              "squared_error": error, "panel_phase_aligned_squared_error": aligned_error,
              "power_error_db": power_error_db, "position_nmse": point_nmse}
    result = {"positions": len(pred), "physical_panels": pred.shape[1],
              "frequency_count": 1, "raw_global_energy_ratio_nmse": raw_nmse,
              "mean_position_nmse": float(point_nmse.mean()),
              "median_position_nmse": float(np.median(point_nmse)),
              "power_mae_db": float(np.abs(power_error_db).mean()),
              "power_error_zero_prediction_floor_db": float(10 * np.log10(floor)),
              "diagnostic_panel_phase_aligned_nmse": float(aligned_error.sum() / energy.sum()),
              "pdp": None, "rms_delay_spread": None,
              "original_pas_pdp_nmse_composite_score": None,
              "array_pas_cosine": None, "dft_beam_loss_db": None}
    result["prediction_snr_from_global_nmse_db"] = float(-10 * np.log10(raw_nmse)) if raw_nmse > 0 else None
    result["perfect_prediction_snr_is_infinite"] = raw_nmse == 0
    if pred.shape[2] * pred.shape[3] > 1:
        p = np.abs(np.fft.fft2(pred, axes=(2, 3))) ** 2
        t = np.abs(np.fft.fft2(truth, axes=(2, 3))) ** 2
        norm = np.sqrt(np.sum(p * p, axis=(2, 3)) * np.sum(t * t, axis=(2, 3)))
        cosine = np.divide(np.sum(p * t, axis=(2, 3)), norm,
                           out=np.zeros_like(norm), where=norm > 0)
        chosen = p.reshape(*p.shape[:2], -1).argmax(axis=-1)
        true_beams = t.reshape(*t.shape[:2], -1)
        selected = np.take_along_axis(true_beams, chosen[..., None], axis=-1)[..., 0]
        beam_loss = -10 * np.log10(np.maximum(selected / true_beams.max(axis=-1), floor))
        result["array_pas_cosine"] = float(cosine.mean())
        result["dft_beam_loss_db"] = float(beam_loss.mean())
        points.update(array_pas_cosine=cosine, dft_beam_loss_db=beam_loss)
    return result, points
