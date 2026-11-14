"""Broadband fixed-codebook metrics. No phase/delay fitting or query feedback."""
from __future__ import annotations
import numpy as np
import torch
LAGS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)

def codebook():
    h, v = np.meshgrid(np.arange(8), np.arange(4), indexing="ij")
    u, w = np.meshgrid(np.arange(8), np.arange(4), indexing="ij")
    return np.exp(2j * np.pi * (h.reshape(-1, 1) * u.reshape(1, -1) / 8 +
                             v.reshape(-1, 1) * w.reshape(1, -1) / 4)) / np.sqrt(32)

def directional_power(channel):
    h = np.asarray(channel)
    if h.ndim != 5 or h.shape[1:4] != (2, 8, 4): raise ValueError("Channel layout")
    if not np.isfinite(h).all(): raise ValueError("Nonfinite channel")
    # Integrate unnormalized per-frequency energy BEFORE any normalization.
    return (np.abs(np.fft.fft2(h.astype(np.complex128), axes=(2, 3), norm="ortho")) ** 2).mean(-1).reshape(len(h), 2, 32)

def torch_directional_power(channel):
    return torch.fft.fft2(channel, dim=(-3, -2), norm="ortho").abs().square().mean(-1).flatten(-2)

def normalized_power(power):
    p = np.asarray(power, dtype=np.float64)
    if p.shape[-1] != 32 or not np.isfinite(p).all() or np.any(p < 0): raise ValueError("Power must be finite, nonnegative and32 beams")
    total = p.sum(-1, keepdims=True)
    return np.divide(p, total, out=np.zeros_like(p), where=total > 0)

def as_power(prediction):
    if "power" in prediction:
        p = np.asarray(prediction["power"], np.float64)
        normalized_power(p)
        if "channel" in prediction and not np.allclose(p, directional_power(prediction["channel"]), rtol=1e-5, atol=1e-15):
            raise ValueError("Declared power and complex output disagree")
        return p
    return directional_power(prediction["channel"])

def select_beam(prediction):
    # Exactly the same deterministic rule for every method, including all-zero ties.
    return as_power(prediction).argmax(-1)

def primary_metrics(prediction, reference_power):
    pred = as_power(prediction)
    truth = np.asarray(reference_power, np.float64)
    normalized_power(truth)
    if pred.shape != truth.shape: raise ValueError("Prediction/reference shape mismatch")
    selected = select_beam(prediction)
    valid = truth.sum(-1) > 0
    numerator = (pred * truth).sum(-1)
    denominator = np.linalg.norm(pred, axis=-1) * np.linalg.norm(truth, axis=-1)
    cosine = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
    best = truth.max(-1)
    received = np.take_along_axis(truth, selected[..., None], axis=-1)[..., 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        loss = 10 * np.log10(best / received)
    loss = np.where(valid, np.maximum(loss, 0.), np.nan)
    return dict(direction_cosine=np.where(valid, np.clip(cosine, 0, 1), np.nan),
                beam_loss_db=loss, beam_id=selected,
                valid_reference=valid, zero_prediction=pred.sum(-1) == 0,
                zero_selected_truth_power=(received == 0) & valid)

def secondary_metrics(prediction, reference_channel):
    if "channel" not in prediction:
        return dict(applicable=False, reason="Power-only output has no covariance or frequency phase information")
    a = np.asarray(prediction["channel"], np.complex128).reshape(len(reference_channel), 2, 32, -1)
    b = np.asarray(reference_channel, np.complex128).reshape(a.shape)
    def covariance(x):
        r = x @ x.conj().swapaxes(-1, -2)
        tr = r.diagonal(axis1=-2, axis2=-1).real.sum(-1)
        return np.divide(r, tr[..., None, None], out=np.zeros_like(r), where=tr[..., None, None] > 0)
    ra, rb = covariance(a), covariance(b)
    denominator = (np.abs(rb)**2).sum((-1,-2))
    with np.errstate(divide="ignore", invalid="ignore"):
        covariance_error = (np.abs(ra-rb)**2).sum((-1,-2)) / denominator
    def correlation(x, lag):
        x1, x0 = x[..., lag:], x[..., :-lag]
        n = np.abs((x1 * x0.conj()).sum((-1, -2)))
        d = np.sqrt((np.abs(x1)**2).sum((-1,-2)) * (np.abs(x0)**2).sum((-1,-2)))
        return np.divide(n, d, out=np.full_like(n, np.nan), where=d>0)
    lags = [lag for lag in LAGS if lag < a.shape[-1]]
    ca = np.stack([correlation(a, lag) for lag in lags], -1)
    cb = np.stack([correlation(b, lag) for lag in lags], -1)
    valid = np.isfinite(ca) & np.isfinite(cb)
    return dict(applicable=True, covariance_error=covariance_error,
                correlation_lags=lags, frequency_correlation_absolute_error=np.abs(ca-cb),
                valid_correlation=valid,reference_valid_correlation=np.isfinite(cb),
                prediction_valid_correlation=np.isfinite(ca))

def training_power_loss(predicted, target):
    predicted = predicted / predicted.sum(-1, keepdim=True).clamp_min(1e-20)
    target = target / target.sum(-1, keepdim=True).clamp_min(1e-20)
    return (1 - torch.nn.functional.cosine_similarity(predicted, target, dim=-1, eps=1e-12)).mean()

def aggregate_primary(points):
    valid = points["valid_reference"]
    cosine, loss = points["direction_cosine"][valid], points["beam_loss_db"][valid]
    infinite = bool(np.isinf(loss).any())
    return dict(reference_panels_total=int(valid.size), reference_panels_valid=int(valid.sum()),
                zero_prediction_panels=int(points["zero_prediction"].sum()),
                undefined_reference_panels=int((~valid).sum()),
                mean_direction_cosine=float(cosine.mean()) if len(cosine) else None,
                mean_beam_loss_db=None if infinite or not len(loss) else float(loss.mean()),
                infinite_mean_beam_loss=infinite,
                p90_beam_loss_db=None if not len(loss) or np.isinf(np.quantile(loss,.9,method="inverted_cdf")) else float(np.quantile(loss,.9,method="inverted_cdf")),
                beam_loss_over3_fraction=float((loss>3).mean()) if len(loss) else None,
                reference_mask_depends_on_prediction=False)

def aggregate_secondary(points):
    if not points["applicable"]:return points
    reference=points["reference_valid_correlation"]
    failed=reference & ~points["prediction_valid_correlation"]
    covariance=points["covariance_error"]
    invalid_covariance=bool(not np.isfinite(covariance).all())
    return dict(applicable=True,reference_correlation_cells=int(reference.sum()),
                failed_prediction_correlation_cells=int(failed.sum()),
                frequency_correlation_MAE=None if failed.any() or not reference.any() else float(points["frequency_correlation_absolute_error"][reference].mean()),
                mean_covariance_error=None if invalid_covariance else float(covariance.mean()),
                invalid_covariance=invalid_covariance,failed_predictions_silently_dropped=False)
