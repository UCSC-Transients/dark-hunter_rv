"""Method fusion v1: bias surfaces, σ inflation, tiered pick, discordance reject.

Produces calibrated adopted columns without mutating raw per-method RVs. No ML scorer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from . import config as dh_config

METHODS: tuple[str, ...] = ("mask_ccf", "template_fft", "strong_lines")

_RV_KEYS: dict[str, str] = {
    "mask_ccf": "mask_rv_kms",
    "template_fft": "template_rv_kms",
    "strong_lines": "strong_lines_rv_kms",
}
_ERR_KEYS: dict[str, str] = {
    "mask_ccf": "mask_err_kms",
    "template_fft": "template_err_kms",
    "strong_lines": "strong_lines_err_kms",
}
_VALID_KEYS: dict[str, str] = {
    "mask_ccf": "mask_valid",
    "template_fft": "template_valid",
    "strong_lines": "strong_lines_valid",
}


@dataclass(frozen=True)
class BiasSurface:
    """Additive bias Δ(Teff) [km/s]; calibrated RV = raw − Δ.

    Use ``constant_kms`` alone, or linear interpolation on ``teff_knots_k`` / ``delta_kms``
    (same length, ≥2 knots). Empty knots → constant only.
    """

    constant_kms: float = 0.0
    teff_knots_k: tuple[float, ...] = ()
    delta_kms: tuple[float, ...] = ()

    def delta(self, teff: float) -> float:
        """Return Δ at ``teff`` (K). Non-finite Teff → ``constant_kms`` only."""
        t = float(teff)
        knots = self.teff_knots_k
        vals = self.delta_kms
        if len(knots) >= 2 and len(knots) == len(vals) and np.isfinite(t):
            return float(np.interp(t, np.asarray(knots, float), np.asarray(vals, float)))
        return float(self.constant_kms)


@dataclass(frozen=True)
class FusionConfig:
    """Tunable fusion policy (opt-in; defaults are conservative starting values)."""

    hot_teff_k: float = float(dh_config.HOT_STAR_TEFF_THRESHOLD)
    mask_snr_min: float = float(10.0 ** float(dh_config.METHOD_REGION_LOG10_SNR_MIN))
    discord_eta_kms: float = 10.0
    discord_sigma_mult: float = 5.0
    inflation_k: float = 1.0
    sigma_floor_mask_kms: float = 0.0
    sigma_floor_template_kms: float = 0.0
    sigma_floor_strong_lines_kms: float = 0.0
    bias_mask: BiasSurface = field(default_factory=BiasSurface)
    bias_template: BiasSurface = field(default_factory=BiasSurface)
    bias_strong_lines: BiasSurface = field(default_factory=BiasSurface)

    def bias_for(self, method: str) -> BiasSurface:
        if method == "mask_ccf":
            return self.bias_mask
        if method == "template_fft":
            return self.bias_template
        if method == "strong_lines":
            return self.bias_strong_lines
        raise KeyError(f"unknown method {method!r}")

    def sigma_floor_kms(self, method: str) -> float:
        if method == "mask_ccf":
            return float(self.sigma_floor_mask_kms)
        if method == "template_fft":
            return float(self.sigma_floor_template_kms)
        if method == "strong_lines":
            return float(self.sigma_floor_strong_lines_kms)
        raise KeyError(f"unknown method {method!r}")


def calibrated_rv_kms(raw_rv: float, method: str, teff: float, cfg: FusionConfig) -> float:
    """Return raw − Δ_m(Teff). Non-finite raw stays non-finite."""
    rv = float(raw_rv)
    if not np.isfinite(rv):
        return float("nan")
    return rv - float(cfg.bias_for(method).delta(teff))


def inter_method_spread_kms(
    calibrated: Mapping[str, float],
    valid: Mapping[str, bool],
) -> float:
    """Max pairwise |Δ| among valid calibrated RVs; NaN if fewer than two valid."""
    vals = [
        float(calibrated[m])
        for m in METHODS
        if bool(valid.get(m, False)) and np.isfinite(float(calibrated.get(m, float("nan"))))
    ]
    if len(vals) < 2:
        return float("nan")
    spread = 0.0
    for i in range(len(vals)):
        for j in range(i + 1, len(vals)):
            spread = max(spread, abs(vals[i] - vals[j]))
    return float(spread)


def inflate_sigma_kms(
    sigma_formal: float,
    method: str,
    inter_method_spread: float,
    cfg: FusionConfig,
    *,
    n_methods_valid: int,
) -> float:
    """σ_eff = sqrt(σ² + floor² + (k·spread)²) when ≥2 methods valid; else sqrt(σ² + floor²)."""
    s = float(sigma_formal)
    if not np.isfinite(s) or s <= 0:
        return float("nan")
    floor = float(cfg.sigma_floor_kms(method))
    extra = 0.0
    if int(n_methods_valid) >= 2 and np.isfinite(float(inter_method_spread)):
        extra = float(cfg.inflation_k) * float(inter_method_spread)
    return float(np.sqrt(s * s + floor * floor + extra * extra))


def select_method_tiered(
    flags: Mapping[str, Any],
    *,
    teff: float,
    cfg: FusionConfig,
) -> str:
    """Tiered policy v1 (no ML).

    Hot Teff → template → strong → mask.
    Else mask if valid and median mask CCF S/N ≥ ``mask_snr_min``;
    else template → strong → mask.
    """
    t = float(teff)
    snr = float(flags.get("median_mask_ccf_peak_snr", float("nan")))
    mask_ok = bool(flags.get("mask_valid"))
    tpl_ok = bool(flags.get("template_valid"))
    sl_ok = bool(flags.get("strong_lines_valid"))
    snr_ok = np.isfinite(snr) and snr >= float(cfg.mask_snr_min)

    if np.isfinite(t) and t > float(cfg.hot_teff_k):
        order = ("template_fft", "strong_lines", "mask_ccf")
    elif mask_ok and snr_ok:
        return "mask_ccf"
    else:
        order = ("template_fft", "strong_lines", "mask_ccf")

    for meth in order:
        if meth == "mask_ccf" and mask_ok:
            return "mask_ccf"
        if meth == "template_fft" and tpl_ok:
            return "template_fft"
        if meth == "strong_lines" and sl_ok:
            return "strong_lines"
    return ""


def discordance_reject(
    calibrated: Mapping[str, float],
    sigma_formal: Mapping[str, float],
    valid: Mapping[str, bool],
    cfg: FusionConfig,
) -> tuple[bool, str]:
    """Reject when any valid pair disagrees more than max(η, c·min(σ_a,σ_b)).

    Returns ``(should_reject, reason)``. Reason empty when accept.
    """
    active = [
        m
        for m in METHODS
        if bool(valid.get(m, False))
        and np.isfinite(float(calibrated.get(m, float("nan"))))
        and np.isfinite(float(sigma_formal.get(m, float("nan"))))
        and float(sigma_formal[m]) > 0
    ]
    if len(active) < 2:
        return False, ""

    eta = float(cfg.discord_eta_kms)
    c = float(cfg.discord_sigma_mult)
    for i, a in enumerate(active):
        for b in active[i + 1 :]:
            d = abs(float(calibrated[a]) - float(calibrated[b]))
            smin = min(float(sigma_formal[a]), float(sigma_formal[b]))
            thr = max(eta, c * smin)
            if d > thr:
                return True, "discordance"
    return False, ""


def fuse_exposure(
    flags: Mapping[str, Any],
    *,
    teff: float | None = None,
    cfg: FusionConfig | None = None,
) -> dict[str, Any]:
    """Fuse one exposure from ``exposure_method_flags``-like dict.

    Returns calibrated per-method RVs, effective σ for the chosen method, and
    ``rv_accepted`` / ``reject_reason``. Raw flag RVs are not modified.
    """
    conf = cfg if cfg is not None else FusionConfig()
    t = float(teff) if teff is not None else float("nan")

    valid = {m: bool(flags.get(_VALID_KEYS[m], False)) for m in METHODS}
    raw_rv = {m: float(flags.get(_RV_KEYS[m], float("nan"))) for m in METHODS}
    raw_err = {m: float(flags.get(_ERR_KEYS[m], float("nan"))) for m in METHODS}

    # Validity still requires finite RV/err; calibrated inherits validity.
    for m in METHODS:
        if not (
            valid[m]
            and np.isfinite(raw_rv[m])
            and np.isfinite(raw_err[m])
            and raw_err[m] > 0
        ):
            valid[m] = False

    calibrated = {
        m: calibrated_rv_kms(raw_rv[m], m, t, conf) if valid[m] else float("nan") for m in METHODS
    }
    n_valid = int(sum(1 for m in METHODS if valid[m]))
    spread = inter_method_spread_kms(calibrated, valid)

    method = select_method_tiered(flags, teff=t, cfg=conf)
    if method and not valid.get(method, False):
        method = ""
    if not method:
        # Fall back: first valid in cascade order (cool preference).
        for m in METHODS:
            if valid[m]:
                method = m
                break

    reject = False
    reason = ""
    if n_valid == 0 or not method:
        reject = True
        reason = "no_valid_method"
    else:
        disc_reject, disc_reason = discordance_reject(calibrated, raw_err, valid, conf)
        if disc_reject:
            reject = True
            reason = disc_reason

    rv_cal = float(calibrated[method]) if method and valid.get(method, False) else float("nan")
    sig_form = float(raw_err[method]) if method and valid.get(method, False) else float("nan")
    sig_eff = (
        inflate_sigma_kms(sig_form, method, spread, conf, n_methods_valid=n_valid)
        if method and np.isfinite(sig_form)
        else float("nan")
    )

    out: dict[str, Any] = {
        "adopted_method_v2": method if not reject else "",
        "rv_calibrated_kms": rv_cal if not reject else float("nan"),
        "sigma_eff_kms": sig_eff if not reject else float("nan"),
        "rv_accepted": bool((not reject) and np.isfinite(rv_cal) and np.isfinite(sig_eff)),
        "reject_reason": reason if reject else "",
        "inter_method_spread_kms": spread,
        "n_methods_valid_fusion": n_valid,
    }
    for m in METHODS:
        out[f"{m}_rv_calibrated_kms"] = calibrated[m]
        out[f"{m}_sigma_eff_kms"] = (
            inflate_sigma_kms(raw_err[m], m, spread, conf, n_methods_valid=n_valid)
            if valid[m]
            else float("nan")
        )
    return out
