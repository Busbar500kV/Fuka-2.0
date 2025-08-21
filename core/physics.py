# core/physics.py
from __future__ import annotations
import numpy as np
from scipy.ndimage import gaussian_filter


# ---------------------------
# Resampling helpers
# ---------------------------
def resample_row(row: np.ndarray, target_len: int) -> np.ndarray:
    """
    Resample a 1D row to `target_len` using linear interpolation.
    Kept for backward-compatibility with the current Engine (1-D).
    """
    row = np.asarray(row, dtype=float)
    src_len = row.shape[0]
    if src_len == target_len:
        return row
    x_src = np.linspace(0.0, 1.0, src_len)
    x_tgt = np.linspace(0.0, 1.0, target_len)
    return np.interp(x_tgt, x_src, row)


def resample_plane(img: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """
    Simple separable linear resample of a 2D array to (H_tgt, W_tgt).
    Not used by the 1-D engine, but provided for 2-D substrates.
    """
    img = np.asarray(img, dtype=float)
    H, W = img.shape
    Ht, Wt = int(target_hw[0]), int(target_hw[1])
    if (H, W) == (Ht, Wt):
        return img

    # resample rows -> (H, Wt)
    x_src = np.linspace(0.0, 1.0, W)
    x_tgt = np.linspace(0.0, 1.0, Wt)
    tmp = np.empty((H, Wt), dtype=float)
    for r in range(H):
        tmp[r] = np.interp(x_tgt, x_src, img[r])

    # resample columns -> (Ht, Wt)
    y_src = np.linspace(0.0, 1.0, H)
    y_tgt = np.linspace(0.0, 1.0, Ht)
    out = np.empty((Ht, Wt), dtype=float)
    for c in range(Wt):
        out[:, c] = np.interp(y_tgt, y_src, tmp[:, c])
    return out


# ---------------------------
# Internal utilities
# ---------------------------
def _mode_for_gaussian(bc: str) -> str:
    """
    Map our boundary condition name to scipy.ndimage.gaussian_filter `mode`.
    """
    return {
        "periodic": "wrap",
        "reflect":  "reflect",
        "absorb":   "constant",
        "wall":     "constant",
    }.get(str(bc).lower(), "reflect")


def _grad_mag(A: np.ndarray) -> np.ndarray:
    """
    Gradient magnitude for 1-D/2-D/N-D arrays using central differences.
    Returns array with same shape as A.
    """
    grads = np.gradient(A)
    if isinstance(grads, list):
        acc = np.zeros_like(A, dtype=float)
        for g in grads:
            acc += g.astype(float) ** 2
        return np.sqrt(acc, dtype=float)
    # 1-D fast path (np.gradient gave a single array)
    return np.abs(grads).astype(float)


# ---------------------------
# Main physics step (emergent boundary)
# ---------------------------
def step_physics(
    prev_S: np.ndarray,
    env_row: np.ndarray,
    k_flux: float,
    k_motor: float,
    diffuse: float,
    decay: float,
    rng: np.random.Generator,
    band: int = 0,               # accepted for compatibility; ignored
    bc: str = "reflect",
    **_unused,                   # accept extra kwargs gracefully
) -> tuple[np.ndarray, float]:
    """
    Emergent-boundary update (works for 1-D or 2-D arrays).

    Idea:
      - No hard-coded gate/band. Coupling strength is driven by where the
        *environment has stronger gradients than the substrate*.
      - Flux term is modulated by w = ReLU(|∇E| - |∇S|) normalized to [0,1].
      - Motor term is noise scaled by current substrate gradient magnitude,
        so exploration concentrates along wherever the substrate already
        has "edges" (proto-boundaries).
      - Diffusion is BC-aware; decay is global.

    Args
    ----
    prev_S : ndarray
        Substrate state at time t-1. Shape can be (X,) or (Y,X).
    env_row : ndarray
        Environment slice aligned with prev_S (same shape).
    k_flux, k_motor, diffuse, decay : floats
        Coupling, exploration, diffusion mix, and loss.
    rng : np.random.Generator
        Random source.
    band : int
        Ignored (kept for API compatibility).
    bc : str
        Boundary condition: 'reflect' | 'periodic' | 'absorb' | 'wall'.

    Returns
    -------
    new_S : ndarray
        Updated substrate at time t (same shape as inputs).
    flux_metric : float
        Mean absolute flux, for plotting.
    """
    S = np.asarray(prev_S, dtype=float)
    E = np.asarray(env_row, dtype=float)
    if S.shape != E.shape:
        raise ValueError(f"step_physics: shape mismatch S{S.shape} vs E{E.shape}")

    # --- Gradient magnitudes (where edges live) ---
    gE = _grad_mag(E)
    gS = _grad_mag(S)

    # Coupling weight: where env edge > substrate edge
    w = np.maximum(gE - gS, 0.0)
    max_gE = float(np.max(gE)) if np.size(gE) else 0.0
    if max_gE > 0.0:
        w = w / (max_gE + 1e-12)

    # --- Flux term: pull S toward E, but only where w>0 ---
    flux = k_flux * w * (E - S)

    # --- Motor/exploration: noise concentrated along current S edges ---
    if k_motor != 0.0:
        max_gS = float(np.max(gS)) if np.size(gS) else 0.0
        if max_gS > 0.0:
            gS_scale = gS / (max_gS + 1e-12)
        else:
            gS_scale = 0.0
        motor = k_motor * rng.standard_normal(size=S.shape) * gS_scale
    else:
        motor = 0.0

    # --- Base update ---
    new_S = S + flux + motor

    # --- Diffusion (BC-aware gaussian blend) ---
    if diffuse > 0.0:
        sigma = 1.0  # small local smoothing in all available dims
        mode = _mode_for_gaussian(bc)
        smoothed = gaussian_filter(new_S, sigma=sigma, mode=mode, cval=0.0)
        new_S = (1.0 - diffuse) * new_S + diffuse * smoothed

    # --- Decay ---
    new_S *= (1.0 - decay)

    # --- Small ambient noise to avoid total stasis ---
    new_S += 1e-3 * rng.standard_normal(size=S.shape)

    # Flux metric for plots (mean absolute flux)
    flux_metric = float(np.mean(np.abs(flux)))
    return new_S, flux_metric