import numpy as np
from scipy.ndimage import gaussian_filter1d

def resample_row(row: np.ndarray, target_len: int) -> np.ndarray:
    """Resample a 1D row to `target_len` using linear interpolation."""
    src_len = len(row)
    if src_len == target_len:
        return row
    x_src = np.linspace(0, 1, src_len)
    x_tgt = np.linspace(0, 1, target_len)
    return np.interp(x_tgt, x_src, row)

def _band_mask(X: int, center: int, half_width: int, bc: str) -> np.ndarray:
    """1 where the gate band is active; 0 elsewhere."""
    mask = np.zeros(X, dtype=float)
    hw = max(0, int(half_width))
    c = int(center)
    if bc == "periodic":
        for dx in range(-hw, hw + 1):
            mask[(c + dx) % X] = 1.0
    else:
        for dx in range(-hw, hw + 1):
            j = c + dx
            if 0 <= j < X:
                mask[j] = 1.0
    return mask

def _mode_for_gaussian(bc: str) -> str:
    return {
        "periodic": "wrap",
        "reflect":  "reflect",
        "absorb":   "constant",
        "wall":     "constant",
    }.get(bc, "reflect")

def step_physics(
    prev_S: np.ndarray,
    env_row: np.ndarray,
    k_flux: float,
    k_motor: float,
    diffuse: float,
    decay: float,
    rng: np.random.Generator,
    band: int = 3,
    bc: str = "reflect",
    center: int = 0,
) -> tuple[np.ndarray, float]:
    """
    One physics update step with a place-able 'gate' band that couples env→substrate.

    - Flux and motor act only on the gate band (center ± band).
    - Diffusion is BC-aware via gaussian_filter1d(mode=...).
    """
    X = len(prev_S)
    gate = _band_mask(X, center=center, half_width=band, bc=bc)

    # External flux (only through gate cells)
    drive = env_row * gate
    flux = k_flux * (drive - prev_S * gate)  # only gate cells are driven

    # Motor/self exploration (random pushes at the gate cells)
    motor = np.zeros_like(prev_S)
    if gate.any() and k_motor != 0.0:
        n = int(gate.sum())
        motor[gate > 0] = k_motor * rng.random(n)

    # Base update
    new_S = prev_S + flux + motor

    # Diffusion: BC-aware smoothing blended by `diffuse`
    if diffuse > 0:
        smoothed = gaussian_filter1d(new_S, sigma=max(1.0, band), mode=_mode_for_gaussian(bc), cval=0.0)
        new_S = (1.0 - diffuse) * new_S + diffuse * smoothed

    # Decay
    new_S *= (1.0 - decay)

    # Tiny ambient noise everywhere (helps break symmetry)
    new_S += 0.01 * rng.standard_normal(size=X)

    # Flux metric for plots
    flux_sum = float(np.sum(np.abs(flux)))
    return new_S, flux_sum