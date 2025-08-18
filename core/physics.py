import numpy as np

def resample_row(row: np.ndarray, target_len: int) -> np.ndarray:
    """Linear resample of a 1D row to target_len."""
    src = row.shape[0]
    if src == target_len:
        return row
    x_src = np.linspace(0.0, 1.0, src)
    x_tgt = np.linspace(0.0, 1.0, target_len)
    return np.interp(x_tgt, x_src, row)

def _neighbors_1d(a: np.ndarray, bc: str) -> tuple[np.ndarray, np.ndarray]:
    """Left/right neighbors under boundary condition."""
    X = a.size
    left  = np.empty_like(a)
    right = np.empty_like(a)
    if bc == "periodic":
        left  = np.roll(a,  1)
        right = np.roll(a, -1)
    elif bc == "reflect":
        left[0] = a[0];          left[1:]  = a[:-1]
        right[-1] = a[-1];       right[:-1] = a[1:]
    elif bc == "absorb" or bc == "wall":
        left[0] = 0.0;           left[1:]  = a[:-1]
        right[-1] = 0.0;         right[:-1] = a[1:]
    else:  # fallback
        left  = np.roll(a,  1)
        right = np.roll(a, -1)
    return left, right

def step_physics(
    prev_S: np.ndarray,
    env_row: np.ndarray,
    k_flux: float,
    k_motor: float,
    diffuse: float,
    decay: float,
    rng: np.random.Generator,
    band: int,
    bc: str,
) -> tuple[np.ndarray, float]:
    """
    One local update step.
      - env→substrate coupling acts only on a 'band' at the left edge (x in [0, band))
      - motor = zero-mean exploration noise at the same band
      - diffusion via discrete Laplacian under requested boundary condition
    """
    X = prev_S.size
    band = max(1, min(int(band), X))
    mask = np.zeros(X, dtype=float)
    mask[:band] = 1.0

    # env flux into boundary band
    drive = env_row - prev_S
    flux = k_flux * drive * mask

    # exploratory motor noise on band
    motor = np.zeros_like(prev_S)
    if k_motor > 0:
        motor[:band] = k_motor * rng.normal(size=band)

    # diffusion (nearest-neighbor Laplacian)
    left, right = _neighbors_1d(prev_S, bc)
    lap = left + right - 2.0 * prev_S

    new_S = prev_S + flux + motor + diffuse * lap
    new_S *= (1.0 - decay)

    # keep non-negative “stored energy”
    new_S = np.maximum(new_S, 0.0)

    # a gentle cap to avoid blow-ups (doesn't usually bind)
    if new_S.max() > 1e6:
        new_S = np.minimum(new_S, 1e6)

    return new_S, float(np.sum(np.abs(flux)))