# core/engine.py
import numpy as np
from .config import Config
from .env import build_env
from .organism import History
from .physics import step_physics, resample_row


def _resample_2d(frame: np.ndarray, new_y: int, new_x: int) -> np.ndarray:
    """
    Resample a 2D array [Y0, X0] -> [new_y, new_x] with separable linear interpolation.
    Keeps periodic structure visually reasonable for our purposes.
    """
    Y0, X0 = frame.shape
    if (Y0, X0) == (new_y, new_x):
        return frame

    # interp along X for each row
    if X0 != new_x:
        x_src = np.linspace(0.0, 1.0, X0)
        x_tgt = np.linspace(0.0, 1.0, new_x)
        tmp = np.zeros((Y0, new_x), dtype=float)
        for y in range(Y0):
            tmp[y] = np.interp(x_tgt, x_src, frame[y])
    else:
        tmp = frame

    # interp along Y for each column
    if Y0 != new_y:
        y_src = np.linspace(0.0, 1.0, Y0)
        y_tgt = np.linspace(0.0, 1.0, new_y)
        out = np.zeros((new_y, new_x), dtype=float)
        for x in range(new_x):
            out[:, x] = np.interp(y_tgt, y_src, tmp[:, x])
        return out
    else:
        return tmp


class Engine:
    """
    Streaming-capable engine.

    - 1D mode: env has shape (T, X_env), substrate S has shape (T, X).
    - 2D mode: env has shape (T, Y_env, X_env), substrate S has shape (T, Y, X),
               where Y = X = cfg.space (square substrate). Each row (y) is updated
               by the existing 1D step_physics, so physics.py does not need changes.
    """
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

        # Build full environment timeline
        self.env = build_env(cfg.env, self.rng)                 # (T, X_env) or (T, Y_env, X_env)
        self.T   = int(cfg.frames)

        # Determine dimensionality
        if self.env.ndim == 2:
            # 1D case
            self.mode = "1d"
            self.X = int(cfg.space)
            self.S = np.zeros((self.T, self.X), dtype=float)
        elif self.env.ndim == 3:
            # 2D case
            self.mode = "2d"
            self.Y = int(cfg.space)          # square substrate by design
            self.X = int(cfg.space)
            self.S = np.zeros((self.T, self.Y, self.X), dtype=float)
        else:
            raise ValueError(f"Unsupported env dimensions: {self.env.shape}")

        self.hist = History()

        # Optional gate center (used only by 1D step_physics that expects it)
        self._gate_center = getattr(self.cfg, "gate_center", 0)

    # ---------- stepping ----------
    def _step_1d(self, t: int):
        e_row = self.env[t]                            # (X_env,)
        e_row = resample_row(e_row, self.X)            # -> (X,)

        prev = self.S[t-1] if t > 0 else self.S[0]
        cur, flux = step_physics(
            prev_S=prev,
            env_row=e_row,
            k_flux=self.cfg.k_flux,
            k_motor=self.cfg.k_motor,
            diffuse=self.cfg.diffuse,
            decay=self.cfg.decay,
            rng=self.rng,
            band=self.cfg.band,
            bc=self.cfg.bc,
            center=self._gate_center,
        )
        self.S[t] = cur

        # bookkeeping (scalars)
        self.hist.t.append(t)
        self.hist.E_cell.append(float(np.mean(cur)))
        self.hist.E_env.append(float(np.mean(e_row)))
        self.hist.E_flux.append(float(flux))

    def _step_2d(self, t: int):
        # env slice (Y_env, X_env) -> resample to (Y, X)
        E_t = self.env[t]
        E_t = _resample_2d(E_t, new_y=self.Y, new_x=self.X)

        # previous substrate (Y, X)
        prev = self.S[t-1] if t > 0 else self.S[0]
        cur  = np.empty_like(prev)
        flux_accum = 0.0

        # Update row-by-row using the existing 1D physics
        for y in range(self.Y):
            row_prev = prev[y]                # (X,)
            row_env  = E_t[y]                 # (X,)
            row_cur, row_flux = step_physics(
                prev_S=row_prev,
                env_row=row_env,
                k_flux=self.cfg.k_flux,
                k_motor=self.cfg.k_motor,
                diffuse=self.cfg.diffuse,
                decay=self.cfg.decay,
                rng=self.rng,
                band=self.cfg.band,
                bc=self.cfg.bc,
                center=self._gate_center,     # keeps compatibility; can remove later
            )
            cur[y] = row_cur
            flux_accum += float(row_flux)

        self.S[t] = cur

        # bookkeeping (scalar summaries)
        self.hist.t.append(t)
        self.hist.E_cell.append(float(np.mean(cur)))
        self.hist.E_env.append(float(np.mean(E_t)))
        # average flux per row (so scale is comparable to 1D)
        self.hist.E_flux.append(float(flux_accum / max(1, self.Y)))

    def step(self, t: int):
        if self.mode == "1d":
            self._step_1d(t)
        else:
            self._step_2d(t)

    # ---------- run ----------
    def run(self, progress_cb=None):
        for t in range(self.T):
            self.step(t)
            if progress_cb is not None:
                progress_cb(t)