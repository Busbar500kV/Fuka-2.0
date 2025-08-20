# core/engine.py
import numpy as np
from typing import Optional, Callable
from .config import Config
from .env import build_env
from .organism import History
from .physics import step_physics, resample_row


class Engine:
    """Streaming-capable 1D engine with optional boundary-condition support.

    Notes
    -----
    - This engine works with either signature of `step_physics`:
        step_physics(..., band=...)
        step_physics(..., band=..., bc="reflect")
      If your physics does not accept `bc`, it's ignored here.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

        # Environment: E[t, x_env]
        self.env = build_env(cfg.env, self.rng)  # shape (T, X_env)
        self.T = int(cfg.frames)
        self.X = int(cfg.space)

        # Substrate timeline S[t, x]
        self.S = np.zeros((self.T, self.X), dtype=float)

        # History buffers
        self.hist = History()

    def step(self, t: int):
        # Resample the environment row to substrate resolution
        e_row = resample_row(self.env[t], self.X)

        prev = self.S[t - 1] if t > 0 else self.S[0]

        # Call physics; try to pass bc if supported
        try:
            cur, flux = step_physics(
                prev_S=prev,
                env_row=e_row,
                k_flux=self.cfg.k_flux,
                k_motor=self.cfg.k_motor,
                diffuse=self.cfg.diffuse,
                decay=self.cfg.decay,
                rng=self.rng,
                band=self.cfg.band,
                bc=self.cfg.bc,        # works if physics supports it
            )
        except TypeError:
            # Fallback for physics without `bc` argument
            cur, flux = step_physics(
                prev_S=prev,
                env_row=e_row,
                k_flux=self.cfg.k_flux,
                k_motor=self.cfg.k_motor,
                diffuse=self.cfg.diffuse,
                decay=self.cfg.decay,
                rng=self.rng,
                band=self.cfg.band,
            )

        self.S[t] = cur

        # Bookkeeping
        self.hist.t.append(t)
        self.hist.E_cell.append(float(np.mean(cur)))
        self.hist.E_env.append(float(np.mean(e_row)))
        self.hist.E_flux.append(float(flux))

    def run(self, progress_cb: Optional[Callable[[int], None]] = None):
        for t in range(self.T):
            self.step(t)
            if progress_cb is not None:
                progress_cb(t)