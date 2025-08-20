import numpy as np
from .config import Config
from .env import build_env
from .organism import History
from .physics import step_physics, resample_row

class Engine:
    """Streaming-capable 1D engine with boundary/gate placement and BCs."""
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

        self.env = build_env(cfg.env, self.rng)  # (T, X_env)
        self.T   = int(cfg.frames)
        self.X   = int(cfg.space)

        self.S   = np.zeros((self.T, self.X), dtype=float)
        self.hist = History()

    def step(self, t: int):
        e_row = resample_row(self.env[t], self.X)

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
            center=self.cfg.gate_center,
        )
        self.S[t] = cur

        # bookkeeping
        self.hist.t.append(t)
        self.hist.E_cell.append(float(np.mean(cur)))
        self.hist.E_env.append(float(np.mean(e_row)))
        self.hist.E_flux.append(float(flux))

    def run(self, progress_cb=None):
        for t in range(self.T):
            self.step(t)
            if progress_cb is not None:
                progress_cb(t)