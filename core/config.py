# core/config.py
from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any

# ---- Environment (can be 1D or 2D via height) ----
@dataclass
class FieldCfg:
    length: int = 512
    frames: int = 1000
    noise_sigma: float = 0.01
    height: int = 1  # if >1 => E[t, y, x], else E[t, x]
    sources: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"kind": "moving_peak", "amp": 1.0, "speed": 0.0, "width": 4.0, "start": 100}
    ])

# ---- Top-level config for the engine ----
@dataclass
class Config:
    seed: int = 0
    frames: int = 5000
    space: int = 64

    # “classic” knobs (still supported)
    k_flux: float = 0.05
    k_motor: float = 0.20
    diffuse: float = 0.05
    decay: float = 0.01
    band: int = 3
    bc: str = "reflect"  # "periodic" | "reflect" | "absorb" | "wall"

    # NEW: anything extra the physics step understands will live here
    physics: Dict[str, Any] = field(default_factory=dict)

    env: FieldCfg = field(default_factory=FieldCfg)

def default_config() -> Dict[str, Any]:
    """Conservative baseline; the app runs strict off defaults.json anyway."""
    return asdict(Config())

# Known physics knobs we’ll hoover up from either a "physics" block or top-level:
_PHYS_HINTS = {
    "alpha_speed", "beta_speed", "flux_limit", "T", "update_mode", "boundary_leak"
}

def make_config_from_dict(d: Dict[str, Any]) -> Config:
    # ---- env ----
    env_d = dict(d.get("env", {}))
    height = int(env_d.get("height", env_d.get("H", 1)))
    fcfg = FieldCfg(
        length=int(env_d.get("length", 512)),
        frames=int(env_d.get("frames", d.get("frames", 5000))),
        noise_sigma=float(env_d.get("noise_sigma", 0.01)),
        height=height,
        sources=env_d.get("sources", FieldCfg().sources),
    )

    # ---- bc normalization ----
    bc = str(d.get("bc", "reflect")).lower()
    if bc not in ("periodic", "reflect", "absorb", "wall"):
        bc = "reflect"

    # ---- collect physics kwargs ----
    physics_block = dict(d.get("physics", {}))  # optional nested block
    # also sweep top-level for hinted keys (like your current defaults.json)
    for k in _PHYS_HINTS:
        if k in d and k not in physics_block:
            physics_block[k] = d[k]

    return Config(
        seed=int(d.get("seed", 0)),
        frames=int(d.get("frames", 5000)),
        space=int(d.get("space", 64)),
        k_flux=float(d.get("k_flux", 0.05)),
        k_motor=float(d.get("k_motor", 0.20)),
        diffuse=float(d.get("diffuse", 0.05)),
        decay=float(d.get("decay", 0.01)),
        band=int(d.get("band", 3)),
        bc=bc,
        physics=physics_block,  # <- passed through by Engine to step_physics
        env=fcfg,
    )