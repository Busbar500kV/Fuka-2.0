# core/config.py
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any

@dataclass
class FieldCfg:
    length: int = 512
    frames: int = 1000
    noise_sigma: float = 0.01
    # 2D env height (Y). If 1 => 1D env, if >1 => 2D env (E[t, y, x]).
    height: int = 1
    # Sources: 1D or 2D descriptors (see env.py for kinds)
    sources: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"kind": "moving_peak", "amp": 1.0, "speed": 0.0, "width": 4.0, "start": 100}
    ])

@dataclass
class Config:
    seed: int = 0
    frames: int = 5000
    space: int = 64

    # physics knobs (legacy/top‑level)
    k_flux: float = 0.05
    k_motor: float = 0.20
    diffuse: float = 0.05
    decay: float = 0.01
    band: int = 3
    bc: str = "reflect"

    # NEW: bag of extra physics kwargs (passed straight to step_physics)
    physics: Dict[str, Any] = field(default_factory=dict)

    env: FieldCfg = field(default_factory=FieldCfg)

def default_config() -> Dict[str, Any]:
    """Return a plain dict suitable for serialization/UI defaults."""
    return asdict(Config())

def make_config_from_dict(d: Dict[str, Any]) -> Config:
    # --- env ---
    env_d = d.get("env", {})
    fcfg = FieldCfg(
        length=int(env_d.get("length", 512)),
        frames=int(env_d.get("frames", d.get("frames", 5000))),
        noise_sigma=float(env_d.get("noise_sigma", 0.01)),
        height=int(env_d.get("height", env_d.get("H", 1))),
        sources=env_d.get("sources", FieldCfg().sources),
    )

    # --- bc ---
    bc = str(d.get("bc", "reflect")).lower()
    if bc not in ("periodic", "reflect", "absorb", "wall"):
        bc = "reflect"

    # --- physics passthrough (any extra kwargs for step_physics) ---
    physics_kwargs = d.get("physics", {})
    if not isinstance(physics_kwargs, dict):
        physics_kwargs = {}

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
        physics=physics_kwargs,
        env=fcfg,
    )