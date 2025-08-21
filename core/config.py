# core/config.py
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any

@dataclass
class FieldCfg:
    length: int = 512
    frames: int = 1000
    noise_sigma: float = 0.01
    # >>> NEW: height for 2D env (Y). If 1 => 1D env, if >1 => 2D env (E[t, y, x]).
    height: int = 1
    # [{"kind":"moving_peak","amp":1.0,"speed":0.0,"width":4.0,"start":100}]
    # or 2D: {"kind":"moving_peak_2d","amp":1.0,"speed_x":0.0,"speed_y":0.0,
    #         "width_x":4.0,"width_y":4.0,"start_x":X//2,"start_y":Y//2}
    sources: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"kind": "moving_peak", "amp": 1.0, "speed": 0.0, "width": 4.0, "start": 100}
    ])

@dataclass
class Config:
    seed: int = 0
    frames: int = 5000
    space: int = 64

    # physics knobs
    k_flux: float = 0.05         # env→substrate coupling
    k_motor: float = 0.20        # exploratory noise
    diffuse: float = 0.05        # local spread
    decay: float = 0.01          # loss per step
    band: int = 3                # (kept for compat; physics ignores for emergent boundary)
    bc: str = "reflect"          # "periodic" | "reflect" | "absorb" | "wall"

    env: FieldCfg = field(default_factory=FieldCfg)

def default_config() -> Dict:
    return asdict(Config())

def make_config_from_dict(d: Dict) -> Config:
    env_d = d.get("env", {})
    # pull height if present; default to 1 (1D)
    height = int(env_d.get("height", env_d.get("H", 1)))

    fcfg = FieldCfg(
        length=int(env_d.get("length", 512)),
        frames=int(env_d.get("frames", d.get("frames", 5000))),
        noise_sigma=float(env_d.get("noise_sigma", 0.01)),
        height=height,
        sources=env_d.get("sources", FieldCfg().sources),
    )

    bc = str(d.get("bc", "reflect")).lower()
    if bc not in ("periodic", "reflect", "absorb", "wall"):
        bc = "reflect"

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
        env=fcfg,
    )