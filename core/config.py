from dataclasses import dataclass, asdict, field
from typing import List, Dict

@dataclass
class FieldCfg:
    length: int = 512
    frames: int = 1000
    noise_sigma: float = 0.01
    # [{"kind":"moving_peak","amp":1.0,"speed":0.0,"width":4.0,"start":100}]
    sources: List[Dict] = field(default_factory=lambda: [
        {"kind": "moving_peak", "amp": 1.0, "speed": 0.0, "width": 4.0, "start": 100}
    ])

@dataclass
class Config:
    seed: int = 0
    frames: int = 5000
    space: int = 64

    # physics knobs
    k_flux: float = 0.05         # env→substrate coupling (at gate band)
    k_motor: float = 0.20        # exploratory noise at gate band
    diffuse: float = 0.05        # local spread
    decay: float = 0.01          # loss per step
    band: int = 3                # gate (boundary) half-width (cells)
    bc: str = "reflect"          # "periodic" | "reflect" | "absorb" | "wall"
    gate_center: int = 0         # index (0..space-1) where the gate band is centered

    env: FieldCfg = field(default_factory=FieldCfg)

def default_config() -> Dict:
    d = asdict(Config())
    # make the default gate sit in the middle by default_config()
    d["gate_center"] = d["space"] // 2
    return d

def make_config_from_dict(d: Dict) -> Config:
    env_d = d.get("env", {})
    fcfg = FieldCfg(
        length=int(env_d.get("length", 512)),
        frames=int(env_d.get("frames", d.get("frames", 5000))),
        noise_sigma=float(env_d.get("noise_sigma", 0.01)),
        sources=env_d.get("sources", FieldCfg().sources),
    )
    bc = str(d.get("bc", "reflect")).lower()
    if bc not in ("periodic", "reflect", "absorb", "wall"):
        bc = "reflect"

    space = int(d.get("space", 64))
    gate_center = int(d.get("gate_center", space // 2)) % max(1, space)

    return Config(
        seed=int(d.get("seed", 0)),
        frames=int(d.get("frames", 5000)),
        space=space,
        k_flux=float(d.get("k_flux", 0.05)),
        k_motor=float(d.get("k_motor", 0.20)),
        diffuse=float(d.get("diffuse", 0.05)),
        decay=float(d.get("decay", 0.01)),
        band=int(d.get("band", 3)),
        bc=bc,
        gate_center=gate_center,
        env=fcfg,
    )