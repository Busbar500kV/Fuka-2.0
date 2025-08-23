# core/config.py
from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Tuple


# ---------------------------
# Dataclass schema (no hardcoded numeric defaults)
# ---------------------------
@dataclass
class FieldCfg:
    length: int
    frames: int
    noise_sigma: float
    height: int               # 1 => 1D env; >1 => 2D env E[t, y, x]
    sources: List[Dict[str, Any]]


@dataclass
class Config:
    seed: int
    frames: int
    space: int

    # physics knobs
    k_flux: float
    k_motor: float
    diffuse: float
    decay: float
    band: int                 # kept for compatibility with UI/engine
    bc: str                   # "periodic" | "reflect" | "absorb" | "wall"

    env: FieldCfg


# ---------------------------
# Public helpers
# ---------------------------
def default_config() -> Dict[str, Any]:
    """
    Minimal fallback used by the app when defaults.json is missing.
    Intentional: returns an empty dict to force the UI to nudge the user
    to supply a proper defaults.json, instead of silently hardcoding values.
    """
    return {}  # NO hidden numeric defaults here.


def required_keys() -> Tuple[List[str], List[str]]:
    """
    Lists of required keys for top-level and env sections.
    Useful for UI warnings.
    """
    top = [
        "seed", "frames", "space",
        "k_flux", "k_motor", "diffuse", "decay",
        "band", "bc", "env"
    ]
    env = ["length", "height", "frames", "noise_sigma", "sources"]
    return top, env


def schema_template() -> Dict[str, Any]:
    """
    A complete example config (for UX/help text). This is NOT used
    to inject defaults; it’s only a template you may show in the UI
    when keys are missing.
    """
    return {
        "seed": 0,
        "frames": 2000,
        "space": 64,

        "k_flux": 0.08,
        "k_motor": 0.20,
        "diffuse": 0.05,
        "decay": 0.05,
        "band": 3,
        "bc": "reflect",

        "env": {
            "length": 256,
            "height": 256,
            "frames": 2000,
            "noise_sigma": 0.01,
            "sources": [
                {
                    "kind": "moving_peak_2d",
                    "amp": 1.0,
                    "speed_x": 0.015,
                    "speed_y": 0.010,
                    "width_x": 6.0,
                    "width_y": 6.0,
                    "start_x": 64,
                    "start_y": 64
                },
                {
                    "kind": "moving_peak_2d",
                    "amp": 0.8,
                    "speed_x": -0.012,
                    "speed_y": 0.018,
                    "width_x": 8.0,
                    "width_y": 8.0,
                    "start_x": 192,
                    "start_y": 160
                },
                {
                    "kind": "moving_peak",
                    "amp": 0.6,
                    "speed": 0.02,
                    "width": 5.0,
                    "start": 128,
                    "y_center": "mid",
                    "width_y": 18.0
                }
            ]
        },

        "chunk": 150,
        "live": True,

        "thr3d": 0.75,
        "max3d": 40000,

        "vis": {
            "heat_floor": 0.10,
            "heat_gamma": 1.0,
            "env_opacity": 1.0,
            "sub_opacity": 0.85
        }
    }


# ---------------------------
# Validation & construction
# ---------------------------
_ALLOWED_BC = {"periodic", "reflect", "absorb", "wall"}


def validate_config_dict(d: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """
    Returns (errors, warnings).
    - errors: missing required keys or invalid values that must be fixed.
    - warnings: non-fatal issues (e.g., unknown bc string which will be lowercased).
    """
    errors: List[str] = []
    warnings: List[str] = []

    if not isinstance(d, dict):
        return [f"Top-level config is not a dict (got {type(d).__name__})."], warnings

    top_req, env_req = required_keys()

    # Missing top-level keys
    for k in top_req:
        if k not in d:
            errors.append(f"Missing top-level key: '{k}'")

    # If env exists, check its inner keys
    env = d.get("env", None)
    if isinstance(env, dict):
        for k in env_req:
            if k not in env:
                errors.append(f"Missing env key: 'env.{k}'")
    else:
        if "env" in d:
            errors.append("Key 'env' must be a dict.")
        # else already reported as missing

    # bc validity (only if present)
    bc = str(d.get("bc", "")).lower()
    if bc and bc not in _ALLOWED_BC:
        errors.append(f"Invalid 'bc'='{d.get('bc')}', allowed: {sorted(_ALLOWED_BC)}")

    # Basic type checks (best-effort, only when keys exist)
    def _need_number(path: str, v: Any):
        if not isinstance(v, (int, float)) and not (isinstance(v, bool)):
            errors.append(f"'{path}' must be a number, got {type(v).__name__}")

    num_paths = [
        "seed", "frames", "space",
        "k_flux", "k_motor", "diffuse", "decay", "band",
        "env.length", "env.height", "env.frames", "env.noise_sigma",
    ]
    for p in num_paths:
        ref = d
        parts = p.split(".")
        ok = True
        for part in parts:
            if isinstance(ref, dict) and part in ref:
                ref = ref[part]
            else:
                ok = False
                break
        if ok:
            _need_number(p, ref)

    # sources must be a list
    if isinstance(env, dict) and "sources" in env and not isinstance(env["sources"], list):
        errors.append("'env.sources' must be a list")

    return errors, warnings


def make_config_from_dict(d: Dict[str, Any], *, strict: bool = True) -> Config:
    """
    Build a Config strictly from a JSON dict.
    If strict=True (default), raises ValueError when required keys are missing/invalid.
    """
    errors, warnings = validate_config_dict(d)
    if errors and strict:
        # Aggregate all errors so the UI can show them at once
        raise ValueError("Invalid configuration:\n  - " + "\n  - ".join(errors))

    # Normalize bc
    bc_in = str(d.get("bc", "")).lower()
    if bc_in not in _ALLOWED_BC:
        # If strict, this would already have raised.
        # If non-strict, fall back to 'reflect'.
        bc_in = "reflect"

    env_d = d.get("env", {})
    fcfg = FieldCfg(
        length=int(env_d["length"]),
        frames=int(env_d["frames"]),
        noise_sigma=float(env_d["noise_sigma"]),
        height=int(env_d["height"]),
        sources=env_d["sources"],
    )

    return Config(
        seed=int(d["seed"]),
        frames=int(d["frames"]),
        space=int(d["space"]),
        k_flux=float(d["k_flux"]),
        k_motor=float(d["k_motor"]),
        diffuse=float(d["diffuse"]),
        decay=float(d["decay"]),
        band=int(d["band"]),
        bc=bc_in,
        env=fcfg,
    )