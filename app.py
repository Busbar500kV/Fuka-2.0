# app.py
from __future__ import annotations
import json, os
from copy import deepcopy
from typing import Any, Dict

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from core.config import default_config, make_config_from_dict
from core.engine import Engine


# ---------- Page ----------
st.set_page_config(page_title="Fuka 2.0 — Free‑Energy Simulation", layout="wide")

# ---------- Load defaults (from file if present) ----------
def load_defaults() -> Dict[str, Any]:
    path = "defaults.json"
    if os.path.exists(path):
        try:
            return json.load(open(path, "r"))
        except Exception:
            pass
    return default_config()

cfg_default = load_defaults()

# ---------- session keys for plot uniqueness ----------
for base in ("combo2d_count", "energy_count", "combo3d_count", "run_id"):
    if base not in st.session_state:
        st.session_state[base] = 0

def new_key(base: str) -> str:
    st.session_state[base + "_count"] += 1
    return f"{base}_{st.session_state['run_id']}_{st.session_state[base + '_count']}"

# ---------- Numeric UI helpers ----------
def _num_step(v: float) -> float:
    """Nice step size based on magnitude."""
    v = abs(float(v)) if v != 0 else 1.0
    m = 10 ** int(np.floor(np.log10(v)))
    return max(10 ** int(np.floor(np.log10(v)) - 2), m * 0.01)

# ---------- Dynamic sidebar renderers (path-keyed) ----------
def render_scalar(label: str, value: Any, path: str):
    """
    Render a single scalar with a unique key derived from JSON path.
    """
    key = f"w:{path}"
    if isinstance(value, bool):
        return st.checkbox(label, value=bool(value), key=key)
    if isinstance(value, int) and not isinstance(value, bool):
        step = int(max(1, round(_num_step(value))))
        # roomy bounds so defaults never violate min/max
        return st.number_input(label, value=int(value), step=step,
                               min_value=-10_000_000, max_value=10_000_000, key=key)
    if isinstance(value, float):
        step = _num_step(value)
        return st.number_input(label, value=float(value), step=step,
                               format="%.6f", key=key)
    if isinstance(value, str):
        return st.text_input(label, value=value, key=key)
    # Fallback: JSON editor for odd types
    return st.text_area(label, value=json.dumps(value, indent=2), key=key)

def render_list(label: str, value: list, path: str):
    """
    For arbitrary lists (e.g., env.sources) we present a JSON editor.
    """
    key = f"w:{path}"
    txt = st.text_area(f"{label} (JSON)", value=json.dumps(value, indent=2), height=220, key=key)
    try:
        return json.loads(txt)
    except Exception as e:
        st.warning(f"{label}: JSON parse error — using previous value. ({e})")
        return value

def render_object(label: str, obj: Dict[str, Any], path: str = "") -> Dict[str, Any]:
    """
    Recursively render a dict. Each widget gets a unique key from its path.
    """
    out: Dict[str, Any] = {}
    for k, v in obj.items():
        child_path = f"{path}.{k}" if path else k
        # Group nested dicts nicely under expanders
        if isinstance(v, dict):
            with st.expander(k, expanded=True):
                out[k] = render_object(k, v, path=child_path)
        elif isinstance(v, list):
            out[k] = render_list(k, v, path=child_path)
        else:
            out[k] = render_scalar(k, v, path=child_path)
    return out

# ---------- Sidebar: fully dynamic from defaults ----------
with st.sidebar:
    st.header("Configuration (auto‑generated)")
    user_cfg = render_object("", deepcopy(cfg_default))

# Pull out streaming & 3D knobs (if present) so they don't go into Engine config
chunk   = int(user_cfg.pop("chunk", 150))
live    = bool(user_cfg.pop("live", True))
thr3d   = float(user_cfg.pop("thr3d", 0.75)) if "thr3d" in cfg_default else 0.75
max3d   = int(user_cfg.pop("max3d", 40_000)) if "max3d" in cfg_default else 40_000

# ---------- Layout placeholders ----------
st.title("Simulation")
combo2d_ph = st.empty()
energy_ph  = st.empty()
plot3d_ph  = st.empty()

# ---------- Plot helpers ----------
def _norm(A: np.ndarray) -> np.ndarray:
    m = float(np.nanmin(A))
    M = float(np.nanmax(A))
    if not np.isfinite(m) or not np.isfinite(M) or M - m < 1e-12:
        return np.zeros_like(A)
    return (A - m) / (M - m + 1e-12)

def _resample_rows(M: np.ndarray, new_len: int) -> np.ndarray:
    T, X = M.shape
    if X == new_len:
        return M
    x_src = np.linspace(0.0, 1.0, X)
    x_tgt = np.linspace(0.0, 1.0, new_len)
    out = np.zeros((T, new_len), dtype=float)
    for t in range(T):
        out[t] = np.interp(x_tgt, x_src, M[t])
    return out

def draw_combined_heatmap(ph, E: np.ndarray, S: np.ndarray, title="Env + Substrate (combined, zoomable)"):
    # S and E may have different X; resample S to E’s width for shared x-axis
    if S.ndim == 3:  # if you later pass 2D substrate timeline: pick a summary or layer
        S = S[..., 0, :]  # example: first row; keep as your current 1D substrate timeline
    if E.ndim == 3:
        E = E[..., 0, :]  # example: show first env row in the combined 2D panel

    if S.shape[1] != E.shape[1]:
        S_res = _resample_rows(S, E.shape[1])
    else:
        S_res = S

    En = _norm(E)
    Sn = _norm(S_res)

    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=En, coloraxis="coloraxis", zsmooth=False, name="Env"))
    fig.add_trace(go.Heatmap(z=Sn, coloraxis="coloraxis2", zsmooth=False, opacity=0.85, name="Substrate"))
    fig.update_layout(
        title=title,
        xaxis_title="x (space)",
        yaxis_title="t (time)",
        coloraxis=dict(colorscale="Viridis", colorbar=dict(title="Env")),
        coloraxis2=dict(colorscale="Inferno", colorbar=dict(title="Substrate", x=1.08)),
        height=620,
        template="plotly_dark",
    )
    ph.plotly_chart(fig, use_container_width=True, theme=None, key=new_key("combo2d"))

def draw_energy_timeseries(ph, t, e_cell, e_env, e_flux, title="Energy vs time"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=e_cell, name="E_cell"))
    fig.add_trace(go.Scatter(x=t, y=e_env,  name="E_env"))
    fig.add_trace(go.Scatter(x=t, y=e_flux, name="E_flux"))
    fig.update_layout(
        xaxis_title="t (frames)",
        yaxis_title="energy",
        title=title,
        height=380,
        template="plotly_dark",
    )
    ph.plotly_chart(fig, use_container_width=True, theme=None, key=new_key("energy"))

def draw_sparse_3d(ph, E: np.ndarray, S: np.ndarray, thr: float, max_points: int):
    # For now: treat E and S as 2D (t,x) slices; if 3D env arrives, summarize one y-slice
    if E.ndim == 3:
        E = E[..., 0, :]
    if S.ndim == 3:
        S = S[..., 0, :]

    En = _norm(E)
    Sn = _norm(S)
    t_idx_E, x_idx_E = np.where(En >= thr)
    t_idx_S, x_idx_S = np.where(Sn >= thr)

    def _sub(x, y, z, vmax):
        if len(x) > vmax:
            idx = np.random.choice(len(x), size=vmax, replace=False)
            return x[idx], y[idx], z[idx]
        return x, y, z

    zE, yE, xE = t_idx_E, np.zeros_like(t_idx_E), x_idx_E
    zS, yS, xS = t_idx_S, np.ones_like(t_idx_S),  x_idx_S
    xE, yE, zE = _sub(xE, yE, zE, max_points // 2)
    xS, yS, zS = _sub(xS, yS, zS, max_points // 2)

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=xE, y=yE, z=zE, mode="markers",
                               marker=dict(size=2), name="Env"))
    fig.add_trace(go.Scatter3d(x=xS, y=yS, z=zS, mode="markers",
                               marker=dict(size=2), name="Substrate"))
    fig.update_layout(
        title="Sparse 3‑D energy (x, layer∈{Env,Sub}, t)",
        scene=dict(xaxis_title="x (space)", yaxis_title="layer", zaxis_title="t (time)"),
        height=640,
        template="plotly_dark",
        showlegend=True,
    )
    ph.plotly_chart(fig, use_container_width=True, theme=None, key=new_key("combo3d"))

# ---------- Run ----------
if st.button("Run / Rerun", use_container_width=True):
    # bump per‑run ids so plot keys are unique every run
    st.session_state["run_id"] += 1
    st.session_state["combo2d_count"] = 0
    st.session_state["energy_count"]  = 0
    st.session_state["combo3d_count"] = 0

    ecfg = make_config_from_dict(user_cfg)
    engine = Engine(ecfg)

    def redraw(upto: int, final: bool = False):
        # 2D combined panel and energy chart
        draw_combined_heatmap(combo2d_ph, engine.env[:upto+1], engine.S[:upto+1])
        draw_energy_timeseries(energy_ph, engine.hist.t, engine.hist.E_cell, engine.hist.E_env, engine.hist.E_flux)
        if final:
            draw_sparse_3d(plot3d_ph, engine.env, engine.S, thr=float(thr3d), max_points=int(max3d))

    if live:
        last = [-1]
        def cb(t: int):
            if t - last[0] >= int(chunk) or t == engine.T - 1:
                last[0] = t
                redraw(t, final=(t == engine.T - 1))
        engine.run(progress_cb=cb)
    else:
        engine.run(progress_cb=None)
        redraw(engine.T - 1, final=True)