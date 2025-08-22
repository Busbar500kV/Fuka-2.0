import json
import os
from copy import deepcopy
from typing import Any, Dict, List

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from core.config import default_config, make_config_from_dict
from core.engine import Engine

# ---------- Page ----------
st.set_page_config(page_title="Fuka 2.0 — Free‑Energy Simulation", layout="wide")

# ---------- Load defaults ----------
def load_defaults() -> Dict[str, Any]:
    path = "defaults.json"
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                data = json.load(f)
                return data
        except Exception:
            pass
    return default_config()

cfg_default: Dict[str, Any] = load_defaults()

# ensure streaming helpers exist (if not provided in defaults)
if "chunk" not in cfg_default:
    cfg_default["chunk"] = 150
if "live" not in cfg_default:
    cfg_default["live"] = True

# ---------- session keys ----------
if "run_id" not in st.session_state:
    st.session_state["run_id"] = 0
for base in ("combo2d", "energy", "combo3d"):
    keyname = f"{base}_count"
    if keyname not in st.session_state:
        st.session_state[keyname] = 0

def new_key(base: str) -> str:
    """Unique key per draw within a run to satisfy Streamlit."""
    st.session_state[base + "_count"] += 1
    return f"{base}_{st.session_state['run_id']}_{st.session_state[base + '_count']}"

# ---------- Dynamic UI builders ----------
def _num_step(v: float | int) -> float:
    """Heuristic step for number_input."""
    try:
        mag = abs(float(v))
        if mag >= 1000: return 100.0
        if mag >= 100:  return 10.0
        if mag >= 10:   return 1.0
        if mag >= 1:    return 0.1
        if mag > 0:     return 0.01
    except Exception:
        pass
    return 1.0

def render_scalar(label: str, value: Any) -> Any:
    """Render a scalar control based on type; return possibly updated value."""
    if isinstance(value, bool):
        return st.checkbox(label, value=value)
    if isinstance(value, int) and not isinstance(value, bool):
        step = int(max(1, round(_num_step(value))))
        # very forgiving limits to avoid BelowMin errors when defaults change
        return st.number_input(label, value=int(value), step=step, min_value=-1_000_000_000, max_value=1_000_000_000)
    if isinstance(value, float):
        step = _num_step(value)
        return st.number_input(label, value=float(value), step=step, min_value=-1e9, max_value=1e9, format="%.6f")
    if isinstance(value, str):
        # small convenience: if looks like BC enum, use a selectbox
        enum = ["reflect","periodic","absorb","wall"]
        if value in enum:
            return st.selectbox(label, options=enum, index=enum.index(value))
        return st.text_input(label, value=value)
    # Fallback: JSON editor for unknown scalars
    txt = st.text_area(label + " (as JSON)", value=json.dumps(value, indent=2), height=120)
    try:
        return json.loads(txt)
    except Exception:
        st.warning(f"Invalid JSON for {label}; keeping previous value.")
        return value

def render_list(label: str, lst: List[Any]) -> List[Any]:
    """
    Lists can be arbitrary (e.g., env.sources). We present as JSON for full flexibility.
    """
    txt = st.text_area(label + " (JSON list)", value=json.dumps(lst, indent=2), height=220)
    try:
        parsed = json.loads(txt)
        if not isinstance(parsed, list):
            st.error(f"{label} must be a JSON list; keeping previous value.")
            return lst
        return parsed
    except Exception as e:
        st.error(f"{label} JSON error: {e}; keeping previous value.")
        return lst

def render_object(title: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively render a dict into widgets. Returns a NEW dict with updated values.
    Special-case: if a key name looks like 'sources' and value is a list, always JSON-edit it.
    """
    st.subheader(title)
    out: Dict[str, Any] = {}
    for k, v in data.items():
        label = f"{title}.{k}" if title else k

        # Group well-known nested objects with a nice header
        if isinstance(v, dict):
            with st.expander(k, expanded=True):
                out[k] = render_object(k, v)
        elif isinstance(v, list):
            # Special-case for 'sources' (or any list): JSON editor for full freedom
            out[k] = render_list(k, v)
        else:
            out[k] = render_scalar(k, v)
    return out

# ---------- Sidebar: fully dynamic from defaults ----------
with st.sidebar:
    st.header("Configuration (auto‑generated)")
    user_cfg = render_object("", deepcopy(cfg_default))

# Pull out streaming knobs for app behavior (not part of Engine Config)
chunk = int(user_cfg.pop("chunk", 150))
live  = bool(user_cfg.pop("live", True))

# ---------- Layout placeholders ----------
st.title("Simulation")
combo2d_ph = st.empty()
energy_ph  = st.empty()
plot3d_ph  = st.empty()

# ---------- Plot helpers ----------
def _norm(A: np.ndarray) -> np.ndarray:
    m, M = float(np.nanmin(A)), float(np.nanmax(A))
    if M - m < 1e-12:
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
    # Handles 1‑D (T,X). If you feed 2‑D (T,Y,X), take mean over Y to keep a single x‑axis.
    if E.ndim == 3:
        E2 = E.mean(axis=1)
    else:
        E2 = E
    if S.ndim == 3:
        S2 = S.mean(axis=1)
    else:
        S2 = S

    if S2.shape[1] != E2.shape[1]:
        S2 = _resample_rows(S2, E2.shape[1])

    En = _norm(E2)
    Sn = _norm(S2)

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
    combo_key = new_key("combo2d")
    ph.plotly_chart(fig, use_container_width=True, theme=None, key=combo_key)

def draw_energy_timeseries(ph, t, e_cell, e_env, e_flux, title="Energy vs time"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=e_cell, name="E_cell"))
    fig.add_trace(go.Scatter(x=t, y=e_env,  name="E_env"))
    fig.add_trace(go.Scatter(x=t, y=e_flux, name="E_flux"))
    fig.update_layout(xaxis_title="t (frames)", yaxis_title="energy", title=title, height=380, template="plotly_dark")
    energy_key = new_key("energy")
    ph.plotly_chart(fig, use_container_width=True, theme=None, key=energy_key)

def draw_sparse_3d(ph, E: np.ndarray, S: np.ndarray, thr: float = 0.75, max_points: int = 40_000):
    # If 2‑D (T,Y,X), keep points over Y and color by source (Env/Sub)
    if E.ndim == 3:
        En = _norm(E)   # (T,Y,X)
        Sn = _norm(S)   # (T,Y,X) expected
        t_idx_E, y_idx_E, x_idx_E = np.where(En >= thr)
        t_idx_S, y_idx_S, x_idx_S = np.where(Sn >= thr)
        xE, yE, zE = x_idx_E, y_idx_E, t_idx_E
        xS, yS, zS = x_idx_S, y_idx_S, t_idx_S
    else:
        # 1‑D fallback (T,X) → put env at y=0, subs at y=1
        En = _norm(E)
        Sn = _norm(S)
        t_idx_E, x_idx_E = np.where(En >= thr)
        t_idx_S, x_idx_S = np.where(Sn >= thr)
        xE, yE, zE = x_idx_E, np.zeros_like(x_idx_E), t_idx_E
        xS, yS, zS = x_idx_S, np.ones_like(x_idx_S),  t_idx_S

    def _sub(x, y, z, vmax):
        n = len(x)
        if n > vmax:
            idx = np.random.choice(n, size=vmax, replace=False)
            return x[idx], y[idx], z[idx]
        return x, y, z

    xE, yE, zE = _sub(xE, yE, zE, max_points // 2)
    xS, yS, zS = _sub(xS, yS, zS, max_points // 2)

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=xE, y=yE, z=zE, mode="markers",
                               marker=dict(size=2, opacity=0.7), name="Env"))
    fig.add_trace(go.Scatter3d(x=xS, y=yS, z=zS, mode="markers",
                               marker=dict(size=2, opacity=0.7), name="Substrate"))
    fig.update_layout(
        title="Sparse 3‑D energy",
        scene=dict(
            xaxis_title="x",
            yaxis_title="y" if E.ndim == 3 else "layer",
            zaxis_title="t",
        ),
        height=640,
        template="plotly_dark",
        showlegend=True,
    )
    key3d = new_key("combo3d")
    ph.plotly_chart(fig, use_container_width=True, theme=None, key=key3d)

# ---------- Run ----------
if st.button("Run / Rerun", use_container_width=True):
    # new run: bump run id and reset per‑run counters
    st.session_state["run_id"] += 1
    for base in ("combo2d", "energy", "combo3d"):
        st.session_state[f"{base}_count"] = 0

    # Build typed Config for engine from the dynamic dict
    ecfg = make_config_from_dict(user_cfg)
    engine = Engine(ecfg)

    # 3‑D sliders (only shown here to keep UI dynamic)
    st.sidebar.subheader("3‑D (sparse) view")
    thr3d = st.sidebar.slider("3‑D energy threshold (0‑1)", 0.0, 1.0, 0.75, 0.01)
    max3d = st.sidebar.number_input("3‑D max points", 1_000, 200_000, 40_000, 1_000)

    def redraw(upto: int, final: bool = False):
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