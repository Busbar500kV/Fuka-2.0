import json
import os
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from core.config import default_config, make_config_from_dict
from core.engine import Engine

# ---------- Page ----------
st.set_page_config(page_title="Fuka 2.0 — Free-Energy Simulation", layout="wide")

# ---------- Load defaults (from file if present) ----------
def load_defaults():
    path = "defaults.json"
    if os.path.exists(path):
        try:
            return json.load(open(path, "r"))
        except Exception:
            pass
    return default_config()

cfg = load_defaults()

# ---------- Sidebar controls ----------
with st.sidebar:
    st.header("Run Controls")
    seed   = st.number_input("Seed", min_value=0, max_value=10_000_000, value=int(cfg["seed"]), step=1)
    frames = st.number_input("Frames", min_value=200, max_value=50_000, value=int(cfg["frames"]), step=200)
    space  = st.number_input("Substrate cells (space)", min_value=16, max_value=1024, value=int(cfg["space"]), step=16)

    st.divider()
    st.subheader("Physics")
    k_flux  = st.slider("k_flux (env→substrate @ boundary)", 0.0, 1.0, float(cfg["k_flux"]), 0.01)
    k_motor = st.slider("k_motor (motor noise @ boundary)", 0.0, 5.0,  float(cfg["k_motor"]), 0.01)
    diffuse = st.slider("diffuse (spread)",                  0.0, 0.5,  float(cfg["diffuse"]), 0.005)
    decay   = st.slider("decay (loss)",                      0.0, 0.2,  float(cfg["decay"]),   0.001)
    band    = st.number_input("band (boundary width)", min_value=1, max_value=max(1,int(space)), value=int(cfg.get("band",3)), step=1)
    bc      = st.selectbox("Boundary condition", options=["reflect","periodic","absorb","wall"], index=["reflect","periodic","absorb","wall"].index(cfg.get("bc","reflect")))

    st.divider()
    st.subheader("Environment")
    env_len   = st.number_input("Env length (x)", min_value=int(space), max_value=4096, value=int(cfg["env"]["length"]), step=int(space))
    env_noise = st.slider("Env noise σ", 0.0, 0.2, float(cfg["env"]["noise_sigma"]), 0.005)

    st.caption("Sources JSON (e.g. moving peaks). Edit freely.")
    default_sources = json.dumps(cfg["env"]["sources"], indent=2)
    sources_text = st.text_area("env.sources JSON", value=default_sources, height=220)
    try:
        sources = json.loads(sources_text)
        st.success("Sources OK")
    except Exception as e:
        sources = cfg["env"]["sources"]
        st.error(f"Sources JSON error: {e}")

    st.divider()
    chunk = st.slider("Update chunk (frames per UI update)", 10, 500, int(cfg.get("chunk",150)), 10)
    live  = st.toggle("Live streaming", value=bool(cfg.get("live", True)))

# ---------- Build config dict ----------
user_cfg = {
    "seed": int(seed),
    "frames": int(frames),
    "space": int(space),
    "k_flux": float(k_flux),
    "k_motor": float(k_motor),
    "diffuse": float(diffuse),
    "decay": float(decay),
    "band": int(band),
    "bc": bc,
    "env": {
        "length": int(env_len),
        "frames": int(frames),
        "noise_sigma": float(env_noise),
        "sources": sources,
    },
}

# ---------- Layout placeholders ----------
st.title("Simulation")
combo_ph = st.empty()
energy_ph = st.empty()

# ---------- Plot helpers ----------
def _norm(A: np.ndarray) -> np.ndarray:
    m, M = float(np.nanmin(A)), float(np.nanmax(A))
    if M - m < 1e-12:
        return np.zeros_like(A)
    return (A - m) / (M - m + 1e-12)

def _resample_rows(M: np.ndarray, new_len: int) -> np.ndarray:
    """Resample each row of 2D array to new_len columns."""
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
    # resample substrate to env width so x-axis matches
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
    )
    ph.plotly_chart(fig, use_container_width=True, theme=None)

def draw_energy_timeseries(ph, t, e_cell, e_env, e_flux, title="Energy vs time"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=e_cell, name="E_cell"))
    fig.add_trace(go.Scatter(x=t, y=e_env,  name="E_env"))
    fig.add_trace(go.Scatter(x=t, y=e_flux, name="E_flux"))
    fig.update_layout(xaxis_title="t (frames)", yaxis_title="energy", title=title, height=380)
    ph.plotly_chart(fig, use_container_width=True, theme=None)

# ---------- Run ----------
if st.button("Run / Rerun", use_container_width=True):
    ecfg = make_config_from_dict(user_cfg)
    engine = Engine(ecfg)

    def redraw(upto: int):
        draw_combined_heatmap(combo_ph, engine.env[:upto+1], engine.S[:upto+1])
        draw_energy_timeseries(energy_ph, engine.hist.t, engine.hist.E_cell, engine.hist.E_env, engine.hist.E_flux)

    if live:
        last = -1
        def cb(t: int):
            nonlocal last
            if t - last >= int(chunk) or t == engine.T - 1:
                last = t
                redraw(t)
        engine.run(progress_cb=cb)
        redraw(engine.T - 1)
    else:
        engine.run(progress_cb=None)
        redraw(engine.T - 1)