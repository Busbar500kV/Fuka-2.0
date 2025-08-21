import json
import os
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from core.config import default_config, make_config_from_dict
from core.engine import Engine

# ---------- Page ----------
st.set_page_config(page_title="Fuka 2.0 — Free‑Energy Simulation", layout="wide")

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

# ---------- session keys ----------
for base in ("run_id", "combo2d_count", "energy_count", "combo3d_count"):
    if base not in st.session_state:
        st.session_state[base] = 0

def new_key(base: str) -> str:
    st.session_state[base + "_count"] += 1
    return f"{base}_{st.session_state['run_id']}_{st.session_state[base + '_count']}"

# ---------- Sidebar controls ----------
with st.sidebar:
    st.header("Run Controls")
    seed   = st.number_input("Seed",   0, 10_000_000, int(cfg["seed"]), 1)
    frames = st.number_input("Frames", 200, 50_000,   int(cfg["frames"]), 200)
    space  = st.number_input("Substrate cells (space)", 16, 1024, int(cfg["space"]), 16)

    st.divider()
    st.subheader("Physics")
    k_flux  = st.slider("k_flux (env→substrate @ gate)", 0.0, 1.0, float(cfg["k_flux"]), 0.01)
    k_motor = st.slider("k_motor (motor noise @ gate)",  0.0, 5.0, float(cfg["k_motor"]), 0.01)
    diffuse = st.slider("diffuse (spread)",               0.0, 0.5, float(cfg["diffuse"]), 0.005)
    decay   = st.slider("decay (loss)",                   0.0, 0.2, float(cfg["decay"]),   0.001)
    band    = st.number_input("gate band half-width", 1, max(1, int(space//2)), int(cfg.get("band", 3)), 1)
    bc      = st.selectbox("Boundary condition",
                           options=["reflect","periodic","absorb","wall"],
                           index=["reflect","periodic","absorb","wall"].index(cfg.get("bc","reflect")))
    gate_center = st.slider("gate_center (index)", 0, int(space)-1,
                            value=int(cfg.get("gate_center", space//2)), step=1)

    st.divider()
    st.subheader("Environment")
    # NEW: env height for 2D
    env_height = st.number_input("Env height (y)", 1, 1024, int(cfg["env"].get("height", 1)), 1)
    env_len    = st.number_input("Env length (x)", min_value=int(space), max_value=4096,
                                 value=int(cfg["env"]["length"]), step=int(space))
    env_noise  = st.slider("Env noise σ", 0.0, 0.2, float(cfg["env"]["noise_sigma"]), 0.005)

    st.caption("Sources JSON (1D or 2D). Examples:\n"
               '1D: {"kind":"moving_peak","amp":1,"speed":0.0,"width":4,"start":100}\n'
               '2D: {"kind":"moving_peak_2d","amp":1,"speed_x":0.0,"speed_y":0.0,"width_x":6,"width_y":6,"start_x":256,"start_y":256}')
    default_sources = json.dumps(cfg["env"]["sources"], indent=2)
    sources_text = st.text_area("env.sources JSON", value=default_sources, height=220)
    try:
        sources = json.loads(sources_text)
        st.success("Sources OK")
    except Exception as e:
        sources = cfg["env"]["sources"]
        st.error(f"Sources JSON error: {e}")

    st.divider()
    st.subheader("Streaming")
    chunk = st.slider("Update chunk (frames per UI update)", 10, 500, int(cfg.get("chunk",150)), 10)
    live  = st.toggle("Live streaming", value=bool(cfg.get("live", True)))

    st.divider()
    st.subheader("3‑D view")
    thr3d = st.slider("3‑D energy threshold (0‑1)", 0.0, 1.0, 0.75, 0.01)
    max3d = st.number_input("3‑D max points", 1_000, 200_000, 40_000, 1_000)

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
    "gate_center": int(gate_center),
    "env": {
        "height": int(env_height),                   # NEW
        "length": int(env_len),
        "frames": int(frames),
        "noise_sigma": float(env_noise),
        "sources": sources,
    },
}

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

def _project_tx(E: np.ndarray) -> np.ndarray:
    """
    Ensure a (T, X) array for the 2D combined heatmap.
    - If E is (T, X): return as is.
    - If E is (T, Y, X): return max over Y -> (T, X).
    """
    if E.ndim == 2:
        return E
    if E.ndim == 3:
        return np.max(E, axis=1)  # (T, X)
    raise ValueError(f"Unexpected env/substrate shape: {E.shape}")

def draw_combined_heatmap(ph, E: np.ndarray, S: np.ndarray, title="Env + Substrate (combined, zoomable)"):
    # For the 2D case, project both to (T, X) via max-over-Y so we can overlay them in a single heatmap.
    E_tx = _project_tx(E)
    S_tx = _project_tx(S)

    # Resample substrate to env width so x-axis matches
    if S_tx.shape[1] != E_tx.shape[1]:
        S_res = _resample_rows(S_tx, E_tx.shape[1])
    else:
        S_res = S_tx

    En = _norm(E_tx)
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
    fig.update_layout(xaxis_title="t (frames)", yaxis_title="energy", title=title, height=380, template="plotly_dark")
    ph.plotly_chart(fig, use_container_width=True, theme=None, key=new_key("energy"))

def draw_sparse_3d(ph, E: np.ndarray, S: np.ndarray, thr: float, max_points: int):
    """
    1D case  : E,S are (T, X) -> plot markers at y=0 (env) and y=1 (substrate)
    2D case  : E,S are (T, Y, X) -> plot true spatial y; env and substrate differ by color.
    """
    def _sub(x, y, z, vmax):
        n = len(x)
        if n <= vmax: 
            return x, y, z
        idx = np.random.choice(n, size=vmax, replace=False)
        return x[idx], y[idx], z[idx]

    if E.ndim == 2:  # 1D
        En = _norm(E)
        Sn = _norm(S if S.ndim == 2 else np.max(S, axis=1))  # safety
        tE, xE = np.where(En >= thr); yE = np.zeros_like(tE)
        tS, xS = np.where(Sn >= thr); yS = np.ones_like(tS)

        xE, yE, zE = _sub(xE, yE, tE, max_points//2)
        xS, yS, zS = _sub(xS, yS, tS, max_points//2)

        fig = go.Figure()
        fig.add_trace(go.Scatter3d(x=xE, y=yE, z=zE, mode="markers",
                                   marker=dict(size=2), name="Env"))
        fig.add_trace(go.Scatter3d(x=xS, y=yS, z=zS, mode="markers",
                                   marker=dict(size=2), name="Substrate"))

        scene = dict(xaxis_title="x (space)", yaxis_title="layer", zaxis_title="t (time)")

    else:            # 2D
        # true 3D points (x,y,t)
        En = _norm(E)
        Sn = _norm(S)
        tE, yE, xE = np.where(En >= thr)
        tS, yS, xS = np.where(Sn >= thr)

        max_half = max_points // 2
        xE, yE, zE = _sub(xE, yE, tE, max_half)
        xS, yS, zS = _sub(xS, yS, tS, max_half)

        fig = go.Figure()
        fig.add_trace(go.Scatter3d(
            x=xE, y=yE, z=zE, mode="markers",
            marker=dict(size=2, opacity=0.5), name="Env"
        ))
        fig.add_trace(go.Scatter3d(
            x=xS, y=yS, z=zS, mode="markers",
            marker=dict(size=2, opacity=0.8), name="Substrate"
        ))

        scene = dict(xaxis_title="x (space)", yaxis_title="y (space)", zaxis_title="t (time)")

    fig.update_layout(
        title="Sparse 3‑D energy",
        scene=scene,
        height=640,
        template="plotly_dark",
        showlegend=True,
    )
    ph.plotly_chart(fig, use_container_width=True, theme=None, key=new_key("combo3d"))

# ---------- Run ----------
if st.button("Run / Rerun", use_container_width=True):
    # new run: bump run id and reset per‑run counters
    st.session_state["run_id"] += 1
    st.session_state["combo2d_count"] = 0
    st.session_state["energy_count"]  = 0
    st.session_state["combo3d_count"] = 0

    ecfg = make_config_from_dict(user_cfg)
    engine = Engine(ecfg)

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