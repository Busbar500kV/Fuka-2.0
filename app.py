# app.py  — Fuka 2.0 UI
import json
import os
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from core.config import default_config, make_config_from_dict
from core.engine  import Engine

# --------------------------
# Page
# --------------------------
st.set_page_config(
    page_title="Fuka 2.0 — Free‑Energy Simulation ~යසස් පොන්වීර~",
    layout="wide"
)

# --------------------------
# Defaults loader
# --------------------------
def load_defaults():
    path = "defaults.json"
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return default_config()

cfg = load_defaults()

# --------------------------
# Sidebar controls
# --------------------------
with st.sidebar:
    st.header("Run Controls")
    seed   = st.number_input("Seed",   0, 10_000_000, int(cfg.get("seed", 0)), step=1)
    frames = st.number_input("Frames", 200, 50_000,   int(cfg.get("frames", 1200)), step=200)
    space  = st.number_input("Substrate cells (space)", 16, 1024, int(cfg.get("space", 64)), step=16)

    st.divider()
    st.subheader("Physics")
    k_flux  = st.slider("k_flux (env→substrate @ boundary)", 0.0, 1.0,  float(cfg.get("k_flux", 0.08)), 0.01)
    k_motor = st.slider("k_motor (motor noise @ boundary)",  0.0, 5.0,  float(cfg.get("k_motor", 0.20)), 0.01)
    diffuse = st.slider("diffuse (spread)",                  0.0, 0.5,  float(cfg.get("diffuse", 0.05)), 0.005)
    decay   = st.slider("decay (loss)",                      0.0, 0.2,  float(cfg.get("decay", 0.05)),   0.001)
    band    = st.number_input("band (boundary width)", 1, max(1, int(space)), int(cfg.get("band", 3)), 1)
    bc      = st.selectbox(
        "Boundary condition",
        options=["reflect", "periodic", "absorb", "wall"],
        index=["reflect", "periodic", "absorb", "wall"].index(cfg.get("bc", "reflect"))
    )

    st.divider()
    st.subheader("Environment")
    env_len   = st.number_input("Env length (x)", int(space), 4096, int(cfg.get("env", {}).get("length", 512)), step=int(space))
    env_noise = st.slider("Env noise σ", 0.0, 0.2, float(cfg.get("env", {}).get("noise_sigma", 0.01)), 0.005)

    st.caption("Sources JSON (e.g. moving peaks). Edit freely.")
    default_sources = json.dumps(cfg.get("env", {}).get("sources", []), indent=2)
    sources_text = st.text_area("env.sources JSON", value=default_sources, height=220)
    try:
        sources = json.loads(sources_text)
        st.success("Sources OK")
    except Exception as e:
        st.error(f"Sources JSON error: {e}")
        sources = cfg.get("env", {}).get("sources", [])

    st.divider()
    chunk = st.slider("Update chunk (frames per UI update)", 10, 500, int(cfg.get("chunk", 150)), 10)
    live  = st.toggle("Live streaming", value=bool(cfg.get("live", True)))

    st.divider()
    st.subheader("3D view (optional)")
    show_3d          = st.toggle("Show rotatable 3D point‑cloud", value=bool(cfg.get("show_3d", True)))
    env_pct          = st.slider("Env percentile threshold",  80.0, 99.9, float(cfg.get("env_percentile", 97.0)), 0.1)
    subs_pct         = st.slider("Substrate percentile threshold", 80.0, 99.9, float(cfg.get("subs_percentile", 97.0)), 0.1)
    env_max_points   = st.number_input("Max env points",  1000, 200_000, int(cfg.get("env_max_points", 60_000)), step=1000)
    subs_max_points  = st.number_input("Max subs points", 1000, 200_000, int(cfg.get("subs_max_points", 60_000)), step=1000)
    z_scale          = st.slider("Z scale (time stretch)", 0.2, 5.0, float(cfg.get("z_scale", 1.0)), 0.1)

# --------------------------
# Build run configuration dict
# --------------------------
user_cfg = {
    "seed":   int(seed),
    "frames": int(frames),
    "space":  int(space),
    "k_flux": float(k_flux),
    "k_motor": float(k_motor),
    "diffuse": float(diffuse),
    "decay":   float(decay),
    "band":    int(band),
    "bc":      bc,
    "env": {
        "length": int(env_len),
        "frames": int(frames),
        "noise_sigma": float(env_noise),
        "sources": sources,
    },
    # 3D UI knobs (harmless for the engine)
    "show_3d": bool(show_3d),
    "env_percentile": float(env_pct),
    "subs_percentile": float(subs_pct),
    "env_max_points": int(env_max_points),
    "subs_max_points": int(subs_max_points),
    "z_scale": float(z_scale),
}

# --------------------------
# Layout placeholders
# --------------------------
st.title("Simulation")
combo_ph  = st.empty()   # combined 2D heatmap (Env+Substrate)
energy_ph = st.empty()   # energy time series
plot3d_ph = st.empty()   # optional 3D plot

# --------------------------
# Small helpers
# --------------------------
def _norm(A: np.ndarray) -> np.ndarray:
    m = float(np.nanmin(A))
    M = float(np.nanmax(A))
    if M - m < 1e-12:
        return np.zeros_like(A)
    return (A - m) / (M - m + 1e-12)

def _resample_rows(M: np.ndarray, new_len: int) -> np.ndarray:
    """Resample each row of a 2D array to new_len columns (linear)."""
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
    # align widths so x axis matches
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
    ph.plotly_chart(fig, use_container_width=True, theme=None, key="combo_heatmap")

def draw_energy_timeseries(ph, t, e_cell, e_env, e_flux, title="Energy vs time"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=e_cell, name="E_cell"))
    fig.add_trace(go.Scatter(x=t, y=e_env,  name="E_env"))
    fig.add_trace(go.Scatter(x=t, y=e_flux, name="E_flux"))
    fig.update_layout(xaxis_title="t (frames)", yaxis_title="energy", title=title, height=380)
    ph.plotly_chart(fig, use_container_width=True, theme=None, key="energy_timeseries")

def draw_3d_pointcloud(ph, E: np.ndarray, S: np.ndarray,
                       env_pct: float, subs_pct: float,
                       env_max_pts: int, subs_max_pts: int,
                       z_scale: float):
    """Rotatable 3D scatter: x horizontal, z=time; high‑energy points only."""
    # Align widths for a shared x
    if S.shape[1] != E.shape[1]:
        S_res = _resample_rows(S, E.shape[1])
    else:
        S_res = S
    En = _norm(E)
    Sn = _norm(S_res)

    T, X = En.shape
    tt, xx = np.meshgrid(np.arange(T), np.arange(X), indexing="ij")

    # thresholds
    e_thr = np.percentile(En, env_pct)
    s_thr = np.percentile(Sn, subs_pct)

    # masks
    mE = En >= e_thr
    mS = Sn >= s_thr

    def sample_mask(mask, max_pts):
        idx = np.argwhere(mask)
        if idx.shape[0] > max_pts:
            sel = np.random.choice(idx.shape[0], size=max_pts, replace=False)
            idx = idx[sel]
        return idx

    idxE = sample_mask(mE, env_max_pts)
    idxS = sample_mask(mS, subs_max_pts)

    fig = go.Figure()

    if idxE.size > 0:
        zE = (idxE[:, 0].astype(float)) * z_scale
        xE =  idxE[:, 1].astype(float)
        fig.add_trace(go.Scatter3d(
            x=xE, y=np.zeros_like(xE), z=zE,
            mode="markers",
            marker=dict(size=2, color=En[idxE[:, 0], idxE[:, 1]], colorscale="Viridis", opacity=0.9),
            name="Env"
        ))

    if idxS.size > 0:
        zS = (idxS[:, 0].astype(float)) * z_scale
        xS =  idxS[:, 1].astype(float)
        fig.add_trace(go.Scatter3d(
            x=xS, y=np.ones_like(xS), z=zS,
            mode="markers",
            marker=dict(size=2, color=Sn[idxS[:, 0], idxS[:, 1]], colorscale="Inferno", opacity=0.8),
            name="Substrate"
        ))

    fig.update_layout(
        title="High‑energy points in (x, t) — rotatable",
        scene=dict(
            xaxis_title="x (space)",
            yaxis_title="trace (0=Env, 1=Substrate)",
            zaxis_title="t (time)",
            aspectmode="cube"
        ),
        height=600
    )
    ph.plotly_chart(fig, use_container_width=True, theme=None, key="scatter3d")

# --------------------------
# Run
# --------------------------
if st.button("Run / Rerun", use_container_width=True):
    ecfg   = make_config_from_dict(user_cfg)
    engine = Engine(ecfg)

    def redraw(upto: int):
        draw_combined_heatmap(combo_ph, engine.env[:upto+1], engine.S[:upto+1])
        draw_energy_timeseries(energy_ph, engine.hist.t, engine.hist.E_cell, engine.hist.E_env, engine.hist.E_flux)
        if show_3d:
            draw_3d_pointcloud(
                plot3d_ph,
                engine.env[:upto+1], engine.S[:upto+1],
                env_pct, subs_pct,
                env_max_points, subs_max_points,
                z_scale
            )

    if live:
        last = [-1]  # mutable so we can update in nested cb without 'nonlocal'
        def cb(t: int):
            if t - last[0] >= int(chunk) or t == engine.T - 1:
                last[0] = t
                redraw(t)
        engine.run(progress_cb=cb)
        redraw(engine.T - 1)
    else:
        engine.run(progress_cb=None)
        redraw(engine.T - 1)