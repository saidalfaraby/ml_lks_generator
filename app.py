import streamlit as st
import numpy as np
import plotly.graph_objects as go
import math

# ─────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Backpropagation Simulator — ML Week 5",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=DM+Sans:wght@300;500;700&display=swap');

  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
  code, .monospace { font-family: 'IBM Plex Mono', monospace; }

  /* Header */
  .sim-title {
    font-size: 26px; font-weight: 700; color: #0da8a8;
    letter-spacing: -0.5px; margin-bottom: 2px;
  }
  .sim-sub { font-size: 13px; color: #8ba0b8; font-family: 'IBM Plex Mono', monospace; }

  /* Cards */
  .card {
    background: #162848; border-radius: 12px;
    padding: 18px 20px; margin-bottom: 14px;
    border: 1px solid rgba(11,140,140,0.25);
  }
  .card-title {
    font-size: 13px; font-weight: 700; color: #0da8a8;
    text-transform: uppercase; letter-spacing: 1px; margin-bottom: 10px;
  }

  /* Step badges */
  .step-badge {
    display: inline-block;
    background: #0b8c8c; color: white;
    border-radius: 50%; width: 28px; height: 28px;
    text-align: center; line-height: 28px;
    font-weight: 700; font-size: 14px; margin-right: 8px;
  }
  .step-done { background: #1ea86a !important; }
  .step-active { background: #e8a020 !important; }
  .step-locked { background: #3a4a5c !important; }

  /* Formula box */
  .formula-box {
    background: #0f1f3d; border-left: 3px solid #0da8a8;
    border-radius: 0 8px 8px 0; padding: 10px 14px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13px; color: #e8eef6; margin: 8px 0;
  }

  /* Result pill */
  .result-ok {
    display: inline-block; background: rgba(30,168,106,0.15);
    border: 1px solid #1ea86a; color: #1ea86a;
    border-radius: 6px; padding: 3px 10px; font-size: 13px;
    font-family: 'IBM Plex Mono', monospace;
  }
  .result-err {
    display: inline-block; background: rgba(217,64,64,0.15);
    border: 1px solid #d94040; color: #d94040;
    border-radius: 6px; padding: 3px 10px; font-size: 13px;
    font-family: 'IBM Plex Mono', monospace;
  }

  /* Answer mode banner */
  .answer-banner {
    background: linear-gradient(90deg,#e8a020,#ffc040);
    color: #0f1f3d; font-weight: 700; font-size: 14px;
    text-align: center; padding: 10px; border-radius: 8px;
    margin-bottom: 16px; letter-spacing: 0.5px;
  }

  /* NIM box */
  .nim-box {
    background: linear-gradient(135deg, #162848, #0f2a3d);
    border: 1px solid #0da8a8; border-radius: 14px;
    padding: 28px 32px; text-align: center; margin: 30px auto;
    max-width: 500px;
  }

  /* Iteration header */
  .iter-header {
    background: linear-gradient(90deg, #0b8c8c22, transparent);
    border-left: 4px solid #0da8a8;
    border-radius: 0 8px 8px 0;
    padding: 10px 16px; margin: 16px 0 12px;
    font-weight: 700; font-size: 16px;
  }

  /* Param table */
  .param-grid {
    display: grid; grid-template-columns: 1fr 1fr;
    gap: 8px;
  }
  .param-item {
    background: #0f1f3d; border-radius: 8px; padding: 8px 12px;
    display: flex; justify-content: space-between; align-items: center;
  }
  .param-key { font-size: 13px; color: #8ba0b8; font-family: 'IBM Plex Mono', monospace; }
  .param-val { font-size: 14px; font-weight: 600; color: #ffc040; font-family: 'IBM Plex Mono', monospace; }

  /* Hide streamlit branding */
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
#  MATH HELPERS
# ─────────────────────────────────────────────────────────────────
def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

def sigmoid_deriv(a):
    return a * (1.0 - a)

def mse_loss(y, y_hat):
    return 0.5 * (y - y_hat) ** 2

# ─────────────────────────────────────────────────────────────────
#  PARAMETER GENERATION FROM NIM
# ─────────────────────────────────────────────────────────────────
def generate_params(nim_seed: int) -> dict:
    """Generate network parameters deterministically from 5-digit NIM seed."""
    rng = np.random.default_rng(nim_seed)

    def rand_weight():
        return round(float(rng.uniform(-0.8, 0.8)), 2)

    def rand_input():
        return round(float(rng.uniform(0.1, 0.9)), 2)

    def rand_bias():
        return round(float(rng.uniform(-0.3, 0.3)), 2)

    def rand_target():
        return round(float(rng.choice([0.0, 1.0])), 1)

    def rand_lr():
        return round(float(rng.choice([0.05, 0.1, 0.15, 0.2])), 2)

    return {
        "x1": rand_input(),
        "x2": rand_input(),
        "w11": rand_weight(),  # x1 -> h1
        "w21": rand_weight(),  # x2 -> h1
        "w12": rand_weight(),  # x1 -> h2
        "w22": rand_weight(),  # x2 -> h2
        "b1_1": rand_bias(),   # bias h1
        "b1_2": rand_bias(),   # bias h2
        "w2_1": rand_weight(), # h1 -> output
        "w2_2": rand_weight(), # h2 -> output
        "b2":   rand_bias(),   # bias output
        "y":    rand_target(),
        "lr":   rand_lr(),
    }

# ─────────────────────────────────────────────────────────────────
#  FULL FORWARD + BACKWARD PASS (one iteration)
# ─────────────────────────────────────────────────────────────────
def compute_iteration(p: dict) -> dict:
    """Return all intermediate values for one backprop iteration."""
    x1, x2 = p["x1"], p["x2"]
    w11, w21, w12, w22 = p["w11"], p["w21"], p["w12"], p["w22"]
    b1_1, b1_2 = p["b1_1"], p["b1_2"]
    w2_1, w2_2, b2 = p["w2_1"], p["w2_2"], p["b2"]
    y, lr = p["y"], p["lr"]

    # Forward pass
    z1_1 = w11 * x1 + w21 * x2 + b1_1
    z1_2 = w12 * x1 + w22 * x2 + b1_2
    a1_1 = sigmoid(z1_1)
    a1_2 = sigmoid(z1_2)
    z2   = w2_1 * a1_1 + w2_2 * a1_2 + b2
    y_hat = sigmoid(z2)
    loss  = mse_loss(y, y_hat)

    # Backward pass
    d_sig2  = sigmoid_deriv(y_hat)
    delta2  = -(y - y_hat) * d_sig2

    d_sig1_1 = sigmoid_deriv(a1_1)
    d_sig1_2 = sigmoid_deriv(a1_2)
    delta1_1 = w2_1 * delta2 * d_sig1_1
    delta1_2 = w2_2 * delta2 * d_sig1_2

    # Update weights
    w2_1_new  = w2_1  - lr * delta2   * a1_1
    w2_2_new  = w2_2  - lr * delta2   * a1_2
    b2_new    = b2    - lr * delta2
    w11_new   = w11   - lr * delta1_1 * x1
    w21_new   = w21   - lr * delta1_1 * x2
    w12_new   = w12   - lr * delta1_2 * x1
    w22_new   = w22   - lr * delta1_2 * x2
    b1_1_new  = b1_1  - lr * delta1_1
    b1_2_new  = b1_2  - lr * delta1_2

    return {
        "z1_1": round(z1_1, 6), "z1_2": round(z1_2, 6),
        "a1_1": round(a1_1, 6), "a1_2": round(a1_2, 6),
        "z2":   round(z2, 6),   "y_hat": round(y_hat, 6),
        "loss": round(loss, 6),
        "d_sig2": round(d_sig2, 6), "delta2": round(delta2, 6),
        "d_sig1_1": round(d_sig1_1, 6), "d_sig1_2": round(d_sig1_2, 6),
        "delta1_1": round(delta1_1, 6), "delta1_2": round(delta1_2, 6),
        "w2_1_new": round(w2_1_new, 6), "w2_2_new": round(w2_2_new, 6),
        "b2_new":   round(b2_new, 6),
        "w11_new":  round(w11_new, 6),  "w21_new": round(w21_new, 6),
        "w12_new":  round(w12_new, 6),  "w22_new": round(w22_new, 6),
        "b1_1_new": round(b1_1_new, 6), "b1_2_new": round(b1_2_new, 6),
    }

# ─────────────────────────────────────────────────────────────────
#  NETWORK VISUALIZATION (Plotly)
# ─────────────────────────────────────────────────────────────────
STEP_ACTIVE_NODES = {
    0: ["h1"],
    1: ["h2"],
    2: ["h1", "h2"],
    3: ["out"],
    4: ["out"],
    5: ["out"],
    6: ["h1"],
}
STEP_ACTIVE_EDGES = {
    0: ["x1h1", "x2h1"],
    1: ["x1h2", "x2h2"],
    2: ["x1h1", "x2h1", "x1h2", "x2h2"],
    3: ["h1out", "h2out"],
    4: ["h1out", "h2out"],
    5: ["h1out", "h2out"],
    6: ["h1out"],
}

def build_node_labels(p: dict, res: dict, steps_done: int) -> dict:
    labels = {
        "x1":  f"x₁={p['x1']}",
        "x2":  f"x₂={p['x2']}",
        "h1":  "h₁\n?",
        "h2":  "h₂\n?",
        "out": "ŷ\n?",
    }
    if steps_done >= 1: labels["h1"] = f"h₁\nz={res['z1_1']:.4f}"
    if steps_done >= 2: labels["h2"] = f"h₂\nz={res['z1_2']:.4f}"
    if steps_done >= 3:
        labels["h1"] = f"h₁\nz={res['z1_1']:.4f}\na={res['a1_1']:.4f}"
        labels["h2"] = f"h₂\nz={res['z1_2']:.4f}\na={res['a1_2']:.4f}"
    if steps_done >= 4:
        labels["out"] = f"ŷ={res['y_hat']:.4f}"
    if steps_done >= 5:
        labels["out"] = f"ŷ={res['y_hat']:.4f}\nL={res['loss']:.4f}"
    return labels


def draw_network(p: dict, res: dict, current_step_in_iter: int, steps_done_in_iter: int):
    BG = "#0f1f3d"

    # Node positions — spread out vertically to give room for edge labels
    pos = {
        "x1":  (0.0, 3.0),
        "x2":  (0.0, 0.8),
        "h1":  (3.0, 3.0),
        "h2":  (3.0, 0.8),
        "out": (6.0, 1.9),
    }

    active_nodes = STEP_ACTIVE_NODES.get(current_step_in_iter, [])
    active_edges = STEP_ACTIVE_EDGES.get(current_step_in_iter, [])

    base_colors = {"x1":"#0b7a7a","x2":"#0b7a7a","h1":"#7a5800","h2":"#7a5800","out":"#0d6640"}
    glow_colors = {"x1":"#0dd4d4","x2":"#0dd4d4","h1":"#ffc040","h2":"#ffc040","out":"#26d980"}

    node_labels = build_node_labels(p, res, steps_done_in_iter)

    # Delta labels — shown only after their steps are done
    # delta2 revealed after step 5 (idx>=6), delta1_1 after step 6 (idx>=7)
    delta2_label   = f"δ²={res['delta2']:.4f}"   if steps_done_in_iter >= 6 else None
    delta1_1_label = f"δ¹₁={res['delta1_1']:.4f}" if steps_done_in_iter >= 7 else None
    delta1_2_label = f"δ¹₂={res['delta1_2']:.4f}" if steps_done_in_iter >= 7 else None

    # Edge definitions:
    # key → (src, dst, weight_text, label_x_frac, label_y_offset)
    # x_frac: how far along edge (0=src, 1=dst) to place label
    # For crossing edges (x1h2, x2h1) we use different fractions so they don't overlap
    edges_def = {
        "x1h1":  ("x1", "h1",  f"w₁₁={p['w11']}",  0.30, +0.18),  # top-left → top-right (direct)
        "x2h2":  ("x2", "h2",  f"w₂₂={p['w22']}",  0.30, -0.18),  # bottom-left → bottom-right (direct)
        "x2h1":  ("x2", "h1",  f"w₂₁={p['w21']}",  0.25, +0.20),  # bottom-left → top-right (cross, label near src)
        "x1h2":  ("x1", "h2",  f"w₁₂={p['w12']}",  0.70, -0.20),  # top-left → bottom-right (cross, label near dst)
        "h1out": ("h1", "out", f"w²₁={p['w2_1']}",  0.35, +0.18),
        "h2out": ("h2", "out", f"w²₂={p['w2_2']}",  0.35, -0.18),
    }

    fig = go.Figure()

    # ── 1. Edges ──
    for ekey, (src, dst, wtext, xfrac, yoff) in edges_def.items():
        x0, y0 = pos[src]
        x1, y1 = pos[dst]
        is_active = ekey in active_edges
        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1], mode="lines",
            line=dict(color="#0dd4d4" if is_active else "#243d5c",
                      width=3.0 if is_active else 1.2),
            hoverinfo="skip", showlegend=False,
        ))
        # Weight label
        tx = x0 + xfrac * (x1 - x0)
        ty = y0 + xfrac * (y1 - y0) + yoff
        fig.add_annotation(
            x=tx, y=ty, text=f"<b>{wtext}</b>" if is_active else wtext,
            showarrow=False,
            font=dict(size=10, color="#0dd4d4" if is_active else "#7a95b0",
                      family="IBM Plex Mono"),
            bgcolor="rgba(0,0,0,0)", borderwidth=0,
        )

    # ── 2. Node circles ──
    for name, (nx, ny) in pos.items():
        is_glow   = name in active_nodes
        fill      = glow_colors[name] if is_glow else base_colors[name]
        bdr_color = "#ffffff" if is_glow else "#2a4a6a"
        fig.add_trace(go.Scatter(
            x=[nx], y=[ny], mode="markers",
            marker=dict(size=74 if is_glow else 64, color=fill,
                        line=dict(color=bdr_color, width=3 if is_glow else 1.5)),
            hoverinfo="skip", showlegend=False,
        ))
        # Label inside node via annotation
        raw = node_labels[name]
        html_text = "<br>".join(raw.split("\n"))
        nlines = len(raw.split("\n"))
        fig.add_annotation(
            x=nx, y=ny, text=f"<b>{html_text}</b>",
            showarrow=False,
            font=dict(size=10 if nlines == 1 else 8.5, color="#ffffff",
                      family="IBM Plex Mono"),
            bgcolor="rgba(0,0,0,0)", borderwidth=0,
        )

    # ── 3. Bias labels below each non-input node ──
    for bname, bx, by, blabel in [
        ("h1",  pos["h1"][0],  pos["h1"][1],  f"b₁={p['b1_1']}"),
        ("h2",  pos["h2"][0],  pos["h2"][1],  f"b₂={p['b1_2']}"),
        ("out", pos["out"][0], pos["out"][1], f"b={p['b2']}"),
    ]:
        fig.add_annotation(
            x=bx, y=by - 0.62, text=blabel, showarrow=False,
            font=dict(size=9, color="#5a7a9a", family="IBM Plex Mono"),
            bgcolor="rgba(0,0,0,0)", borderwidth=0,
        )

    # ── 4. Delta annotations — shown beside the relevant node/edge ──
    # delta2 floats beside the output→hidden edge area (right side of hidden layer)
    if delta2_label:
        fig.add_annotation(
            x=pos["out"][0] + 0.3, y=pos["out"][1] + 0.55,
            text=f"<b>{delta2_label}</b>",
            showarrow=True, ax=-40, ay=0,
            arrowhead=2, arrowsize=1, arrowwidth=1.5, arrowcolor="#d94040",
            font=dict(size=10, color="#ff6060", family="IBM Plex Mono"),
            bgcolor="#1a0a0a", bordercolor="#d94040", borderwidth=1, borderpad=5,
        )
    if delta1_1_label:
        fig.add_annotation(
            x=pos["h1"][0] - 0.3, y=pos["h1"][1] + 0.55,
            text=f"<b>{delta1_1_label}</b>",
            showarrow=True, ax=40, ay=0,
            arrowhead=2, arrowsize=1, arrowwidth=1.5, arrowcolor="#d94040",
            font=dict(size=10, color="#ff6060", family="IBM Plex Mono"),
            bgcolor="#1a0a0a", bordercolor="#d94040", borderwidth=1, borderpad=5,
        )
    if delta1_2_label:
        fig.add_annotation(
            x=pos["h2"][0] - 0.3, y=pos["h2"][1] - 0.55,
            text=f"<b>{delta1_2_label}</b>",
            showarrow=True, ax=40, ay=0,
            arrowhead=2, arrowsize=1, arrowwidth=1.5, arrowcolor="#d94040",
            font=dict(size=10, color="#ff6060", family="IBM Plex Mono"),
            bgcolor="#1a0a0a", bordercolor="#d94040", borderwidth=1, borderpad=5,
        )

    # ── 5. Loss/Target info box ──
    loss_display = f"{res['loss']:.6f}" if steps_done_in_iter >= 5 else "?"
    fig.add_annotation(
        x=7.4, y=1.9,
        text=f"<b>Loss</b><br>{loss_display}<br><br><b>Target</b><br>y = {p['y']}",
        showarrow=False,
        font=dict(size=11, color="#e8eef6", family="IBM Plex Mono"),
        align="center",
        bgcolor="#162848", bordercolor="#0da8a8", borderwidth=1.5, borderpad=10,
    )

    # ── 6. Layer labels at top ──
    for lx, ltxt, lc in [(0.0,"INPUT","#0da8a8"),(3.0,"HIDDEN","#e8a020"),(6.0,"OUTPUT","#26d980")]:
        fig.add_annotation(
            x=lx, y=4.2, text=f"<b>{ltxt}</b>", showarrow=False,
            font=dict(size=11, color=lc, family="DM Sans"),
            bgcolor="rgba(0,0,0,0)", borderwidth=0,
        )

    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG,
        xaxis=dict(visible=False, range=[-1.0, 9.2]),
        yaxis=dict(visible=False, range=[0.0, 4.5]),
        margin=dict(l=0, r=0, t=0, b=0),
        height=400,
        dragmode=False,
    )
    return fig

# ─────────────────────────────────────────────────────────────────
#  LOSS CURVE PLOT
# ─────────────────────────────────────────────────────────────────
def draw_loss_curve(losses: list):
    iters = list(range(len(losses)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=iters, y=losses,
        mode="lines+markers",
        line=dict(color="#0da8a8", width=2.5),
        marker=dict(size=8, color="#ffc040", line=dict(color="#0da8a8", width=2)),
        name="Loss",
        hovertemplate="Iterasi %{x}<br>Loss: %{y:.6f}<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,31,61,0.8)",
        xaxis=dict(
            title="Iterasi", tickmode="linear", tick0=0, dtick=1,
            gridcolor="rgba(11,140,140,0.15)", color="#8ba0b8",
            title_font=dict(color="#8ba0b8"), tickfont=dict(color="#8ba0b8"),
        ),
        yaxis=dict(
            title="Loss (MSE)",
            gridcolor="rgba(11,140,140,0.15)", color="#8ba0b8",
            title_font=dict(color="#8ba0b8"), tickfont=dict(color="#8ba0b8"),
        ),
        margin=dict(l=10, r=10, t=10, b=10),
        height=220,
        showlegend=False,
    )
    return fig

# ─────────────────────────────────────────────────────────────────
#  STEP DEFINITIONS  (2 iterations × 7 sub-steps each)
# ─────────────────────────────────────────────────────────────────
# Each step: (step_id, label, field_key, formula_html, answer_key, unit_hint)
def get_steps(it: int):
    """Return step list for iteration `it` (1 or 2)."""
    prefix = f"it{it}"
    return [
        (f"{prefix}_z1_1",  "Hitung z[1]₁ (pra-aktivasi h₁)",
         "z1_1",
         "z[1]₁ = W[1]₁₁·x₁ + W[1]₂₁·x₂ + b[1]₁",
         "z1_1", ""),
        (f"{prefix}_z1_2",  "Hitung z[1]₂ (pra-aktivasi h₂)",
         "z1_2",
         "z[1]₂ = W[1]₁₂·x₁ + W[1]₂₂·x₂ + b[1]₂",
         "z1_2", ""),
        (f"{prefix}_a1",    "Hitung a[1]₁ dan a[1]₂  (aktivasi sigmoid hidden layer)",
         "a1",
         "a[1]ⱼ = σ(z[1]ⱼ) = 1 / (1 + e^(−z[1]ⱼ))  →  masukkan a[1]₁ saja",
         "a1_1", "a[1]₁ = ?"),
        (f"{prefix}_yhat",  "Hitung z[2] dan ŷ (output layer)",
         "yhat",
         "z[2] = W[2]₁·a[1]₁ + W[2]₂·a[1]₂ + b[2]  →  ŷ = σ(z[2])",
         "y_hat", "ŷ = ?"),
        (f"{prefix}_loss",  "Hitung Loss (MSE)",
         "loss",
         "L = ½ · (y − ŷ)²",
         "loss", "L = ?"),
        (f"{prefix}_delta2","Hitung δ[2] (gradien output layer)",
         "delta2",
         "δ[2] = −(y − ŷ) · σ'(z[2])  di mana σ'(z[2]) = ŷ·(1−ŷ)",
         "delta2", "δ[2] = ?"),
        (f"{prefix}_delta1_1", "Hitung δ[1]₁ (gradien hidden neuron h₁)",
         "delta1_1",
         "δ[1]₁ = W[2]₁ · δ[2] · σ'(z[1]₁)  di mana σ'(z[1]₁) = a[1]₁·(1−a[1]₁)",
         "delta1_1", "δ[1]₁ = ?"),
        (f"{prefix}_done",  "Bobot ter-update! Verifikasi hasil Iterasi",
         None, None, None, None),
    ]


# ─────────────────────────────────────────────────────────────────
#  TAB 2 — GRADIENT DESCENT EXPLORER HELPERS
# ─────────────────────────────────────────────────────────────────
def generate_gd_params(nim_seed: int) -> dict:
    """Generate GD landscape params from NIM seed (different stream from backprop)."""
    rng = np.random.default_rng(nim_seed + 99999)  # offset so different from backprop
    w1_opt = round(float(rng.uniform(-3.0, 3.0)), 2)
    w2_opt = round(float(rng.uniform(-3.0, 3.0)), 2)
    # Start far enough from optimum
    w1_init = round(float(w1_opt + rng.choice([-1,1]) * rng.uniform(2.0, 4.0)), 2)
    w2_init = round(float(w2_opt + rng.choice([-1,1]) * rng.uniform(2.0, 4.0)), 2)
    lr      = round(float(rng.choice([0.05, 0.1, 0.15, 0.2])), 2)
    return {"w1_init": w1_init, "w2_init": w2_init,
            "w1_opt": w1_opt,  "w2_opt": w2_opt, "lr": lr}


def gd_loss(w1: float, w2: float, w1_opt: float, w2_opt: float) -> float:
    return (w1 - w1_opt)**2 + (w2 - w2_opt)**2


def gd_gradient(w1: float, w2: float, w1_opt: float, w2_opt: float):
    """Returns (dL/dw1, dL/dw2)."""
    return 2*(w1 - w1_opt), 2*(w2 - w2_opt)


def draw_gd_landscape(gd_p: dict, history: list, active_step: int):
    """
    Draw 2D loss contour + trajectory + current position.
    history: list of (w1, w2, loss)
    active_step: index of current manual step being worked on (for glow)
    """
    BG = "#0f1f3d"
    w1_opt, w2_opt = gd_p["w1_opt"], gd_p["w2_opt"]

    # Grid for contour
    margin = 5.0
    w1_min = min(w1_opt, gd_p["w1_init"]) - margin
    w1_max = max(w1_opt, gd_p["w1_init"]) + margin
    w2_min = min(w2_opt, gd_p["w2_init"]) - margin
    w2_max = max(w2_opt, gd_p["w2_init"]) + margin

    n = 120
    w1_grid = np.linspace(w1_min, w1_max, n)
    w2_grid = np.linspace(w2_min, w2_max, n)
    W1, W2 = np.meshgrid(w1_grid, w2_grid)
    Z = (W1 - w1_opt)**2 + (W2 - w2_opt)**2

    fig = go.Figure()

    # Contour
    fig.add_trace(go.Contour(
        x=w1_grid, y=w2_grid, z=Z,
        colorscale=[
            [0.0,  "#0a2540"],
            [0.15, "#0b5a6e"],
            [0.35, "#0b8c8c"],
            [0.6,  "#e8a020"],
            [0.8,  "#d94040"],
            [1.0,  "#ffffff"],
        ],
        contours=dict(showlabels=True, labelfont=dict(size=9, color="white"),
                      coloring="heatmap", showlines=True),
        line=dict(width=0.6, color="rgba(255,255,255,0.3)"),
        colorbar=dict(
            title=dict(text="Loss", font=dict(color="#8ba0b8", size=11)),
            tickfont=dict(color="#8ba0b8"),
            len=0.7,
        ),
        hovertemplate="w₁=%{x:.3f}<br>w₂=%{y:.3f}<br>Loss=%{z:.4f}<extra></extra>",
        showlegend=False,
    ))

    # Optimal point
    fig.add_trace(go.Scatter(
        x=[w1_opt], y=[w2_opt], mode="markers",
        marker=dict(size=14, color="#26d980", symbol="star",
                    line=dict(color="white", width=1.5)),
        name="Optimal (w₁*, w₂*)",
        hovertemplate=f"Optimal<br>w₁*={w1_opt}<br>w₂*={w2_opt}<extra></extra>",
    ))

    # Trajectory line
    if len(history) >= 2:
        traj_w1 = [h[0] for h in history]
        traj_w2 = [h[1] for h in history]
        fig.add_trace(go.Scatter(
            x=traj_w1, y=traj_w2, mode="lines",
            line=dict(color="rgba(255,192,64,0.7)", width=2, dash="dot"),
            name="Lintasan", hoverinfo="skip", showlegend=False,
        ))

    # Past positions (gray dots)
    if len(history) > 1:
        past_w1 = [h[0] for h in history[:-1]]
        past_w2 = [h[1] for h in history[:-1]]
        past_labels = [f"Step {i}<br>w₁={h[0]:.4f}<br>w₂={h[1]:.4f}<br>Loss={h[2]:.4f}"
                       for i, h in enumerate(history[:-1])]
        fig.add_trace(go.Scatter(
            x=past_w1, y=past_w2, mode="markers",
            marker=dict(size=8, color="rgba(200,200,200,0.5)",
                        line=dict(color="white", width=1)),
            name="Posisi sebelumnya",
            text=past_labels, hoverinfo="text",
            showlegend=False,
        ))

    # Current position (glowing)
    if history:
        cur_w1, cur_w2, cur_loss = history[-1]
        fig.add_trace(go.Scatter(
            x=[cur_w1], y=[cur_w2], mode="markers",
            marker=dict(size=16, color="#ffc040",
                        line=dict(color="white", width=2.5)),
            name="Posisi saat ini",
            hovertemplate=f"Posisi saat ini<br>w₁={cur_w1:.4f}<br>w₂={cur_w2:.4f}<br>Loss={cur_loss:.4f}<extra></extra>",
        ))

        # Gradient arrow (only during manual steps and before auto)
        if active_step < 3:
            dw1, dw2 = gd_gradient(cur_w1, cur_w2, w1_opt, w2_opt)
            scale = 0.3
            fig.add_annotation(
                x=cur_w1 - scale*dw1, y=cur_w2 - scale*dw2,
                ax=cur_w1, ay=cur_w2,
                xref="x", yref="y", axref="x", ayref="y",
                text="", showarrow=True,
                arrowhead=3, arrowsize=1.5, arrowwidth=2.5, arrowcolor="#d94040",
            )
            fig.add_annotation(
                x=cur_w1 + 0.15, y=cur_w2 + 0.15,
                text=f"<b>−∇L</b>",
                showarrow=False,
                font=dict(size=11, color="#ff6060", family="IBM Plex Mono"),
                bgcolor="rgba(0,0,0,0)", borderwidth=0,
            )

    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG,
        xaxis=dict(title="w₁", gridcolor="rgba(11,140,140,0.1)",
                   color="#8ba0b8", title_font=dict(color="#8ba0b8"),
                   tickfont=dict(color="#8ba0b8"), zeroline=False),
        yaxis=dict(title="w₂", gridcolor="rgba(11,140,140,0.1)",
                   color="#8ba0b8", title_font=dict(color="#8ba0b8"),
                   tickfont=dict(color="#8ba0b8"), zeroline=False,
                   scaleanchor="x", scaleratio=1),
        legend=dict(font=dict(color="#e8eef6", size=10), bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=10, r=10, t=10, b=10),
        height=420,
        dragmode="pan",
    )
    return fig


def draw_gd_curves(history: list):
    """Draw loss curve and w1/w2 trajectory over steps."""
    steps = list(range(len(history)))
    losses = [h[2] for h in history]
    w1s    = [h[0] for h in history]
    w2s    = [h[1] for h in history]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=steps, y=losses, mode="lines+markers", name="Loss",
        line=dict(color="#0da8a8", width=2.5),
        marker=dict(size=7, color="#ffc040", line=dict(color="#0da8a8", width=1.5)),
        hovertemplate="Step %{x}<br>Loss: %{y:.6f}<extra></extra>",
        yaxis="y1",
    ))
    fig.add_trace(go.Scatter(x=steps, y=w1s, mode="lines+markers", name="w₁",
        line=dict(color="#0dd4d4", width=1.8, dash="dot"),
        marker=dict(size=5, color="#0dd4d4"),
        hovertemplate="Step %{x}<br>w₁: %{y:.4f}<extra></extra>",
        yaxis="y2",
    ))
    fig.add_trace(go.Scatter(x=steps, y=w2s, mode="lines+markers", name="w₂",
        line=dict(color="#e8a020", width=1.8, dash="dot"),
        marker=dict(size=5, color="#e8a020"),
        hovertemplate="Step %{x}<br>w₂: %{y:.4f}<extra></extra>",
        yaxis="y2",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,31,61,0.8)",
        xaxis=dict(title="Step", tickmode="linear", tick0=0, dtick=1,
                   gridcolor="rgba(11,140,140,0.15)", color="#8ba0b8",
                   title_font=dict(color="#8ba0b8"), tickfont=dict(color="#8ba0b8")),
        yaxis=dict(title="Loss", gridcolor="rgba(11,140,140,0.15)", color="#0da8a8",
                   title_font=dict(color="#0da8a8"), tickfont=dict(color="#8ba0b8"),
                   side="left"),
        yaxis2=dict(title="Nilai w", overlaying="y", side="right",
                    color="#0dd4d4", title_font=dict(color="#0dd4d4"),
                    tickfont=dict(color="#8ba0b8"), showgrid=False),
        legend=dict(font=dict(color="#e8eef6", size=10), bgcolor="rgba(0,0,0,0)",
                    orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=10, r=10, t=30, b=10),
        height=240,
    )
    return fig

# ─────────────────────────────────────────────────────────────────
#  SESSION STATE INIT
# ─────────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "nim": "",
        "nim_confirmed": False,
        "params": None,
        "res1": None,
        "res2": None,
        "current_step": 0,   # 0..7 = iter1 steps, 8..15 = iter2 steps, 16 = done
        "losses": [],
        "answer_mode": False,
        "input_errors": {},
        # Auto-iteration state (unlocked after manual iter 2 done)
        "auto_params": None,   # current weights for next auto iteration
        "auto_iter":   0,      # how many auto iterations have been run
        # Tab 2 — Gradient Descent Explorer
        "gd_params":   None,   # {w1_init, w2_init, w1_opt, w2_opt, lr, alpha}
        "gd_history":  [],     # list of (w1, w2, loss) per step
        "gd_manual_done": 0,   # how many manual steps completed (need 3)
        "gd_checkpoint": None, # snapshot of history after 3 manual steps (for reset)
        "gd_auto_alpha": None, # alpha used in auto mode (can differ from manual)
        "gd_errors":   {},     # validation errors per field
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ─────────────────────────────────────────────────────────────────
#  HELPER: check answer
# ─────────────────────────────────────────────────────────────────
TOLERANCE = 0.001

def check(user_val: float, correct: float) -> bool:
    return abs(user_val - correct) <= TOLERANCE

# ─────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sim-title">🧠 BackProp</div>', unsafe_allow_html=True)
    st.markdown('<div class="sim-sub">ML Week 5 · Simulator</div>', unsafe_allow_html=True)
    st.markdown("---")

    if st.session_state.nim_confirmed and st.session_state.params:
        p = st.session_state.params
        st.markdown("**Parameter Jaringan**")
        st.markdown(f"""
        <div class="card">
          <div class="card-title">Input</div>
          <div class="param-grid">
            <div class="param-item"><span class="param-key">x₁</span><span class="param-val">{p['x1']}</span></div>
            <div class="param-item"><span class="param-key">x₂</span><span class="param-val">{p['x2']}</span></div>
          </div>
        </div>
        <div class="card">
          <div class="card-title">Bobot Layer 1</div>
          <div class="param-grid">
            <div class="param-item"><span class="param-key">W[1]₁₁</span><span class="param-val">{p['w11']}</span></div>
            <div class="param-item"><span class="param-key">W[1]₂₁</span><span class="param-val">{p['w21']}</span></div>
            <div class="param-item"><span class="param-key">W[1]₁₂</span><span class="param-val">{p['w12']}</span></div>
            <div class="param-item"><span class="param-key">W[1]₂₂</span><span class="param-val">{p['w22']}</span></div>
            <div class="param-item"><span class="param-key">b[1]₁</span><span class="param-val">{p['b1_1']}</span></div>
            <div class="param-item"><span class="param-key">b[1]₂</span><span class="param-val">{p['b1_2']}</span></div>
          </div>
        </div>
        <div class="card">
          <div class="card-title">Bobot Layer 2</div>
          <div class="param-grid">
            <div class="param-item"><span class="param-key">W[2]₁</span><span class="param-val">{p['w2_1']}</span></div>
            <div class="param-item"><span class="param-key">W[2]₂</span><span class="param-val">{p['w2_2']}</span></div>
            <div class="param-item"><span class="param-key">b[2]</span><span class="param-val">{p['b2']}</span></div>
          </div>
        </div>
        <div class="card">
          <div class="card-title">Target & LR</div>
          <div class="param-grid">
            <div class="param-item"><span class="param-key">y</span><span class="param-val">{p['y']}</span></div>
            <div class="param-item"><span class="param-key">α</span><span class="param-val">{p['lr']}</span></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        progress_val = min(st.session_state.current_step / 14, 1.0)
        step_label = st.session_state.current_step
        st.progress(progress_val, text=f"Progress: step {step_label}/14")

        if st.button("🔄 Reset Simulasi", use_container_width=True):
            for k in ["nim","nim_confirmed","params","res1","res2",
                      "current_step","losses","answer_mode","input_errors"]:
                del st.session_state[k]
            st.rerun()

    

# ─────────────────────────────────────────────────────────────────
#  MAIN AREA
# ─────────────────────────────────────────────────────────────────

# Answer mode banner
if st.session_state.answer_mode:
    st.markdown(
        '<div class="answer-banner">⚠️ MODE JAWABAN AKTIF — Semua langkah ditampilkan</div>',
        unsafe_allow_html=True
    )

# ── STEP 0: NIM INPUT ──
if not st.session_state.nim_confirmed:
    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        st.markdown("""
        <div class="nim-box">
          <div style="font-size:40px;margin-bottom:12px">🧠</div>
          <div style="font-size:22px;font-weight:700;color:#0da8a8;margin-bottom:6px">
            Backpropagation Simulator
          </div>
          <div style="font-size:13px;color:#8ba0b8;margin-bottom:24px;font-family:'IBM Plex Mono',monospace">
            Machine Learning — Week 5
          </div>
          <div style="font-size:14px;color:#e8eef6;margin-bottom:8px">
            Masukkan NIM Anda untuk generate parameter soal unik.
          </div>
        </div>
        """, unsafe_allow_html=True)

        nim_input = st.text_input(
            "NIM (5 digit terakhir akan digunakan sebagai seed):",
            placeholder="contoh: 2311234567",
            key="nim_input_field",
        )

        if st.button("🎲 Generate Parameter", use_container_width=True, type="primary"):
            if nim_input.strip():
                nim_clean = nim_input.strip()
                seed_str = nim_clean[-5:]
                try:
                    seed = int(seed_str)
                    st.session_state.nim = nim_clean
                    st.session_state.params = generate_params(seed)
                    st.session_state.nim_confirmed = True
                    # Pre-compute both iterations
                    p = st.session_state.params
                    r1 = compute_iteration(p)
                    st.session_state.res1 = r1
                    # Iter 2 uses updated weights
                    p2 = {**p,
                          "w11": r1["w11_new"], "w21": r1["w21_new"],
                          "w12": r1["w12_new"], "w22": r1["w22_new"],
                          "b1_1": r1["b1_1_new"], "b1_2": r1["b1_2_new"],
                          "w2_1": r1["w2_1_new"], "w2_2": r1["w2_2_new"],
                          "b2": r1["b2_new"]}
                    st.session_state.res2 = compute_iteration(p2)
                    st.session_state.losses = []
                    st.rerun()
                except ValueError:
                    st.error("NIM tidak valid — pastikan berisi angka.")
            else:
                st.warning("Masukkan NIM terlebih dahulu.")
    st.stop()

# ── CONFIRMED — show main simulator ──
# ── TABS ──
tab1, tab2 = st.tabs(["🧠 Backpropagation Simulator", "📉 Gradient Descent Explorer"])

with tab1:
    p   = st.session_state.params
    r1  = st.session_state.res1
    r2  = st.session_state.res2

    # Determine which iteration's params / results to use for network viz
    cur = st.session_state.current_step
    is_iter2 = cur >= 7

    if is_iter2:
        p_viz = {**p,
                 "w11": r1["w11_new"], "w21": r1["w21_new"],
                 "w12": r1["w12_new"], "w22": r1["w22_new"],
                 "b1_1": r1["b1_1_new"], "b1_2": r1["b1_2_new"],
                 "w2_1": r1["w2_1_new"], "w2_2": r1["w2_2_new"],
                 "b2": r1["b2_new"]}
        r_viz        = r2
        step_in_iter = min(cur - 7, 6)
        steps_done   = cur - 7
    else:
        p_viz        = p
        r_viz        = r1
        step_in_iter = min(cur, 6)
        steps_done   = cur

    # ── TOP ROW: Title + NIM ──
    iter_label = "Iterasi 2" if is_iter2 else "Iterasi 1"
    st.markdown(f"""
    <div class="sim-title">🧠 Backpropagation Simulator</div>
    <div class="sim-sub">NIM: {st.session_state.nim} &nbsp;|&nbsp; {iter_label} &nbsp;|&nbsp; Step {cur}/14</div>
    """, unsafe_allow_html=True)

    # ── MAIN LAYOUT: viz panel sticky, steps scrollable ──
    st.markdown("""
    <style>
      /* Outer row: children align to top, required for sticky to work */
      [data-testid="stHorizontalBlock"] {
        align-items: flex-start !important;
      }
      /* Stick every column except the last (scrollable steps column) */
      [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:not(:last-child) {
        position: sticky !important;
        top: 60px !important;
        align-self: flex-start !important;
        overflow: visible !important;
        z-index: 10;
      }
    </style>
    """, unsafe_allow_html=True)

    col_net, col_steps = st.columns([1.15, 1])

    # ── LEFT: sticky network viz + loss curve ──
    with col_net:
        st.markdown("**Visualisasi Jaringan**")
        fig_net = draw_network(p_viz, r_viz, current_step_in_iter=step_in_iter, steps_done_in_iter=steps_done)
        st.plotly_chart(fig_net, use_container_width=True, config={"displayModeBar": False})

        if st.session_state.losses:
            st.markdown("**Kurva Loss**")
            fig_loss = draw_loss_curve(st.session_state.losses)
            st.plotly_chart(fig_loss, use_container_width=True, config={"displayModeBar": False})

    # ── RIGHT: scrollable step-by-step ──
    with col_steps:
        steps_it1 = get_steps(1)
        steps_it2 = get_steps(2)

        def render_iteration(steps, res, params_used, iter_num, base_idx):
            iter_complete = st.session_state.current_step >= base_idx + len(steps)
            iter_active   = (base_idx <= st.session_state.current_step < base_idx + len(steps))

            label_color = "#1ea86a" if iter_complete else ("#e8a020" if iter_active else "#3a5070")
            st.markdown(
                f'<div class="iter-header" style="border-color:{label_color}">Iterasi {iter_num}</div>',
                unsafe_allow_html=True
            )

            for i, (step_id, label, field_key, formula, ans_key, hint) in enumerate(steps):
                global_idx = base_idx + i
                is_done_step = (field_key is None)

                if is_done_step:
                    if st.session_state.current_step > global_idx - 1:
                        # Show updated weights summary
                        with st.expander(f"✅ Iterasi {iter_num} selesai — Lihat bobot baru", expanded=(st.session_state.current_step == global_idx)):
                            st.markdown(f"""
                            | Bobot | Lama | Baru |
                            |-------|------|------|
                            | W[1]₁₁ | `{params_used['w11']:.4f}` | `{res['w11_new']:.4f}` |
                            | W[1]₂₁ | `{params_used['w21']:.4f}` | `{res['w21_new']:.4f}` |
                            | W[1]₁₂ | `{params_used['w12']:.4f}` | `{res['w12_new']:.4f}` |
                            | W[1]₂₂ | `{params_used['w22']:.4f}` | `{res['w22_new']:.4f}` |
                            | W[2]₁  | `{params_used['w2_1']:.4f}` | `{res['w2_1_new']:.4f}` |
                            | W[2]₂  | `{params_used['w2_2']:.4f}` | `{res['w2_2_new']:.4f}` |
                            | b[1]₁  | `{params_used['b1_1']:.4f}` | `{res['b1_1_new']:.4f}` |
                            | b[1]₂  | `{params_used['b1_2']:.4f}` | `{res['b1_2_new']:.4f}` |
                            | b[2]   | `{params_used['b2']:.4f}`   | `{res['b2_new']:.4f}`   |
                            """)
                            loss_before = st.session_state.losses[iter_num - 1] if len(st.session_state.losses) >= iter_num else "-"
                            st.info(f"Loss Iterasi {iter_num}: **{res['loss']:.6f}**")
                    continue

                # Step state
                if st.session_state.current_step > global_idx:
                    badge = "step-done"
                    icon  = "✓"
                elif st.session_state.current_step == global_idx:
                    badge = "step-active"
                    icon  = str(i + 1)
                else:
                    badge = "step-locked"
                    icon  = "🔒"

                with st.expander(
                    f"{'✓' if badge=='step-done' else ('▶' if badge=='step-active' else '🔒')}  Step {i+1}: {label}",
                    expanded=(st.session_state.current_step == global_idx)
                ):
                    if badge == "step-locked":
                        st.caption("Selesaikan langkah sebelumnya terlebih dahulu.")
                        continue

                    # Formula
                    st.markdown(f'<div class="formula-box">{formula}</div>', unsafe_allow_html=True)

                    correct_val = res[ans_key]

                    # Answer mode: show answer directly
                    if st.session_state.answer_mode:
                        st.markdown(
                            f'<div class="result-ok">Jawaban: {correct_val:.6f}</div>',
                            unsafe_allow_html=True
                        )
                        # Extra detail for some steps
                        if ans_key == "a1_1":
                            st.markdown(f'<div class="formula-box">a[1]₁ = {res["a1_1"]:.6f} &nbsp;|&nbsp; a[1]₂ = {res["a1_2"]:.6f}</div>', unsafe_allow_html=True)
                        if ans_key == "delta2":
                            st.markdown(f'<div class="formula-box">σ\'(z[2]) = {res["d_sig2"]:.6f}</div>', unsafe_allow_html=True)
                        if ans_key == "delta1_1":
                            st.markdown(f'<div class="formula-box">δ[1]₁ = {res["delta1_1"]:.6f} &nbsp;|&nbsp; δ[1]₂ = {res["delta1_2"]:.6f}</div>', unsafe_allow_html=True)

                    # Input field (always shown if not locked)
                    if badge in ("step-active", "step-done"):
                        user_input = st.number_input(
                            label=hint or f"Masukkan nilai ({ans_key}):",
                            value=None,
                            format="%.6f",
                            key=f"input_{step_id}",
                            placeholder="0.000000",
                            label_visibility="visible",
                        )

                        if badge == "step-active":
                            err_key = f"err_{step_id}"
                            if st.button(f"✔ Cek & Lanjut", key=f"btn_{step_id}", type="primary"):
                                if user_input is None:
                                    st.session_state.input_errors[err_key] = "Masukkan nilai terlebih dahulu."
                                elif check(user_input, correct_val):
                                    # Correct!
                                    st.session_state.input_errors.pop(err_key, None)
                                    st.session_state.current_step += 1
                                    # Record loss after loss step
                                    if ans_key == "loss":
                                        st.session_state.losses.append(correct_val)
                                    st.rerun()
                                else:
                                    st.session_state.input_errors[err_key] = (
                                        f"Belum tepat. Selisih: {abs(user_input - correct_val):.6f}  "
                                        f"(toleransi ±{TOLERANCE}). Cek kembali perhitungan Anda."
                                    )

                            err_msg = st.session_state.input_errors.get(err_key)
                            if err_msg:
                                st.error(err_msg)

                        elif badge == "step-done":
                            st.markdown(
                                f'<div class="result-ok">✓ Jawaban Anda diterima</div>',
                                unsafe_allow_html=True
                            )

        # Render Iterasi 1
        render_iteration(steps_it1, r1, p, 1, base_idx=0)

        # Render Iterasi 2 (unlocks after iter1 done step = step index 7)
        if st.session_state.current_step >= 7:
            p2 = {**p,
                  "w11": r1["w11_new"], "w21": r1["w21_new"],
                  "w12": r1["w12_new"], "w22": r1["w22_new"],
                  "b1_1": r1["b1_1_new"], "b1_2": r1["b1_2_new"],
                  "w2_1": r1["w2_1_new"], "w2_2": r1["w2_2_new"],
                  "b2": r1["b2_new"]}
            render_iteration(steps_it2, r2, p2, 2, base_idx=7)
        else:
            st.markdown("""
            <div class="card" style="opacity:0.4;text-align:center;padding:20px">
              🔒 <b>Iterasi 2</b> akan terbuka setelah Iterasi 1 selesai.
            </div>
            """, unsafe_allow_html=True)

    # ── COMPLETION BANNER + AUTO ITERATION ──
    if st.session_state.current_step >= 14:
        st.success("🎉 **Selamat!** Anda telah menyelesaikan 2 iterasi backpropagation penuh.")

        # Show loss summary for manual iterations
        if len(st.session_state.losses) >= 2:
            delta = st.session_state.losses[0] - st.session_state.losses[1]
            pct = (delta / st.session_state.losses[0]) * 100 if st.session_state.losses[0] != 0 else 0
            col1, col2, col3 = st.columns(3)
            col1.metric("Loss Iterasi 1", f"{st.session_state.losses[0]:.6f}")
            col2.metric("Loss Iterasi 2", f"{st.session_state.losses[1]:.6f}",
                        delta=f"{-delta:.6f}", delta_color="inverse")
            col3.metric("Penurunan Loss (iter 1→2)", f"{pct:.2f}%")

        st.markdown("---")
        st.markdown("### ▶ Lanjutkan Iterasi Otomatis")
        st.caption(
            "Klik tombol di bawah untuk menjalankan satu iterasi berikutnya secara otomatis "
            "dan amati bagaimana loss terus berkurang."
        )

        # Initialise auto_params from the weights after manual iter 2
        if st.session_state.auto_params is None:
            st.session_state.auto_params = {
                **p,
                "w11":  r1["w11_new"], "w21":  r1["w21_new"],
                "w12":  r1["w12_new"], "w22":  r1["w22_new"],
                "b1_1": r1["b1_1_new"], "b1_2": r1["b1_2_new"],
                "w2_1": r1["w2_1_new"], "w2_2": r1["w2_2_new"],
                "b2":   r1["b2_new"],
            }
            # Apply iter 2 update so auto_params starts from weights AFTER iter 2
            r_temp = compute_iteration(st.session_state.auto_params)
            st.session_state.auto_params = {
                **st.session_state.auto_params,
                "w11":  r_temp["w11_new"],  "w21":  r_temp["w21_new"],
                "w12":  r_temp["w12_new"],  "w22":  r_temp["w22_new"],
                "b1_1": r_temp["b1_1_new"], "b1_2": r_temp["b1_2_new"],
                "w2_1": r_temp["w2_1_new"], "w2_2": r_temp["w2_2_new"],
                "b2":   r_temp["b2_new"],
            }

        # Button: run one more iteration
        btn_col, info_col = st.columns([1, 2])
        with btn_col:
            if st.button("⚡ Jalankan 1 Iterasi Lagi", type="primary", use_container_width=True):
                r_auto = compute_iteration(st.session_state.auto_params)
                st.session_state.losses.append(r_auto["loss"])
                st.session_state.auto_iter += 1
                # Update weights for next click
                st.session_state.auto_params = {
                    **st.session_state.auto_params,
                    "w11":  r_auto["w11_new"],  "w21":  r_auto["w21_new"],
                    "w12":  r_auto["w12_new"],  "w22":  r_auto["w22_new"],
                    "b1_1": r_auto["b1_1_new"], "b1_2": r_auto["b1_2_new"],
                    "w2_1": r_auto["w2_1_new"], "w2_2": r_auto["w2_2_new"],
                    "b2":   r_auto["b2_new"],
                }
                st.rerun()

        with info_col:
            if st.session_state.auto_iter > 0:
                latest_loss = st.session_state.losses[-1]
                prev_loss   = st.session_state.losses[-2]
                delta_auto  = prev_loss - latest_loss
                pct_auto    = (delta_auto / prev_loss * 100) if prev_loss != 0 else 0
                st.markdown(
                    f'<div class="formula-box">'
                    f'Iterasi ke-<b>{2 + st.session_state.auto_iter}</b> &nbsp;|&nbsp; '
                    f'Loss: <b>{latest_loss:.6f}</b> &nbsp;|&nbsp; '
                    f'Δ dari sebelumnya: <b>{"−" if delta_auto >= 0 else "+"}{abs(delta_auto):.6f}</b> '
                    f'({pct_auto:.2f}%)'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        # Show full loss curve including auto iterations
        if len(st.session_state.losses) > 2:
            st.markdown("**Kurva Loss — Semua Iterasi**")
            all_losses = st.session_state.losses
            fig_all = go.Figure()
            # Manual segment (iter 1-2) in teal
            fig_all.add_trace(go.Scatter(
                x=list(range(1, 3)), y=all_losses[:2],
                mode="lines+markers",
                name="Manual",
                line=dict(color="#0da8a8", width=2.5),
                marker=dict(size=9, color="#ffc040", line=dict(color="#0da8a8", width=2)),
                hovertemplate="Iterasi %{x}<br>Loss: %{y:.6f}<extra>Manual</extra>",
            ))
            # Auto segment in amber
            if len(all_losses) > 2:
                auto_x = list(range(2, len(all_losses) + 1))
                auto_y = all_losses[1:]   # overlap at iter 2 for continuity
                fig_all.add_trace(go.Scatter(
                    x=auto_x, y=auto_y,
                    mode="lines+markers",
                    name="Otomatis",
                    line=dict(color="#e8a020", width=2.5, dash="dot"),
                    marker=dict(size=8, color="#e8a020"),
                    hovertemplate="Iterasi %{x}<br>Loss: %{y:.6f}<extra>Otomatis</extra>",
                ))
            fig_all.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,31,61,0.8)",
                xaxis=dict(title="Iterasi", tickmode="linear", tick0=1, dtick=1,
                           gridcolor="rgba(11,140,140,0.15)", color="#8ba0b8",
                           title_font=dict(color="#8ba0b8"), tickfont=dict(color="#8ba0b8")),
                yaxis=dict(title="Loss (MSE)",
                           gridcolor="rgba(11,140,140,0.15)", color="#8ba0b8",
                           title_font=dict(color="#8ba0b8"), tickfont=dict(color="#8ba0b8")),
                legend=dict(font=dict(color="#e8eef6"), bgcolor="rgba(0,0,0,0)"),
                margin=dict(l=10, r=10, t=10, b=10),
                height=260,
            )
            st.plotly_chart(fig_all, use_container_width=True, config={"displayModeBar": False})


# ══════════════════════════════════════════════════════════════════
#  TAB 2 — GRADIENT DESCENT EXPLORER
# ══════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("""
    <div class="sim-title">📉 Gradient Descent Explorer</div>
    <div class="sim-sub">Visualisasi kontur loss 2D · L(w₁,w₂) = (w₁−w₁*)² + (w₂−w₂*)²</div>
    """, unsafe_allow_html=True)

    # ── Init GD params from NIM ──
    if st.session_state.gd_params is None:
        nim_clean = st.session_state.nim.strip()
        seed_str  = nim_clean[-5:]
        gd_seed   = int(seed_str)
        gd_p      = generate_gd_params(gd_seed)
        st.session_state.gd_params  = gd_p
        st.session_state.gd_auto_alpha = gd_p["lr"]
        # Seed history with starting point
        init_loss = gd_loss(gd_p["w1_init"], gd_p["w2_init"], gd_p["w1_opt"], gd_p["w2_opt"])
        st.session_state.gd_history = [(gd_p["w1_init"], gd_p["w2_init"], round(init_loss, 6))]

    gd_p    = st.session_state.gd_params
    gd_hist = st.session_state.gd_history
    gd_done = st.session_state.gd_manual_done

    # ── Layout ──
    col_gd_left, col_gd_right = st.columns([1.3, 1])

    with col_gd_left:
        st.markdown("**Loss Landscape (Kontur 2D)**")
        fig_landscape = draw_gd_landscape(gd_p, gd_hist, active_step=gd_done)
        st.plotly_chart(fig_landscape, use_container_width=True, config={"displayModeBar": True, "scrollZoom": True})

        if len(gd_hist) > 1:
            st.markdown("**Kurva Konvergensi**")
            fig_curves = draw_gd_curves(gd_hist)
            st.plotly_chart(fig_curves, use_container_width=True, config={"displayModeBar": False})

    with col_gd_right:
        # ── Parameter info ──
        st.markdown(f"""
        <div class="card">
          <div class="card-title">Parameter (dari NIM)</div>
          <div class="param-grid">
            <div class="param-item"><span class="param-key">w₁ awal</span><span class="param-val">{gd_p['w1_init']}</span></div>
            <div class="param-item"><span class="param-key">w₂ awal</span><span class="param-val">{gd_p['w2_init']}</span></div>
            <div class="param-item"><span class="param-key">w₁*</span><span class="param-val">{gd_p['w1_opt']}</span></div>
            <div class="param-item"><span class="param-key">w₂*</span><span class="param-val">{gd_p['w2_opt']}</span></div>
            <div class="param-item"><span class="param-key">α (manual)</span><span class="param-val">{gd_p['lr']}</span></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Formula box ──
        st.markdown("""
        <div class="formula-box">
        L(w₁,w₂) = (w₁ − w₁*)² + (w₂ − w₂*)<br><br>
        ∂L/∂w₁ = 2·(w₁ − w₁*)<br>
        ∂L/∂w₂ = 2·(w₂ − w₂*)<br><br>
        w₁_baru = w₁ − α · ∂L/∂w₁<br>
        w₂_baru = w₂ − α · ∂L/∂w₂
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # ── MANUAL STEPS (3 steps required) ──
        if gd_done < 3:
            cur_w1, cur_w2, cur_loss = gd_hist[-1]
            correct_dw1, correct_dw2 = gd_gradient(cur_w1, cur_w2, gd_p["w1_opt"], gd_p["w2_opt"])
            correct_w1_new = round(cur_w1 - gd_p["lr"] * correct_dw1, 6)
            correct_w2_new = round(cur_w2 - gd_p["lr"] * correct_dw2, 6)

            st.markdown(
                f'<div class="iter-header" style="border-color:#e8a020">'
                f'Langkah Manual {gd_done + 1} / 3</div>',
                unsafe_allow_html=True
            )
            st.markdown(f"""
            <div class="formula-box">
            Posisi saat ini:<br>
            &nbsp;&nbsp;w₁ = <b>{cur_w1:.6f}</b><br>
            &nbsp;&nbsp;w₂ = <b>{cur_w2:.6f}</b><br>
            &nbsp;&nbsp;Loss = <b>{cur_loss:.6f}</b>
            </div>
            """, unsafe_allow_html=True)

            with st.expander("▶ Hitung Gradien & Update Bobot", expanded=True):
                st.markdown("**Step A — Hitung gradien:**")
                col_a1, col_a2 = st.columns(2)
                with col_a1:
                    val_dw1 = st.number_input("∂L/∂w₁ =", value=None, format="%.6f",
                                               key=f"gd_dw1_{gd_done}", placeholder="0.000000")
                with col_a2:
                    val_dw2 = st.number_input("∂L/∂w₂ =", value=None, format="%.6f",
                                               key=f"gd_dw2_{gd_done}", placeholder="0.000000")

                if st.session_state.answer_mode:
                    st.markdown(
                        f'<div class="result-ok">∂L/∂w₁ = {correct_dw1:.6f} &nbsp;|&nbsp; ∂L/∂w₂ = {correct_dw2:.6f}</div>',
                        unsafe_allow_html=True)

                st.markdown("**Step B — Hitung bobot baru:**")
                col_b1, col_b2 = st.columns(2)
                with col_b1:
                    val_w1_new = st.number_input("w₁_baru =", value=None, format="%.6f",
                                                  key=f"gd_w1n_{gd_done}", placeholder="0.000000")
                with col_b2:
                    val_w2_new = st.number_input("w₂_baru =", value=None, format="%.6f",
                                                  key=f"gd_w2n_{gd_done}", placeholder="0.000000")

                if st.session_state.answer_mode:
                    st.markdown(
                        f'<div class="result-ok">w₁_baru = {correct_w1_new:.6f} &nbsp;|&nbsp; w₂_baru = {correct_w2_new:.6f}</div>',
                        unsafe_allow_html=True)

                err_key = f"gd_err_{gd_done}"
                if st.button("✔ Cek & Lanjut", key=f"gd_btn_{gd_done}", type="primary"):
                    errors = []
                    if val_dw1 is None or val_dw2 is None or val_w1_new is None or val_w2_new is None:
                        errors.append("Isi semua field terlebih dahulu.")
                    else:
                        if not check(val_dw1, correct_dw1):
                            errors.append(f"∂L/∂w₁ belum tepat (selisih {abs(val_dw1-correct_dw1):.6f})")
                        if not check(val_dw2, correct_dw2):
                            errors.append(f"∂L/∂w₂ belum tepat (selisih {abs(val_dw2-correct_dw2):.6f})")
                        if not check(val_w1_new, correct_w1_new):
                            errors.append(f"w₁_baru belum tepat (selisih {abs(val_w1_new-correct_w1_new):.6f})")
                        if not check(val_w2_new, correct_w2_new):
                            errors.append(f"w₂_baru belum tepat (selisih {abs(val_w2_new-correct_w2_new):.6f})")

                    if errors:
                        st.session_state.gd_errors[err_key] = " · ".join(errors)
                    else:
                        # Correct! Advance
                        st.session_state.gd_errors.pop(err_key, None)
                        new_loss = gd_loss(correct_w1_new, correct_w2_new, gd_p["w1_opt"], gd_p["w2_opt"])
                        st.session_state.gd_history.append(
                            (round(correct_w1_new, 6), round(correct_w2_new, 6), round(new_loss, 6))
                        )
                        st.session_state.gd_manual_done += 1
                        # Save checkpoint right after 3rd manual step completes
                        if st.session_state.gd_manual_done == 3:
                            st.session_state.gd_checkpoint = list(st.session_state.gd_history)
                        st.rerun()

                if err_key in st.session_state.gd_errors:
                    st.error(st.session_state.gd_errors[err_key])

        # ── AUTO MODE (after 3 manual steps) ──
        else:
            cur_w1, cur_w2, cur_loss = gd_hist[-1]
            st.success(f"✅ **3 langkah manual selesai!** Loss saat ini: **{cur_loss:.6f}**")
            st.markdown("---")

            st.markdown("### ⚡ Mode Otomatis")
            st.caption("Ganti learning rate lalu klik untuk menjalankan iterasi berikutnya dari posisi saat ini.")

            # col_lr, col_btn = st.columns([1.2, 1])
            # with col_lr:
            #     new_alpha = st.slider(
            #         "Learning Rate (α)",
            #         min_value=0.001, max_value=5.0,
            #         value=float(st.session_state.gd_auto_alpha),
            #         step=0.001, format="%.3f",
            #         key="gd_alpha_slider",
            #     )
            #     st.session_state.gd_auto_alpha = new_alpha
            col_lr, col_btn = st.columns([1.2, 1])
            with col_lr:
                st.markdown("**Learning Rate (α)**")
                minus_col, input_col, plus_col = st.columns([1, 1.6, 1])
                with minus_col:
                    if st.button("−0.1", key="gd_lr_m1", use_container_width=True):
                        st.session_state.gd_auto_alpha = round(max(0.001, st.session_state.gd_auto_alpha - 0.1), 3)
                        st.rerun()
                    if st.button("−0.01", key="gd_lr_m2", use_container_width=True):
                        st.session_state.gd_auto_alpha = round(max(0.001, st.session_state.gd_auto_alpha - 0.01), 3)
                        st.rerun()
                with input_col:
                    new_alpha = st.number_input(
                        "α", label_visibility="collapsed",
                        min_value=0.001, max_value=5.0,
                        value=float(st.session_state.gd_auto_alpha),
                        step=0.001, format="%.3f",
                        key="gd_alpha_input",
                    )
                    st.session_state.gd_auto_alpha = new_alpha
                with plus_col:
                    if st.button("+0.1", key="gd_lr_p1", use_container_width=True):
                        st.session_state.gd_auto_alpha = round(min(5.0, st.session_state.gd_auto_alpha + 0.1), 3)
                        st.rerun()
                    if st.button("+0.01", key="gd_lr_p2", use_container_width=True):
                        st.session_state.gd_auto_alpha = round(min(5.0, st.session_state.gd_auto_alpha + 0.01), 3)
                        st.rerun()

            with col_btn:
                st.markdown("<br>", unsafe_allow_html=True)
                n_auto = st.number_input("Jumlah iterasi:", min_value=1, max_value=50,
                                          value=1, step=1, key="gd_n_auto")
                if st.button("▶ Jalankan", type="primary", use_container_width=True, key="gd_run_auto"):
                    w1_cur, w2_cur = gd_hist[-1][0], gd_hist[-1][1]
                    for _ in range(int(n_auto)):
                        dw1, dw2 = gd_gradient(w1_cur, w2_cur, gd_p["w1_opt"], gd_p["w2_opt"])
                        w1_cur = round(w1_cur - new_alpha * dw1, 6)
                        w2_cur = round(w2_cur - new_alpha * dw2, 6)
                        new_l  = round(gd_loss(w1_cur, w2_cur, gd_p["w1_opt"], gd_p["w2_opt"]), 6)
                        st.session_state.gd_history.append((w1_cur, w2_cur, new_l))
                    st.rerun()

            # Live metrics
            if len(gd_hist) > 1:
                prev_loss = gd_hist[-2][2]
                d_loss = prev_loss - cur_loss
                col_m1, col_m2, col_m3 = st.columns(3)
                col_m1.metric("Loss Saat Ini", f"{cur_loss:.6f}")
                col_m2.metric("w₁", f"{cur_w1:.4f}",
                              delta=f"{cur_w1 - gd_hist[-2][0]:.4f}")
                col_m3.metric("w₂", f"{cur_w2:.4f}",
                              delta=f"{cur_w2 - gd_hist[-2][1]:.4f}")

            # Reset GD — dua opsi
            st.markdown("---")
            rc1, rc2 = st.columns(2)
            with rc1:
                if st.button("🔄 Reset ke posisi manual", key="gd_reset",
                             use_container_width=True,
                             help="Kembalikan ke posisi akhir langkah manual ke-3. Learning rate tetap."):
                    if st.session_state.gd_checkpoint is not None:
                        st.session_state.gd_history = list(st.session_state.gd_checkpoint)
                    st.rerun()
            with rc2:
                if st.button("🗑 Reset penuh (ulang dari awal)", key="gd_reset_full",
                             use_container_width=True,
                             help="Hapus semua progress termasuk langkah manual. Mulai dari titik awal NIM."):
                    for k in ["gd_params","gd_history","gd_manual_done",
                              "gd_auto_alpha","gd_errors","gd_checkpoint"]:
                        if k in st.session_state:
                            del st.session_state[k]
                    st.rerun()
