"""
bullwhip_app.py
═══════════════════════════════════════════════════════════════════════
Interactive Bullwhip Effect Simulator — BAT 3307

3-echelon supply chain: Retailer → Wholesaler → Manufacturer

Run:
    pip install streamlit plotly numpy pandas
    streamlit run bullwhip_app.py
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TU BAT 3307 - Bullwhip Effect Simulator",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme colors ──────────────────────────────────────────────────────────────
CLR = {
    "consumer":     "#64B5F6",
    "retailer":     "#E9C46A",
    "wholesaler":   "#F4A261",
    "manufacturer": "#E76F51",
    "bg":           "#0D1B2A",
    "panel":        "#1A2E44",
    "text":         "#F0F4F8",
    "grid":         "#2A3F55",
}

# ═══════════════════════════════════════════════════════════════════════
#  SIMULATION ENGINE
# ═══════════════════════════════════════════════════════════════════════

def simulate_bullwhip(
    n_periods: int,
    demand_mean: float,
    demand_std: float,
    lead_time_r: int,
    lead_time_w: int,
    lead_time_m: int,
    ma_window: int,
    ss_factor: float,
    holding_cost: float,
    stockout_cost: float,
    order_cost: float,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Simulate a 3-echelon (Retailer–Wholesaler–Manufacturer) supply chain.

    Each echelon:
      1. Receives shipments from upstream (with lead-time delay).
      2. Observes demand / orders from downstream.
      3. Forecasts next-period demand using a moving average of span ma_window.
      4. Places a new order = forecast + safety stock adjustment.
      5. Ships what it can from on-hand inventory.

    Returns a DataFrame with one row per period containing inventory,
    orders, shipments, and cost metrics for every echelon.
    """
    rng = np.random.default_rng(seed)
    consumer_demand = np.maximum(
        0, rng.normal(demand_mean, demand_std, n_periods)
    ).round()

    # ── state vectors ─────────────────────────────────────────────────
    inv   = {"R": demand_mean * 2, "W": demand_mean * 3, "M": demand_mean * 4}
    bl    = {"R": 0.0, "W": 0.0, "M": 0.0}   # backlog
    pipe  = {                                   # in-transit pipeline
        "R": [demand_mean] * lead_time_r,
        "W": [demand_mean] * lead_time_w,
        "M": [demand_mean] * lead_time_m,
    }
    hist  = {"R": [demand_mean] * ma_window,
             "W": [demand_mean] * ma_window,
             "M": [demand_mean] * ma_window}

    # ── output lists ──────────────────────────────────────────────────
    rows = []

    for t in range(n_periods):
        row = {"period": t + 1, "consumer_demand": consumer_demand[t]}

        # Receive shipments (front of pipeline)
        for e in ["R", "W", "M"]:
            arrival = pipe[e].pop(0)
            inv[e]  = max(0.0, inv[e] + arrival - bl[e])
            bl[e]   = max(0.0, bl[e] - (inv[e] + arrival))

        # Each echelon faces downstream demand
        d = {"R": consumer_demand[t], "W": 0.0, "M": 0.0}

        orders = {}
        for e in ["R", "W", "M"]:
            demand_e = d[e]
            hist[e].append(demand_e)
            if len(hist[e]) > ma_window:
                hist[e].pop(0)

            # Moving-average forecast
            forecast = np.mean(hist[e])

            # Safety stock = ss_factor × forecast std  (use rolling window std)
            fc_std  = np.std(hist[e]) if len(hist[e]) > 1 else demand_std
            ss      = ss_factor * fc_std * np.sqrt(
                {"R": lead_time_r, "W": lead_time_w, "M": lead_time_m}[e]
            )

            # Order-up-to level
            lt   = {"R": lead_time_r, "W": lead_time_w, "M": lead_time_m}[e]
            target = forecast * (lt + 1) + ss

            # Pipeline inventory (what's already ordered but not received)
            pipeline_inv = sum(pipe[e])
            order = max(0.0, target - inv[e] - pipeline_inv + bl[e])
            orders[e] = round(order)

            # Downstream echelon faces this as demand
            if e == "R":
                d["W"] = orders["R"]
            elif e == "W":
                d["M"] = orders["W"]

        # Ship (min of inventory + backlog-adjusted stock)
        shipped = {}
        for e in ["R", "W", "M"]:
            can_ship = max(0.0, inv[e])
            shipped[e] = min(can_ship, d[e] + bl[e])
            inv[e]    -= shipped[e]
            bl[e]      = max(0.0, d[e] - shipped[e])

        # Push orders into pipelines
        for e in ["R", "W", "M"]:
            pipe[e].append(orders[e])

        # ── Costs ─────────────────────────────────────────────────────
        costs = {}
        for e, label in [("R", "retailer"), ("W", "wholesaler"), ("M", "manufacturer")]:
            hc = holding_cost * inv[e]
            sc = stockout_cost * bl[e]
            oc = order_cost if orders[e] > 0 else 0.0
            costs[label] = hc + sc + oc
            row[f"inv_{label}"]   = round(inv[e], 1)
            row[f"bl_{label}"]    = round(bl[e], 1)
            row[f"order_{label}"] = orders[e]
            row[f"cost_{label}"]  = round(hc + sc + oc, 2)
            row[f"hcost_{label}"] = round(hc, 2)
            row[f"scost_{label}"] = round(sc, 2)

        row["total_cost"] = round(sum(costs.values()), 2)
        rows.append(row)

    return pd.DataFrame(rows)


def bullwhip_ratio(df: pd.DataFrame) -> dict:
    """Variance ratio of orders vs consumer demand per echelon."""
    vd = df["consumer_demand"].var()
    if vd == 0:
        return {"retailer": 1.0, "wholesaler": 1.0, "manufacturer": 1.0}
    return {
        "retailer":     round(df["order_retailer"].var()     / vd, 2),
        "wholesaler":   round(df["order_wholesaler"].var()   / vd, 2),
        "manufacturer": round(df["order_manufacturer"].var() / vd, 2),
    }


def lee_bound(lead_time: int, ma_window: int) -> float:
    """Lee et al. (1997) theoretical lower bound for variance amplification."""
    L, p = lead_time, ma_window
    return round(1 + 2 * L / p + 2 * (L / p) ** 2, 2)


# ═══════════════════════════════════════════════════════════════════════
#  SIDEBAR — PARAMETERS
# ═══════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## ⚙️ Parameters")
    st.markdown("---")

    st.markdown("### 📊 Demand (Consumer)")
    demand_mean = st.slider("Mean demand / period",   20, 300, 100, 5)
    demand_std  = st.slider("Demand std deviation",    1,  80,  20, 1)
    n_periods   = st.slider("Simulation periods",     20, 200,  60, 5)

    st.markdown("---")
    st.markdown("### 🔗 Lead Times (periods)")
    lt_r = st.slider("Retailer lead time",      1, 8, 2)
    lt_w = st.slider("Wholesaler lead time",    1, 8, 3)
    lt_m = st.slider("Manufacturer lead time",  1, 8, 4)

    st.markdown("---")
    st.markdown("### 🔮 Forecasting")
    ma_window = st.slider(
        "Moving-average window (T)",
        1, 12, 4,
        help="Shorter window = noisier forecasts = stronger bullwhip"
    )
    ss_factor = st.slider(
        "Safety-stock factor (z)",
        0.0, 3.0, 1.5, 0.1,
        help="z × σ × √L added as buffer. Higher z = more over-ordering."
    )

    st.markdown("---")
    st.markdown("### 💰 Unit Costs (per period)")
    holding_cost  = st.slider("Holding cost / unit",  0.1, 5.0, 1.0, 0.1)
    stockout_cost = st.slider("Stockout cost / unit", 1.0, 20.0, 5.0, 0.5)
    order_cost    = st.slider("Fixed order cost",     0.0, 200.0, 50.0, 10.0)

    st.markdown("---")
    seed = st.number_input("Random seed", 0, 9999, 42, 1)

# ─── Run simulation ───────────────────────────────────────────────────────────
df  = simulate_bullwhip(
    n_periods, demand_mean, demand_std,
    lt_r, lt_w, lt_m,
    ma_window, ss_factor,
    holding_cost, stockout_cost, order_cost,
    seed=int(seed),
)
bwr = bullwhip_ratio(df)
lb  = {
    "retailer":     lee_bound(lt_r, ma_window),
    "wholesaler":   lee_bound(lt_w, ma_window),
    "manufacturer": lee_bound(lt_m, ma_window),
}

# ═══════════════════════════════════════════════════════════════════════
#  HEADER
# ═══════════════════════════════════════════════════════════════════════

st.markdown(
    """
    <h1 style='color:#E9C46A; font-family:Georgia; margin-bottom:0'>
        📦 Bullwhip Effect Simulator
    </h1>
    <p style='color:#94A3B8; margin-top:4px'>
        3-Echelon Supply Chain &nbsp;·&nbsp; Retailer → Wholesaler → Manufacturer
    </p>
    """,
    unsafe_allow_html=True,
)
st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════
#  KPI CARDS — ROW 1
# ═══════════════════════════════════════════════════════════════════════

k1, k2, k3, k4, k5, k6 = st.columns(6)

def kpi(col, label, value, delta=None, help_txt=None):
    col.metric(label, value, delta=delta, help=help_txt)

kpi(k1, "🛒 Retailer BWR",     f"{bwr['retailer']}×",
    delta=f"Theory: ≥{lb['retailer']}×")
kpi(k2, "🏭 Wholesaler BWR",   f"{bwr['wholesaler']}×",
    delta=f"Theory: ≥{lb['wholesaler']}×")
kpi(k3, "🔧 Manufacturer BWR", f"{bwr['manufacturer']}×",
    delta=f"Theory: ≥{lb['manufacturer']}×")
kpi(k4, "💵 Total Chain Cost",
    f"${df['total_cost'].sum():,.0f}",
    help_txt="Sum of holding + stockout + ordering costs over all periods")
kpi(k5, "📉 Avg Stockout/Period",
    f"{(df['bl_retailer'] + df['bl_wholesaler'] + df['bl_manufacturer']).mean():.1f} units")
kpi(k6, "📈 Demand CV",
    f"{demand_std/demand_mean*100:.1f}%",
    help_txt="Coefficient of Variation of consumer demand")

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════
#  CHART 1 — ORDERS VS CONSUMER DEMAND  (the core bullwhip chart)
# ═══════════════════════════════════════════════════════════════════════

st.markdown("### 📈 Orders Across the Supply Chain vs. Consumer Demand")
st.caption(
    "The hallmark of the bullwhip effect: small consumer demand swings amplify into "
    "large order swings at each upstream tier."
)

fig1 = go.Figure()
series = [
    ("consumer_demand", "Consumer Demand", CLR["consumer"],     "dot",    2),
    ("order_retailer",  "Retailer Orders", CLR["retailer"],     "solid",  1.8),
    ("order_wholesaler","Wholesaler Orders",CLR["wholesaler"],  "solid",  1.8),
    ("order_manufacturer","Manufacturer Orders",CLR["manufacturer"],"solid",1.8),
]
for col, name, color, dash, width in series:
    fig1.add_trace(go.Scatter(
        x=df["period"], y=df[col],
        mode="lines", name=name,
        line=dict(color=color, dash=dash, width=width),
    ))

fig1.update_layout(
    height=360,
    plot_bgcolor=CLR["bg"], paper_bgcolor=CLR["bg"],
    font_color=CLR["text"],
    legend=dict(orientation="h", y=1.12, bgcolor="rgba(0,0,0,0)"),
    xaxis=dict(title="Period", gridcolor=CLR["grid"], showgrid=True),
    yaxis=dict(title="Units",  gridcolor=CLR["grid"], showgrid=True),
    hovermode="x unified",
    margin=dict(l=50, r=20, t=30, b=40),
)
st.plotly_chart(fig1, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════
#  CHART 2 — INVENTORY LEVELS  |  CHART 3 — BACKLOGS
# ═══════════════════════════════════════════════════════════════════════

col_a, col_b = st.columns(2)

def hex_to_rgba(hex_color: str, alpha: float = 0.12) -> str:
    """Convert '#RRGGBB' to 'rgba(r,g,b,alpha)'."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

with col_a:
    st.markdown("### 🏬 Inventory Levels")
    fig2 = go.Figure()
    for col, name, color in [
        ("inv_retailer",     "Retailer",     CLR["retailer"]),
        ("inv_wholesaler",   "Wholesaler",   CLR["wholesaler"]),
        ("inv_manufacturer", "Manufacturer", CLR["manufacturer"]),
    ]:
        fig2.add_trace(go.Scatter(
            x=df["period"], y=df[col],
            mode="lines", name=name,
            line=dict(color=color, width=1.8),
            fill="tozeroy",
            fillcolor=hex_to_rgba(color, 0.12),
        ))
    fig2.update_layout(
        height=310,
        plot_bgcolor=CLR["bg"], paper_bgcolor=CLR["bg"],
        font_color=CLR["text"],
        legend=dict(orientation="h", y=1.12, bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(title="Period", gridcolor=CLR["grid"]),
        yaxis=dict(title="Units on hand", gridcolor=CLR["grid"]),
        hovermode="x unified",
        margin=dict(l=50, r=20, t=10, b=40),
    )
    st.plotly_chart(fig2, use_container_width=True)

with col_b:
    st.markdown("### ⚠️ Backlog (Unfilled Demand)")
    fig3 = go.Figure()
    for col, name, color in [
        ("bl_retailer",     "Retailer",     CLR["retailer"]),
        ("bl_wholesaler",   "Wholesaler",   CLR["wholesaler"]),
        ("bl_manufacturer", "Manufacturer", CLR["manufacturer"]),
    ]:
        fig3.add_trace(go.Bar(
            x=df["period"], y=df[col],
            name=name, marker_color=color,
            opacity=0.85,
        ))
    fig3.update_layout(
        barmode="stack",
        height=310,
        plot_bgcolor=CLR["bg"], paper_bgcolor=CLR["bg"],
        font_color=CLR["text"],
        legend=dict(orientation="h", y=1.12, bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(title="Period", gridcolor=CLR["grid"]),
        yaxis=dict(title="Units backordered", gridcolor=CLR["grid"]),
        hovermode="x unified",
        margin=dict(l=50, r=20, t=10, b=40),
    )
    st.plotly_chart(fig3, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════
#  CHART 4 — COST BREAKDOWN OVER TIME
# ═══════════════════════════════════════════════════════════════════════

st.markdown("### 💰 Total Supply Chain Cost per Period")

fig4 = make_subplots(
    rows=1, cols=2,
    column_widths=[0.65, 0.35],
    subplot_titles=["Period-by-Period Cost (Stacked)", "Cumulative Cost Share by Echelon"],
    specs=[[{"type": "xy"}, {"type": "domain"}]],
)

# Stacked area cost over time
for col, name, color in [
    ("cost_retailer",     "Retailer",     CLR["retailer"]),
    ("cost_wholesaler",   "Wholesaler",   CLR["wholesaler"]),
    ("cost_manufacturer", "Manufacturer", CLR["manufacturer"]),
]:
    fig4.add_trace(go.Scatter(
        x=df["period"], y=df[col],
        mode="lines", name=name,
        stackgroup="one",
        line=dict(color=color, width=0.5),
        fillcolor=color,
    ), row=1, col=1)

# Pie — total cost share
totals = {
    "Retailer":     df["cost_retailer"].sum(),
    "Wholesaler":   df["cost_wholesaler"].sum(),
    "Manufacturer": df["cost_manufacturer"].sum(),
}
fig4.add_trace(go.Pie(
    labels=list(totals.keys()),
    values=list(totals.values()),
    marker_colors=[CLR["retailer"], CLR["wholesaler"], CLR["manufacturer"]],
    textinfo="label+percent",
    hole=0.45,
    showlegend=False,
), row=1, col=2)

fig4.update_layout(
    height=340,
    plot_bgcolor=CLR["bg"], paper_bgcolor=CLR["bg"],
    font_color=CLR["text"],
    legend=dict(orientation="h", y=1.1, bgcolor="rgba(0,0,0,0)"),
    hovermode="x unified",
    margin=dict(l=50, r=20, t=40, b=40),
)
fig4.update_xaxes(gridcolor=CLR["grid"], row=1, col=1)
fig4.update_yaxes(gridcolor=CLR["grid"], row=1, col=1)
for ann in fig4.layout.annotations:
    ann.font.color = CLR["text"]

st.plotly_chart(fig4, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════
#  CHART 5 — COST BREAKDOWN: HOLDING vs STOCKOUT
# ═══════════════════════════════════════════════════════════════════════

st.markdown("### 🔍 Cost Composition: Holding vs. Stockout per Echelon")

cost_data = pd.DataFrame({
    "Echelon":  ["Retailer", "Wholesaler", "Manufacturer"] * 2,
    "Type":     ["Holding"] * 3 + ["Stockout"] * 3,
    "Cost": [
        df["hcost_retailer"].sum(),  df["hcost_wholesaler"].sum(),  df["hcost_manufacturer"].sum(),
        df["scost_retailer"].sum(),  df["scost_wholesaler"].sum(),  df["scost_manufacturer"].sum(),
    ],
})

fig5 = px.bar(
    cost_data, x="Echelon", y="Cost", color="Type", barmode="group",
    color_discrete_map={"Holding": CLR["retailer"], "Stockout": CLR["manufacturer"]},
)
fig5.update_layout(
    height=300,
    plot_bgcolor=CLR["bg"], paper_bgcolor=CLR["bg"],
    font_color=CLR["text"],
    legend=dict(orientation="h", y=1.12, bgcolor="rgba(0,0,0,0)"),
    xaxis=dict(gridcolor=CLR["grid"]),
    yaxis=dict(title="Total Cost ($)", gridcolor=CLR["grid"]),
    margin=dict(l=50, r=20, t=20, b=40),
)
st.plotly_chart(fig5, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════
#  CHART 6 — VARIANCE AMPLIFICATION WATERFALL
# ═══════════════════════════════════════════════════════════════════════

st.markdown("### 📊 Order Variance Amplification (Bullwhip Ratios)")
col_left, col_right = st.columns([1.2, 1])

with col_left:
    st.caption(
        "Bars = simulated Var(order)/Var(demand). "
        "Dashed line = Lee et al. (1997) theoretical lower bound."
    )
    echelons  = ["Retailer", "Wholesaler", "Manufacturer"]
    sim_vals  = [bwr["retailer"], bwr["wholesaler"], bwr["manufacturer"]]
    th_vals   = [lb["retailer"],  lb["wholesaler"],  lb["manufacturer"]]

    fig6 = go.Figure()
    fig6.add_trace(go.Bar(
        x=echelons, y=sim_vals,
        name="Simulated BWR",
        marker_color=[CLR["retailer"], CLR["wholesaler"], CLR["manufacturer"]],
        text=[f"{v}×" for v in sim_vals],
        textposition="outside",
        textfont=dict(color=CLR["text"]),
    ))
    fig6.add_trace(go.Scatter(
        x=echelons, y=th_vals,
        mode="markers+lines+text",
        name="Lee et al. Lower Bound",
        marker=dict(color=CLR["consumer"], size=10, symbol="diamond"),
        line=dict(color=CLR["consumer"], dash="dash", width=1.5),
        text=[f"≥{v}×" for v in th_vals],
        textposition="top center",
        textfont=dict(color=CLR["consumer"]),
    ))
    fig6.update_layout(
        height=320,
        plot_bgcolor=CLR["bg"], paper_bgcolor=CLR["bg"],
        font_color=CLR["text"],
        legend=dict(orientation="h", y=1.12, bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor=CLR["grid"]),
        yaxis=dict(title="Var(order) / Var(demand)", gridcolor=CLR["grid"]),
        margin=dict(l=50, r=20, t=20, b=40),
    )
    st.plotly_chart(fig6, use_container_width=True)

with col_right:
    st.markdown("#### 📐 Lee et al. Formula")
    st.latex(r"\frac{\text{Var}(q)}{\text{Var}(d)} \geq 1 + \frac{2L}{p} + 2\left(\frac{L}{p}\right)^2")
    st.markdown(
        f"""
        | Parameter | Value |
        |-----------|-------|
        | Review period *p* | {ma_window} |
        | Retailer lead time *L* | {lt_r} → bound ≥ **{lb["retailer"]}×** |
        | Wholesaler *L* | {lt_w} → bound ≥ **{lb["wholesaler"]}×** |
        | Manufacturer *L* | {lt_m} → bound ≥ **{lb["manufacturer"]}×** |
        | Safety stock factor *z* | {ss_factor} |
        """
    )
    st.info(
        "💡 Longer lead times and shorter MA windows increase the "
        "theoretical lower bound. Try **L=6, T=1** for an extreme case."
    )

# ═══════════════════════════════════════════════════════════════════════
#  CHART 7 — ORDER QUANTITY DISTRIBUTION (histograms)
# ═══════════════════════════════════════════════════════════════════════

st.markdown("### 📉 Order Quantity Distributions")
st.caption("Wider distributions indicate more volatile ordering — a direct measure of the bullwhip effect.")

fig7 = go.Figure()
for col, name, color in [
    ("consumer_demand",    "Consumer Demand",  CLR["consumer"]),
    ("order_retailer",     "Retailer",         CLR["retailer"]),
    ("order_wholesaler",   "Wholesaler",        CLR["wholesaler"]),
    ("order_manufacturer", "Manufacturer",     CLR["manufacturer"]),
]:
    fig7.add_trace(go.Histogram(
        x=df[col], name=name,
        marker_color=color, opacity=0.65,
        nbinsx=30,
    ))
fig7.update_layout(
    barmode="overlay",
    height=300,
    plot_bgcolor=CLR["bg"], paper_bgcolor=CLR["bg"],
    font_color=CLR["text"],
    legend=dict(orientation="h", y=1.12, bgcolor="rgba(0,0,0,0)"),
    xaxis=dict(title="Order Quantity (units)", gridcolor=CLR["grid"]),
    yaxis=dict(title="Frequency",              gridcolor=CLR["grid"]),
    margin=dict(l=50, r=20, t=10, b=40),
)
st.plotly_chart(fig7, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════
#  SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════════

st.markdown("### 📋 Echelon Summary Statistics")

summary = pd.DataFrame({
    "Metric": [
        "Mean Order (units)", "Std Dev Orders (units)",
        "Bullwhip Ratio (×)", "Lee Bound (×)",
        "Avg Inventory (units)", "Avg Backlog (units)",
        "Total Holding Cost ($)", "Total Stockout Cost ($)", "Total Cost ($)",
    ],
    "Consumer Demand": [
        f"{df['consumer_demand'].mean():.1f}", f"{df['consumer_demand'].std():.1f}",
        "1.00", "1.00", "—", "—", "—", "—", "—",
    ],
    "Retailer": [
        f"{df['order_retailer'].mean():.1f}", f"{df['order_retailer'].std():.1f}",
        f"{bwr['retailer']}", f"{lb['retailer']}",
        f"{df['inv_retailer'].mean():.1f}", f"{df['bl_retailer'].mean():.1f}",
        f"${df['hcost_retailer'].sum():,.0f}", f"${df['scost_retailer'].sum():,.0f}",
        f"${df['cost_retailer'].sum():,.0f}",
    ],
    "Wholesaler": [
        f"{df['order_wholesaler'].mean():.1f}", f"{df['order_wholesaler'].std():.1f}",
        f"{bwr['wholesaler']}", f"{lb['wholesaler']}",
        f"{df['inv_wholesaler'].mean():.1f}", f"{df['bl_wholesaler'].mean():.1f}",
        f"${df['hcost_wholesaler'].sum():,.0f}", f"${df['scost_wholesaler'].sum():,.0f}",
        f"${df['cost_wholesaler'].sum():,.0f}",
    ],
    "Manufacturer": [
        f"{df['order_manufacturer'].mean():.1f}", f"{df['order_manufacturer'].std():.1f}",
        f"{bwr['manufacturer']}", f"{lb['manufacturer']}",
        f"{df['inv_manufacturer'].mean():.1f}", f"{df['bl_manufacturer'].mean():.1f}",
        f"${df['hcost_manufacturer'].sum():,.0f}", f"${df['scost_manufacturer'].sum():,.0f}",
        f"${df['cost_manufacturer'].sum():,.0f}",
    ],
})

st.dataframe(
    summary.set_index("Metric"),
    use_container_width=True,
    height=360,
)

# ═══════════════════════════════════════════════════════════════════════
#  FOOTER
# ═══════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown(
    """
    <div style='text-align:center; color:#64748B; font-size:12px'>
    Based on Lee, Padmanabhan & Whang (1997) · <em>The Bullwhip Effect in Supply Chains</em>, Sloan Management Review<br>
    BAT 3307 Supply Management · Trinity University
    </div>
    """,
    unsafe_allow_html=True,
)
