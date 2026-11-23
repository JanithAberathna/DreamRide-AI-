"""Editorial Light design system, custom CSS, Plotly theme, and HTML UI components
for DreamRide AI (Sri Lankan Vehicle Price Prediction).
"""

from __future__ import annotations

import html
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Color Palette — Clean Editorial Light (Navy & Emerald Showroom)
# ---------------------------------------------------------------------------
PALETTE = {
    "bg": "#F8FAFC",
    "surface": "#FFFFFF",
    "surface_subtle": "#F1F5F9",
    "border": "#E2E8F0",
    "border_strong": "#CBD5E1",
    "navy": "#0F294A",
    "navy_mid": "#1E3A5F",
    "navy_soft": "#EFF6FF",
    "emerald": "#059669",
    "emerald_bright": "#10B981",
    "emerald_soft": "#ECFDF5",
    "teal": "#0D9488",
    "blue": "#2563EB",
    "amber": "#D97706",
    "amber_soft": "#FFFBEB",
    "rose": "#DC2626",
    "rose_soft": "#FEF2F2",
    "ink": "#0F172A",
    "ink_secondary": "#334155",
    "ink_muted": "#64748B",
}

MODEL_COLORS = {
    "Stacking Regressor": PALETTE["emerald"],
    "Gradient Boosting": PALETTE["navy"],
    "Random Forest": PALETTE["blue"],
    "Linear Regression": "#94A3B8",
}

MODEL_ROLES = {
    "Stacking Regressor": "Hybrid Meta-Ensemble (RidgeCV)",
    "Gradient Boosting": "Boosting (Bias Reduction)",
    "Random Forest": "Bagging (Variance Reduction)",
    "Linear Regression": "Linear Baseline",
}

PLOTLY_CONFIG = {
    "displayModeBar": False,
    "responsive": True,
}


def inject_editorial_css() -> None:
    """Inject crisp editorial light CSS, typography, and custom card styles."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

        :root {
            --bg: #F8FAFC;
            --surface: #FFFFFF;
            --surface-subtle: #F1F5F9;
            --border: #E2E8F0;
            --navy: #0F294A;
            --navy-mid: #1E3A5F;
            --emerald: #059669;
            --emerald-soft: #ECFDF5;
            --ink: #0F172A;
            --ink-muted: #64748B;
        }

        html, body, [class*="css"], .stApp {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
            color: var(--ink);
            background-color: var(--bg);
        }

        .main .block-container {
            padding-top: 1.4rem;
            padding-bottom: 3rem;
            max-width: 1380px;
        }

        /* Hide default Streamlit chrome clutter */
        #MainMenu { visibility: hidden; }
        footer { visibility: hidden; }
        header[data-testid="stHeader"] {
            background: rgba(248, 250, 252, 0.85);
            backdrop-filter: blur(8px);
        }

        /* Editorial Masthead */
        .dr-masthead {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 16px;
            padding: 1.25rem 1.65rem;
            margin-bottom: 1.25rem;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04), 0 8px 24px rgba(15, 23, 42, 0.03);
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            position: relative;
            overflow: hidden;
        }
        .dr-masthead::before {
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(90deg, #0F294A 0%, #059669 65%, #10B981 100%);
        }
        .dr-brand-wrap {
            display: flex;
            align-items: center;
            gap: 0.9rem;
        }
        .dr-logo-mark {
            width: 46px;
            height: 46px;
            border-radius: 12px;
            background: linear-gradient(135deg, #0F294A 0%, #1E3A5F 100%);
            color: #FFFFFF;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 800;
            font-size: 1.15rem;
            letter-spacing: -0.03em;
            box-shadow: 0 4px 12px rgba(15, 41, 74, 0.18);
            flex-shrink: 0;
        }
        .dr-brand-eyebrow {
            font-size: 0.7rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.11em;
            color: #059669;
            margin-bottom: 0.1rem;
        }
        .dr-brand-title {
            font-size: 1.48rem;
            font-weight: 800;
            color: #0F294A;
            letter-spacing: -0.03em;
            line-height: 1.15;
            margin: 0;
        }
        .dr-brand-sub {
            font-size: 0.85rem;
            color: #64748B;
            margin-top: 0.15rem;
        }
        .dr-status-pills {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            align-items: center;
        }
        .dr-pill {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.73rem;
            font-weight: 600;
            padding: 0.35rem 0.7rem;
            border-radius: 999px;
            background: #F8FAFC;
            border: 1px solid #E2E8F0;
            color: #334155;
            white-space: nowrap;
        }
        .dr-pill-emerald {
            background: #ECFDF5;
            border-color: #A7F3D0;
            color: #047857;
        }
        .dr-pill-navy {
            background: #EFF6FF;
            border-color: #BFDBFE;
            color: #1E3A8A;
        }

        /* Editorial Tabs */
        div[data-testid="stTabs"] button[role="tab"] {
            font-family: 'Plus Jakarta Sans', sans-serif;
            font-weight: 600;
            font-size: 0.89rem;
            color: #475569;
            padding: 0.65rem 1.05rem;
            border-radius: 10px 10px 0 0;
            transition: all 0.16s ease;
        }
        div[data-testid="stTabs"] button[role="tab"]:hover {
            color: #0F294A;
            background-color: #F1F5F9;
        }
        div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
            color: #0F294A;
            font-weight: 700;
            background-color: #FFFFFF;
        }
        div[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
            background-color: #059669 !important;
            height: 3px !important;
        }
        div[data-testid="stTabs"] [data-baseweb="tab-list"] {
            gap: 0.25rem;
            border-bottom: 1px solid #E2E8F0;
            margin-bottom: 1.1rem;
        }

        /* Section Header */
        .dr-sec-header {
            margin: 0.4rem 0 1rem 0;
        }
        .dr-sec-eyebrow {
            font-size: 0.7rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: #059669;
            margin-bottom: 0.18rem;
        }
        .dr-sec-title {
            font-size: 1.24rem;
            font-weight: 800;
            color: #0F294A;
            letter-spacing: -0.02em;
            margin: 0;
        }
        .dr-sec-desc {
            font-size: 0.88rem;
            color: #64748B;
            margin-top: 0.22rem;
            line-height: 1.5;
        }

        /* Editorial KPI Card */
        .dr-kpi-card {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 14px;
            padding: 1.05rem 1.2rem;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03);
            height: 100%;
            position: relative;
            overflow: hidden;
        }
        .dr-kpi-card.accent-emerald::before {
            content: "";
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 3px;
            background: #059669;
        }
        .dr-kpi-card.accent-navy::before {
            content: "";
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 3px;
            background: #0F294A;
        }
        .dr-kpi-label {
            font-size: 0.74rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.07em;
            color: #64748B;
            margin-bottom: 0.45rem;
        }
        .dr-kpi-value {
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.52rem;
            font-weight: 700;
            color: #0F294A;
            letter-spacing: -0.03em;
            line-height: 1.15;
        }
        .dr-kpi-sub {
            font-size: 0.79rem;
            color: #475569;
            margin-top: 0.4rem;
            line-height: 1.35;
        }
        .dr-kpi-badge {
            display: inline-block;
            font-size: 0.71rem;
            font-weight: 700;
            padding: 0.14rem 0.48rem;
            border-radius: 6px;
            margin-top: 0.4rem;
        }
        .dr-badge-emerald { background: #ECFDF5; color: #047857; }
        .dr-badge-navy { background: #EFF6FF; color: #1E3A8A; }
        .dr-badge-amber { background: #FFFBEB; color: #B45309; }
        .dr-badge-rose { background: #FEF2F2; color: #B91C1C; }

        /* Hero Valuation Card (Deep Navy Showroom Showcase) */
        .dr-hero-valuation {
            background: linear-gradient(135deg, #0B1F3A 0%, #123158 62%, #173F6E 100%);
            color: #FFFFFF;
            border-radius: 18px;
            padding: 1.55rem 1.75rem;
            box-shadow: 0 12px 32px rgba(15, 41, 74, 0.18), 0 2px 6px rgba(15, 41, 74, 0.08);
            margin-bottom: 1rem;
            position: relative;
            overflow: hidden;
        }
        .dr-hero-valuation::after {
            content: "";
            position: absolute;
            top: -40%;
            right: -12%;
            width: 280px;
            height: 280px;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(16, 185, 129, 0.22) 0%, rgba(16, 185, 129, 0) 70%);
            pointer-events: none;
        }
        .dr-hero-top {
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 0.6rem;
            margin-bottom: 0.65rem;
        }
        .dr-hero-eyebrow {
            font-size: 0.72rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.11em;
            color: #6EE7B7;
        }
        .dr-hero-conf {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.72rem;
            font-weight: 600;
            background: rgba(16, 185, 129, 0.18);
            border: 1px solid rgba(110, 231, 183, 0.35);
            color: #A7F3D0;
            padding: 0.24rem 0.65rem;
            border-radius: 999px;
        }
        .dr-hero-vehicle {
            font-size: 1.05rem;
            font-weight: 600;
            color: #E2E8F0;
            margin-bottom: 0.35rem;
        }
        .dr-hero-price-row {
            display: flex;
            align-items: baseline;
            flex-wrap: wrap;
            gap: 1rem;
            margin-bottom: 0.9rem;
        }
        .dr-hero-price {
            font-family: 'JetBrains Mono', monospace;
            font-size: 2.45rem;
            font-weight: 700;
            color: #FFFFFF;
            letter-spacing: -0.04em;
            line-height: 1.05;
        }
        .dr-hero-millions {
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.1rem;
            font-weight: 600;
            color: #6EE7B7;
        }
        .dr-hero-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.75rem;
            padding-top: 0.9rem;
            border-top: 1px solid rgba(255, 255, 255, 0.13);
        }
        .dr-hero-stat-label {
            font-size: 0.69rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #94A3B8;
            margin-bottom: 0.2rem;
        }
        .dr-hero-stat-val {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.93rem;
            font-weight: 600;
            color: #F8FAFC;
        }
        .dr-hero-tags {
            display: flex;
            flex-wrap: wrap;
            gap: 0.4rem;
            margin-top: 0.95rem;
        }
        .dr-hero-tag {
            font-size: 0.74rem;
            font-weight: 500;
            color: #CBD5E1;
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.12);
            padding: 0.2rem 0.58rem;
            border-radius: 6px;
        }

        /* Editorial Surface Cards & Callouts */
        .dr-card {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 14px;
            padding: 1.2rem 1.35rem;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03);
            height: 100%;
        }
        .dr-card-title {
            font-size: 0.95rem;
            font-weight: 700;
            color: #0F294A;
            margin-bottom: 0.45rem;
        }
        .dr-card-body {
            font-size: 0.86rem;
            color: #475569;
            line-height: 1.58;
        }
        .dr-callout {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-left: 4px solid #059669;
            border-radius: 10px;
            padding: 0.95rem 1.15rem;
            margin: 0.75rem 0 1rem 0;
            font-size: 0.87rem;
            color: #334155;
            line-height: 1.55;
        }
        .dr-callout.navy {
            border-left-color: #0F294A;
        }
        .dr-callout strong {
            color: #0F294A;
        }

        /* Deal Analyzer Card */
        .dr-deal-banner {
            border-radius: 14px;
            padding: 1rem 1.25rem;
            border: 1px solid #E2E8F0;
            margin-bottom: 1rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 0.8rem;
        }
        .dr-deal-great { background: #ECFDF5; border-color: #6EE7B7; }
        .dr-deal-fair { background: #EFF6FF; border-color: #93C5FD; }
        .dr-deal-above { background: #FFFBEB; border-color: #FCD34D; }
        .dr-deal-high { background: #FEF2F2; border-color: #FCA5A5; }
        .dr-deal-verdict {
            font-size: 1rem;
            font-weight: 800;
            letter-spacing: -0.01em;
        }
        .dr-deal-delta {
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.1rem;
            font-weight: 700;
        }

        /* Pipeline Step Diagram */
        .dr-pipeline {
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap: 0.7rem;
            margin: 0.8rem 0 1.2rem 0;
        }
        .dr-pipe-step {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 12px;
            padding: 0.95rem 0.95rem;
            position: relative;
        }
        .dr-pipe-num {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.68rem;
            font-weight: 700;
            color: #059669;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.3rem;
        }
        .dr-pipe-title {
            font-size: 0.87rem;
            font-weight: 700;
            color: #0F294A;
            margin-bottom: 0.32rem;
        }
        .dr-pipe-desc {
            font-size: 0.77rem;
            color: #64748B;
            line-height: 1.42;
        }
        .dr-pipe-step.final {
            background: #0F294A;
            border-color: #0F294A;
        }
        .dr-pipe-step.final .dr-pipe-num { color: #6EE7B7; }
        .dr-pipe-step.final .dr-pipe-title { color: #FFFFFF; }
        .dr-pipe-step.final .dr-pipe-desc { color: #CBD5E1; }

        /* Streamlit native containers & widgets refinement */
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background-color: #FFFFFF;
            border-color: #E2E8F0 !important;
            border-radius: 14px !important;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03);
        }
        .stButton > button {
            border-radius: 9px;
            font-weight: 600;
            font-size: 0.82rem;
            border: 1px solid #CBD5E1;
            transition: all 0.15s ease;
        }
        .stButton > button:hover {
            border-color: #0F294A;
            color: #0F294A;
            background-color: #F1F5F9;
        }
        .stButton > button[kind="primary"] {
            background-color: #0F294A;
            border-color: #0F294A;
            color: #FFFFFF;
        }
        .stButton > button[kind="primary"]:hover {
            background-color: #163861;
            border-color: #163861;
            color: #FFFFFF;
        }
        div[data-testid="stDataFrame"] {
            border: 1px solid #E2E8F0;
            border-radius: 12px;
            overflow: hidden;
        }

        @media (max-width: 900px) {
            .dr-pipeline { grid-template-columns: 1fr; }
            .dr-hero-grid { grid-template-columns: 1fr; }
            .dr-hero-price { font-size: 1.9rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_masthead(n_listings: int, n_brands: int, best_r2: float, best_mae: float) -> None:
    """Render the clean editorial showroom masthead."""
    st.markdown(
        f"""
        <div class="dr-masthead">
            <div class="dr-brand-wrap">
                <div class="dr-logo-mark">DR</div>
                <div>
                    <div class="dr-brand-eyebrow">Sri Lankan Automotive Valuation Intelligence</div>
                    <h1 class="dr-brand-title">DreamRide AI</h1>
                    <div class="dr-brand-sub">Ensemble Learning Price Valuation &amp; Market Analytics · Prices in LKR Lakhs (1 Lakh = Rs. 100,000)</div>
                </div>
            </div>
            <div class="dr-status-pills">
                <span class="dr-pill dr-pill-navy">{n_listings:,} Listings · {n_brands} Brands</span>
                <span class="dr-pill dr-pill-emerald">Stacking R² {best_r2:.4f}</span>
                <span class="dr-pill">Test MAE Rs. {best_mae:.2f}L</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_header(eyebrow: str, title: str, description: str = "") -> None:
    """Render a consistent editorial section title block."""
    desc_html = f'<div class="dr-sec-desc">{html.escape(description)}</div>' if description else ""
    st.markdown(
        f"""
        <div class="dr-sec-header">
            <div class="dr-sec-eyebrow">{html.escape(eyebrow)}</div>
            <h2 class="dr-sec-title">{html.escape(title)}</h2>
            {desc_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_card(
    label: str,
    value: str,
    subtext: str = "",
    badge: str = "",
    badge_tone: str = "emerald",
    accent: str = "navy",
) -> None:
    """Render a single editorial KPI metric card."""
    badge_html = (
        f'<div class="dr-kpi-badge dr-badge-{badge_tone}">{html.escape(badge)}</div>'
        if badge
        else ""
    )
    sub_html = f'<div class="dr-kpi-sub">{html.escape(subtext)}</div>' if subtext else ""
    st.markdown(
        f"""
        <div class="dr-kpi-card accent-{accent}">
            <div class="dr-kpi-label">{html.escape(label)}</div>
            <div class="dr-kpi-value">{html.escape(value)}</div>
            {sub_html}
            {badge_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_info_card(title: str, body_html: str, tag: str = "", tag_tone: str = "emerald") -> None:
    """Render a crisp editorial content card."""
    tag_html = (
        f'<div class="dr-kpi-badge dr-badge-{tag_tone}" style="margin-top:0;margin-bottom:0.55rem;">'
        f"{html.escape(tag)}</div>"
        if tag
        else ""
    )
    st.markdown(
        f"""
        <div class="dr-card">
            {tag_html}
            <div class="dr-card-title">{html.escape(title)}</div>
            <div class="dr-card-body">{body_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_callout(text_html: str, tone: str = "emerald") -> None:
    """Render an editorial insight callout bar."""
    cls = "dr-callout navy" if tone == "navy" else "dr-callout"
    st.markdown(f'<div class="{cls}">{text_html}</div>', unsafe_allow_html=True)


def style_plotly(
    fig: go.Figure,
    title: str = "",
    height: int = 380,
    x_title: str = "",
    y_title: str = "",
    show_legend: bool = True,
) -> go.Figure:
    """Apply consistent Clean Editorial Light styling to any Plotly figure."""
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>" if title else "",
            font=dict(family="Plus Jakarta Sans, sans-serif", size=14, color=PALETTE["navy"]),
            x=0.0,
            xanchor="left",
            pad=dict(b=8),
        ),
        font=dict(family="Plus Jakarta Sans, sans-serif", size=12, color=PALETTE["ink_secondary"]),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        height=height,
        margin=dict(l=16, r=16, t=44 if title else 20, b=36),
        showlegend=show_legend,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            bgcolor="rgba(255,255,255,0.8)",
            font=dict(size=11, color=PALETTE["ink_secondary"]),
        ),
        hoverlabel=dict(
            bgcolor="#0F294A",
            bordercolor="#0F294A",
            font=dict(family="Plus Jakarta Sans, sans-serif", size=12, color="#FFFFFF"),
        ),
    )
    fig.update_xaxes(
        title_text=x_title,
        title_font=dict(size=11, color=PALETTE["ink_muted"]),
        gridcolor="#F1F5F9",
        linecolor="#E2E8F0",
        zerolinecolor="#E2E8F0",
        tickfont=dict(size=11, color=PALETTE["ink_muted"]),
    )
    fig.update_yaxes(
        title_text=y_title,
        title_font=dict(size=11, color=PALETTE["ink_muted"]),
        gridcolor="#F1F5F9",
        linecolor="#E2E8F0",
        zerolinecolor="#E2E8F0",
        tickfont=dict(size=11, color=PALETTE["ink_muted"]),
    )
    return fig
