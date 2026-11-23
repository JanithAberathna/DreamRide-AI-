"""Interactive dashboard sections for DreamRide AI:
1. Live Valuation Studio (Predictor + Deal Checker + Depreciation Simulator + Comparables)
2. Executive Overview & Data Audit
3. Market Explorer (Interactive EDA)
4. Ensemble & Stacking Lab (Test, 5-Fold CV, RidgeCV Meta-Weights, Split Robustness)
5. Explainability & Diagnostics (Permutation Importance, Parity, Residuals, Tier Errors)
"""

from __future__ import annotations

import html
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data_utils import (
    REFERENCE_YEAR,
    build_feature_row,
    format_lkr_lakhs,
    format_lkr_millions,
)
from src.ui_theme import (
    MODEL_COLORS,
    MODEL_ROLES,
    PALETTE,
    PLOTLY_CONFIG,
    render_callout,
    render_info_card,
    render_kpi_card,
    render_section_header,
    style_plotly,
)

MODEL_ORDER = [
    "Stacking Regressor",
    "Gradient Boosting",
    "Random Forest",
    "Linear Regression",
]


def _pick_preset_model(meta: dict, brand: str, candidates: list[str]) -> tuple[str, str] | None:
    """Safely find an existing (brand, model) pair in dropdown metadata."""
    if brand not in meta["brand_to_models"]:
        return None
    available = meta["brand_to_models"][brand]
    for cand in candidates:
        if cand in available:
            return brand, cand
    return brand, available[0]


def _predict_single(models: dict, input_df: pd.DataFrame) -> dict[str, float]:
    """Run live inference across all four saved pipelines."""
    return {
        name: float(np.clip(models[name].predict(input_df)[0], 0.0, None))
        for name in MODEL_ORDER
        if name in models
    }


def _apply_preset(meta: dict, brand: str, model_name: str, overrides: dict | None = None) -> None:
    """Write preset vehicle values into session state for instant showcase demos."""
    defaults = meta["model_defaults"].get(
        f"{brand}||{model_name}",
        {"median_yom": 2016, "median_engine": 1500.0, "mode_gear": "Automatic", "mode_fuel": "Petrol"},
    )
    overrides = overrides or {}
    st.session_state["pred_brand"] = brand
    st.session_state["pred_model"] = model_name
    st.session_state["pred_yom"] = int(overrides.get("yom", defaults["median_yom"]))
    st.session_state["pred_engine"] = float(overrides.get("engine", defaults["median_engine"]))
    st.session_state["pred_gear"] = overrides.get("gear", defaults["mode_gear"])
    st.session_state["pred_fuel"] = overrides.get("fuel", defaults["mode_fuel"])
    st.session_state["pred_town"] = overrides.get("town", "Colombo")
    st.session_state["pred_condition"] = overrides.get("condition", "USED")
    st.session_state["pred_leasing"] = overrides.get("leasing", "No Leasing")
    st.session_state["pred_ac"] = True
    st.session_state["pred_ps"] = True
    st.session_state["pred_pm"] = True
    st.session_state["pred_pw"] = True
    st.session_state["pred_asking"] = float(overrides.get("asking", 0.0))


def _ensure_predictor_state(meta: dict) -> None:
    """Initialize predictor session state cleanly on first load."""
    if "pred_brand" not in st.session_state:
        init_brand = "TOYOTA" if "TOYOTA" in meta["brands"] else meta["brands"][0]
        init_models = meta["brand_to_models"][init_brand]
        init_model = "VITZ" if "VITZ" in init_models else init_models[0]
        _apply_preset(meta, init_brand, init_model)


# ===========================================================================
# TAB 1: LIVE VALUATION STUDIO
# ===========================================================================
def render_predictor_tab(models: dict, meta: dict, df: pd.DataFrame, summary: dict | None) -> None:
    _ensure_predictor_state(meta)

    render_section_header(
        "Live Valuation Studio",
        "Instant Sri Lankan Vehicle Price Estimator",
        "Configure any vehicle specification below for real-time inference across all four trained pipelines, "
        "plus a Deal Checker, multi-year Depreciation Forecast, and real comparable market listings.",
    )

    # Quick-select Sri Lankan Showroom Presets
    presets_spec = [
        ("Toyota Vitz (Compact)", "TOYOTA", ["VITZ", "AQUA", "AXIO"], {"town": "Colombo", "asking": 58.0}),
        ("Honda Fit GP5 (Hybrid)", "HONDA", ["FIT GP5", "GRACE", "VEZEL"], {"town": "Nugegoda", "asking": 89.0}),
        ("Suzuki Wagon R FZ", "SUZUKI", ["WAGON R FZ", "WAGON R", "SWIFT"], {"town": "Gampaha", "asking": 64.0}),
        ("Mercedes-Benz C200 (Luxury)", "MERCEDES-BENZ", ["C200", "C180", "E200"], {"town": "Colombo", "asking": 195.0}),
    ]
    preset_cols = st.columns([1.1, 1, 1, 1, 1], gap="small")
    preset_cols[0].markdown(
        "<div style='padding-top:0.45rem;font-size:0.78rem;font-weight:700;"
        "text-transform:uppercase;letter-spacing:0.08em;color:#64748B;'>Quick Showroom Demos</div>",
        unsafe_allow_html=True,
    )
    for idx, (label, b_name, m_cands, ovr) in enumerate(presets_spec):
        picked = _pick_preset_model(meta, b_name, m_cands)
        if picked and preset_cols[idx + 1].button(label, key=f"preset_{idx}", width="stretch"):
            _apply_preset(meta, picked[0], picked[1], overrides=ovr)
            st.rerun()

    st.markdown("<div style='height:0.55rem'></div>", unsafe_allow_html=True)

    left_col, right_col = st.columns([0.42, 0.58], gap="large")

    # -----------------------------------------------------------------------
    # Left Column: Vehicle Configuration Card
    # -----------------------------------------------------------------------
    with left_col:
        with st.container(border=True):
            st.markdown(
                "<div style='font-size:0.75rem;font-weight:700;text-transform:uppercase;"
                "letter-spacing:0.09em;color:#059669;margin-bottom:0.2rem;'>Step 1 · Vehicle Identity</div>"
                "<div style='font-size:1.05rem;font-weight:800;color:#0F294A;margin-bottom:0.7rem;'>"
                "Specification Sheet</div>",
                unsafe_allow_html=True,
            )

            # Brand selection — when changed, reset model + model defaults smoothly
            prev_brand = st.session_state["pred_brand"]
            brand = st.selectbox(
                "Make / Brand",
                meta["brands"],
                index=meta["brands"].index(prev_brand) if prev_brand in meta["brands"] else 0,
                key="ui_brand",
            )
            brand_models = meta["brand_to_models"][brand]
            if brand != prev_brand:
                st.session_state["pred_brand"] = brand
                first_model = brand_models[0]
                st.session_state["pred_model"] = first_model
                d = meta["model_defaults"].get(f"{brand}||{first_model}")
                if d:
                    st.session_state["pred_yom"] = int(d["median_yom"])
                    st.session_state["pred_engine"] = float(d["median_engine"])
                    st.session_state["pred_gear"] = d["mode_gear"]
                    st.session_state["pred_fuel"] = d["mode_fuel"]

            prev_model = st.session_state["pred_model"]
            if prev_model not in brand_models:
                prev_model = brand_models[0]
                st.session_state["pred_model"] = prev_model

            model_name = st.selectbox(
                "Model / Trim",
                brand_models,
                index=brand_models.index(prev_model),
                key="ui_model",
            )
            if model_name != prev_model:
                st.session_state["pred_model"] = model_name
                d = meta["model_defaults"].get(f"{brand}||{model_name}")
                if d:
                    st.session_state["pred_yom"] = int(d["median_yom"])
                    st.session_state["pred_engine"] = float(d["median_engine"])
                    st.session_state["pred_gear"] = d["mode_gear"]
                    st.session_state["pred_fuel"] = d["mode_fuel"]

            # Year & Engine sliders/inputs
            y_min, y_max = int(meta["yom_min"]), int(meta["yom_max"])
            cur_yom = int(np.clip(st.session_state.get("pred_yom", 2016), y_min, y_max))
            yom = st.slider(
                "Year of Manufacture (YOM)",
                min_value=y_min,
                max_value=y_max,
                value=cur_yom,
                step=1,
            )
            st.session_state["pred_yom"] = yom
            vehicle_age = max(0, REFERENCE_YEAR - int(yom))

            c_eng, c_age = st.columns([0.62, 0.38])
            cur_eng = float(np.clip(st.session_state.get("pred_engine", 1500.0), 500.0, 5000.0))
            engine_cc = c_eng.number_input(
                "Engine Capacity (cc)",
                min_value=500.0,
                max_value=5000.0,
                value=cur_eng,
                step=50.0,
            )
            st.session_state["pred_engine"] = engine_cc
            c_age.markdown(
                f"""
                <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:10px;
                            padding:0.42rem 0.7rem;margin-top:1.58rem;text-align:center;">
                    <span style="font-size:0.68rem;font-weight:700;text-transform:uppercase;color:#64748B;">Age</span>
                    <div style="font-family:'JetBrains Mono',monospace;font-size:0.95rem;font-weight:700;color:#0F294A;">
                        {vehicle_age} yrs
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Powertrain & Market details
            p1, p2 = st.columns(2)
            gear_cur = st.session_state.get("pred_gear", meta["gears"][0])
            gear = p1.selectbox(
                "Transmission",
                meta["gears"],
                index=meta["gears"].index(gear_cur) if gear_cur in meta["gears"] else 0,
            )
            st.session_state["pred_gear"] = gear

            fuel_cur = st.session_state.get("pred_fuel", meta["fuel_types"][0])
            fuel_type = p2.selectbox(
                "Fuel Type",
                meta["fuel_types"],
                index=meta["fuel_types"].index(fuel_cur) if fuel_cur in meta["fuel_types"] else 0,
            )
            st.session_state["pred_fuel"] = fuel_type

            m1, m2 = st.columns(2)
            town_cur = st.session_state.get("pred_town", "Colombo")
            town = m1.selectbox(
                "Listing Town",
                meta["towns"],
                index=meta["towns"].index(town_cur) if town_cur in meta["towns"] else 0,
            )
            st.session_state["pred_town"] = town

            cond_cur = st.session_state.get("pred_condition", "USED")
            condition = m2.selectbox(
                "Condition",
                meta["conditions"],
                index=meta["conditions"].index(cond_cur) if cond_cur in meta["conditions"] else 0,
            )
            st.session_state["pred_condition"] = condition

            lease_cur = st.session_state.get("pred_leasing", meta["leasing_options"][0])
            leasing = st.selectbox(
                "Leasing Status",
                meta["leasing_options"],
                index=meta["leasing_options"].index(lease_cur) if lease_cur in meta["leasing_options"] else 0,
            )
            st.session_state["pred_leasing"] = leasing

            # Equipment / Facilities
            st.markdown(
                "<div style='font-size:0.74rem;font-weight:700;text-transform:uppercase;"
                "letter-spacing:0.08em;color:#64748B;margin:0.65rem 0 0.35rem 0;'>Cabin &amp; Power Equipment</div>",
                unsafe_allow_html=True,
            )
            e1, e2 = st.columns(2)
            air_condition = e1.toggle("Air Conditioning", value=bool(st.session_state.get("pred_ac", True)))
            power_steering = e2.toggle("Power Steering", value=bool(st.session_state.get("pred_ps", True)))
            power_mirror = e1.toggle("Power Mirrors", value=bool(st.session_state.get("pred_pm", True)))
            power_window = e2.toggle("Power Windows", value=bool(st.session_state.get("pred_pw", True)))
            st.session_state["pred_ac"] = air_condition
            st.session_state["pred_ps"] = power_steering
            st.session_state["pred_pm"] = power_mirror
            st.session_state["pred_pw"] = power_window
            fac_count = int(air_condition) + int(power_steering) + int(power_mirror) + int(power_window)

            st.divider()

            # Optional Deal Checker
            st.markdown(
                "<div style='font-size:0.75rem;font-weight:700;text-transform:uppercase;"
                "letter-spacing:0.09em;color:#059669;margin-bottom:0.15rem;'>Step 2 · Optional Deal Analyzer</div>"
                "<div style='font-size:0.8rem;color:#64748B;margin-bottom:0.45rem;'>"
                "Enter a seller's asking price (Lakhs) to check if a listing is a bargain or overpriced. Set to 0 to hide.</div>",
                unsafe_allow_html=True,
            )
            asking_price = st.number_input(
                "Seller Asking Price (LKR Lakhs)",
                min_value=0.0,
                max_value=1000.0,
                value=float(st.session_state.get("pred_asking", 0.0)),
                step=1.0,
                help="1 Lakh = Rs. 100,000. Example: 65.0 = Rs. 6.5 Million.",
            )
            st.session_state["pred_asking"] = asking_price

    # -----------------------------------------------------------------------
    # Compute Live Predictions
    # -----------------------------------------------------------------------
    input_df = build_feature_row(
        brand=brand,
        model=model_name,
        yom=int(yom),
        engine_cc=float(engine_cc),
        gear=gear,
        fuel_type=fuel_type,
        town=town,
        leasing=leasing,
        condition=condition,
        air_condition=air_condition,
        power_steering=power_steering,
        power_mirror=power_mirror,
        power_window=power_window,
    )
    preds = _predict_single(models, input_df)
    best_price = preds["Stacking Regressor"]
    rf_price = preds["Random Forest"]
    gb_price = preds["Gradient Boosting"]

    # Calibrated Fair Market Range combining model spread + test MedAE (2.32L)
    test_medae = (
        summary["test_metrics"]["Stacking Regressor"]["MedAE"]
        if summary and "test_metrics" in summary
        else 2.32
    )
    tree_spread = abs(rf_price - gb_price) / 2.0
    half_band = max(test_medae, tree_spread, best_price * 0.045)
    low_band = max(1.0, best_price - half_band)
    high_band = best_price + half_band

    agreement_pct = max(0.0, 100.0 - (abs(rf_price - gb_price) / max(best_price, 1.0)) * 100.0)
    if agreement_pct >= 94:
        conf_label = f"High Ensemble Consensus · {agreement_pct:.1f}%"
    elif agreement_pct >= 86:
        conf_label = f"Strong Consensus · {agreement_pct:.1f}%"
    else:
        conf_label = f"Moderate Spread · {agreement_pct:.1f}%"

    # -----------------------------------------------------------------------
    # Right Column: Hero Valuation Showcase + Deal Checker + Ensemble Chart
    # -----------------------------------------------------------------------
    with right_col:
        st.markdown(
            f"""
            <div class="dr-hero-valuation">
                <div class="dr-hero-top">
                    <div class="dr-hero-eyebrow">Stacking Regressor · Primary Valuation</div>
                    <div class="dr-hero-conf">{html.escape(conf_label)}</div>
                </div>
                <div class="dr-hero-vehicle">{int(yom)} {html.escape(brand)} {html.escape(model_name)}</div>
                <div class="dr-hero-price-row">
                    <div class="dr-hero-price">{format_lkr_lakhs(best_price)}</div>
                    <div class="dr-hero-millions">{format_lkr_millions(best_price)}</div>
                </div>
                <div class="dr-hero-grid">
                    <div>
                        <div class="dr-hero-stat-label">Full Sri Lankan Rupees</div>
                        <div class="dr-hero-stat-val">Rs. {best_price * 100_000:,.0f}</div>
                    </div>
                    <div>
                        <div class="dr-hero-stat-label">Fair Market Range</div>
                        <div class="dr-hero-stat-val">{low_band:,.1f}L – {high_band:,.1f}L</div>
                    </div>
                    <div>
                        <div class="dr-hero-stat-label">RF vs GB Spread</div>
                        <div class="dr-hero-stat-val">± Rs. {abs(rf_price - gb_price) / 2:,.2f}L</div>
                    </div>
                </div>
                <div class="dr-hero-tags">
                    <span class="dr-hero-tag">{engine_cc:,.0f} cc</span>
                    <span class="dr-hero-tag">{html.escape(gear)}</span>
                    <span class="dr-hero-tag">{html.escape(fuel_type)}</span>
                    <span class="dr-hero-tag">{html.escape(condition)}</span>
                    <span class="dr-hero-tag">{html.escape(town)}</span>
                    <span class="dr-hero-tag">Options: {fac_count}/4</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Deal Analyzer Banner (when seller asking price is provided)
        if asking_price > 0:
            diff_lakhs = asking_price - best_price
            diff_pct = (diff_lakhs / max(best_price, 0.1)) * 100.0
            if diff_pct <= -6.0:
                deal_cls, verdict, tone_color = "dr-deal-great", "Great Deal · Below Market", "#047857"
                advice = f"Priced Rs. {abs(diff_lakhs):,.2f}L ({abs(diff_pct):.1f}%) below AI fair valuation."
            elif diff_pct <= 6.0:
                deal_cls, verdict, tone_color = "dr-deal-fair", "Fair Market Price", "#1E3A8A"
                advice = f"Within ±6% of expected Sri Lankan market value ({diff_pct:+.1f}%)."
            elif diff_pct <= 15.0:
                deal_cls, verdict, tone_color = "dr-deal-above", "Slightly Above Market", "#B45309"
                advice = f"Asking Rs. {diff_lakhs:,.2f}L ({diff_pct:+.1f}%) over estimate — room to negotiate."
            else:
                deal_cls, verdict, tone_color = "dr-deal-high", "Overpriced Listing", "#B91C1C"
                advice = f"Asking Rs. {diff_lakhs:,.2f}L ({diff_pct:+.1f}%) above fair value — inspect trim/upgrades carefully."

            st.markdown(
                f"""
                <div class="dr-deal-banner {deal_cls}">
                    <div>
                        <div style="font-size:0.7rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:{tone_color};">
                            Deal Analyzer · Asking Rs. {asking_price:,.2f}L ({format_lkr_millions(asking_price)})
                        </div>
                        <div class="dr-deal-verdict" style="color:{tone_color};">{verdict}</div>
                        <div style="font-size:0.82rem;color:#334155;margin-top:0.15rem;">{advice}</div>
                    </div>
                    <div class="dr-deal-delta" style="color:{tone_color};">
                        {diff_lakhs:+.2f}L ({diff_pct:+.1f}%)
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # 4-Model Live Comparison Chart
        with st.container(border=True):
            chart_models = MODEL_ORDER[::-1]
            chart_vals = [preds[m] for m in chart_models]
            chart_colors = [MODEL_COLORS[m] for m in chart_models]

            fig_ens = go.Figure()
            fig_ens.add_trace(
                go.Bar(
                    y=chart_models,
                    x=chart_vals,
                    orientation="h",
                    marker=dict(color=chart_colors, line=dict(width=0)),
                    text=[f"Rs. {v:,.2f}L ({v/10:.2f}M)" for v in chart_vals],
                    textposition="outside",
                    textfont=dict(family="JetBrains Mono, monospace", size=11, color=PALETTE["navy"]),
                    hovertemplate="<b>%{y}</b><br>Estimate: Rs. %{x:,.2f} Lakhs<extra></extra>",
                )
            )
            if asking_price > 0:
                fig_ens.add_vline(
                    x=asking_price,
                    line_dash="dash",
                    line_color=PALETTE["amber"],
                    line_width=2,
                    annotation_text=f"Asking {asking_price:.1f}L",
                    annotation_position="top",
                    annotation_font=dict(size=11, color=PALETTE["amber"]),
                )
            max_x = max(chart_vals + ([asking_price] if asking_price > 0 else [0])) * 1.24
            style_plotly(
                fig_ens,
                title="Live Multi-Model Ensemble Comparison (LKR Lakhs)",
                height=260,
                x_title="Predicted Price (LKR Lakhs)",
                show_legend=False,
            )
            fig_ens.update_xaxes(range=[0, max(10.0, max_x)])
            st.plotly_chart(fig_ens, width="stretch", config=PLOTLY_CONFIG)

            # Compact comparison table
            comp_rows = [
                {
                    "Model": name,
                    "Family": MODEL_ROLES[name],
                    "LKR Lakhs": round(preds[name], 2),
                    "LKR Millions": format_lkr_millions(preds[name]),
                    "vs Stacking": "Primary" if name == "Stacking Regressor" else f"{preds[name] - best_price:+.2f}L",
                }
                for name in MODEL_ORDER
            ]
            st.dataframe(pd.DataFrame(comp_rows), width="stretch", hide_index=True)

    # -----------------------------------------------------------------------
    # Bottom Row: Depreciation What-If Curve + Real Comparable Listings
    # -----------------------------------------------------------------------
    st.markdown("<div style='height:0.65rem'></div>", unsafe_allow_html=True)
    b_left, b_right = st.columns([0.48, 0.52], gap="large")

    with b_left:
        with st.container(border=True):
            # Simulate YOM - 5 to min(y_max, YOM + 4) in a single batched DataFrame inference!
            sim_start = max(int(meta["yom_min"]), int(yom) - 5)
            sim_end = min(int(meta["yom_max"]), int(yom) + 4)
            sim_years = list(range(sim_start, sim_end + 1))
            sim_rows = pd.concat(
                [
                    build_feature_row(
                        brand=brand,
                        model=model_name,
                        yom=yr,
                        engine_cc=float(engine_cc),
                        gear=gear,
                        fuel_type=fuel_type,
                        town=town,
                        leasing=leasing,
                        condition=condition,
                        air_condition=air_condition,
                        power_steering=power_steering,
                        power_mirror=power_mirror,
                        power_window=power_window,
                    )
                    for yr in sim_years
                ],
                ignore_index=True,
            )
            sim_stack = np.clip(models["Stacking Regressor"].predict(sim_rows), 0.0, None)
            sim_rf = np.clip(models["Random Forest"].predict(sim_rows), 0.0, None)
            sim_gb = np.clip(models["Gradient Boosting"].predict(sim_rows), 0.0, None)

            fig_dep = go.Figure()
            fig_dep.add_trace(
                go.Scatter(
                    x=sim_years,
                    y=sim_rf,
                    mode="lines",
                    name="Random Forest",
                    line=dict(color=PALETTE["blue"], width=1.6, dash="dot"),
                )
            )
            fig_dep.add_trace(
                go.Scatter(
                    x=sim_years,
                    y=sim_gb,
                    mode="lines",
                    name="Gradient Boosting",
                    line=dict(color=PALETTE["navy"], width=1.6, dash="dash"),
                )
            )
            fig_dep.add_trace(
                go.Scatter(
                    x=sim_years,
                    y=sim_stack,
                    mode="lines+markers",
                    name="Stacking (Primary)",
                    line=dict(color=PALETTE["emerald"], width=3),
                    marker=dict(size=7, color=PALETTE["emerald"]),
                    hovertemplate="YOM %{x}<br>Stacking: Rs. %{y:,.2f}L<extra></extra>",
                )
            )
            # Highlight selected YOM
            fig_dep.add_trace(
                go.Scatter(
                    x=[int(yom)],
                    y=[best_price],
                    mode="markers",
                    name="Selected Year",
                    marker=dict(size=13, color="#FFFFFF", line=dict(color=PALETTE["navy"], width=3)),
                    hovertemplate="<b>Selected %{x}</b><br>Rs. %{y:,.2f}L<extra></extra>",
                )
            )
            style_plotly(
                fig_dep,
                title=f"Year-of-Manufacture Sensitivity Curve ({brand} {model_name})",
                height=345,
                x_title="Year of Manufacture (YOM)",
                y_title="Estimated Price (LKR Lakhs)",
            )
            fig_dep.update_xaxes(dtick=1)
            st.plotly_chart(fig_dep, width="stretch", config=PLOTLY_CONFIG)

            if len(sim_years) >= 2:
                avg_yr_step = float((sim_stack[-1] - sim_stack[0]) / (sim_years[-1] - sim_years[0]))
                st.caption(
                    f"Simulated vintage value shift for this exact spec across {sim_years[0]}–{sim_years[-1]}: "
                    f"avg **{avg_yr_step:+.2f} Lakhs / year**."
                )

    with b_right:
        with st.container(border=True):
            exact_matches = df[(df["Brand"] == brand) & (df["Model"] == model_name)].copy()
            match_scope = f"{brand} {model_name}"
            if exact_matches.empty:
                exact_matches = df[df["Brand"] == brand].copy()
                match_scope = f"{brand} (Brand-Level Fallback)"

            exact_matches["yom_diff"] = (exact_matches["YOM"] - int(yom)).abs()
            exact_matches["eng_diff"] = (exact_matches["Engine (cc)"] - float(engine_cc)).abs()
            exact_matches["gear_diff"] = (exact_matches["Gear"] != gear).astype(int)
            exact_matches["fuel_diff"] = (exact_matches["Fuel Type"] != fuel_type).astype(int)
            similar = exact_matches.sort_values(
                ["gear_diff", "fuel_diff", "yom_diff", "eng_diff"]
            ).head(8)

            med_comp = float(similar["Price"].median()) if not similar.empty else best_price
            st.markdown(
                f"""
                <div style="display:flex;justify-content:space-between;align-items:baseline;flex-wrap:wrap;margin-bottom:0.65rem;">
                    <div>
                        <div style="font-size:0.93rem;font-weight:700;color:#0F294A;">
                            Comparable Real Listings · {html.escape(match_scope)}
                        </div>
                        <div style="font-size:0.78rem;color:#64748B;">
                            {len(exact_matches):,} matching records in dataset · Showing top {len(similar)} closest specs
                        </div>
                    </div>
                    <div style="font-family:'JetBrains Mono',monospace;font-size:0.82rem;font-weight:700;color:#059669;">
                        Comp Median: Rs. {med_comp:,.2f}L
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            display_sim = similar[
                ["YOM", "Engine (cc)", "Gear", "Fuel Type", "Town", "Condition", "Price"]
            ].copy()
            display_sim["vs Estimate"] = display_sim["Price"].apply(lambda p: f"{p - best_price:+.1f}L")
            display_sim = display_sim.rename(columns={"Price": "Listed (Lakhs)", "Engine (cc)": "CC"})
            display_sim["Listed (Lakhs)"] = display_sim["Listed (Lakhs)"].round(2)
            st.dataframe(display_sim, width="stretch", hide_index=True, height=305)


# ===========================================================================
# TAB 2: EXECUTIVE OVERVIEW & DATA AUDIT
# ===========================================================================
def render_overview_tab(summary: dict | None) -> None:
    render_section_header(
        "Executive Overview",
        "Controlled Ensemble Learning Study for Sri Lankan Vehicle Valuation",
        "DreamRide AI benchmarks linear modeling against Bagging, Boosting, and Stacking hybridization "
        "on 9,770 audited Sri Lankan car listings scraped between December 2024 and February 2025.",
    )

    if not summary:
        st.info("Run the training pipeline to populate `artifacts/evaluation_summary.json`.")
        return

    audit = summary["dataset_audit"]
    test_m = summary["test_metrics"]
    lr_mae = test_m["Linear Regression"]["MAE"]
    st_mae = test_m["Stacking Regressor"]["MAE"]
    mae_drop_pct = (lr_mae - st_mae) / lr_mae * 100.0

    # Top 4 KPI Cards
    k1, k2, k3, k4 = st.columns(4, gap="medium")
    with k1:
        render_kpi_card(
            "Audited Dataset",
            f"{audit['clean_rows']:,}",
            f"{audit['raw_rows']:,} raw listings · {audit['duplicates_removed']} exact duplicates removed",
            badge="0 Missing Values",
            badge_tone="emerald",
            accent="navy",
        )
    with k2:
        render_kpi_card(
            "Market Coverage",
            f"{audit['unique_brands']} Brands",
            f"{audit['unique_models']:,} unique trims across {audit['unique_towns']} Sri Lankan towns",
            badge=f"YOM {audit['yom_min']}–{audit['yom_max']}",
            badge_tone="navy",
            accent="navy",
        )
    with k3:
        render_kpi_card(
            "Champion Test R²",
            f"{test_m['Stacking Regressor']['R2']:.4f}",
            f"Stacking Regressor · Chronological R²: {summary['split_robustness']['chronological_80_20']['metrics']['Stacking Regressor']['R2']:.4f}",
            badge="Outperforms RF & GB",
            badge_tone="emerald",
            accent="emerald",
        )
    with k4:
        render_kpi_card(
            "Error Reduction",
            f"-{mae_drop_pct:.1f}% MAE",
            f"Reduced from Rs. {lr_mae:.2f}L (Linear) to Rs. {st_mae:.2f}L (Stacking)",
            badge=f"Median Error: Rs. {test_m['Stacking Regressor']['MedAE']:.2f}L",
            badge_tone="emerald",
            accent="emerald",
        )

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # Visual End-to-End Architecture Diagram
    st.markdown(
        """
        <div style="font-size:0.74rem;font-weight:700;text-transform:uppercase;letter-spacing:0.09em;
                    color:#059669;margin-bottom:0.35rem;">Leak-Free ML Architecture</div>
        <div class="dr-pipeline">
            <div class="dr-pipe-step">
                <div class="dr-pipe-num">Stage 01</div>
                <div class="dr-pipe-title">Audit &amp; Clean</div>
                <div class="dr-pipe-desc">Drop index ID, deduplicate 18 exact reposts, isolate Date for chronological test only.</div>
            </div>
            <div class="dr-pipe-step">
                <div class="dr-pipe-num">Stage 02</div>
                <div class="dr-pipe-title">Feature Engineering</div>
                <div class="dr-pipe-desc">Compute Vehicle_Age, binary facility flags, Facility_Count, and composite Brand_Model.</div>
            </div>
            <div class="dr-pipe-step">
                <div class="dr-pipe-num">Stage 03</div>
                <div class="dr-pipe-title">Dual Encoding</div>
                <div class="dr-pipe-desc">StandardScaler + OHE (min_freq=3) + 5-fold CV TargetEncoder for 1,555 models.</div>
            </div>
            <div class="dr-pipe-step">
                <div class="dr-pipe-num">Stage 04</div>
                <div class="dr-pipe-title">Base Ensembles</div>
                <div class="dr-pipe-desc">Random Forest (200 trees, Bagging) + Gradient Boosting (350 trees, Boosting).</div>
            </div>
            <div class="dr-pipe-step final">
                <div class="dr-pipe-num">Stage 05</div>
                <div class="dr-pipe-title">RidgeCV Stacking</div>
                <div class="dr-pipe-desc">5-fold out-of-fold meta-learner optimally blends RF (0.53) + GB (0.48) into final LKR valuation.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Three Critical Data Audit Cards (Matches Presentation Script)
    render_section_header(
        "Pre-Modeling Data Audit",
        "Three Critical Forensic Findings",
        "Careful dataset inspection before modeling prevented multicollinearity, temporal leakage, and flawed outlier clipping.",
    )
    a1, a2, a3 = st.columns(3, gap="medium")
    with a1:
        render_info_card(
            "1. Price Scale & Luxury Tail Verified",
            f"Target prices are recorded in <strong>LKR Lakhs</strong> (100 Lakhs = LKR 10M). "
            f"Median is <strong>Rs. {audit['price_median_lakhs']:.1f}L</strong>, mean is <strong>Rs. {audit['price_mean_lakhs']:.2f}L</strong>, "
            f"99th percentile is <strong>Rs. {audit['price_p99_lakhs']:.1f}L</strong>, and extreme values up to "
            f"<strong>Rs. {audit['price_max_lakhs']:.1f}L</strong> represent genuine Mercedes S-Class, BMW 7-Series, and Land Rover luxury listings—retained intact rather than clipped.",
            tag="Domain Validation",
            tag_tone="navy",
        )
    with a2:
        render_info_card(
            "2. Synthetic Mileage Multicollinearity",
            f"Audit uncovered that <code>Millage(KM)</code> equals <code>(2025 − YOM) × 11,000</code> for "
            f"<strong>{audit['synthetic_mileage_match_pct']:.1f}%</strong> of rows (Pearson <strong>r = {audit['yom_mileage_correlation']:.4f}</strong>). "
            f"Dropping synthetic mileage and engineering <code>Vehicle_Age</code> eliminates exact collinearity and prevents artificial feature-importance dilution.",
            tag="Critical Discovery",
            tag_tone="amber",
        )
    with a3:
        render_info_card(
            "3. High-Cardinality & Leak-Free Splits",
            f"With <strong>{audit['unique_models']:,}</strong> distinct model strings across 50 brands, naive one-hot encoding explodes dimensionality. "
            f"Embedding scikit-learn's <strong>cross-validated TargetEncoder</strong> inside the pipeline prevents target leakage across both 5-fold CV and holdout evaluations.",
            tag="Pipeline Rigor",
            tag_tone="emerald",
        )

    st.markdown("<div style='height:0.85rem'></div>", unsafe_allow_html=True)

    # Feature Schema Summary
    with st.expander("View Full Engineered Feature Schema & Excluded Columns"):
        f_col1, f_col2 = st.columns(2, gap="large")
        feats = summary["features"]
        with f_col1:
            st.markdown("**Included Model Features (15 Inputs)**")
            st.markdown(f"- **Numerical & Binary (7):** `{', '.join(feats['numerical'])}`")
            st.markdown(f"- **One-Hot Encoded (6):** `{', '.join(feats['categorical_ohe'])}`")
            st.markdown(f"- **CV Target Encoded (2):** `{', '.join(feats['categorical_target_encoded'])}`")
        with f_col2:
            st.markdown("**Intentionally Excluded Columns (3)**")
            for col_name, reason in feats["excluded_features"].items():
                st.markdown(f"- **`{col_name}`:** {reason}")


# ===========================================================================
# TAB 3: MARKET EXPLORER (EDA)
# ===========================================================================
def render_eda_tab(df: pd.DataFrame) -> None:
    render_section_header(
        "Market Explorer",
        "Interactive Sri Lankan Automotive Market Intelligence",
        "Slice 9,770 real listings by brand, powertrain, transmission, manufacturing year, and budget tier.",
    )

    # Interactive Filter Bar
    with st.container(border=True):
        f1, f2, f3, f4, f5 = st.columns([1.4, 1.0, 0.9, 1.1, 1.1])
        top_brands = df["Brand"].value_counts().index.tolist()
        sel_brands = f1.multiselect("Brands (empty = All)", top_brands, default=[])
        sel_fuels = f2.multiselect("Fuel Type", sorted(df["Fuel Type"].unique()), default=[])
        sel_gears = f3.multiselect("Transmission", sorted(df["Gear"].unique()), default=[])
        price_cap = f4.slider(
            "Max Price (Lakhs)",
            min_value=20,
            max_value=int(np.ceil(df["Price"].max())),
            value=250,
            step=10,
        )
        yom_range = f5.slider(
            "YOM Range",
            min_value=int(df["YOM"].min()),
            max_value=int(df["YOM"].max()),
            value=(1990, int(df["YOM"].max())),
        )

    fdf = df.copy()
    if sel_brands:
        fdf = fdf[fdf["Brand"].isin(sel_brands)]
    if sel_fuels:
        fdf = fdf[fdf["Fuel Type"].isin(sel_fuels)]
    if sel_gears:
        fdf = fdf[fdf["Gear"].isin(sel_gears)]
    fdf = fdf[(fdf["Price"] <= price_cap) & (fdf["YOM"].between(yom_range[0], yom_range[1]))]

    if fdf.empty:
        st.warning("No listings match the selected filters. Broaden one of the filters above.")
        return

    # Dynamic KPI strip
    s1, s2, s3, s4 = st.columns(4, gap="medium")
    with s1:
        render_kpi_card("Matching Listings", f"{len(fdf):,}", f"{len(fdf)/len(df)*100:.1f}% of audited dataset", accent="navy")
    with s2:
        render_kpi_card("Median Asking Price", f"Rs. {fdf['Price'].median():,.1f}L", format_lkr_millions(fdf["Price"].median()), accent="emerald")
    with s3:
        render_kpi_card("Mean Asking Price", f"Rs. {fdf['Price'].mean():,.1f}L", f"IQR: {fdf['Price'].quantile(0.25):.1f}L – {fdf['Price'].quantile(0.75):.1f}L", accent="navy")
    with s4:
        render_kpi_card("Median Year / Engine", f"{int(fdf['YOM'].median())}", f"Median engine: {fdf['Engine (cc)'].median():,.0f} cc", accent="emerald")

    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

    # Row 1: Price Distribution + Top Brands
    r1_c1, r1_c2 = st.columns(2, gap="medium")
    with r1_c1:
        with st.container(border=True):
            fig_hist = px.histogram(
                fdf,
                x="Price",
                nbins=45,
                marginal="box",
                color_discrete_sequence=[PALETTE["navy"]],
                opacity=0.88,
            )
            fig_hist.add_vline(
                x=float(fdf["Price"].median()),
                line_dash="dash",
                line_color=PALETTE["emerald"],
                line_width=2,
                annotation_text=f"Median {fdf['Price'].median():.1f}L",
                annotation_font=dict(color=PALETTE["emerald"], size=11),
            )
            style_plotly(
                fig_hist,
                title=f"Asking Price Distribution (≤ Rs. {price_cap}L)",
                height=370,
                x_title="Price (LKR Lakhs)",
                y_title="Listing Count",
                show_legend=False,
            )
            st.plotly_chart(fig_hist, width="stretch", config=PLOTLY_CONFIG)

    with r1_c2:
        with st.container(border=True):
            brand_stats = (
                fdf.groupby("Brand")
                .agg(Count=("Price", "size"), Median_Price=("Price", "median"))
                .sort_values("Count", ascending=False)
                .head(12)
                .sort_values("Count", ascending=True)
                .reset_index()
            )
            fig_brands = px.bar(
                brand_stats,
                x="Count",
                y="Brand",
                orientation="h",
                color="Median_Price",
                color_continuous_scale=[[0, "#BFDBFE"], [0.5, "#1E3A5F"], [1, "#059669"]],
                hover_data={"Median_Price": ":.1f", "Count": ":,"},
                text="Count",
            )
            fig_brands.update_traces(textposition="outside", textfont=dict(size=11))
            style_plotly(
                fig_brands,
                title="Top Brands by Listing Volume (Shaded by Median Price)",
                height=370,
                x_title="Number of Listings",
                y_title="",
                show_legend=False,
            )
            fig_brands.update_coloraxes(colorbar_title="Med. L")
            st.plotly_chart(fig_brands, width="stretch", config=PLOTLY_CONFIG)

    # Row 2: Depreciation Curve by Fuel + Powertrain Breakdown
    r2_c1, r2_c2 = st.columns(2, gap="medium")
    with r2_c1:
        with st.container(border=True):
            dep_df = (
                fdf.groupby(["YOM", "Fuel Type"])
                .agg(Median_Price=("Price", "median"), Count=("Price", "size"))
                .reset_index()
            )
            dep_df = dep_df[dep_df["Count"] >= 3]
            fuel_colors = {
                "Petrol": PALETTE["navy"],
                "Hybrid": PALETTE["emerald"],
                "Diesel": PALETTE["amber"],
                "Electric": PALETTE["blue"],
            }
            fig_yom = px.line(
                dep_df,
                x="YOM",
                y="Median_Price",
                color="Fuel Type",
                markers=True,
                color_discrete_map=fuel_colors,
                hover_data={"Count": True, "Median_Price": ":.1f"},
            )
            style_plotly(
                fig_yom,
                title="Median Price Trajectory by Year & Fuel Type (n ≥ 3)",
                height=360,
                x_title="Year of Manufacture (YOM)",
                y_title="Median Price (LKR Lakhs)",
            )
            st.plotly_chart(fig_yom, width="stretch", config=PLOTLY_CONFIG)

    with r2_c2:
        with st.container(border=True):
            fig_box = px.box(
                fdf,
                x="Fuel Type",
                y="Price",
                color="Gear",
                color_discrete_map={"Automatic": PALETTE["navy"], "Manual": PALETTE["emerald"]},
                points=False,
            )
            style_plotly(
                fig_box,
                title="Valuation Spread by Fuel Type & Transmission",
                height=360,
                x_title="Fuel Type",
                y_title="Price (LKR Lakhs)",
            )
            st.plotly_chart(fig_box, width="stretch", config=PLOTLY_CONFIG)

    # Row 3: Regional Market Towns + Engine vs Price
    r3_c1, r3_c2 = st.columns(2, gap="medium")
    with r3_c1:
        with st.container(border=True):
            town_stats = (
                fdf.groupby("Town")
                .agg(Count=("Price", "size"), Median_Price=("Price", "median"))
                .query("Count >= 15")
                .sort_values("Median_Price", ascending=False)
                .head(12)
                .sort_values("Median_Price", ascending=True)
                .reset_index()
            )
            if town_stats.empty:
                town_stats = (
                    fdf.groupby("Town")
                    .agg(Count=("Price", "size"), Median_Price=("Price", "median"))
                    .sort_values("Count", ascending=False)
                    .head(10)
                    .sort_values("Median_Price", ascending=True)
                    .reset_index()
                )
            fig_towns = px.bar(
                town_stats,
                x="Median_Price",
                y="Town",
                orientation="h",
                color_discrete_sequence=[PALETTE["emerald"]],
                hover_data={"Count": ":,", "Median_Price": ":.1f"},
                text=town_stats["Median_Price"].map(lambda v: f"Rs. {v:.1f}L"),
            )
            fig_towns.update_traces(textposition="outside", textfont=dict(size=10))
            style_plotly(
                fig_towns,
                title="Highest-Valued Sri Lankan Towns (Active Markets)",
                height=360,
                x_title="Median Price (LKR Lakhs)",
                y_title="",
                show_legend=False,
            )
            st.plotly_chart(fig_towns, width="stretch", config=PLOTLY_CONFIG)

    with r3_c2:
        with st.container(border=True):
            sample_df = fdf.sample(min(len(fdf), 1500), random_state=42)
            fig_eng = px.scatter(
                sample_df,
                x="Engine (cc)",
                y="Price",
                color="Vehicle_Age",
                color_continuous_scale=[[0, "#059669"], [0.5, "#2563EB"], [1, "#94A3B8"]],
                opacity=0.65,
                hover_data=["Brand", "Model", "YOM", "Fuel Type", "Gear"],
            )
            style_plotly(
                fig_eng,
                title="Engine Capacity vs Price (Colored by Vehicle Age)",
                height=360,
                x_title="Engine Capacity (cc)",
                y_title="Price (LKR Lakhs)",
                show_legend=False,
            )
            fig_eng.update_coloraxes(colorbar_title="Age (yr)")
            st.plotly_chart(fig_eng, width="stretch", config=PLOTLY_CONFIG)


# ===========================================================================
# TAB 4: ENSEMBLE & STACKING LAB
# ===========================================================================
def render_ensemble_tab(summary: dict | None) -> None:
    render_section_header(
        "Ensemble & Stacking Lab",
        "Bagging vs. Boosting vs. Hybrid Stacking Performance",
        "All four models share identical preprocessing and 80/20 splits. Compare holdout accuracy, "
        "5-fold cross-validation stability, RidgeCV meta-learner weights, and split robustness.",
    )

    if not summary:
        st.info("Evaluation summary artifact is missing.")
        return

    test_m = summary["test_metrics"]
    train_m = summary["train_metrics"]
    cv_m = summary["cv_5fold"]
    times = summary["train_times_sec"]

    # Leaderboard Cards
    cols = st.columns(4, gap="medium")
    for col, name in zip(cols, MODEL_ORDER):
        m = test_m[name]
        is_best = name == "Stacking Regressor"
        with col:
            render_kpi_card(
                name,
                f"R² {m['R2']:.4f}",
                f"MAE: Rs. {m['MAE']:.2f}L · RMSE: Rs. {m['RMSE']:.2f}L · MedAE: Rs. {m['MedAE']:.2f}L",
                badge="Champion Hybrid" if is_best else MODEL_ROLES[name],
                badge_tone="emerald" if is_best else "navy",
                accent="emerald" if is_best else "navy",
            )

    st.markdown("<div style='height:0.85rem'></div>", unsafe_allow_html=True)

    # Row 1: Holdout Error Metrics + Train vs Test Generalization
    c1, c2 = st.columns([0.56, 0.44], gap="medium")
    with c1:
        with st.container(border=True):
            fig_err = go.Figure()
            for metric_key, color in [("MAE", PALETTE["navy"]), ("RMSE", PALETTE["emerald"]), ("MedAE", PALETTE["blue"])]:
                fig_err.add_trace(
                    go.Bar(
                        name=f"{metric_key} (Lakhs)",
                        x=MODEL_ORDER,
                        y=[test_m[m][metric_key] for m in MODEL_ORDER],
                        marker_color=color,
                        text=[f"{test_m[m][metric_key]:.2f}" for m in MODEL_ORDER],
                        textposition="outside",
                        textfont=dict(family="JetBrains Mono, monospace", size=10),
                    )
                )
            style_plotly(
                fig_err,
                title="Test Set Error Metrics — Lower is Better (LKR Lakhs)",
                height=360,
                y_title="Error (LKR Lakhs)",
            )
            fig_err.update_layout(barmode="group")
            st.plotly_chart(fig_err, width="stretch", config=PLOTLY_CONFIG)

    with c2:
        with st.container(border=True):
            # Stacking Meta-Learner Weights Visual
            meta_info = summary.get("stacking_meta_learner", {})
            weights = meta_info.get("base_weights", {})
            w_names = list(weights.keys())
            w_vals = list(weights.values())
            total_w = sum(w_vals) if sum(w_vals) > 0 else 1.0
            w_pcts = [v / total_w * 100 for v in w_vals]

            fig_meta = go.Figure(
                go.Bar(
                    x=w_names,
                    y=w_vals,
                    marker_color=[PALETTE["blue"], PALETTE["navy"]],
                    width=0.45,
                    text=[f"w = {v:.4f}<br>({p:.1f}% share)" for v, p in zip(w_vals, w_pcts)],
                    textposition="outside",
                    textfont=dict(family="JetBrains Mono, monospace", size=11, color=PALETTE["navy"]),
                )
            )
            style_plotly(
                fig_meta,
                title=f"Stacking Meta-Learner ({meta_info.get('meta_model', 'RidgeCV')}) Learned Weights",
                height=265,
                y_title="Ridge Coefficient Weight",
                show_legend=False,
            )
            fig_meta.update_yaxes(range=[0, max(w_vals) * 1.38])
            st.plotly_chart(fig_meta, width="stretch", config=PLOTLY_CONFIG)
            render_callout(
                f"<strong>Why Stacking Wins:</strong> RidgeCV (<code>alpha={meta_info.get('selected_alpha', 1000.0):.0f}</code>, "
                f"intercept <code>{meta_info.get('intercept', 0):.2f}L</code>) learns an almost balanced blend of "
                f"<strong>Random Forest (Bagging · {w_pcts[0]:.1f}%)</strong> and <strong>Gradient Boosting (Boosting · {w_pcts[1]:.1f}%)</strong>, "
                f"beating both base learners simultaneously on MAE, RMSE, and R².",
                tone="emerald",
            )

    # Row 2: 5-Fold Cross-Validation Stability + Train vs Test Table
    r2_1, r2_2 = st.columns([0.52, 0.48], gap="medium")
    with r2_1:
        with st.container(border=True):
            cv_metric = st.radio(
                "5-Fold CV Metric",
                ["R2_folds", "MAE_folds", "RMSE_folds"],
                format_func=lambda k: {"R2_folds": "R² Score", "MAE_folds": "MAE (Lakhs)", "RMSE_folds": "RMSE (Lakhs)"}[k],
                horizontal=True,
            )
            fig_cv = go.Figure()
            for name in MODEL_ORDER:
                vals = cv_m[name][cv_metric]
                fig_cv.add_trace(
                    go.Box(
                        y=vals,
                        name=name.replace(" Regressor", ""),
                        boxpoints="all",
                        jitter=0.25,
                        pointpos=0,
                        marker=dict(color=MODEL_COLORS[name], size=7),
                        line=dict(color=MODEL_COLORS[name]),
                    )
                )
            style_plotly(
                fig_cv,
                title="5-Fold Cross-Validation Distribution Across Folds",
                height=320,
                y_title=cv_metric.replace("_folds", ""),
                show_legend=False,
            )
            st.plotly_chart(fig_cv, width="stretch", config=PLOTLY_CONFIG)

    with r2_2:
        with st.container(border=True):
            st.markdown(
                "<div style='font-size:0.93rem;font-weight:700;color:#0F294A;margin-bottom:0.55rem;'>"
                "Complete Holdout, Generalization &amp; Runtime Scorecard</div>",
                unsafe_allow_html=True,
            )
            rows = []
            for name in MODEL_ORDER:
                tm = test_m[name]
                tr = train_m[name]
                cv = cv_m[name]
                rows.append(
                    {
                        "Model": name,
                        "Train R²": round(tr["R2"], 4),
                        "Test R²": round(tm["R2"], 4),
                        "5-Fold R²": f"{cv['R2_mean']:.4f} ± {cv['R2_std']:.3f}",
                        "Test MAE": round(tm["MAE"], 2),
                        "MAPE %": f"{tm['MAPE_pct']:.1f}%",
                        "Fit Time": f"{times[name]:.1f}s",
                    }
                )
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
            render_callout(
                "<strong>Bias–Variance Takeaway:</strong> Linear Regression underfits non-linear depreciation (Test R² 0.689). "
                "Random Forest cuts variance aggressively (MAE 5.66L), Gradient Boosting sharpens residual fit (RMSE 19.25L), "
                "and Stacking harmonizes both into the lowest Test MAE (<strong>5.57L</strong>), RMSE (<strong>18.83L</strong>), and highest R² (<strong>0.8507</strong>).",
                tone="navy",
            )

    # Row 3: Split Robustness Stress-Test
    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
    render_section_header(
        "Methodological Rigor",
        "Split Robustness Verification (Random vs. Chronological vs. Unseen Models)",
        "Validating that model accuracy holds across time with zero temporal leakage, while honestly measuring error on unseen Brand+Model trims.",
    )
    rob = summary.get("split_robustness", {})
    if rob:
        split_labels = {
            "random_80_20": "Random 80/20",
            "chronological_80_20": "Chronological (≥ Feb 2)",
            "grouped_brand_model": "Unseen Brand+Model",
        }
        rob_rows = []
        for s_key, s_label in split_labels.items():
            if s_key in rob:
                for m_name in MODEL_ORDER:
                    m_res = rob[s_key]["metrics"][m_name]
                    rob_rows.append(
                        {
                            "Split Strategy": s_label,
                            "Model": m_name,
                            "R2": m_res["R2"],
                            "MAE": m_res["MAE"],
                            "RMSE": m_res["RMSE"],
                        }
                    )
        rob_df = pd.DataFrame(rob_rows)
        rb1, rb2 = st.columns(2, gap="medium")
        with rb1:
            with st.container(border=True):
                fig_rob_r2 = px.bar(
                    rob_df,
                    x="Split Strategy",
                    y="R2",
                    color="Model",
                    barmode="group",
                    color_discrete_map=MODEL_COLORS,
                    text=rob_df["R2"].map(lambda v: f"{v:.3f}"),
                )
                fig_rob_r2.update_traces(textposition="outside", textfont=dict(size=10))
                style_plotly(fig_rob_r2, title="R² Across Evaluation Split Strategies", height=340, y_title="R² Score")
                fig_rob_r2.update_yaxes(range=[0.5, 1.02])
                st.plotly_chart(fig_rob_r2, width="stretch", config=PLOTLY_CONFIG)
        with rb2:
            with st.container(border=True):
                fig_rob_mae = px.bar(
                    rob_df,
                    x="Split Strategy",
                    y="MAE",
                    color="Model",
                    barmode="group",
                    color_discrete_map=MODEL_COLORS,
                    text=rob_df["MAE"].map(lambda v: f"{v:.1f}L"),
                )
                fig_rob_mae.update_traces(textposition="outside", textfont=dict(size=10))
                style_plotly(fig_rob_mae, title="MAE (Lakhs) Across Split Strategies", height=340, y_title="MAE (Lakhs)")
                st.plotly_chart(fig_rob_mae, width="stretch", config=PLOTLY_CONFIG)


# ===========================================================================
# TAB 5: EXPLAINABILITY & DIAGNOSTICS
# ===========================================================================
def render_explainability_tab(fi_df: pd.DataFrame | None, preds_df: pd.DataFrame | None) -> None:
    render_section_header(
        "Explainability & Diagnostics",
        "What Drives Sri Lankan Car Prices & Where Models Excel",
        "Inspect permutation feature importance, parity scatter plots on all 1,954 held-out test vehicles, "
        "residual error distributions, and accuracy broken down by price tier.",
    )

    sel_col1, sel_col2 = st.columns([0.45, 0.55])
    selected_model = sel_col1.selectbox(
        "Inspect Diagnostics For Model",
        MODEL_ORDER,
        index=0,
        key="exp_model_select",
    )
    cap_parity = sel_col2.checkbox(
        "Focus Parity Plot on Core Market (≤ Rs. 250 Lakhs)",
        value=True,
        help="Uncheck to view extreme luxury listings up to Rs. 790 Lakhs.",
    )

    # Row 1: Permutation Feature Importance + Parity Plot
    c1, c2 = st.columns([0.48, 0.52], gap="medium")
    with c1:
        with st.container(border=True):
            if fi_df is not None and not fi_df.empty:
                fi_model = selected_model if selected_model in fi_df["Model"].unique() else "Stacking Regressor"
                sub_fi = (
                    fi_df[fi_df["Model"] == fi_model]
                    .sort_values("RMSE_Increase_Mean", ascending=True)
                    .tail(10)
                )
                fig_fi = go.Figure(
                    go.Bar(
                        y=sub_fi["Feature"],
                        x=sub_fi["RMSE_Increase_Mean"],
                        orientation="h",
                        error_x=dict(type="data", array=sub_fi["RMSE_Increase_Std"], color="#64748B", thickness=1.5),
                        marker_color=[
                            PALETTE["emerald"] if i >= len(sub_fi) - 3 else PALETTE["navy"]
                            for i in range(len(sub_fi))
                        ],
                        text=sub_fi["RMSE_Increase_Mean"].map(lambda v: f"+{v:.2f}L"),
                        textposition="outside",
                        textfont=dict(family="JetBrains Mono, monospace", size=10),
                    )
                )
                style_plotly(
                    fig_fi,
                    title=f"Top 10 Drivers — Permutation RMSE Increase ({fi_model})",
                    height=410,
                    x_title="RMSE Increase When Shuffled (LKR Lakhs)",
                    show_legend=False,
                )
                fig_fi.update_xaxes(range=[0, sub_fi["RMSE_Increase_Mean"].max() * 1.25])
                st.plotly_chart(fig_fi, width="stretch", config=PLOTLY_CONFIG)
                if selected_model == "Linear Regression":
                    st.caption("Permutation importance was recorded for the 3 tree/stacking ensembles; displaying Stacking Regressor.")
            else:
                st.info("Feature importance file (`artifacts/feature_importance.csv`) not found.")

    with c2:
        with st.container(border=True):
            if preds_df is not None and not preds_df.empty:
                pred_col = f"Pred_{selected_model}"
                pdf = preds_df.copy()
                pdf["Predicted"] = pdf[pred_col].clip(lower=0.0)
                pdf["Residual"] = pdf["Price"] - pdf["Predicted"]
                pdf["AbsError"] = pdf["Residual"].abs()

                plot_df = pdf[pdf["Price"] <= 250] if cap_parity else pdf
                max_val = float(max(plot_df["Price"].max(), plot_df["Predicted"].max())) * 1.04

                fig_par = px.scatter(
                    plot_df,
                    x="Price",
                    y="Predicted",
                    color="AbsError",
                    color_continuous_scale=[[0, "#059669"], [0.35, "#2563EB"], [1, "#DC2626"]],
                    opacity=0.68,
                    hover_data={
                        "Brand": True,
                        "Model": True,
                        "YOM": True,
                        "Price": ":.2f",
                        "Predicted": ":.2f",
                        "Residual": ":+.2f",
                        "AbsError": False,
                    },
                )
                fig_par.add_trace(
                    go.Scatter(
                        x=[0, max_val],
                        y=[0, max_val],
                        mode="lines",
                        name="Perfect Prediction (y = x)",
                        line=dict(color=PALETTE["navy"], width=2, dash="dash"),
                    )
                )
                style_plotly(
                    fig_par,
                    title=f"Actual vs. Predicted Test Listings ({selected_model} · n={len(plot_df):,})",
                    height=410,
                    x_title="Actual Asking Price (LKR Lakhs)",
                    y_title="Predicted Price (LKR Lakhs)",
                    show_legend=False,
                )
                fig_par.update_coloraxes(colorbar_title="|Err| L")
                st.plotly_chart(fig_par, width="stretch", config=PLOTLY_CONFIG)
            else:
                st.info("Test predictions file (`artifacts/test_predictions.csv`) not found.")

    # Row 2: Residual Distribution + Accuracy by Price Tier
    if preds_df is not None and not preds_df.empty:
        pred_col = f"Pred_{selected_model}"
        pdf = preds_df.copy()
        pdf["Predicted"] = pdf[pred_col].clip(lower=0.0)
        pdf["Residual"] = pdf["Price"] - pdf["Predicted"]
        pdf["AbsError"] = pdf["Residual"].abs()
        pdf["APE_pct"] = (pdf["AbsError"] / pdf["Price"].clip(lower=1.0)) * 100.0

        r2_c1, r2_c2 = st.columns([0.48, 0.52], gap="medium")
        with r2_c1:
            with st.container(border=True):
                clipped_res = pdf["Residual"].clip(-50, 50)
                fig_res = px.histogram(
                    x=clipped_res,
                    nbins=50,
                    color_discrete_sequence=[PALETTE["navy"]],
                    opacity=0.88,
                )
                fig_res.add_vline(x=0, line_color=PALETTE["emerald"], line_width=2.5)
                style_plotly(
                    fig_res,
                    title=f"Residual Error Distribution ({selected_model}, Clipped ±50L)",
                    height=330,
                    x_title="Actual − Predicted (LKR Lakhs)",
                    y_title="Test Vehicles",
                    show_legend=False,
                )
                st.plotly_chart(fig_res, width="stretch", config=PLOTLY_CONFIG)

        with r2_c2:
            with st.container(border=True):
                bins = [0, 30, 75, 150, 1000]
                labels = ["Budget (< 30L)", "Mid-Range (30–75L)", "Upper Mid (75–150L)", "Luxury (150L+)"]
                pdf["Price_Tier"] = pd.cut(pdf["Price"], bins=bins, labels=labels, right=False)
                tier_df = (
                    pdf.groupby("Price_Tier", observed=False)
                    .agg(
                        Count=("Price", "size"),
                        MAE=("AbsError", "mean"),
                        MedAE=("AbsError", "median"),
                        MAPE_pct=("APE_pct", "mean"),
                    )
                    .reset_index()
                )
                fig_tier = px.bar(
                    tier_df,
                    x="Price_Tier",
                    y="MedAE",
                    color_discrete_sequence=[PALETTE["emerald"]],
                    text=tier_df.apply(lambda r: f"MedAE {r['MedAE']:.1f}L<br>({r['MAPE_pct']:.1f}%)", axis=1),
                )
                fig_tier.update_traces(textposition="outside", textfont=dict(size=10))
                style_plotly(
                    fig_tier,
                    title=f"Median Error & MAPE by Price Segment ({selected_model})",
                    height=330,
                    x_title="Sri Lankan Market Tier",
                    y_title="Median Absolute Error (Lakhs)",
                    show_legend=False,
                )
                fig_tier.update_yaxes(range=[0, max(5.0, float(tier_df["MedAE"].max()) * 1.35)])
                st.plotly_chart(fig_tier, width="stretch", config=PLOTLY_CONFIG)

        # Row 3: Inspect Best Predictions vs Hardest Outliers
        with st.expander(f"Inspect Closest Predictions & Hardest Luxury Edge Cases ({selected_model})"):
            e_c1, e_c2 = st.columns(2, gap="medium")
            cols_show = ["Brand", "Model", "YOM", "Engine (cc)", "Price", "Predicted", "Residual"]
            with e_c1:
                st.markdown("**Most Accurate Test Predictions**")
                best_tbl = pdf.nsmallest(8, "AbsError")[cols_show].copy()
                best_tbl[["Price", "Predicted", "Residual"]] = best_tbl[["Price", "Predicted", "Residual"]].round(2)
                st.dataframe(best_tbl, width="stretch", hide_index=True)
            with e_c2:
                st.markdown("**Largest Residual Edge Cases (Rare / Bespoke Trims)**")
                worst_tbl = pdf.nlargest(8, "AbsError")[cols_show].copy()
                worst_tbl[["Price", "Predicted", "Residual"]] = worst_tbl[["Price", "Predicted", "Residual"]].round(2)
                st.dataframe(worst_tbl, width="stretch", hide_index=True)
