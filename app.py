# """
# app.py
# ------
# Intelligent Automatic Grade Change System — Operator Advisory Dashboard
# Built with Streamlit + XGBoost + SHAP for a papermaking hackathon demo.

# Run with:  streamlit run app.py
# """

# # ─────────────────────────── Standard imports ───────────────────────────────
# import os
# import datetime
# import warnings
# warnings.filterwarnings("ignore")

# import numpy as np
# import pandas as pd
# import matplotlib
# matplotlib.use("Agg")                     # non-interactive backend for Streamlit
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# import seaborn as sns
# import xgboost as xgb
# import shap
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_absolute_error, r2_score

# import streamlit as st

# # Internal module
# from database import insert_feedback, fetch_all_feedback, get_feedback_summary

# # ─────────────────────────── Page configuration ────────────────────────────
# st.set_page_config(
#     page_title="PaperMill IQ — Grade Change Advisor",
#     page_icon="📄",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )

# # ─────────────────────────── Custom CSS ─────────────────────────────────────
# st.markdown("""
# <style>
# /* Dark industrial theme */
# [data-testid="stAppViewContainer"] {
#     background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
# }
# [data-testid="stSidebar"] {
#     background: #0d1117;
#     border-right: 1px solid #30363d;
# }
# h1, h2, h3, h4 { color: #e6edf3; }
# p, li { color: #8b949e; }

# /* KPI cards */
# .kpi-card {
#     background: linear-gradient(135deg, #1c2128, #21262d);
#     border: 1px solid #30363d;
#     border-radius: 12px;
#     padding: 18px 22px;
#     text-align: center;
#     margin-bottom: 8px;
# }
# .kpi-label  { font-size: 0.75rem; color: #8b949e; text-transform: uppercase; letter-spacing: 1px; }
# .kpi-value  { font-size: 2rem;   font-weight: 700; color: #58a6ff; margin: 4px 0; }
# .kpi-unit   { font-size: 0.7rem;  color: #6e7681; }

# /* Alert banners */
# .alert-critical {
#     background: linear-gradient(90deg, #3d1a1a, #2d1010);
#     border-left: 4px solid #f85149;
#     border-radius: 8px;
#     padding: 16px 20px;
#     margin: 12px 0;
# }
# .alert-ok {
#     background: linear-gradient(90deg, #0d2218, #0a1f15);
#     border-left: 4px solid #3fb950;
#     border-radius: 8px;
#     padding: 16px 20px;
#     margin: 12px 0;
# }
# .alert-title { font-size: 1.05rem; font-weight: 600; margin-bottom: 4px; }
# .alert-body  { font-size: 0.85rem; color: #c9d1d9; }

# /* Recommendation chip */
# .rec-chip {
#     display: inline-block;
#     background: #1f6feb22;
#     border: 1px solid #1f6feb;
#     color: #58a6ff;
#     border-radius: 20px;
#     padding: 4px 14px;
#     font-size: 0.8rem;
#     margin: 4px 4px;
# }

# /* Section divider */
# .section-header {
#     font-size: 0.7rem;
#     text-transform: uppercase;
#     letter-spacing: 2px;
#     color: #8b949e;
#     border-bottom: 1px solid #21262d;
#     padding-bottom: 4px;
#     margin-bottom: 16px;
# }
# </style>
# """, unsafe_allow_html=True)

# # ─────────────────────────── Constants ─────────────────────────────────────
# DATA_PATH        = "paper_mill_data.csv"
# FEATURE_COLS     = ["stock_flow", "steam_pressure", "machine_speed", "moisture", "caliper"]
# TARGET_COL       = "basis_weight"
# OFFSPEC_THRESH   = 0.025   # 2.5 % deviation from target
# ANOMALY_START    = 400
# ANOMALY_END      = 500

# # ═══════════════════════════════════════════════════════════════════════════
# # SECTION 1 — DATA LOADING & MODEL TRAINING (cached)
# # ═══════════════════════════════════════════════════════════════════════════

# @st.cache_data(show_spinner="Loading dataset …")
# def load_data() -> pd.DataFrame:
#     """Load the synthesised CSV; generate it on the fly if missing."""
#     if not os.path.exists(DATA_PATH):
#         st.warning("Dataset not found — generating it now …")
#         from data_generator import generate_dataset
#         df = generate_dataset()
#         df.to_csv(DATA_PATH, index=False)
#     df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
#     return df


# @st.cache_resource(show_spinner="Training XGBoost model …")
# def train_model(df: pd.DataFrame):
#     """
#     Trains an XGBoost Regressor on the full dataset and returns
#     the model, SHAP explainer, and train/test metrics.
#     """
#     X = df[FEATURE_COLS].values
#     y = df[TARGET_COL].values

#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42, shuffle=False
#     )

#     model = xgb.XGBRegressor(
#         n_estimators=300,
#         max_depth=5,
#         learning_rate=0.08,
#         subsample=0.8,
#         colsample_bytree=0.8,
#         random_state=42,
#         verbosity=0,
#     )
#     model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

#     y_pred = model.predict(X_test)
#     metrics = {
#         "mae":  round(mean_absolute_error(y_test, y_pred), 4),
#         "r2":   round(r2_score(y_test, y_pred), 4),
#     }

#     # SHAP Explainer (TreeExplainer is fastest for XGBoost)
#     explainer   = shap.TreeExplainer(model)
#     shap_values = explainer(df[FEATURE_COLS])   # SHAP for every row

#     return model, explainer, shap_values, metrics


# # ─────────────────────────── Load everything ────────────────────────────────
# df              = load_data()
# model, explainer, shap_values, train_metrics = train_model(df)

# # Pre-compute full predictions once
# df["predicted_bw"] = model.predict(df[FEATURE_COLS].values)

# # ═══════════════════════════════════════════════════════════════════════════
# # SIDEBAR — Controls & Model info
# # ═══════════════════════════════════════════════════════════════════════════
# with st.sidebar:
#     st.image(
#         "https://img.icons8.com/color/96/paper-roll.png",
#         width=60,
#     )
#     st.title("PaperMill IQ")
#     st.caption("Intelligent Grade Change Advisor")
#     st.divider()

#     st.markdown("### ⚙️ Simulation Controls")
#     selected_row = st.slider(
#         "Current time-step (row index)",
#         min_value=0,
#         max_value=len(df) - 1,
#         value=450,
#         step=1,
#         help="Drag to simulate different points in time along the process run.",
#     )
#     st.divider()

#     st.markdown("### 📊 Model Performance")
#     st.metric("MAE (g/m²)", train_metrics["mae"])
#     st.metric("R² Score",   train_metrics["r2"])
#     st.divider()

#     with st.expander("📁 Raw Feedback Log"):
#         feedback_log = fetch_all_feedback()
#         if feedback_log:
#             st.dataframe(pd.DataFrame(feedback_log), use_container_width=True, height=250)
#         else:
#             st.info("No feedback logged yet.")

#     summary = get_feedback_summary()
#     c1, c2 = st.columns(2)
#     c1.metric("✅ Accepted", summary["accepted"])
#     c2.metric("❌ Rejected", summary["rejected"])

# # ═══════════════════════════════════════════════════════════════════════════
# # MAIN HEADER
# # ═══════════════════════════════════════════════════════════════════════════
# st.markdown(
#     "<h1 style='margin-bottom:0'>📄 Intelligent Grade Change System</h1>",
#     unsafe_allow_html=True,
# )
# st.markdown(
#     "<p style='color:#8b949e; margin-top:4px;'>Real-time Operator Advisory Dashboard — "
#     "Paper Machine Grade Change: <b style='color:#58a6ff'>86 → 72 g/m²</b></p>",
#     unsafe_allow_html=True,
# )
# st.divider()

# # ─── Current row snapshot ───────────────────────────────────────────────────
# current   = df.iloc[selected_row]
# bw_actual = float(current["basis_weight"])
# bw_pred   = float(current["predicted_bw"])
# bw_target = float(current["bw_target"])
# phase     = str(current["phase"])

# deviation_pct = (bw_actual - bw_target) / bw_target * 100
# is_offspec    = abs(deviation_pct) > (OFFSPEC_THRESH * 100)

# # ═══════════════════════════════════════════════════════════════════════════
# # SECTION 2 — LIVE MONITORING CHART
# # ═══════════════════════════════════════════════════════════════════════════
# st.markdown("<div class='section-header'>SECTION 2 — LIVE PROCESS MONITORING</div>", unsafe_allow_html=True)

# # KPI row
# k1, k2, k3, k4, k5 = st.columns(5)
# def kpi(col, label, value, unit=""):
#     col.markdown(
#         f"<div class='kpi-card'>"
#         f"<div class='kpi-label'>{label}</div>"
#         f"<div class='kpi-value'>{value}</div>"
#         f"<div class='kpi-unit'>{unit}</div>"
#         f"</div>",
#         unsafe_allow_html=True,
#     )

# kpi(k1, "Basis Weight (Actual)",  f"{bw_actual:.2f}",  "g/m²")
# kpi(k2, "Basis Weight (Pred)",    f"{bw_pred:.2f}",    "g/m²")
# kpi(k3, "Target Setpoint",        f"{bw_target:.2f}",  "g/m²")
# kpi(k4, "Deviation",              f"{deviation_pct:+.2f}", "%")
# kpi(k5, "Phase",                  phase,               "")

# # Chart window: show up to the selected row (max 600 rows for readability)
# window_start = max(0, selected_row - 600)
# chart_df = df.iloc[window_start : selected_row + 1].copy()

# fig, ax = plt.subplots(figsize=(13, 4), facecolor="#0d1117")
# ax.set_facecolor("#161b22")

# # Basis weight actual trace
# ax.plot(
#     chart_df.index, chart_df["basis_weight"],
#     color="#58a6ff", linewidth=1.4, label="Actual BW", zorder=3,
# )
# # Predicted trace
# ax.plot(
#     chart_df.index, chart_df["predicted_bw"],
#     color="#f0883e", linewidth=1.2, linestyle="--", alpha=0.85, label="XGBoost Pred", zorder=3,
# )
# # Target setpoint
# ax.plot(
#     chart_df.index, chart_df["bw_target"],
#     color="#3fb950", linewidth=1.6, linestyle="-.", label="Target Setpoint", zorder=2,
# )
# # ±2.5 % safe operating envelope
# upper_bound = chart_df["bw_target"] * (1 + OFFSPEC_THRESH)
# lower_bound = chart_df["bw_target"] * (1 - OFFSPEC_THRESH)
# ax.fill_between(
#     chart_df.index, lower_bound, upper_bound,
#     alpha=0.12, color="#3fb950", label="±2.5% Safe Zone",
# )

# # Anomaly zone shading
# anomaly_x = chart_df.index[
#     (chart_df.index >= ANOMALY_START) & (chart_df.index < ANOMALY_END)
# ]
# if len(anomaly_x) > 0:
#     ax.axvspan(anomaly_x[0], anomaly_x[-1], alpha=0.15, color="#f85149", label="Anomaly Zone")

# # Current position marker
# ax.axvline(selected_row, color="#d2a8ff", linewidth=1.5, linestyle=":", alpha=0.9, label="Now")

# ax.set_xlabel("Row (Time Index)", color="#8b949e", fontsize=9)
# ax.set_ylabel("Basis Weight (g/m²)", color="#8b949e", fontsize=9)
# ax.tick_params(colors="#8b949e")
# for spine in ax.spines.values():
#     spine.set_edgecolor("#30363d")
# ax.legend(fontsize=8, facecolor="#21262d", edgecolor="#30363d", labelcolor="#c9d1d9",
#           loc="upper right", ncol=3)
# ax.set_title("Basis Weight — Live Monitoring", color="#e6edf3", fontsize=11, pad=10)
# plt.tight_layout()
# st.pyplot(fig)
# plt.close(fig)

# # ═══════════════════════════════════════════════════════════════════════════
# # SECTION 3 — PREDICTIVE ALERTING & OPTIMISATION
# # ═══════════════════════════════════════════════════════════════════════════
# st.markdown("<div class='section-header'>SECTION 3 — PREDICTIVE ALERTING & OPTIMISATION</div>", unsafe_allow_html=True)

# # ── Compute recommended setpoints ──────────────────────────────────────────
# # Simple physics-based correction:
# #   To bring BW back on-spec reduce steam pressure (lowers moisture) and
# #   adjust stock_flow proportionally to the target BW.
# current_steam = float(current["steam_pressure"])
# current_stock = float(current["stock_flow"])

# recommended_steam = round(current_steam * 0.98, 2)       # −2 %
# recommended_stock = round(current_stock * 1.015, 2)      # +1.5 %

# # ── Alert banner ────────────────────────────────────────────────────────────
# if is_offspec:
#     st.markdown(
#         f"""<div class='alert-critical'>
#         <div class='alert-title' style='color:#f85149'>
#             🚨 HIGH PRIORITY — BASIS WEIGHT OFF-SPEC
#         </div>
#         <div class='alert-body'>
#             Predicted basis weight <b>{bw_pred:.2f} g/m²</b> deviates 
#             <b>{deviation_pct:+.2f}%</b> from target <b>{bw_target:.2f} g/m²</b>.
#             Threshold: ±2.5%. Immediate corrective action recommended.
#         </div></div>""",
#         unsafe_allow_html=True,
#     )

#     st.markdown("#### 🔧 Recommended Corrective Setpoints")
#     rc1, rc2, rc3 = st.columns(3)
#     rc1.metric(
#         "Steam Pressure",
#         f"{recommended_steam} kPa",
#         f"{recommended_steam - current_steam:.2f} kPa",
#         delta_color="inverse",
#     )
#     rc2.metric(
#         "Stock Flow",
#         f"{recommended_stock} L/min",
#         f"{recommended_stock - current_stock:.2f} L/min",
#     )
#     rc3.metric(
#         "Expected Correction",
#         f"~{abs(deviation_pct) * 0.7:.1f}% ↓",
#         help="Estimated reduction in deviation after applying setpoints.",
#     )

#     st.markdown(
#         "<span class='rec-chip'>▼ Reduce Steam Pressure 2%</span>"
#         "<span class='rec-chip'>▲ Increase Stock Flow 1.5%</span>"
#         "<span class='rec-chip'>⚡ Monitor Moisture Closely</span>",
#         unsafe_allow_html=True,
#     )
# else:
#     bw_upper = bw_target * (1 + OFFSPEC_THRESH)
#     bw_lower = bw_target * (1 - OFFSPEC_THRESH)
#     st.markdown(
#         f"""<div class='alert-ok'>
#         <div class='alert-title' style='color:#3fb950'>
#             ✅ SYSTEM NOMINAL — Basis Weight Within Spec
#         </div>
#         <div class='alert-body'>
#             Predicted basis weight <b>{bw_pred:.2f} g/m²</b> is within the 
#             safe operating band [{bw_lower:.2f} – {bw_upper:.2f}] g/m².
#             Deviation: <b>{deviation_pct:+.2f}%</b>.
#         </div></div>""",
#         unsafe_allow_html=True,
#     )

# # ═══════════════════════════════════════════════════════════════════════════
# # SECTION 4 — EXPLAINABLE AI (SHAP + Correlation Heatmap)
# # ═══════════════════════════════════════════════════════════════════════════
# st.divider()
# st.markdown("<div class='section-header'>SECTION 4 — EXPLAINABLE AI (XAI)</div>", unsafe_allow_html=True)

# col_shap, col_heat = st.columns([1, 1])

# # ── SHAP Waterfall for the selected row ─────────────────────────────────────
# with col_shap:
#     st.markdown("#### 🧠 SHAP Waterfall — Why This Prediction?")
#     st.caption(
#         "Each bar shows a feature's contribution (positive = pushes BW higher, "
#         "negative = pushes BW lower). Start from the expected value (ℰ[f])."
#     )

#     fig_shap, ax_shap = plt.subplots(figsize=(6, 4), facecolor="#0d1117")
#     plt.rcParams.update({"text.color": "#e6edf3"})

#     shap_row = shap_values[selected_row]
#     feature_names = FEATURE_COLS
#     vals  = shap_row.values
#     order = np.argsort(np.abs(vals))[::-1]

#     colors = ["#f85149" if v > 0 else "#3fb950" for v in vals[order]]
#     y_pos  = np.arange(len(feature_names))

#     ax_shap.barh(
#         y_pos,
#         vals[order],
#         color=colors,
#         edgecolor="#30363d",
#         height=0.6,
#     )
#     ax_shap.set_yticks(y_pos)
#     ax_shap.set_yticklabels([feature_names[i] for i in order], color="#e6edf3", fontsize=9)
#     ax_shap.set_xlabel("SHAP value (impact on BW prediction)", color="#8b949e", fontsize=8)
#     ax_shap.set_facecolor("#161b22")
#     ax_shap.tick_params(colors="#8b949e")
#     for spine in ax_shap.spines.values():
#         spine.set_edgecolor("#30363d")
#     ax_shap.axvline(0, color="#8b949e", linewidth=0.8)
#     ax_shap.set_title(f"SHAP Waterfall — Row {selected_row}", color="#e6edf3", fontsize=10)

#     red_patch   = mpatches.Patch(color="#f85149", label="Pushes BW ↑")
#     green_patch = mpatches.Patch(color="#3fb950", label="Pushes BW ↓")
#     ax_shap.legend(handles=[red_patch, green_patch], fontsize=8,
#                    facecolor="#21262d", edgecolor="#30363d", labelcolor="#c9d1d9")

#     plt.tight_layout()
#     st.pyplot(fig_shap)
#     plt.close(fig_shap)

#     # SHAP summary bar chart (global feature importance)
#     st.markdown("##### Global Feature Importance (SHAP Mean |values|)")
#     fig_imp, ax_imp = plt.subplots(figsize=(6, 2.5), facecolor="#0d1117")
#     mean_shap = np.abs(shap_values.values).mean(axis=0)
#     imp_order = np.argsort(mean_shap)

#     ax_imp.barh(
#         [FEATURE_COLS[i] for i in imp_order],
#         mean_shap[imp_order],
#         color="#58a6ff",
#         edgecolor="#30363d",
#         height=0.5,
#     )
#     ax_imp.set_facecolor("#161b22")
#     ax_imp.tick_params(colors="#8b949e", labelsize=8)
#     ax_imp.set_xlabel("Mean |SHAP value|", color="#8b949e", fontsize=8)
#     for spine in ax_imp.spines.values():
#         spine.set_edgecolor("#30363d")
#     ax_imp.set_title("Global Feature Importance", color="#e6edf3", fontsize=9)
#     plt.tight_layout()
#     st.pyplot(fig_imp)
#     plt.close(fig_imp)

# # ── Correlation Heatmap ──────────────────────────────────────────────────────
# with col_heat:
#     st.markdown("#### 🔥 Process Variable Correlation Heatmap")
#     st.caption(
#         "Pearson correlation across all process variables. "
#         "Strong positive/negative correlations reveal control loop interactions."
#     )

#     heat_cols = FEATURE_COLS + [TARGET_COL]
#     corr = df[heat_cols].corr()

#     fig_heat, ax_heat = plt.subplots(figsize=(6, 5.5), facecolor="#0d1117")
#     ax_heat.set_facecolor("#161b22")

#     mask = np.zeros_like(corr, dtype=bool)
#     mask[np.triu_indices_from(mask)] = True   # upper triangle only

#     sns.heatmap(
#         corr,
#         mask=mask,
#         annot=True,
#         fmt=".2f",
#         cmap="coolwarm",
#         center=0,
#         linewidths=0.5,
#         linecolor="#30363d",
#         ax=ax_heat,
#         annot_kws={"size": 8, "color": "#e6edf3"},
#         cbar_kws={"shrink": 0.8},
#     )
#     ax_heat.set_xticklabels(
#         ax_heat.get_xticklabels(), rotation=35, ha="right",
#         color="#8b949e", fontsize=8,
#     )
#     ax_heat.set_yticklabels(
#         ax_heat.get_yticklabels(), rotation=0,
#         color="#8b949e", fontsize=8,
#     )
#     ax_heat.set_title("Correlation Matrix — All Process Variables", color="#e6edf3", fontsize=10)
#     plt.tight_layout()
#     st.pyplot(fig_heat)
#     plt.close(fig_heat)

# # ═══════════════════════════════════════════════════════════════════════════
# # SECTION 5 — OPERATOR ACTION & FEEDBACK LOOP
# # ═══════════════════════════════════════════════════════════════════════════
# st.divider()
# st.markdown("<div class='section-header'>SECTION 5 — OPERATOR DECISION & FEEDBACK LOOP</div>", unsafe_allow_html=True)
# st.markdown("#### 🧑‍💼 Review AI Recommendation & Log Decision")

# if not is_offspec:
#     st.info(
#         "No active off-spec event at the current time-step. "
#         "Drag the slider to row 400–500 to simulate the anomaly.",
#         icon="ℹ️",
#     )

# st.markdown(
#     f"""
#     | Parameter              | Current Value         | Recommended Value     | Change    |
#     |------------------------|-----------------------|-----------------------|-----------|
#     | Steam Pressure (kPa)   | {current_steam:.2f}   | {recommended_steam}   | −2 %      |
#     | Stock Flow (L/min)     | {current_stock:.2f}   | {recommended_stock}   | +1.5 %    |
#     | Basis Weight Pred.     | {bw_pred:.2f} g/m²    | ≈ {bw_target:.2f} g/m²| ↓ to target |
#     """
# )

# # Action buttons
# b_accept, b_reject, _ = st.columns([1, 1, 3])

# if b_accept.button("✅  Accept Recommendation", type="primary", use_container_width=True):
#     row_id = insert_feedback(
#         timestamp               = datetime.datetime.now().isoformat(),
#         current_basis_weight    = bw_actual,
#         predicted_basis_weight  = bw_pred,
#         suggested_stock_flow    = recommended_stock,
#         suggested_steam_pressure= recommended_steam,
#         operator_decision       = "Accepted",
#     )
#     st.success(
#         f"✅ Decision **Accepted** logged successfully (Record #{row_id}). "
#         "Setpoints transmitted to DCS.",
import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from database import init_db, log_feedback, get_feedback_history

# Page Configuration
st.set_page_config(page_title="Honeywell QCS - Grade Change Intelligence", layout="wide", initial_sidebar_state="expanded")
init_db()

# Custom Industrial CSS
st.markdown("""
    <style>
    .main { background-color: #0E1117; }
    .stMetric { background-color: #1E232A; padding: 12px; border-radius: 8px; border-left: 4px solid #00D4B1; }
    .alert-box { background-color: #3D1414; padding: 15px; border-radius: 8px; border-left: 5px solid #FF4B4B; color: white; }
    .success-box { background-color: #113322; padding: 15px; border-radius: 8px; border-left: 5px solid #00C853; color: white; }
    </style>
""", unsafe_allow_html=True)

# 1. DATA PIPELINE & FEATURE ENGINEERING (Lag Features for Dead-time)
@st.cache_data
def load_and_preprocess_data():
    df = pd.read_csv("paper_mill_data.csv")
    
    # Target variable: Basis Weight 5 steps into the future (Forecasting t+5)
    FORECAST_HORIZON = 5
    df['target_bw_future'] = df['basis_weight'].shift(-FORECAST_HORIZON)
    
    # Feature Engineering: Lag features and rate-of-change
    features = ['stock_flow', 'steam_pressure', 'machine_speed', 'filler_flow', 'moisture', 'ash_content', 'caliper']
    for f in features:
        df[f'{f}_lag3'] = df[f].shift(3)
        df[f'{f}_lag5'] = df[f].shift(5)
        df[f'{f}_roc'] = df[f].diff(3)
        
    df = df.dropna().reset_index(drop=True)
    return df

df = load_and_preprocess_data()

# 2. MODEL TRAINING (Strict Temporal Split - Train on Stable Phase, Test on Transition)
feature_cols = [c for c in df.columns if c not in ['timestamp', 'basis_weight', 'target_bw_future', 'alarm_state']]

@st.cache_resource
def train_forecaster(data):
    # Train strictly on baseline historical data (rows 0-250) to prevent target/anomaly leakage
    train_df = data.iloc[:250]
    
    X_train = train_df[feature_cols]
    y_train = train_df['target_bw_future']
    
    model = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train)
    
    explainer = shap.TreeExplainer(model)
    return model, explainer

model, explainer = train_forecaster(df)

# 3. REAL AI RECOMMENDATION ENGINE (Inverse Model Optimization via Grid Search)
def optimize_setpoints(current_row, target_bw):
    """
    Evaluates 121 control action permutations over stock_flow and steam_pressure
    using the trained XGBoost model as a Digital Twin surrogate.
    Returns setpoints that minimize error to target while minimizing actuator wear.

    ROOT-CAUSE FIX: The model is trained on lag features (stock_flow_lag3,
    stock_flow_lag5, stock_flow_roc, etc.). Simply changing the raw stock_flow /
    steam_pressure columns leaves all lags unchanged, so the model predicts the
    same basis-weight for every grid candidate and the optimizer trivially returns
    the current value as "best". We now propagate each perturbation to all related
    lag and rate-of-change columns so the surrogate sees a meaningfully different
    operating point.
    """
    best_loss = float('inf')
    best_stock = current_row['stock_flow']
    best_steam = current_row['steam_pressure']

    # Guard against divide-by-zero
    cur_stock = current_row['stock_flow']  if current_row['stock_flow']  != 0 else 1e-6
    cur_steam = current_row['steam_pressure'] if current_row['steam_pressure'] != 0 else 1e-6

    # Search grid around current operating values (±10 % stock, ±15 % steam)
    stock_candidates = np.linspace(cur_stock * 0.90, cur_stock * 1.10, 11)
    steam_candidates = np.linspace(cur_steam * 0.85, cur_steam * 1.15, 11)

    base_features = current_row[feature_cols].copy()

    for stock in stock_candidates:
        for steam in steam_candidates:
            test_feat = base_features.copy()

            # --- Update raw setpoint columns ---
            test_feat['stock_flow']     = stock
            test_feat['steam_pressure'] = steam

            # --- Propagate to lag features (simulate sustained setpoint change) ---
            stock_ratio = stock / cur_stock
            steam_ratio = steam / cur_steam

            for lag in ['lag3', 'lag5']:
                sf_lag = f'stock_flow_{lag}'
                sp_lag = f'steam_pressure_{lag}'
                if sf_lag in test_feat.index:
                    test_feat[sf_lag] = base_features[sf_lag] * stock_ratio
                if sp_lag in test_feat.index:
                    test_feat[sp_lag] = base_features[sp_lag] * steam_ratio

            # --- Update rate-of-change (delta from current baseline) ---
            if 'stock_flow_roc'     in test_feat.index:
                test_feat['stock_flow_roc']     = stock - cur_stock
            if 'steam_pressure_roc' in test_feat.index:
                test_feat['steam_pressure_roc'] = steam - cur_steam

            # Predict using ML surrogate
            pred_bw = model.predict(pd.DataFrame([test_feat]))[0]

            # Loss = tracking error + control-effort penalty
            error_penalty   = (pred_bw - target_bw) ** 2
            control_penalty = 0.01 * ((stock - cur_stock) ** 2 + (steam - cur_steam) ** 2)
            total_loss = error_penalty + control_penalty

            if total_loss < best_loss:
                best_loss  = total_loss
                best_stock = stock
                best_steam = steam

    return round(best_stock, 2), round(best_steam, 2)

# --- DASHBOARD UI LAYOUT ---
st.title("⚡ Honeywell QCS: Predictive Grade Change Intelligence")
st.caption("Machine Direction (MD) Advanced Advisory System | Edge-Deployed Surrogates & XAI")

# Time Slider for Operator Replay / Stream Simulation
st.sidebar.header("Control Panel")
time_idx = st.sidebar.slider("Simulation Time Step (Minutes)", min_value=10, max_value=len(df)-1, value=440, step=1)

current_sample = df.iloc[time_idx]
current_X = pd.DataFrame([current_sample[feature_cols]])

# Generate 5-Step Ahead Forecast
predicted_future_bw = model.predict(current_X)[0]
actual_target = current_sample['recipe_target_bw']
current_bw = current_sample['basis_weight']
deviation_pct = abs(predicted_future_bw - actual_target) / actual_target * 100

# Top KPI Metric Cards
c1, c2, c3, c4 = st.columns(4)
c1.metric("Current Basis Weight", f"{current_bw:.2f} g/m²")
c2.metric("Target Setpoint", f"{actual_target:.2f} g/m²")
c3.metric("5-Min Forecast (Y_t+5)", f"{predicted_future_bw:.2f} g/m²", delta=f"{predicted_future_bw - actual_target:.2f}")
c4.metric("Predicted Deviation", f"{deviation_pct:.2f}%", delta_color="inverse")

st.markdown("---")

# Main Plot: Trajectory Tracking & Anomaly Prediction
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("📈 Machine Direction Trajectory & Predictive Horizon")
    
    window_df = df.iloc[max(0, time_idx-60):min(len(df), time_idx+20)].copy()
    
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(window_df['timestamp'], window_df['basis_weight'], label="Actual Basis Weight", color="#00D4B1", lw=2)
    ax.plot(window_df['timestamp'], window_df['recipe_target_bw'], label="Recipe Target", color="#FFFFFF", linestyle="--", alpha=0.7)
    
    # Upper/Lower 2.5% Bounds
    ax.fill_between(window_df['timestamp'], window_df['recipe_target_bw']*1.025, window_df['recipe_target_bw']*0.975, color="#00D4B1", alpha=0.1, label="±2.5% Spec Band")
    
    # Highlight Current Step & Forecast Step
    ax.axvline(current_sample['timestamp'], color="#FFCC00", linestyle=":", label="Current Time (t)")
    ax.scatter(df.iloc[min(len(df)-1, time_idx+5)]['timestamp'], predicted_future_bw, color="#FF4B4B", s=80, zorder=5, label="Forecast (t+5)")
    
    ax.set_facecolor("#1E232A")
    fig.patch.set_facecolor("#0E1117")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.legend(facecolor="#1E232A", labelcolor="white")
    st.pyplot(fig)

with col_right:
    st.subheader("🤖 Advisory & Optimization Engine")
    
    if deviation_pct > 2.5:
        st.markdown("""
        <div class="alert-box">
            <b>⚠️ OFF-SPEC RISK DETECTED</b><br>
            Basis Weight forecast exceeds 2.5% tolerance limit in T+5 minutes.
        </div>
        """, unsafe_allow_html=True)
        
        # Trigger Inverse Model Optimizer
        opt_stock, opt_steam = optimize_setpoints(current_sample, actual_target)
        
        st.write("### Recommended Setpoint Adjustments:")
        st.write(f"• **Thick Stock Flow:** `{current_sample['stock_flow']}` ➔ **`{opt_stock}`**")
        st.write(f"• **Dryer Steam Pressure:** `{current_sample['steam_pressure']}` ➔ **`{opt_steam}`**")
        st.caption("Optimization Source: Inverse XGBoost Constraint Solver")
        
        col_btn1, col_btn2 = st.columns(2)
        if col_btn1.button("✅ Accept Advisory"):
            log_feedback(current_bw, predicted_future_bw, actual_target, opt_stock, opt_steam, "ACCEPTED")
            st.success("Setpoints dispatched to QCS Controller. Logged in DB.")
            
        if col_btn2.button("❌ Reject Advisory"):
            log_feedback(current_bw, predicted_future_bw, actual_target, opt_stock, opt_steam, "REJECTED")
            st.error("Advisory rejected. Feedback stored for retraining.")
    else:
        st.markdown("""
        <div class="success-box">
            <b>✅ SYSTEM OPTIMAL</b><br>
            Process trajectory is within 2.5% quality bounds. No intervention required.
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# Explainability & Diagnostic Section
col_shap, col_corr = st.columns(2)

with col_shap:
    st.subheader("🔍 Rationale (SHAP Feature Attribution)")
    shap_values = explainer(current_X)
    
    fig_shap, ax_shap = plt.subplots(figsize=(6, 4))
    # Waterfall / Bar plot for localized explanation
    vals = shap_values.values[0]
    features_names = current_X.columns
    top_idx = np.argsort(np.abs(vals))[-6:]
    
    ax_shap.barh(range(6), vals[top_idx], color=np.where(vals[top_idx]>0, "#FF4B4B", "#00D4B1"))
    ax_shap.set_yticks(range(6))
    ax_shap.set_yticklabels([features_names[i] for i in top_idx])
    ax_shap.set_title("Feature Contribution to T+5 Forecast Deviation", color="white")
    ax_shap.set_facecolor("#1E232A")
    fig_shap.patch.set_facecolor("#0E1117")
    ax_shap.tick_params(colors="white")
    st.pyplot(fig_shap)

with col_corr:
    st.subheader("📊 Multivariable Correlation Matrix")
    fig_corr, ax_corr = plt.subplots(figsize=(6, 4))
    corr_cols = ['stock_flow', 'steam_pressure', 'machine_speed', 'moisture', 'basis_weight']
    sns.heatmap(df[corr_cols].corr(), annot=True, fmt=".2f", cmap="vlag", ax=ax_corr, cbar=False)
    ax_corr.set_facecolor("#1E232A")
    fig_corr.patch.set_facecolor("#0E1117")
    ax_corr.tick_params(colors="white")
    st.pyplot(fig_corr)

# Database Audit Trail Log
with st.expander("📁 View Closed-Loop Operator Audit Trail (SQLite)"):
    logs_df = get_feedback_history()
    st.dataframe(logs_df, use_container_width=True)