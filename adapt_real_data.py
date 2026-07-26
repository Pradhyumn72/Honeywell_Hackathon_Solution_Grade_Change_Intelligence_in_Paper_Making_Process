"""
adapt_real_data.py
------------------
Real-World Data Pipeline Integration Script
Ingests sampledata.csv (100 rows, anonymized sensor readings),
tiles it 10× with injected noise, maps to physical paper-mill variables,
synthesises basis_weight with a 5-step transport lag, and writes the
final paper_mill_data.csv consumed by the Streamlit dashboard (app.py).

Run:  myenv/bin/python adapt_real_data.py
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
import sys

# ─────────────────────────── Configuration ─────────────────────────────────
INPUT_FILE   = "sampledata.csv"
OUTPUT_FILE  = "paper_mill_data.csv"
START_TS     = "2026-07-26 00:00"
N_TILES      = 10           # 100 rows × 10 = 1 000 rows
SEED         = 2026
RNG          = np.random.default_rng(SEED)

# Grade-change recipe definition
BW_HIGH       = 80.0        # g/m²  — initial grade
BW_LOW        = 65.0        # g/m²  — final grade
PHASE1_END    = 300
TRANSITION_END = 700

# ─────────────────────────── Step 1 · Load raw sample data ─────────────────
print(f"[1/7] Reading {INPUT_FILE} …")
if not os.path.exists(INPUT_FILE):
    sys.exit(f"ERROR: {INPUT_FILE} not found in working directory.")

raw = pd.read_csv(INPUT_FILE)
# Ensure expected columns
for col in ["y", "x", "x1", "x2"]:
    if col not in raw.columns:
        sys.exit(f"ERROR: Expected column '{col}' missing from {INPUT_FILE}.")

print(f"       {len(raw)} rows loaded — columns: {list(raw.columns)}")

# ─────────────────────────── Step 2 · Tile × 10 with noise ─────────────────
print(f"[2/7] Tiling {N_TILES}× and injecting industrial sensor noise …")

tiles = []
for tile_idx in range(N_TILES):
    tile = raw[["y", "x", "x1", "x2"]].copy()
    # Each tile gets slightly different noise amplitude to avoid perfect repetition
    noise_scale = 0.03 + tile_idx * 0.005
    tile["y"]  += RNG.normal(0, abs(raw["y"].std())  * noise_scale, len(tile))
    tile["x"]  += RNG.normal(0, abs(raw["x"].std())  * noise_scale, len(tile))
    tile["x1"] += RNG.normal(0, abs(raw["x1"].std()) * noise_scale, len(tile))
    tile["x2"] += RNG.normal(0, abs(raw["x2"].std()) * noise_scale, len(tile))
    tiles.append(tile)

df = pd.concat(tiles, ignore_index=True)
N = len(df)   # = 1 000
print(f"       Dataset expanded to {N} rows.")

# ─────────────────────────── Step 3 · New timestamp column ─────────────────
print("[3/7] Generating 1-minute interval timestamps …")
base_ts = datetime.strptime(START_TS, "%Y-%m-%d %H:%M")
df["timestamp"] = [
    (base_ts + timedelta(minutes=i)).strftime("%Y-%m-%d %H:%M:%S")
    for i in range(N)
]

# ─────────────────────────── Step 4 · Map to physical variables ─────────────
print("[4/7] Mapping anonymized variables → physical paper-mill variables …")

# Micro-noise helper (instrument measurement uncertainty)
def inoise(n, sigma):
    return RNG.normal(0, sigma, n)

# PRE-NORMALISE to zero-mean / unit-std so the mapping coefficients (15, 5, 2)
# behave as designed. Raw data has std(x)~9, std(x1)~31, std(x2)~14 which
# would create ±150 L/min stock_flow swings and 90%+ alarm rate if used raw.
x_z  = (df["x"]  - df["x"].mean())  / (df["x"].std()  + 1e-9)
x1_z = (df["x1"] - df["x1"].mean()) / (df["x1"].std() + 1e-9)
x2_z = (df["x2"] - df["x2"].mean()) / (df["x2"].std() + 1e-9)

df["machine_speed"]  = (x_z  * 15) + 1200 + inoise(N, 3.0)
df["stock_flow"]     = (x1_z *  5) + 400  + inoise(N, 2.5)
df["steam_pressure"] = (x2_z *  2) + 35   + inoise(N, 0.8)

# ── Grade-change setpoint correction on stock_flow ──────────────────────────
# In a real grade change, the operator ramps stock flow DOWN to reach BW_LOW.
# We build a correction signal that reduces stock_flow by ~100 L/min by row 700.
idx = np.arange(N)
gc_correction = np.where(
    idx <= PHASE1_END, 0.0,
    np.where(
        idx >= TRANSITION_END, -100.0,
        -100.0 * (idx - PHASE1_END) / (TRANSITION_END - PHASE1_END)
    )
)
df["stock_flow"] += gc_correction
df["machine_speed"] -= gc_correction * 0.15   # speed also drops slightly

# Clip to physically realistic operating ranges (tight, matching the low variance)
df["machine_speed"]  = df["machine_speed"].clip(1150, 1260)
df["stock_flow"]     = df["stock_flow"].clip(280,  430)
df["steam_pressure"] = df["steam_pressure"].clip(29,   42)

# ─────────────────────────── Step 5 · Derive secondary variables ────────────
print("[5/7] Deriving filler_flow, moisture, ash_content, caliper …")

# filler_flow (L/min) — typically 5-8% of stock, slightly more at lower grades
df["filler_flow"] = (
    df["stock_flow"] * 0.065
    + gc_correction * (-0.01)       # filler fraction increases at low BW
    + inoise(N, 0.5)
).clip(10, 60)

# moisture (%) — inversely related to steam pressure; higher steam → drier sheet
df["moisture"] = (
    8.0
    - (df["steam_pressure"] - 35) * 0.06
    + inoise(N, 0.25)
).clip(3.5, 12.0)

# ash_content (%) — derived from filler proportion
df["ash_content"] = (
    df["filler_flow"] * 0.38
    + inoise(N, 0.4)
).clip(0.5, 25.0)

# ─────────────────────────── Step 6 · recipe_target_bw ─────────────────────
print("[6/7] Building recipe_target_bw with grade-change ramp …")

recipe_target = np.where(
    idx <= PHASE1_END, BW_HIGH,
    np.where(
        idx >= TRANSITION_END, BW_LOW,
        BW_HIGH + (BW_LOW - BW_HIGH) * (idx - PHASE1_END) / (TRANSITION_END - PHASE1_END)
    )
)
df["recipe_target_bw"] = np.round(recipe_target, 4)

# ─────────────────────────── Step 7 · basis_weight with transport lag ────────
print("[7/7] Synthesising basis_weight with 5-step transport delay …")

sf  = df["stock_flow"].values
sp  = df["steam_pressure"].values
x1v = df["x1"].values

# Normalise raw sensor signals to zero-mean / unit-std BEFORE applying formula.
# The prescribed coefficients (0.8 and 0.2) imply unit-scale inputs — using
# raw values (std~10 for y, std~30 for x1) would create ±30 g/m² swings
# and trigger alarms on 90 %+ of rows.
y_norm  = (df["y"].values  - df["y"].mean())  / (df["y"].std()  + 1e-9)
x1_norm = (df["x1"].values - df["x1"].mean()) / (df["x1"].std() + 1e-9)

basis_weight = np.zeros(N)

for t in range(N):
    # 5-step dead-time lag on stock_flow, 2-step on steam_pressure
    sf_lag = sf[t - 5] if t >= 5 else sf[0]
    sp_lag = sp[t - 2] if t >= 2 else sp[0]

    # Physics-based BW from process variables
    # Formula: bw_physics = (stock_flow[t-5] * 0.15) + (steam_pressure[t-2] * 0.3) + 9.5
    bw_physics = (sf_lag * 0.15) + (sp_lag * 0.3) + 9.5

    # Residual error + real sensor contribution (normalised inputs)
    # Formula: basis_weight[t] = bw_physics + (error * 0.8) + (x1[t] * 0.2)
    basis_weight[t] = bw_physics + (y_norm[t] * 0.8) + (x1_norm[t] * 0.2)

# Clip to physically realistic paper-weight range
basis_weight = np.clip(basis_weight, 50.0, 120.0)
df["basis_weight"] = np.round(basis_weight, 4)

# caliper (µm) — derived from basis weight after BW is known
df["caliper"] = (
    df["basis_weight"] * 1.25
    + inoise(N, 1.2)
).clip(60, 130)
df["caliper"] = df["caliper"].round(3)

# ─────────────────────────── Step 8 · alarm_state ──────────────────────────
deviation_pct = (
    (df["basis_weight"] - df["recipe_target_bw"]).abs()
    / df["recipe_target_bw"]
    * 100
)
df["alarm_state"] = (deviation_pct > 2.5).astype(int)

# ─────────────────────────── Final column order & export ───────────────────
final_cols = [
    "timestamp",
    "machine_speed",
    "stock_flow",
    "steam_pressure",
    "filler_flow",
    "moisture",
    "ash_content",
    "caliper",
    "basis_weight",
    "recipe_target_bw",
    "alarm_state",
]
df_out = df[final_cols].round(4)
df_out.to_csv(OUTPUT_FILE, index=False)

# ─────────────────────────── Verification summary ──────────────────────────
print(f"\n{'='*60}")
print(f"✓  Output written → {OUTPUT_FILE}  ({len(df_out)} rows × {len(final_cols)} cols)")
print(f"\nColumn ranges:")
for col in final_cols[1:]:
    s = df_out[col]
    print(f"  {col:<22}  min={s.min():.3f}  mean={s.mean():.3f}  max={s.max():.3f}")

alarm_rows = df_out["alarm_state"].sum()
print(f"\nAlarm rows (deviation > 2.5%): {alarm_rows}  ({alarm_rows/N*100:.1f}%)")

# Phase distribution
p1  = (df_out.index <= PHASE1_END).sum()
tr  = ((df_out.index > PHASE1_END) & (df_out.index <= TRANSITION_END)).sum()
p2  = (df_out.index > TRANSITION_END).sum()
print(f"\nPhase split:  Stable-1={p1}  Transition={tr}  Stable-2={p2}")

bw_p1 = df_out.loc[df_out.index <= PHASE1_END,  "basis_weight"].mean()
bw_tr = df_out.loc[(df_out.index > PHASE1_END) & (df_out.index <= TRANSITION_END), "basis_weight"].mean()
bw_p2 = df_out.loc[df_out.index > TRANSITION_END, "basis_weight"].mean()
print(f"Mean BW:      Stable-1={bw_p1:.2f}  Transition={bw_tr:.2f}  Stable-2={bw_p2:.2f}  g/m²")
print(f"{'='*60}\n")
