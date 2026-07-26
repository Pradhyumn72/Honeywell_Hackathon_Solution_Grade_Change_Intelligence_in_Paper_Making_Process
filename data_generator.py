import numpy as np
import pandas as pd

def generate_industrial_data():
    np.random.seed(42)
    n_steps = 1000
    timestamps = pd.date_range(start="2026-07-25 00:00", periods=n_steps, freq="1min")
    
    # Recipe Targets
    recipe_target_bw = np.zeros(n_steps)
    recipe_target_bw[:301] = 80.0
    recipe_target_bw[301:701] = np.linspace(80.0, 65.0, 400)
    recipe_target_bw[701:] = 65.0
    
    # Base Control Variables
    machine_speed = np.zeros(n_steps)
    machine_speed[:301] = 1200.0
    machine_speed[301:701] = np.linspace(1200.0, 1350.0, 400)
    machine_speed[701:] = 1350.0
    machine_speed += np.random.normal(0, 1.5, n_steps)
    
    # Stock Flow follows target + noise
    stock_flow = recipe_target_bw * 5.2 + np.random.normal(0, 0.8, n_steps)
    
    # Filler & Ash
    filler_flow = stock_flow * 0.12 + np.random.normal(0, 0.1, n_steps)
    ash_content = (filler_flow / stock_flow) * 100 + np.random.normal(0, 0.05, n_steps)
    
    # Steam Pressure with ANOMALY injection (rows 420 to 520)
    steam_pressure = recipe_target_bw * 0.45 + np.random.normal(0, 0.2, n_steps)
    steam_pressure[420:520] -= 3.5  # Sudden pressure drop causing thermal imbalance
    
    # Physics Simulation for Basis Weight (with 5-step transport delay / lag)
    basis_weight = np.zeros(n_steps)
    basis_weight[:5] = 80.0
    for t in range(5, n_steps):
        # Basis weight depends on delayed stock flow, steam pressure efficiency, and speed
        bw_physics = (stock_flow[t-5] * 0.18) + (steam_pressure[t-2] * 0.1) - (machine_speed[t] * 0.005) + 38.5
        basis_weight[t] = bw_physics + np.random.normal(0, 0.3)
        
    # Caliper and Moisture
    moisture = 7.0 - (steam_pressure * 0.15) + np.random.normal(0, 0.1, n_steps)
    caliper = (basis_weight * 1.4) - (machine_speed * 0.01) + np.random.normal(0, 0.5, n_steps)
    
    # Alarm state trigger (>2.5% deviation)
    dev = np.abs(basis_weight - recipe_target_bw) / recipe_target_bw
    alarm_state = np.where(dev > 0.025, 1, 0)
    
    df = pd.DataFrame({
        "timestamp": timestamps,
        "stock_flow": np.round(stock_flow, 2),
        "steam_pressure": np.round(steam_pressure, 2),
        "machine_speed": np.round(machine_speed, 2),
        "filler_flow": np.round(filler_flow, 2),
        "moisture": np.round(moisture, 2),
        "ash_content": np.round(ash_content, 2),
        "caliper": np.round(caliper, 2),
        "basis_weight": np.round(basis_weight, 2),
        "recipe_target_bw": np.round(recipe_target_bw, 2),
        "alarm_state": alarm_state
    })
    
    df.to_csv("paper_mill_data.csv", index=False)
    print("Dataset paper_mill_data.csv generated successfully.")

if __name__ == "__main__":
    generate_industrial_data()
