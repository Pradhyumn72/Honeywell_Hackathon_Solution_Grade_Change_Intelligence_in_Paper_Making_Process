
# Grade-Change Intelligence & Predictive Advisory (GCIPA) System

**Honeywell Hackathon Solution: AI-Driven Grade Change Advisory System for Paper Manufacturing**

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?logo=streamlit)
![XGBoost](https://img.shields.io/badge/XGBoost-ML-F37626)
![Status](https://img.shields.io/badge/Status-Prototype_Complete-success)

---

## Problem Statement

During a paper machine grade change, multiple process variables such as stock flow, steam pressure, machine speed, and dryer conditions must be adjusted simultaneously. Because of complex nonlinear process interactions and thermal delays, the machine can temporarily deviate from its desired operating specifications, resulting in off-spec paper production, increased waste, reduced productivity, and inconsistent product quality.

Traditional Quality Control Systems (QCS) primarily execute predefined control strategies based on process models. While effective for routine operation, they have limited capability to predict future process deviations or continuously learn from historical operating behaviour during dynamic grade transitions.

Honeywell's challenge is to develop an intelligent automatic grade change system capable of predicting process deviations before they occur and assisting operators with timely, data-driven recommendations that help maintain product quality throughout the transition.

---

## Our Solution

The **Grade-Change Intelligence & Predictive Advisory (GCIPA) System** is an edge-deployable, AI-driven advisory platform that works alongside Honeywell's existing Quality Control System rather than replacing it.

Our system continuously monitors critical machine parameters, analyzes historical operating behaviour, and predicts future process deviations up to five minutes in advance using a machine learning forecasting model. Whenever a potential quality deviation is detected, the system intelligently evaluates multiple control strategies and recommends the optimal machine setpoints required to keep the process within specification.

To increase operator confidence, every recommendation is accompanied by an Explainable AI (SHAP) visualization that clearly identifies which process variables contributed most to the prediction. Operator decisions are then stored in a local database, enabling continuous analysis and future model improvement.

The complete solution runs entirely on an edge computer without requiring cloud connectivity, making it suitable for deployment inside industrial control rooms.

---

## Core Architecture

### 1)Predictive Time-Series Forecasting

The system uses an XGBoost Regressor trained with engineered lag features to forecast the future Basis Weight trajectory five minutes ahead (T+5). This enables operators to identify process deviations before they actually occur instead of reacting after quality has already deteriorated.

### 2)AI-Based Prescriptive Optimization

When the predicted Basis Weight exceeds the acceptable deviation threshold, the optimization engine freezes the current machine state and simulates 121 possible combinations of Stock Flow and Steam Pressure. The combination that minimizes the predicted quality error is recommended to the operator as the optimal corrective action.

### 3)Explainable AI (SHAP)

Industrial operators require transparency before trusting AI recommendations. SHAP Waterfall plots explain the contribution of every process variable toward the prediction, allowing operators to understand exactly why the model expects the process to deviate.

### 4)Closed-Loop Operator Feedback

Every recommendation can either be accepted or rejected by the operator. These decisions are stored inside a lightweight SQLite database, creating an audit trail that supports continuous improvement and future model retraining.

---

## Technology Stack

| Component        | Technology         |
| ---------------- | ------------------ |
| Frontend         | Streamlit          |
| Machine Learning | XGBoost            |
| Explainable AI   | SHAP               |
| Data Processing  | Pandas, NumPy      |
| Database         | SQLite             |
| Visualization    | Matplotlib, Plotly |
| Deployment       | Edge Computing     |

---

## How It Works

1. Machine sensors continuously send process data.
2. Historical sensor readings are converted into lag features.
3. XGBoost predicts the Basis Weight five minutes into the future.
4. If an off-spec deviation is predicted, the optimizer evaluates multiple control actions.
5. The optimal Steam Pressure and Stock Flow settings are recommended.
6. SHAP explains the reasoning behind the prediction.
7. The operator accepts or rejects the recommendation.
8. The decision is stored in SQLite for continuous improvement.

---

## Key Features

- Predictive process deviation forecasting (T+5)
- AI-based recommendation of optimal machine setpoints
- Explainable AI using SHAP Waterfall plots
- Real-time process monitoring dashboard
- Closed-loop operator feedback logging
- Zero data leakage through temporal train-test splitting
- Fully edge-deployable architecture
-  Lightweight local SQLite database

---

## Expected Industrial Impact

- Reduce off-spec paper during grade transitions
- Improve process stability
- Reduce operator dependency on manual experience
- Increase confidence through Explainable AI
- Support faster and safer grade changes
- Complement existing Honeywell Quality Control Systems



# Project Setup Guide

## Prerequisites

Before running the project, ensure the following are installed on your system:

- Python 3.10 or above (Recommended: Python 3.12)
- Git
- pip (comes with Python)
- A modern web browser (Chrome, Edge, Firefox, or Safari)

Verify your installation:

```bash
python --version
pip --version
git --version
```

---

# Setup for macOS

## Step 1: Clone the Repository

```bash
git clone https://github.com/Pradhyumn72/Honeywell_Hackathon_Solution_Grade_Change_Intelligence_in_Paper_Making_Process.git

cd Honeywell_Hackathon_Solution_Grade_Change_Intelligence_in_Paper_Making_Process
```

## Step 2: Create a Virtual Environment

```bash
python3 -m venv venv
```

## Step 3: Activate the Virtual Environment

```bash
source venv/bin/activate
```

You should now see:

```
(venv)
```

at the beginning of your terminal.

## Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 5: Generate the Dataset

```bash
python adapt_real_data.py
```

## Step 6: Initialize the Database

```bash
python database.py
```

## Step 7: Launch the Dashboard

```bash
streamlit run app.py
```

Open your browser and visit:

```
http://localhost:8501
```

---

# Setup for Windows

## Step 1: Clone the Repository

Open **Command Prompt** or **PowerShell**.

```powershell
git clone https://github.com/Pradhyumn72/Honeywell_Hackathon_Solution_Grade_Change_Intelligence_in_Paper_Making_Process.git

cd Honeywell_Hackathon_Solution_Grade_Change_Intelligence_in_Paper_Making_Process
```

## Step 2: Create a Virtual Environment

```powershell
python -m venv venv
```

## Step 3: Activate the Virtual Environment

### Command Prompt

```cmd
venv\Scripts\activate
```

### PowerShell

```powershell
venv\Scripts\Activate.ps1
```

## Step 4: Install Dependencies

```powershell
pip install -r requirements.txt
```

## Step 5: Generate the Dataset

```powershell
python adapt_real_data.py
```

## Step 6: Initialize the Database

```powershell
python database.py
```

## Step 7: Launch the Dashboard

```powershell
streamlit run app.py
```

Open your browser and navigate to:

```
http://localhost:8501
```

---

# Project Structure

```
Honeywell_Hackathon_Solution/
│
├── app.py                     # Main Streamlit Dashboard
├── model.py                   # XGBoost Forecasting Model
├── optimizer.py               # AI Optimization Engine
├── database.py                # SQLite Database Initialization
├── adapt_real_data.py         # Dataset Generation & Preprocessing
├── requirements.txt           # Python Dependencies
├── operator_feedback.db       # SQLite Database (Generated)
├── data/
│   ├── processed_dataset.csv
│   └── raw_dataset.csv
├── models/
│   └── xgboost_model.pkl
├── assets/
│   ├── dashboard.png
│   └── architecture.png
└── README.md
```

---

# Installing Dependencies Manually (Optional)

If `requirements.txt` is unavailable, install the required packages manually:

```bash
pip install streamlit
pip install pandas
pip install numpy
pip install matplotlib
pip install plotly
pip install scikit-learn
pip install xgboost
pip install shap
pip install joblib
```

---

# Verifying the Installation

Run:

```bash
streamlit run app.py
```

If everything is configured correctly, the dashboard should open automatically in your default browser at:

```
http://localhost:8501
```

You should be able to:

- View live process variables
- Generate process deviation predictions
- Receive AI-based advisory recommendations
- Visualize SHAP explanations
- Log operator feedback into SQLite

---

# Common Issues

## ModuleNotFoundError

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## streamlit: command not found

Run:

```bash
python -m streamlit run app.py
```

---

## Port 8501 Already in Use

Launch Streamlit on another port:

```bash
streamlit run app.py --server.port 8502
```

---

## Virtual Environment Not Activating (Windows)

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Restart PowerShell and activate the virtual environment again.

---

# Deactivate the Virtual Environment

When you're done, run:

```bash
deactivate
```

---

## Author

**Pradhyumn**

VIT Bhopal University

Built for the **Honeywell Hackathon 2026**
