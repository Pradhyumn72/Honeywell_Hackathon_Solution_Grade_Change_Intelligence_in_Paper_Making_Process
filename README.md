
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

---

## Author

**Pradhyumn**

VIT Bhopal University

Built for the **Honeywell Hackathon 2026**
