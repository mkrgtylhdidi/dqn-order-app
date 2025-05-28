# 🏗️ DQN Ordering Recommendation for Construction Logistics

This Streamlit web app provides **daily order quantity recommendations** for steel reinforcement deliveries on construction sites using a trained **Deep Q-Network (DQN)** model.

The model is designed to optimise delivery decisions under uncertainty (e.g. demand variability, lead time, and on-site constraints), aiming to minimise:
- Total cost (holding, transport, shortage)
- Overstocking near project end
- Delivery inefficiencies

---

## 🔗 Live App
👉 [Click here to try the app](https://dqn-order-app-5bckqeamh48k7bn8r9rfjt.streamlit.app/)

---

## 🧠 Model Overview

The DQN model was trained using a custom Gym environment that simulates:
- Variable demand based on planned activities and stochastic disruptions (weather, crane, labour)
- Lead time variability (1–2 days)
- Limited site storage (max 60 tons)
- Discrete truck-based ordering (10-ton increments)

**State features (8 total):**
- Current inventory level
- Current day of the episode
- Backlog (unfulfilled demand)
- Expected demand today
- Lead time
- Expected demand tomorrow
- Trucks used on previous 2 days

**Action space:**  
Discrete order quantities: `[0, 10, 20, 30, 40]` tons

---

## ⚙️ How to Use

1. Input observed state values from the site (e.g. inventory, demand forecast)
2. Click `Get Recommendation`
3. Receive the **optimal order quantity** for today, as recommended by the trained DQN

---

## 📁 Files Included

- `streamlit_app2.py`: The main Streamlit app script
- `trained_dqn_model.pth`: Saved PyTorch model weights
- `requirements.txt`: Python dependencies
- `Untitled5-Copy5.ipynb`: Training notebook for the DQN model (for reference)

---

## 📦 Requirements

To run locally:

```bash
pip install -r requirements.txt
streamlit run streamlit_app2.py
