# streamlit_app.ipynb or you can rename to streamlit_app.py later for deployment

import streamlit as st
import torch
import numpy as np
import torch.nn as nn

# Load trained model class (same as in your training code)
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Instantiate and load the trained model
model = DQN(input_dim=8, output_dim=5)
model.load_state_dict(torch.load("trained_dqn_model.pth", map_location=torch.device("cpu")))
model.eval()

# Define the valid order quantities used during training
valid_order_quantities = [0, 10, 20, 30, 40]
max_inventory = 60

# Streamlit UI
st.title("DQN Ordering Recommendation for Construction Logistics")
st.write("Input today's observed state below to get the recommended order quantity.")

# Input form
inventory = st.slider("Current Inventory Level (tons)", 0, max_inventory, 10, step=1)
day = st.slider("Current Day (0-10)", 0, 10, 2, step=1)
backlog = st.slider("Backlog (tons)", 0, 2 * max_inventory, 0, step=1)
expected_today = st.slider("Expected Demand Today (tons)", 0, 20, 12, step=1)
lead_time = st.selectbox("Lead Time observed for most current delivery (days)", [1, 2], index=0)
expected_tomorrow = st.slider("Expected Demand Tomorrow (tons)", 0, 20, 14, step=1)
trucks_used_yesterday = st.selectbox("Trucks Used on Day t-1", [0, 1, 2, 3, 4], index=1)
trucks_used_day_before = st.selectbox("Trucks Used on Day t-2", [0, 1, 2, 3, 4], index=0)

if st.button("Get Recommendation"):
    # Prepare input state
    state = np.array([
        inventory,
        day,
        backlog,
        expected_today,
        lead_time,
        expected_tomorrow,
        trucks_used_yesterday,
        trucks_used_day_before
    ], dtype=np.float32)

    with torch.no_grad():
        state_tensor = torch.tensor(state).unsqueeze(0)  # shape (1, 8)
        q_values = model(state_tensor).squeeze(0)

        # Only allow valid actions that won't exceed inventory cap
        possible_actions = [i for i, qty in enumerate(valid_order_quantities) if inventory + qty <= max_inventory]
        if inventory >= max_inventory:
            possible_actions = [0]

        valid_q_values = [q_values[i] for i in possible_actions]
        best_action_idx = possible_actions[torch.argmax(torch.tensor(valid_q_values)).item()]
        best_order_quantity = valid_order_quantities[best_action_idx]

    st.success(f"Recommended Order Quantity: **{best_order_quantity} tons**")
