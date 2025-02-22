import random

def predict_cost(question):
    epc_data = {
        "solar project cost": "A 100 MW solar power plant costs approximately ₹400 crore.",
        "wind project cost": "A 100 MW wind power plant costs approximately ₹600 crore.",
        "bess cost": "Battery Energy Storage Systems (BESS) cost around ₹1.5 crore per MW."
    }
    return epc_data.get(question, f"Estimated cost: ₹{random.randint(100, 500)} crore based on project size.")
