import random

def predict_failures(question):
    failures = {
        "wind turbine health": "Vibration analysis indicates a 10% chance of gearbox failure in 30 days.",
        "transformer maintenance": "Thermal sensors detect overheating; maintenance required within 2 weeks."
    }
    return failures.get(question, "No maintenance issues detected.")
