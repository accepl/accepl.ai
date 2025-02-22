import random

def forecast_load(question):
    forecasts = {
        "next week demand": "Expected power demand: 1500 MW based on historical load trends.",
        "grid optimization": "AI recommends load shifting to off-peak hours to reduce energy waste."
    }
    return forecasts.get(question, f"Predicted load: {random.randint(1000, 5000)} MW.")
