import random

def monitor_pipelines():
    monitoring = {
        "pipeline leak detection": "Pressure drop detected; potential leak at pipeline segment 34A.",
        "refinery health check": "All refinery sensors indicate normal operations.",
        "offshore drilling": "Increased vibration detected on offshore drilling unit; maintenance recommended."
    }
    return monitoring.get(random.choice(list(monitoring.keys())), "AI is analyzing pipeline conditions.")
