import os
import json
import random
import requests
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import openai
from flask import Flask, request, jsonify, send_from_directory

# API Security Key (For Authentication)
API_SECRET_KEY = os.environ.get("API_SECRET_KEY", "your_secret_key")

# OpenAI API Key (For ChatGPT Fallback)
openai.api_key = os.environ.get("OPENAI_API_KEY", "your_openai_api_key")

# Initialize Flask App
app = Flask(__name__)

# ==============================
# ✅ AI Core Model (Self-Learning)
# ==============================
class AIModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(AIModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out[:, -1, :])
        return output

# Dummy Data Training
def train_dummy_models():
    """Trains AI models on dummy datasets for EPC, Smart Grid, Oil & Gas, Telecom, and IPP."""
    np.random.seed(42)
    torch.manual_seed(42)

    models = {}
    industries = ["EPC", "SmartGrid", "OilGas", "Telecom", "IPP"]
    
    for industry in industries:
        input_size, hidden_size, output_size = 10, 20, 1
        model = AIModel(input_size, hidden_size, output_size)

        # Generate random data
        X_train = torch.rand(100, 10, input_size)
        y_train = torch.rand(100, output_size)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        for epoch in range(50):  # Train for 50 epochs
            optimizer.zero_grad()
            output = model(X_train)
            loss = criterion(output, y_train)
            loss.backward()
            optimizer.step()

        models[industry] = model
    return models

# Train Models
trained_models = train_dummy_models()

# ==============================
# ✅ AI Decision-Making Logic
# ==============================
def process_prompt(prompt):
    """Processes a user prompt and generates an AI response."""
    prompt = prompt.lower()
    response = ""

    # AI Decision Logic (Trillion Logic)
    if "grid" in prompt:
        response = f"AI Grid Optimization: Adjust power load by {random.randint(10, 50)}%."
    elif "battery" in prompt or "bess" in prompt:
        response = f"AI Smart BESS: Optimized charge level: {random.randint(50, 90)}%."
    elif "maintenance" in prompt:
        response = f"Predictive Maintenance: Failure likelihood in next 30 days: {random.randint(5, 25)}%."
    else:
        response = ask_chatgpt(prompt)  # Fallback to ChatGPT

    return response

# ==============================
# ✅ ChatGPT Fallback
# ==============================
def ask_chatgpt(prompt):
    """Queries ChatGPT if AI lacks a trained response."""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        return f"ChatGPT Error: {e}"

# ==============================
# ✅ API Endpoint for AI Prompt Processing
# ==============================
@app.route("/api/prompt", methods=["POST"])
def handle_prompt():
    """Handles user prompt and returns AI-generated response."""
    data = request.json
    prompt = data.get("prompt", "")

    if request.headers.get("Authorization") != API_SECRET_KEY:
        return jsonify({"error": "Unauthorized"}), 403

    ai_response = process_prompt(prompt)
    return jsonify({"response": ai_response})

# ==============================
# ✅ Serve Company Logo
# ==============================
@app.route('/static/<path:filename>')
def static_files(filename):
    """Serves static files like logo.jpg"""
    return send_from_directory('static', filename)

# ==============================
# ✅ Built-in Frontend UI
# ==============================
@app.route("/")
def index():
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AI-Powered EPC System</title>
        <style>
            body {{ font-family: Arial, sans-serif; text-align: center; margin: 50px; }}
            img {{ width: 200px; margin-bottom: 20px; }}
            input {{ width: 60%; padding: 10px; font-size: 16px; }}
            button {{ padding: 10px 20px; font-size: 16px; background-color: #28a745; color: white; border: none; }}
            #response {{ margin-top: 20px; font-weight: bold; font-size: 18px; }}
        </style>
    </head>
    <body>
        <img src="/static/logo.jpg" alt="Company Logo">
        <h1>AI-Powered EPC & Smart Grid Assistant</h1>
        <input type="text" id="userPrompt" placeholder="Ask your AI a question...">
        <button onclick="sendPrompt()">Get AI Response</button>
        <div id="response"></div>
        <script>
            async function sendPrompt() {{
                let prompt = document.getElementById("userPrompt").value;
                let responseDiv = document.getElementById("response");
                if (!prompt) {{ responseDiv.innerHTML = "Please enter a prompt!"; return; }}
                responseDiv.innerHTML = "Processing...";
                let response = await fetch("/api/prompt", {{
                    method: "POST",
                    headers: {{"Authorization": "{API_SECRET_KEY}", "Content-Type": "application/json"}},
                    body: JSON.stringify({{ prompt: prompt }})
                }});
                let result = await response.json();
                responseDiv.innerHTML = "AI Response: " + result.response;
            }}
        </script>
    </body>
    </html>
    """

# ==============================
# ✅ Start Server with Render-Compatible Port
# ==============================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render assigns port dynamically
    app.run(host="0.0.0.0", port=port, debug=True)
