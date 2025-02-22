from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn
import os

app = FastAPI()

# Ensure static folder exists for logo
static_dir = "accepl.ai/static"
if not os.path.exists(static_dir):
    os.makedirs(static_dir)

# AI logic to process queries
def process_query(query):
    if "EPC" in query:
        return "AI Prediction: The EPC market is projected to grow significantly."
    elif "Grid Load" in query:
        return "AI Prediction: Grid load optimization suggests reducing peak demand by 15%."
    elif "Oil & Gas" in query:
        return "AI Prediction: Oil prices expected to stabilize at $80 per barrel."
    elif "Telecom" in query:
        return "AI Prediction: 5G rollout will increase spectrum efficiency by 30%."
    elif "IPP" in query:
        return "AI Prediction: Renewable IPP projects will dominate the market by 2027."
    else:
        return "AI Response: I'm still learning! Try asking about EPC, Grid Load, Oil & Gas, Telecom, or IPP."

# ✅ API Endpoint for Programmatic Access
@app.get("/api/ask", response_class=JSONResponse)
async def ask_api(query: str):
    response = process_query(query)
    return {"query": query, "ai_response": response}

# ✅ Web-Based Chat UI
@app.get("/", response_class=HTMLResponse)
async def home():
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Accepl.AI Chat</title>
        <style>
            body {{ font-family: Arial, sans-serif; text-align: center; padding: 50px; }}
            .container {{ width: 50%; margin: auto; padding: 20px; background: #f9f9f9; border-radius: 10px; }}
            input, button {{ padding: 10px; margin: 10px; font-size: 16px; }}
            .response {{ margin-top: 20px; font-weight: bold; }}
            img {{ width: 150px; margin-bottom: 20px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <img src="/static/logo.jpg" alt="Accepl.AI Logo">
            <h2>Ask Accepl.AI Anything!</h2>
            <form action="/ask" method="post">
                <input type="text" name="query" placeholder="Type your question here..." required>
                <button type="submit">Ask AI</button>
            </form>
            <div class="response" id="response"></div>
        </div>
    </body>
    </html>
    """

# ✅ Web UI: Handle User Input and Return Response
@app.post("/ask", response_class=HTMLResponse)
async def ask_ui(request: Request, query: str = Form(...)):
    response = process_query(query)
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Accepl.AI Chat</title>
        <style>
            body {{ font-family: Arial, sans-serif; text-align: center; padding: 50px; }}
            .container {{ width: 50%; margin: auto; padding: 20px; background: #f9f9f9; border-radius: 10px; }}
            input, button {{ padding: 10px; margin: 10px; font-size: 16px; }}
            .response {{ margin-top: 20px; font-weight: bold; }}
            img {{ width: 150px; margin-bottom: 20px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <img src="/static/logo.jpg" alt="Accepl.AI Logo">
            <h2>Ask Accepl.AI Anything!</h2>
            <form action="/ask" method="post">
                <input type="text" name="query" placeholder="Type your question here..." required>
                <button type="submit">Ask AI</button>
            </form>
            <div class="response">AI Response: {response}</div>
        </div>
    </body>
    </html>
    """

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
