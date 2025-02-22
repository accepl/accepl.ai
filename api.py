ffrom fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import requests
import json
import openai  # ChatGPT API
import os

# Initialize FastAPI
app = FastAPI()

# Serve Static Files (For Logo)
app.mount("/static", StaticFiles(directory="static"), name="static")

# ChatGPT API Key (Replace with your own)
OPENAI_API_KEY = "your_openai_api_key"

# Datasheet (Industry-Specific Information)
DATASHEET = {
    "EPC": "Engineering, Procurement, and Construction (EPC) involves designing and building infrastructure.",
    "Grid Load": "Grid Load Balancing ensures stable electricity distribution across networks.",
    "Oil & Gas": "The Oil & Gas sector focuses on extraction, refining, and distribution of fuels.",
    "Telecom": "Telecommunications includes 5G, fiber optics, and digital infrastructure.",
    "IPP": "Independent Power Producers (IPP) generate and sell electricity to utilities."
}

# HTML Page with Prompt Input & Logo
@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <html>
        <head>
            <title>AI Assistant</title>
        </head>
        <body>
            <div style="text-align: center;">
                <img src="/static/logo.jpg" alt="Company Logo" width="200"><br>
                <h2>Ask AI Anything</h2>
                <form action="/predict" method="get">
                    <input type="text" name="query" placeholder="Ask me anything...">
                    <button type="submit">Submit</button>
                </form>
            </div>
        </body>
    </html>
    """

# AI Prediction Route with Datasheet Support
@app.get("/predict")
async def predict(query: str = Query(..., description="Ask AI a question")):
    query_lower = query.lower()

    # Check Datasheet First
    for key in DATASHEET:
        if key.lower() in query_lower:
            return {"AI Response": DATASHEET[key]}

    # Search Google if No Datasheet Entry
    google_results = search_google(query)
    if google_results:
        return {"AI Response": google_results}

    # Ask ChatGPT if Google Fails
    chatgpt_response = ask_chatgpt(query)
    return {"AI Response": chatgpt_response}

# Google Search Integration
def search_google(query):
    try:
        response = requests.get(f"https://www.google.com/search?q={query}", headers={"User-Agent": "Mozilla/5.0"})
        return f"Google Search Results for '{query}': {response.url}"
    except Exception:
        return None

# ChatGPT API Integration
def ask_chatgpt(query):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": query}],
            api_key=OPENAI_API_KEY
        )
        return response["choices"][0]["message"]["content"]
    except Exception:
        return "ChatGPT is currently unavailable."

# Health Check
@app.get("/health")
async def health_check():
    return {"status": "API is running successfully"}

# Custom Error Handling
@app.exception_handler(Exception)
async def exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"error": "An unexpected error occurred. Please try again later."}
    )

# Run Uvicorn Server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
