import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from api import app

# ✅ Serve static files (HTML, CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
