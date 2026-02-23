import uvicorn
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from backend.app import app

# Paths relative to this file
BASE_DIR = Path(__file__).resolve().parent

# Mount static files
app.mount("/resources/images", StaticFiles(directory=str(BASE_DIR / "frontend" / "images")), name="images")
app.mount("/resources", StaticFiles(directory=str(BASE_DIR / "backend" / "resources")), name="resources")

@app.get("/")
def serve_ui():
    return FileResponse(str(BASE_DIR / "frontend" / "templates" / "ui.html"))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=10805)
