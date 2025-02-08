# Standard library imports
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Local imports
from src.api.barista import router as barista_router
from src.api.data_loader import router as data_loader_router

APP_CONFIG = {
    "TEMP_DIR": "temp_uploads",
    "OUTPUT_TYPE": "html",
    "HOST": "127.0.0.1",
    "PORT": 8000,
}

# get constants
HOST = APP_CONFIG["HOST"]
PORT = APP_CONFIG["PORT"]

# create the fastapi app
app = FastAPI(
    title="Resume Parser",
    description="Advanced Resume Parsing API with LLM capabilities",
    version="1.0.0",
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(barista_router, prefix="/api/v1")
app.include_router(data_loader_router, prefix="/api/v1")


if __name__ == "__main__":
    uvicorn.run("main:app", host=HOST, port=PORT, log_level="info", reload=True)
