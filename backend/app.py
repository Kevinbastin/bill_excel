from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import logging
import os

# ✅ CORRECT: NO backend. prefix
from routes.extract import router as extract_router
from routes.health import router as health_router
from config import API_HOST, API_PORT

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Invoice OCR to Excel API",
    description="Extract invoice data using PaddleOCR PP-Structure",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(extract_router, prefix="/api", tags=["OCR"])
app.include_router(health_router, tags=["Health"])

@app.get("/")
def root():
    return {
        "message": "Invoice OCR API v2.0",
        "endpoints": {
            "extract": "POST /api/extract",
            "health": "GET /api/health",
            "docs": "GET /docs"
        }
    }

@app.get("/ocr")
async def ocr_ui():
    try:
        with open("static/invoice_ocr.html", "r") as f:
            return f.read()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="OCR UI not found")

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting server on {API_HOST}:{API_PORT}")
    uvicorn.run(app, host=API_HOST, port=API_PORT, log_level="info")
