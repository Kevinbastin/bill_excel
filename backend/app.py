"""
FastAPI Application - Invoice OCR API
Main entry point for the backend server
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
from pathlib import Path

# ✅ CORRECT: Use relative imports from same package
from routes.extract import router as extract_router
from routes.health import router as health_router
from config import API_HOST, API_PORT


# ==================== LOGGING ====================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==================== FASTAPI APP ====================

app = FastAPI(
    title="Invoice OCR to Excel API",
    description="Extract invoice data using PaddleOCR PP-Structure",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)


# ==================== MIDDLEWARE ====================

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== ROUTERS ====================

# Include OCR extraction routes
app.include_router(extract_router, prefix="/api", tags=["OCR Extraction"])

# Include health check routes
app.include_router(health_router, tags=["Health"])


# ==================== ROOT ENDPOINT ====================

@app.get("/")
def root():
    """
    Root endpoint - API information
    """
    return {
        "name": "Invoice OCR to Excel API",
        "version": "2.0.0",
        "description": "Extract invoice data using PaddleOCR PP-Structure",
        "endpoints": {
            "extract_invoice": "POST /api/extract/invoice",
            "batch_extract": "POST /api/extract/batch",
            "health_check": "GET /api/health",
            "documentation": "GET /docs",
            "redoc": "GET /redoc"
        },
        "status": "🟢 Running"
    }


@app.get("/status")
def status():
    """
    System status endpoint
    """
    return {
        "status": "healthy",
        "service": "Invoice OCR API",
        "version": "2.0.0",
        "timestamp": __import__("datetime").datetime.now().isoformat()
    }


# ==================== ERROR HANDLERS ====================

@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Handle 404 errors"""
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not found",
            "message": f"Endpoint {request.url.path} does not exist",
            "documentation": "See /docs for available endpoints"
        }
    )


@app.exception_handler(500)
async def server_error_handler(request, exc):
    """Handle 500 errors"""
    logger.exception(f"Server error on {request.url.path}: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Server error",
            "message": "An unexpected error occurred",
            "contact": "Check server logs for details"
        }
    )


# ==================== STARTUP/SHUTDOWN ====================

@app.on_event("startup")
async def startup_event():
    """
    Tasks to run on application startup
    """
    logger.info("="*80)
    logger.info("🚀 Invoice OCR API Starting")
    logger.info("="*80)
    logger.info(f"📝 Version: 2.0.0")
    logger.info(f"🌐 Host: {API_HOST}")
    logger.info(f"🔌 Port: {API_PORT}")
    logger.info(f"📚 API Docs: http://{API_HOST}:{API_PORT}/docs")
    logger.info("="*80)


@app.on_event("shutdown")
async def shutdown_event():
    """
    Tasks to run on application shutdown
    """
    logger.info("🛑 Invoice OCR API Shutting down")


# ==================== MAIN ====================

if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"Starting FastAPI on {API_HOST}:{API_PORT}")
    logger.info(f"Documentation: http://{API_HOST}:{API_PORT}/docs")
    
    uvicorn.run(
        app,
        host=API_HOST,
        port=API_PORT,
        log_level="info",
        reload=True  # Set to False in production
    )
