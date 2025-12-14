from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask
import shutil
import os
from pathlib import Path
import logging
import json
from datetime import datetime
from typing import Optional


from config import UPLOAD_DIR, OUTPUT_DIR
from ocr.paddle_extractor import PaddleExtractor
from ocr.excel_writer import ExcelWriter


logger = logging.getLogger(__name__)
router = APIRouter()


# Initialize OCR extractor (persistent across requests)
try:
    extractor = PaddleExtractor(use_gpu=False)  # Set to True if you have GPU
    logger.info("✅ PaddleExtractor initialized")
except Exception as e:
    logger.exception(f"❌ Failed to initialize PaddleExtractor: {e}")
    extractor = None


# ==================== UTILITY FUNCTIONS ====================

def _remove_file(path: Optional[str]) -> None:
    """Safely remove file"""
    try:
        if path and os.path.exists(path):
            os.remove(path)
            logger.debug(f"✓ Removed temporary file: {path}")
    except Exception as e:
        logger.warning(f"⚠️  Could not remove file {path}: {e}")


def _ensure_dirs() -> None:
    """Ensure output directories exist"""
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==================== API ENDPOINTS ====================

@router.post("/extract/invoice")
async def extract_invoice(file: UploadFile = File(...)):
    """
    Extract invoice data and return Excel file
    
    Args:
        file: Uploaded invoice image (PNG, JPG, JPEG, PDF)
        
    Returns:
        Excel file with extracted data
        
    Raises:
        HTTPException: On validation or processing error
    """
    upload_path = None
    excel_path = None

    try:
        # Check if extractor initialized
        if extractor is None:
            raise HTTPException(
                status_code=500,
                detail="OCR engine not initialized. Check server logs."
            )

        # Validate file type
        allowed_types = {
            "image/png",
            "image/jpeg",
            "image/jpg",
            "application/pdf"
        }

        if not file.content_type:
            raise HTTPException(
                status_code=400,
                detail="Missing content_type. Please upload a PNG/JPG/PDF file."
            )

        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file.content_type}. "
                        f"Allowed: PNG, JPG, JPEG, PDF"
            )

        # Validate file size (max 50MB)
        max_size = 50 * 1024 * 1024  # 50MB
        if file.size and file.size > max_size:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Max size: 50MB. Got: {file.size // 1024 // 1024}MB"
            )

        # Ensure directories exist
        _ensure_dirs()

        # Save uploaded file
        safe_name = Path(file.filename).name
        upload_path = os.path.join(UPLOAD_DIR, safe_name)

        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info(f"📤 File uploaded: {file.filename} ({file.content_type})")

        # Extract using OCR
        logger.info("🔄 Starting extraction...")
        result = extractor.extract_from_image(upload_path)
        
        if result.get("status") != "success":
            error_msg = result.get("message", "Extraction failed")
            logger.error(f"❌ Extraction failed: {error_msg}")
            raise HTTPException(
                status_code=500,
                detail=f"Extraction failed: {error_msg}"
            )

        logger.info(f"✅ Extraction successful")
        extracted_data = result["data"]

        # Save Excel file
        excel_filename = f"{Path(safe_name).stem}_extracted.xlsx"
        excel_path = os.path.join(OUTPUT_DIR, excel_filename)
        
        ExcelWriter.write_invoice_to_excel(extracted_data, excel_path)
        logger.info(f"✅ Excel file created: {excel_filename}")

        # Save JSON metadata (optional, for debugging)
        json_filename = f"{Path(safe_name).stem}_extracted.json"
        json_path = os.path.join(OUTPUT_DIR, json_filename)
        
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "filename": file.filename,
                        "content_type": file.content_type,
                        "extraction_result": extracted_data,
                    },
                    f,
                    indent=2,
                    ensure_ascii=False
                )
            logger.info(f"✅ JSON metadata saved: {json_filename}")
        except Exception as e:
            logger.warning(f"⚠️  Could not save JSON metadata: {e}")

        # Log summary
        logger.info(
            f"📊 Extraction summary: "
            f"tables={extracted_data.get('table_count', 0)}, "
            f"text_items={extracted_data.get('text_items', 0)}, "
            f"confidence={extracted_data.get('confidence_score', 0):.1%}"
        )

        # Return Excel file
        return FileResponse(
            excel_path,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            filename=excel_filename,
            background=BackgroundTask(_remove_file, excel_path),
        )

    except HTTPException as e:
        logger.warning(f"HTTPException {e.status_code}: {e.detail}")
        raise e

    except Exception as e:
        logger.exception(f"❌ Unhandled server error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Server error: {repr(e)}"
        )

    finally:
        # Clean up uploaded file
        if upload_path and os.path.exists(upload_path):
            try:
                os.remove(upload_path)
                logger.debug(f"✓ Cleaned up upload: {upload_path}")
            except Exception as e:
                logger.warning(f"⚠️  Could not clean upload: {e}")


@router.post("/extract/batch")
async def extract_batch(files: list[UploadFile] = File(...)):
    """
    Extract multiple invoices (batch processing)
    
    Args:
        files: Multiple invoice images
        
    Returns:
        JSON with extraction results for all files
    """
    results = []
    
    for file in files:
        try:
            logger.info(f"Processing: {file.filename}")
            # Call single extraction for each file
            # Note: In production, use async task queue for large batches
            result = await extract_invoice(file)
            results.append({
                "filename": file.filename,
                "status": "success"
            })
        except Exception as e:
            logger.error(f"Batch extraction failed for {file.filename}: {e}")
            results.append({
                "filename": file.filename,
                "status": "error",
                "error": str(e)
            })
    
    return {"results": results}


@router.get("/health")
def health_check():
    """
    Health check endpoint
    
    Returns:
        Health status and service info
    """
    status = "healthy" if extractor is not None else "degraded"
    return {
        "status": status,
        "service": "Invoice OCR API",
        "version": "1.0.0",
        "ocr_engine": "PaddleOCR PP-Structure"
    }


@router.get("/")
def root():
    """
    API root endpoint
    
    Returns:
        API info
    """
    return {
        "name": "Invoice OCR API",
        "version": "1.0.0",
        "endpoints": {
            "POST /api/extract/invoice": "Extract single invoice",
            "POST /api/extract/batch": "Extract multiple invoices",
            "GET /api/health": "Health check"
        },
        "docs": "/docs",
        "redoc": "/redoc"
    }
