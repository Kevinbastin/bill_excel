from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask
import shutil
import os
from pathlib import Path
import logging
import json
from datetime import datetime

from config import UPLOAD_DIR, OUTPUT_DIR
from ocr.paddle_extractor import InvoiceExtractor
from ocr.excel_writer import ExcelWriter

logger = logging.getLogger(__name__)
router = APIRouter()

extractor = InvoiceExtractor(use_gpu=False)


def _remove_file(path: str) -> None:
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


@router.post("/extract")
async def extract_invoice(file: UploadFile = File(...)):
    """Extract invoice data and return Excel file"""
    upload_path = None
    excel_path = None

    try:
        allowed_types = {"image/png", "image/jpeg", "image/jpg", "application/pdf"}

        if not file.content_type:
            raise HTTPException(
                status_code=400,
                detail="Missing content_type from upload. Please upload a valid PNG/JPG/PDF file."
            )

        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file.content_type}. Allowed: PNG/JPG/JPEG/PDF"
            )

        # Save upload
        safe_name = Path(file.filename).name
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        upload_path = os.path.join(UPLOAD_DIR, safe_name)

        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info(f"Processing: filename={file.filename} content_type={file.content_type}")

        # Extract using OCR
        result = extractor.extract_from_image(upload_path)
        if result.get("status") != "success":
            raise HTTPException(status_code=500, detail=result.get("message", "Extraction failed"))

        # Save Excel
        excel_filename = f"{Path(safe_name).stem}_extracted.xlsx"
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        excel_path = os.path.join(OUTPUT_DIR, excel_filename)
        ExcelWriter.write_invoice_to_excel(result["data"], excel_path)

        # Save JSON metadata (optional)
        json_filename = f"{Path(safe_name).stem}_extracted.json"
        json_path = os.path.join(OUTPUT_DIR, json_filename)
        with open(json_path, "w") as f:
            json.dump(
                {
                    "timestamp": datetime.now().isoformat(),
                    "filename": file.filename,
                    "content_type": file.content_type,
                    "extraction_result": result["data"],
                },
                f,
                indent=2,
            )

        logger.info(f"Extracted: {excel_filename}")

        return FileResponse(
            excel_path,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            filename=excel_filename,
            background=BackgroundTask(_remove_file, excel_path),  # delete after response is sent [web:655]
        )

    except HTTPException as e:
        logger.warning(f"HTTPException {e.status_code}: {e.detail}")
        raise e

    except Exception as e:
        logger.exception("Unhandled server error")
        raise HTTPException(status_code=500, detail=repr(e))

    finally:
        # Remove uploaded file (safe; not used after OCR finishes)
        if upload_path and os.path.exists(upload_path):
            try:
                os.remove(upload_path)
            except Exception:
                pass


@router.get("/health")
def health_check():
    return {"status": "healthy", "service": "Invoice OCR API"}
