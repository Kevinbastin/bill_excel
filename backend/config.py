import os
from dotenv import load_dotenv

load_dotenv()

# ===== PaddleOCR Configuration =====
PADDLE_USE_GPU = os.getenv("PADDLE_USE_GPU", "False").lower() == "true"
PADDLE_USE_GPN = os.getenv("PADDLE_USE_GPN", "False").lower() == "true"

# ===== File Paths =====
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./outputs")
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB

# ===== API Configuration =====
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# ===== Image Processing =====
ENABLE_PREPROCESSING = os.getenv("ENABLE_PREPROCESSING", "True").lower() == "true"
PREPROCESSING_ENHANCE_CONTRAST = os.getenv("PREPROCESSING_ENHANCE_CONTRAST", "True").lower() == "true"
PREPROCESSING_DENOISE = os.getenv("PREPROCESSING_DENOISE", "True").lower() == "true"

# Create directories
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"✓ Config Loaded | GPU: {PADDLE_USE_GPU} | Preprocessing: {ENABLE_PREPROCESSING}")
