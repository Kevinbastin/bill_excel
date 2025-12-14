import cv2
import numpy as np
from skimage import exposure
import logging


logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """Enhance image quality for better OCR accuracy"""

    @staticmethod
    def enhance_image(
        image: np.ndarray,
        enhance_contrast: bool = True,
        denoise: bool = True
    ) -> np.ndarray:
        """
        Enhance image for better OCR accuracy
        
        Processing steps:
        1. Convert to grayscale
        2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
        3. Denoising
        4. Adaptive thresholding
        
        Args:
            image: Input image (BGR or grayscale)
            enhance_contrast: Apply CLAHE contrast enhancement
            denoise: Apply denoising filter
            
        Returns:
            Enhanced image (grayscale or binary)
        """
        try:
            # Step 1: Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                logger.debug("✓ Converted to grayscale")
            else:
                gray = image

            # Step 2: Contrast enhancement (CLAHE)
            if enhance_contrast:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                gray = clahe.apply(gray)
                logger.debug("✓ CLAHE contrast enhancement applied")

            # Step 3: Denoising
            if denoise:
                gray = cv2.fastNlMeansDenoising(
                    gray,
                    h=10,
                    templateWindowSize=7,
                    searchWindowSize=21
                )
                logger.debug("✓ Denoising applied")

            # Step 4: Adaptive thresholding for better edge detection
            binary = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11,
                2
            )
            logger.debug("✓ Adaptive thresholding applied")
            
            return binary

        except Exception as e:
            logger.exception(f"❌ Preprocessing error: {e}")
            return image

    @staticmethod
    def auto_rotate_image(image: np.ndarray) -> np.ndarray:
        """
        Auto-rotate image to correct orientation using Hough line detection
        
        Detects skew angle from document edges and corrects rotation.
        
        Args:
            image: Input image (BGR)
            
        Returns:
            Rotated image
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            # Detect edges
            edges = cv2.Canny(gray, 50, 150)

            # Find lines using Hough transform
            lines = cv2.HoughLinesP(
                edges,
                rho=1,
                theta=np.pi / 180,
                threshold=100,
                minLineLength=100,
                maxLineGap=10,
            )

            if lines is None or len(lines) == 0:
                logger.debug("⚠️  No lines detected for rotation")
                return image

            # Extract angles from lines
            angles = []
            for line in lines:
                # HoughLinesP returns shape (1,4) per line
                x1, y1, x2, y2 = line.reshape(-1)[:4]

                # Calculate angle
                dx = x2 - x1
                dy = y2 - y1
                
                if dx == 0 and dy == 0:
                    continue

                angle = np.degrees(np.arctan2(dy, dx))

                # Keep only near-horizontal lines (reduce false rotations)
                if -45 <= angle <= 45:
                    angles.append(angle)

            if not angles:
                logger.debug("⚠️  No suitable angles detected")
                return image

            # Calculate median angle
            median_angle = float(np.median(angles))

            # Normalize angle
            if median_angle > 45:
                median_angle -= 90
            elif median_angle < -45:
                median_angle += 90

            # Only rotate if significant
            if abs(median_angle) > 2:
                h, w = image.shape[:2]
                rotation_matrix = cv2.getRotationMatrix2D(
                    (w / 2, h / 2),
                    median_angle,
                    1.0
                )
                rotated = cv2.warpAffine(
                    image,
                    rotation_matrix,
                    (w, h),
                    flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_REFLECT,
                )
                logger.debug(f"✓ Rotated {median_angle:.2f}°")
                return rotated
            else:
                logger.debug(f"⚠️  Rotation < 2° (skipped)")
                return image

        except Exception as e:
            logger.exception(f"❌ Auto-rotation error: {e}")
            return image

    @staticmethod
    def upscale_image(
        image: np.ndarray,
        scale_factor: float = 1.5
    ) -> np.ndarray:
        """
        Upscale small images for better OCR accuracy
        
        Small images benefit from upscaling before OCR.
        
        Args:
            image: Input image
            scale_factor: Scaling factor (1.5 = 1.5x larger)
            
        Returns:
            Upscaled image or original if already large
        """
        try:
            h, w = image.shape[:2]
            
            # Only upscale if smaller than threshold
            if h < 500 or w < 500:
                new_w = int(w * scale_factor)
                new_h = int(h * scale_factor)
                image = cv2.resize(
                    image,
                    (new_w, new_h),
                    interpolation=cv2.INTER_CUBIC
                )
                logger.debug(f"✓ Upscaled to {new_w}x{new_h}")
            else:
                logger.debug(f"ℹ️  Image already large ({w}x{h}), skipped upscaling")
            
            return image
            
        except Exception as e:
            logger.exception(f"❌ Upscaling error: {e}")
            return image
