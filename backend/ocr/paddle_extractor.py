import os
import re
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from io import StringIO
from paddleocr import PPStructure

from .image_processor import ImagePreprocessor
from config import ENABLE_PREPROCESSING

logger = logging.getLogger(__name__)


class PaddleExtractor:
    """
    PP-Structure table extractor (restored first-pass behavior + optional crop second-pass).

    - First pass: same config that worked for you (lang="ch", recovery=False, layout=True).
    - Second pass: optional, runs on cropped table bbox (layout=False, recovery=True) only if needed.
    - Returns both 'canonicalrows' and 'canonical_rows' to satisfy ExcelWriter.
    """

    HEADER_KEYWORDS = (
        "id", "item", "code", "desc", "description",
        "qty", "quantity", "rate", "price", "amount", "total",
        "product", "service", "hsn", "sac", "taxable", "igst", "cgst", "sgst", "cess"
    )

    def __init__(self, use_gpu: bool = False):
        self.use_gpu = use_gpu
        self.preprocessor = ImagePreprocessor()

        # First pass: keep your original working config
        self.doc_engine = PPStructure(
            use_cuda=use_gpu,
            use_pdf2image_cuda=use_gpu,
            lang="ch",          # your previous default that worked
            recovery=False,     # previous default
            layout=True,
            table=True,
            ocr=True,
        )

        # Second pass: table-only engine (used only if first pass fails)
        self.table_engine = PPStructure(
            use_cuda=use_gpu,
            use_pdf2image_cuda=use_gpu,
            lang="ch",          # stay consistent to minimize regressions
            recovery=True,      # stronger table recovery when zoomed on table
            layout=False,
            table=True,
            ocr=True,
        )

        logger.info(f"✅ PaddleOCR PP-Structure initialized | GPU: {use_gpu}")

    # ==================== MAIN ====================

    def extract_from_image(self, image_path: str) -> Dict[str, Any]:
        preprocessed_path = None
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Failed to read image: {image_path}")
            logger.info(f"📸 Image loaded: {img.shape}")

            if ENABLE_PREPROCESSING:
                logger.info("🔄 Preprocessing enabled...")
                img = self.preprocessor.auto_rotate_image(img)
                img = self.preprocessor.enhance_image(img)
                p = Path(image_path)
                preprocessed_path = str(p.with_name(p.stem + "_preprocessed" + p.suffix))
                cv2.imwrite(preprocessed_path, img)
                logger.info("✅ Preprocessing complete")

            logger.info("🔄 Running PP-Structure extraction (first pass)...")
            doc_result = self.doc_engine(img, return_ocr_result_in_table=True)
            logger.info(f"✅ PP-Structure complete: {len(doc_result)} regions found")

            # Safe: img is optional; second pass will use it if provided
            table_blocks = self.extract_and_reconstruct_tables(doc_result, img)
            extracted_data = self._parse_results(doc_result, table_blocks)
            return {"status": "success", "data": extracted_data}

        except Exception as e:
            logger.exception(f"❌ Extraction error: {e}")
            return {"status": "error", "message": str(e)}
        finally:
            if preprocessed_path and os.path.exists(preprocessed_path):
                try:
                    os.remove(preprocessed_path)
                except Exception as e:
                    logger.warning(f"⚠️ Could not remove temp file: {e}")

    # ==================== TABLES ====================

    def extract_and_reconstruct_tables(
        self,
        docresult: List[Dict[str, Any]],
        img: Optional[np.ndarray] = None
    ) -> List[Dict[str, Any]]:
        """
        1) First-pass region: rec_res + cell_bbox/boxes -> grid
        2) If grid missing and image available: crop bbox and run second-pass engine
        3) Else HTML
        4) Else tokens fallback
        """
        out: List[Dict[str, Any]] = []
        logger.info("=" * 80)
        logger.info("📊 TABLE EXTRACTION")
        logger.info("=" * 80)

        table_count = 0
        for region in docresult:
            if region.get("type") != "table":
                continue

            table_count += 1
            logger.info(f"\n📋 TABLE {table_count}")

            res = region.get("res") or {}
            html = res.get("html")
            rec_res = res.get("rec_res") or res.get("recres") or []
            boxes = res.get("boxes") or res.get("box") or []
            cell_bbox = res.get("cell_bbox") or res.get("cellBbox") or []

            canonical_rows: Optional[List[List[str]]] = None
            raw_grid: Optional[List[List[str]]] = None

            # METHOD 1: first-pass cell reconstruction
            cells = self._cells_from_pp_table(rec_res, cell_bbox=cell_bbox, boxes=boxes)
            if cells:
                logger.info(f"  🔄 METHOD 1: OCR cells (cells={len(cells)})...")
                raw_grid = self._grid_from_cells(cells)
                if raw_grid:
                    canonical_rows = self._ensure_header_row(raw_grid)
                    logger.info(f"  ✅ SUCCESS: {len(canonical_rows)} rows × {max(len(r) for r in canonical_rows)} cols")

            # METHOD 1b: optional second pass on crop if first pass failed
            if canonical_rows is None and img is not None:
                crop = self._safe_crop(img, region.get("bbox"), pad=10)
                if crop is not None:
                    try:
                        logger.info("  🔄 METHOD 1b: second-pass table_engine on crop...")
                        second = self.table_engine(crop, return_ocr_result_in_table=True)
                        res2 = self._first_table_res(second)
                        if res2:
                            rec_res2 = res2.get("rec_res") or res2.get("recres") or []
                            boxes2 = res2.get("boxes") or res2.get("box") or []
                            cell_bbox2 = res2.get("cell_bbox") or res2.get("cellBbox") or []
                            cells2 = self._cells_from_pp_table(rec_res2, cell_bbox=cell_bbox2, boxes=boxes2)
                            if cells2:
                                raw_grid = self._grid_from_cells(cells2)
                                if raw_grid:
                                    canonical_rows = self._ensure_header_row(raw_grid)
                                    logger.info("  ✅ SECOND PASS SUCCESS")
                    except Exception as e:
                        logger.debug(f"  ⚠️ Second pass failed: {e}")

            # METHOD 2: HTML
            if canonical_rows is None and html:
                logger.info(f"  🔄 METHOD 2: HTML ({len(html)} chars)...")
                raw_grid = self._parse_html_table(html)
                if raw_grid:
                    canonical_rows = self._ensure_header_row(raw_grid)
                    logger.info(f"  ✅ HTML rows: {len(canonical_rows)}")

            # METHOD 3: tokens
            if canonical_rows is None and rec_res:
                logger.info("  🔄 METHOD 3: token fallback...")
                tokens = self._tokens_from_rec_res(rec_res)
                if tokens:
                    raw_grid = [["Item", "Text"]] + [[str(i + 1), t] for i, t in enumerate(tokens)]
                    canonical_rows = raw_grid
                    logger.warning(f"  ⚠️ FALLBACK tokens: {len(tokens)}")

            if canonical_rows is None:
                logger.error(f"  ❌ TABLE {table_count}: no usable grid")
                continue

            table_block = {
                "canonicalrows": canonical_rows,
                "canonical_rows": canonical_rows,
                "raw_grid": raw_grid or canonical_rows,
                "html": self.grid_to_html(canonical_rows),
                "rec_res": rec_res,
                "cell_bbox": cell_bbox,
                "boxes": boxes,
                "row_count": len(canonical_rows),
                "col_count": max((len(r) for r in canonical_rows), default=0),
                "bbox": region.get("bbox"),
            }
            out.append(table_block)
            logger.info(f"  📤 Added table {table_count}")

        logger.info("\n" + "=" * 80)
        logger.info(f"📊 Total tables extracted: {len(out)}")
        logger.info("=" * 80)
        return out

    # ==================== HELPERS ====================

    @staticmethod
    def _first_table_res(result: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(result, list):
            return None
        for r in result:
            if isinstance(r, dict) and r.get("type") == "table":
                return r.get("res") or {}
        return None

    @staticmethod
    def _safe_crop(img: np.ndarray, bbox: Any, pad: int = 0) -> Optional[np.ndarray]:
        if img is None or bbox is None or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return None
        h, w = img.shape[:2]
        try:
            x1, y1, x2, y2 = map(int, bbox)
        except Exception:
            return None
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
        if x2 <= x1 or y2 <= y1:
            return None
        return img[y1:y2, x1:x2].copy()

    def _cells_from_pp_table(self, rec_res: List[Any], cell_bbox: List[Any], boxes: Any) -> List[Dict[str, Any]]:
        if not rec_res:
            return []
        bbox_list = None
        if isinstance(cell_bbox, list) and len(cell_bbox) == len(rec_res):
            bbox_list = cell_bbox
        elif isinstance(boxes, (list, tuple)) and len(boxes) == len(rec_res):
            bbox_list = boxes
        if bbox_list is None:
            return []

        cells: List[Dict[str, Any]] = []
        for t, b in zip(rec_res, bbox_list):
            text, score = self._extract_text_score(t)
            if not text:
                continue
            xyxy = self._bbox_to_xyxy(b)
            if xyxy is None:
                continue
            x1, y1, x2, y2 = xyxy
            cells.append(
                {"text": text, "score": score, "x1": float(x1), "y1": float(y1),
                 "x2": float(x2), "y2": float(y2), "xc": (x1 + x2) / 2.0, "yc": (y1 + y2) / 2.0}
            )
        return cells

    def _grid_from_cells(self, cells: List[Dict[str, Any]]) -> Optional[List[List[str]]]:
        if not cells:
            return None
        cells_sorted = sorted(cells, key=lambda c: (c["yc"], c["xc"]))
        y_thr = self._auto_y_threshold([c["yc"] for c in cells_sorted])

        rows: List[List[Dict[str, Any]]] = []
        cur: List[Dict[str, Any]] = [cells_sorted[0]]
        cur_y = cells_sorted[0]["yc"]
        for cell in cells_sorted[1:]:
            if abs(cell["yc"] - cur_y) <= y_thr:
                cur.append(cell)
                cur_y = float(np.median([c["yc"] for c in cur]))
            else:
                rows.append(cur); cur = [cell]; cur_y = cell["yc"]
        rows.append(cur)
        for r in rows:
            r.sort(key=lambda c: c["xc"])

        anchor_row = max(rows, key=lambda r: len(r))
        col_x = [c["xc"] for c in anchor_row]
        if not col_x:
            return None
        col_x = self._merge_close_1d(col_x, min_gap=max(8.0, float(np.median(np.diff(sorted(col_x))) if len(col_x) > 1 else 15.0) * 0.35))
        ncols = len(col_x)
        if ncols <= 0:
            return None

        grid: List[List[str]] = []
        for r in rows:
            row_out = [""] * ncols
            for cell in r:
                j = int(np.argmin([abs(cell["xc"] - x) for x in col_x]))
                row_out[j] = (row_out[j] + " " + cell["text"]).strip() if row_out[j] else cell["text"]
            if any(v.strip() for v in row_out):
                grid.append(row_out)
        return grid if grid else None

    @staticmethod
    def _merge_close_1d(xs: List[float], min_gap: float) -> List[float]:
        xs = sorted(xs)
        out = [xs[0]]
        for x in xs[1:]:
            if abs(x - out[-1]) <= min_gap:
                out[-1] = (out[-1] + x) / 2.0
            else:
                out.append(x)
        return out

    @staticmethod
    def _auto_y_threshold(y_coords: List[float]) -> float:
        if not y_coords or len(y_coords) < 2:
            return 15.0
        ys = sorted(set(float(y) for y in y_coords))
        if len(ys) < 2:
            return 15.0
        diffs = np.diff(ys)
        med = float(np.median(diffs)) if len(diffs) else 15.0
        return max(6.0, min(30.0, med * 0.45))

    # ---------- HTML / tokens / fields (unchanged) ----------

    def _ensure_header_row(self, grid: List[List[str]]) -> List[List[str]]:
        if not grid:
            return [["COL1"]]
        first = [self._clean_cell(x) for x in grid[0]]
        if self._looks_like_header(first):
            return [first] + [[self._clean_cell(x) for x in r] for r in grid[1:]]
        ncols = max((len(r) for r in grid), default=len(first))
        header = [f"COL{i+1}" for i in range(ncols)]
        fixed: List[List[str]] = [header]
        for r in grid:
            rr = [self._clean_cell(x) for x in r] + [""] * (ncols - len(r))
            fixed.append(rr[:ncols])
        return fixed

    def _looks_like_header(self, row: List[str]) -> bool:
        text = " ".join(row).lower()
        hits = sum(1 for kw in self.HEADER_KEYWORDS if kw in text)
        return hits >= 2

    @staticmethod
    def _clean_cell(x: Any) -> str:
        if x is None:
            return ""
        return str(x).strip()

    @staticmethod
    def _extract_text_score(item: Any) -> Tuple[str, Optional[float]]:
        if item is None:
            return "", None
        if isinstance(item, dict):
            txt = item.get("text") or item.get("transcription") or ""
            score = item.get("score") or item.get("confidence")
            try:
                score = float(score) if score is not None else None
            except Exception:
                score = None
            return str(txt).strip(), score
        if isinstance(item, (list, tuple)):
            if len(item) >= 2 and isinstance(item[0], str):
                txt = item[0]; score = item[1]
                try: score = float(score) if score is not None else None
                except Exception: score = None
                return str(txt).strip(), score
            if len(item) >= 1 and isinstance(item[0], (list, tuple)):
                inner = item[0]
                if len(inner) >= 2 and isinstance(inner[0], str):
                    txt = inner[0]; score = inner[1]
                    try: score = float(score) if score is not None else None
                    except Exception: score = None
                    return str(txt).strip(), score
            if len(item) >= 1 and isinstance(item[0], str):
                return item[0].strip(), None
        return "", None

    @staticmethod
    def _bbox_to_xyxy(bbox: Any) -> Optional[Tuple[float, float, float, float]]:
        if bbox is None:
            return None
        if isinstance(bbox, (list, tuple)) and len(bbox) == 8 and all(isinstance(v, (int, float)) for v in bbox):
            xs = [float(bbox[i]) for i in range(0, 8, 2)]
            ys = [float(bbox[i]) for i in range(1, 8, 2)]
            return min(xs), min(ys), max(xs), max(ys)
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4 and all(isinstance(v, (int, float)) for v in bbox):
            x1, y1, x2, y2 = bbox
            return float(x1), float(y1), float(x2), float(y2)
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4 and all(isinstance(p, (list, tuple)) and len(p) == 2 for p in bbox):
            xs = [float(p[0]) for p in bbox]; ys = [float(p[1]) for p in bbox]
            return min(xs), min(ys), max(xs), max(ys)
        return None

    def _parse_html_table(self, html: str) -> Optional[List[List[str]]]:
        try:
            if not html or "<table" not in html.lower():
                return None
            dfs = pd.read_html(StringIO(html))
            if not dfs:
                return None
            df = dfs[0]
            header = [str(c).strip() for c in df.columns]
            rows_list = [header]
            for _, row in df.iterrows():
                rows_list.append([str(v).strip() for v in row.values])
            return rows_list if len(rows_list) > 1 else None
        except Exception as e:
            logger.debug(f"⚠️ HTML parsing failed: {e}")
            return None

    @staticmethod
    def _tokens_from_rec_res(rec_res: List[Any]) -> List[str]:
        tokens: List[str] = []
        for item in rec_res or []:
            txt, _ = PaddleExtractor._extract_text_score(item)
            if txt:
                tokens.append(txt)
        return tokens

    def _parse_results(self, doc_result: List[Dict[str, Any]], table_blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
        text_lines: List[str] = []
        confidences: List[float] = []
        for region in doc_result:
            if region.get("type") == "table":
                continue
            region_res = region.get("res")
            if isinstance(region_res, tuple) and len(region_res) == 2:
                _boxes, rec_res = region_res
                for item in rec_res or []:
                    txt, score = self._extract_text_score(item)
                    if txt: text_lines.append(txt)
                    if score is not None: confidences.append(float(score))
            elif isinstance(region_res, list):
                for item in region_res:
                    txt, score = self._extract_text_score(item)
                    if txt: text_lines.append(txt)
                    if score is not None: confidences.append(float(score))
            elif isinstance(region_res, dict):
                txt = region_res.get("text") or region_res.get("transcription")
                if txt: text_lines.append(str(txt).strip())
                score = region_res.get("score") or region_res.get("confidence")
                if score is not None:
                    try: confidences.append(float(score))
                    except Exception: pass
        full_text_str = "\n".join([t for t in text_lines if t]).strip()
        fields = self._extract_key_fields(full_text_str)
        avg_confidence = float(np.mean(confidences)) if confidences else 0.0
        return {
            "tables": table_blocks,
            "full_text": full_text_str,
            "fields": fields,
            "confidence_score": avg_confidence,
            "table_count": len(table_blocks),
            "text_items": len(text_lines),
        }
    @staticmethod
    def grid_to_html(grid: List[List[str]]) -> str:
        if not grid:
            return ""
        html_parts = ['<table border="1" cellpadding="5" cellspacing="0">']
        for row in grid:
            html_parts.append("<tr>")
            for cell_text in row:
                safe_text = (
                    str(cell_text)
                    .replace("&", "&amp;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;")
                    .replace('"', "&quot;")
                )
                html_parts.append(f"<td>{safe_text}</td>")
            html_parts.append("</tr>")
        html_parts.append("</table>")
        return "\n".join(html_parts)

    @staticmethod
    def _extract_key_fields(text: str) -> Dict[str, Optional[str]]:
        fields: Dict[str, Optional[str]] = {}
        inv_patterns = [
            r"(?:Invoice|Bill|Invoice No|Inv|INV)[\s#:]*([A-Z0-9\-\/]+)",
            r"(?:Invoice Number)[\s:]*([0-9A-Z\-\/]+)",
        ]
        fields["invoice_number"] = None
        for pattern in inv_patterns:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                fields["invoice_number"] = m.group(1).strip()
                break
        date_patterns = [
            r"(?:Date|Dated)[\s:]*(\d{1,2}[-/]\d{1,2}[-/]\d{4})",
            r"(\d{4}-\d{1,2}-\d{1,2})",
        ]
        fields["date"] = None
        for pattern in date_patterns:
            m = re.search(pattern, text)
            if m:
                fields["date"] = m.group(1)
                break
        amount_patterns = [
            r"(?:Total|Grand Total|Amount Due|TOTAL)[\s:]*[₹$]?\s*([0-9,]+\.?\d*)",
            r"[₹$]\s*([0-9,]+\.?\d*)",
        ]
        fields["total_amount"] = None
        for pattern in amount_patterns:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                fields["total_amount"] = m.group(1).strip()
                break
        gstin_match = re.search(r"(?:GSTIN|GST)[\s:]*([0-9A-Z]{15})", text, re.IGNORECASE)
        fields["gstin"] = gstin_match.group(1) if gstin_match else None
        po_match = re.search(r"(?:PO|P\.O\.|Purchase Order)[\s#:]*([A-Z0-9\-]+)", text, re.IGNORECASE)
        fields["po_number"] = po_match.group(1) if po_match else None
        due_date_patterns = [r"(?:Due Date|DueDate|Payment Due)[\s:]*(\d{1,2}[-/]\d{1,2}[-/]\d{4})"]
        fields["due_date"] = None
        for pattern in due_date_patterns:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                fields["due_date"] = m.group(1)
                break
        vendor_lines = [line.strip() for line in text.split("\n") if len(line.strip()) > 5][:2]
        fields["vendor"] = " ".join(vendor_lines) if vendor_lines else None
        return fields
