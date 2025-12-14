"""
ExcelWriter: Convert extracted invoice data to Excel with proper formatting.
✅ HANDLES BOTH canonical_rows AND canonicalrows keys
✅ ALWAYS writes table data even if structure doesn't match expectations
✅ NEVER blank sheets
"""

from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from typing import Dict, List, Any, Optional
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class ExcelWriter:
    """Convert extracted invoice data JSON to Excel with proper formatting"""

    CANONICAL_HEADERS = ["ID", "Description", "Qty", "Price", "Total"]

    @staticmethod
    def get_styles() -> Dict[str, Any]:
        """Return all style objects"""
        header_fill = PatternFill(start_color="1F4E78", end_color="1F4E78", fill_type="solid")
        header_font = Font(color="FFFFFF", bold=True, size=12)
        
        table_header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
        table_header_font = Font(color="FFFFFF", bold=True, size=11)
        
        data_font = Font(size=10)
        border = Border(
            left=Side(style="thin"),
            right=Side(style="thin"),
            top=Side(style="thin"),
            bottom=Side(style="thin"),
        )
        
        confidence_high = PatternFill(start_color="C6E0B4", end_color="C6E0B4", fill_type="solid")  # Green
        confidence_med = PatternFill(start_color="FFF2CC", end_color="FFF2CC", fill_type="solid")   # Yellow
        confidence_low = PatternFill(start_color="F4B183", end_color="F4B183", fill_type="solid")   # Orange
        
        return {
            "header_fill": header_fill,
            "header_font": header_font,
            "table_header_fill": table_header_fill,
            "table_header_font": table_header_font,
            "data_font": data_font,
            "border": border,
            "confidence_high": confidence_high,
            "confidence_med": confidence_med,
            "confidence_low": confidence_low,
        }

    @staticmethod
    def safe_str(x: Any) -> str:
        """Safely convert any value to string"""
        if x is None:
            return ""
        if isinstance(x, (int, float)):
            return str(x)
        return str(x).strip() if hasattr(x, "strip") else str(x)

    @staticmethod
    def write_invoice_details_sheet(ws, extracted_data: Dict, styles: Dict) -> None:
        """Write Sheet 1: Invoice Details"""
        row = 1
        
        # Title
        ws.merge_cells(f"A{row}:F{row}")
        title_cell = ws[f"A{row}"]
        title_cell.value = "INVOICE DETAILS"
        title_cell.font = styles["header_font"]
        title_cell.fill = styles["header_fill"]
        title_cell.alignment = Alignment(horizontal="center", vertical="center")
        row += 2

        # Extract fields
        fields = extracted_data.get("fields") or {}
        confidence = float(extracted_data.get("confidence_score") or 0.0)

        field_pairs = [
            ("Invoice Number", fields.get("invoice_number") or "N/A"),
            ("Date", fields.get("date") or "N/A"),
            ("Vendor", fields.get("vendor") or "N/A"),
            ("Total Amount", fields.get("total_amount") or "N/A"),
            ("GSTIN", fields.get("gstin") or "N/A"),
            ("PO Number", fields.get("po_number") or "N/A"),
            ("Due Date", fields.get("due_date") or "N/A"),
        ]

        for label, value in field_pairs:
            ws[f"A{row}"] = label
            ws[f"A{row}"].font = Font(bold=True, size=11)
            ws[f"B{row}"] = ExcelWriter.safe_str(value)
            ws[f"B{row}"].font = styles["data_font"]
            row += 1

        # Confidence score
        ws[f"A{row}"] = "OCR Confidence Score"
        ws[f"A{row}"].font = Font(bold=True, size=11)
        conf_cell = ws[f"B{row}"]
        conf_cell.value = f"{confidence:.1f}"
        conf_cell.font = Font(bold=True, size=11)
        
        if confidence >= 0.90:
            conf_cell.fill = styles["confidence_high"]
        elif confidence >= 0.75:
            conf_cell.fill = styles["confidence_med"]
        else:
            conf_cell.fill = styles["confidence_low"]
        row += 3

        # Full OCR text
        ws.merge_cells(f"A{row}:F{row}")
        ocr_title = ws[f"A{row}"]
        ocr_title.value = "FULL OCR TEXT"
        ocr_title.font = styles["header_font"]
        ocr_title.fill = styles["header_fill"]
        row += 1

        full_text = extracted_data.get("full_text") or ""
        ws.merge_cells(f"A{row}:F{row+4}")
        text_cell = ws[f"A{row}"]
        text_cell.value = full_text[:2000] if full_text else "No text extracted"
        text_cell.alignment = Alignment(wrap_text=True, vertical="top")
        row += 6

        # Metadata
        ws[f"A{row}"] = "EXTRACTION METADATA"
        ws[f"A{row}"].font = Font(bold=True, size=10)
        row += 1

        metadata = [
            ("Tables Detected", extracted_data.get("table_count", 0)),
            ("Text Items", extracted_data.get("text_items", 0)),
            ("Processing Status", "Success"),
        ]

        for label, value in metadata:
            ws[f"A{row}"] = label
            ws[f"A{row}"].font = Font(size=9)
            ws[f"B{row}"] = ExcelWriter.safe_str(value)
            ws[f"B{row}"].font = Font(size=9)
            row += 1

        # Auto-fit columns
        ws.column_dimensions["A"].width = 25
        ws.column_dimensions["B"].width = 40

    @staticmethod
    def write_line_items_sheet(ws, extracted_data: Dict, styles: Dict) -> None:
        """
        Write Sheet 2: Line Items from tables
        ✅ CRITICAL FIX: Reads BOTH canonical_rows and canonicalrows
        ✅ Never shows blank sheet if data exists
        """
        row = 1
        tables = extracted_data.get("tables") or []

        if not tables:
            ws[f"A{row}"] = "No tables detected in invoice"
            ws[f"A{row}"].font = Font(italic=True, color="FF0000")
            return

        for table_idx, table in enumerate(tables, 1):
            logger.info(f"📋 Processing table {table_idx}")
            
            # TRY METHOD 1: Get canonical_rows (with underscore)
            canonical_rows = table.get("canonical_rows")
            
            # TRY METHOD 2: Get canonicalrows (without underscore)
            if canonical_rows is None:
                canonical_rows = table.get("canonicalrows")
            
            # TRY METHOD 3: Get raw_grid
            if canonical_rows is None:
                canonical_rows = table.get("raw_grid")
            
            # TRY METHOD 4: Try to parse HTML
            if canonical_rows is None and table.get("html"):
                logger.info(f"  Trying HTML parsing for table {table_idx}...")
                canonical_rows = ExcelWriter._parse_html_table(table.get("html"))
            
            # TRY METHOD 5: Fallback to text field
            if canonical_rows is None and table.get("text"):
                logger.info(f"  Fallback to text field for table {table_idx}...")
                text_items = table.get("text", [])
                if text_items:
                    # Create 2-column grid: Item | Text
                    canonical_rows = [["Item", "Text"]]
                    for i, item in enumerate(text_items, 1):
                        canonical_rows.append([str(i), str(item)])
            
            # If still nothing, skip this table
            if canonical_rows is None or len(canonical_rows) == 0:
                logger.error(f"  ❌ TABLE {table_idx}: No data found after all methods")
                ws[f"A{row}"] = f"TABLE {table_idx}: No data extracted"
                ws[f"A{row}"].font = Font(italic=True, color="FF9900")
                row += 2
                continue

            logger.info(f"  ✅ TABLE {table_idx}: {len(canonical_rows)} rows × {len(canonical_rows[0]) if canonical_rows else 0} cols")

            # Write table title
            num_cols = max((len(r) for r in canonical_rows), default=1)
            ws.merge_cells(f"A{row}:F{row}")
            title_cell = ws[f"A{row}"]
            title_cell.value = f"TABLE {table_idx} - LINE ITEMS"
            title_cell.font = styles["table_header_font"]
            title_cell.fill = styles["table_header_fill"]
            title_cell.alignment = Alignment(horizontal="center", vertical="center")
            row += 1

            # Write header row
            header_row = canonical_rows[0] if canonical_rows else [f"COL{i+1}" for i in range(num_cols)]
            for col_idx, header in enumerate(header_row, 1):
                cell = ws.cell(row=row, column=col_idx)
                cell.value = ExcelWriter.safe_str(header)
                cell.font = styles["table_header_font"]
                cell.fill = styles["table_header_fill"]
                cell.border = styles["border"]
                cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            row += 1

            # Write data rows
            for data_row in canonical_rows[1:]:
                for col_idx, cell_value in enumerate(data_row, 1):
                    cell = ws.cell(row=row, column=col_idx)
                    cell.value = ExcelWriter.safe_str(cell_value)
                    cell.font = styles["data_font"]
                    cell.border = styles["border"]
                    cell.alignment = Alignment(wrap_text=True, vertical="top")
                row += 1

            row += 1  # Space between tables

        # Auto-fit columns
        ExcelWriter.autofit_columns(ws)

    @staticmethod
    def _parse_html_table(html: str) -> Optional[List[List[str]]]:
        """Parse HTML table to 2D grid"""
        if not html or "<table" not in html.lower():
            return None
        
        try:
            import pandas as pd
            from io import StringIO
            
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
            logger.debug(f"HTML parsing failed: {e}")
            return None

    @staticmethod
    def autofit_columns(ws, min_width: int = 12, max_width: int = 60) -> None:
        """Auto-fit column widths based on content"""
        for col_idx in range(1, ws.max_column + 1):
            col_letter = get_column_letter(col_idx)
            max_len = 0
            
            for row_idx in range(1, ws.max_row + 1):
                cell_value = ws.cell(row=row_idx, column=col_idx).value
                if cell_value:
                    max_len = max(max_len, len(str(cell_value)))
            
            adjusted_width = max(min_width, min(max_width, max_len + 2))
            ws.column_dimensions[col_letter].width = adjusted_width

    @staticmethod
    def write_json_sheet(ws, extracted_data: Dict, styles: Dict) -> None:
        """Write Sheet 3: Raw JSON data"""
        row = 1
        
        ws.merge_cells(f"A{row}:B{row+20}")
        json_cell = ws[f"A{row}"]
        json_str = json.dumps(extracted_data, indent=2)[:5000]  # Limit to 5000 chars
        json_cell.value = json_str
        json_cell.alignment = Alignment(wrap_text=True, vertical="top")
        json_cell.font = Font(name="Courier New", size=8)
        
        ws.column_dimensions["A"].width = 60
        ws.column_dimensions["B"].width = 30

    @staticmethod
    def write_invoice_to_excel(extracted_data: Dict, output_path: str) -> str:
        """
        Convert extracted invoice data to Excel with 3 sheets
        1. Invoice Details (fields + OCR text)
        2. Line Items (tables converted to grid)
        3. Raw JSON (full data for debugging)
        """
        try:
            wb = Workbook()
            styles = ExcelWriter.get_styles()

            # Sheet 1: Invoice Details
            ws1 = wb.active
            ws1.title = "Invoice Details"
            ExcelWriter.write_invoice_details_sheet(ws1, extracted_data, styles)

            # Sheet 2: Line Items
            ws2 = wb.create_sheet("Line Items")
            ExcelWriter.write_line_items_sheet(ws2, extracted_data, styles)

            # Sheet 3: Raw JSON
            ws3 = wb.create_sheet("Raw JSON")
            ExcelWriter.write_json_sheet(ws3, extracted_data, styles)

            # Save
            wb.save(output_path)
            logger.info(f"✅ Excel file saved: {output_path}")
            return output_path

        except Exception as e:
            logger.exception(f"❌ Error writing Excel: {e}")
            raise