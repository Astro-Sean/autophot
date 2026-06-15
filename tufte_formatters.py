"""
Tufte-inspired logging formatters and console output utilities.

Principles from Edward Tufte's work:
1. Maximize data-ink ratio (minimize non-data ink)
2. Layering and separation of information
3. High-density displays
4. Small multiples for comparison
5. Remove chartjunk (decorative elements)
"""

import logging
import sys
import os
from typing import Optional


class TufteFormatter(logging.Formatter):
    """
    Minimal, high-density log formatter inspired by Tufte's principles.
    
    Removes redundant decoration while maximizing information density:
    - No timestamp repetition for same-second messages
    - Compact level indicators (single char: ⚠, ✗, ✓, ○)
    - Right-aligned data values for easy scanning
    - Minimal separators
    
    Example output:
        11:32:15  Science   FWHM=3.5px  ZP=25.34  Sources=47  ✓
        11:32:16  Template  FWHM=3.4px  ZP=25.31  Sources=52  ✓
        11:32:18  Subtract  σ_diff=12.4  Quality=0.98  ⚠ anomaly
    """
    
    # Single-character level indicators (data-dense, Tufte-approved)
    LEVEL_MARKS = {
        logging.DEBUG: '○',      # Open circle - background info
        logging.INFO: '·',       # Middle dot - routine progress  
        logging.WARNING: '⚠',    # Warning triangle - attention needed
        logging.ERROR: '✗',      # X mark - failure
        logging.CRITICAL: '◆',  # Diamond - critical failure
    }
    
    def __init__(self, compact: bool = True, align_values: bool = True):
        super().__init__()
        self.compact = compact
        self.align_values = align_values
        self._last_time: Optional[str] = None
        
    def format(self, record: logging.LogRecord) -> str:
        # Extract time, suppress if same as previous (reduces non-data ink)
        time_str = self.formatTime(record, "%H:%M:%S")
        if time_str == self._last_time and self.compact:
            time_display = "        "  # 8 spaces to align
        else:
            time_display = time_str
            self._last_time = time_str
            
        # Compact level indicator
        level_mark = self.LEVEL_MARKS.get(record.levelno, '?')
        
        # Clean up message: remove redundant prefixes, normalize
        msg = record.getMessage()
        msg = self._clean_message(msg)
        
        # Format: "HH:MM:SS  level  message"
        if self.compact and record.levelno <= logging.INFO:
            # For routine messages, just show time and content (maximize data-ink)
            return f"{time_display}  {msg}"
        else:
            # For warnings/errors, include level indicator
            return f"{time_display} {level_mark} {msg}"
    
    def _clean_message(self, msg: str) -> str:
        """Remove redundant decoration from messages."""
        # Remove common redundant prefixes
        prefixes_to_strip = [
            "INFO - ", "WARNING - ", "ERROR - ", "DEBUG - ",
            "— ", " —",
            "[", "]",
        ]
        for prefix in prefixes_to_strip:
            msg = msg.replace(prefix, "")
        
        # Collapse multiple spaces
        while "  " in msg:
            msg = msg.replace("  ", " ")
            
        return msg.strip()


class PipelineProgressFormatter(logging.Formatter):
    """
    Dense multi-column progress display for pipeline stages.
    
    Inspired by Tufte's small multiples - consistent columns for easy scanning.
    
    Columns: [Time] [Stage] [KeyMetric1] [KeyMetric2] [KeyMetric3] [Status]
    
    Example:
        11:32:15  INIT   files=12/12  threads=4   │
        11:32:16  WCS    stars=47     rmse=0.3"   │
        11:32:18  PSF    fwhm=3.5px   β=2.1      │
        11:32:20  PHOT   zp=25.34     σ_zp=0.02  ✓ SN2024pba_r
    """
    
    COLUMN_WIDTHS = [9, 7, 12, 12, 12]  # Time, Stage, Metric1, Metric2, Status
    
    def __init__(self):
        super().__init__()
        self._column_data: dict = {}
        
    def format(self, record: logging.LogRecord) -> str:
        time_str = self.formatTime(record, "%H:%M:%S")
        msg = record.getMessage()
        
        # Try to extract structured data from message
        parts = self._parse_structured(msg)
        
        if parts:
            return self._format_columns(time_str, parts)
        else:
            # Fallback to simple format
            return f"{time_str}  {msg}"
    
    def _parse_structured(self, msg: str) -> Optional[dict]:
        """Extract key=value pairs from common pipeline patterns."""
        import re
        
        # Pattern: "STAGE key=value key=value ..."
        stage_match = re.match(r'(\w+)\s+(.+)', msg)
        if not stage_match:
            return None
            
        stage = stage_match.group(1)
        rest = stage_match.group(2)
        
        # Extract key=value pairs
        pairs = re.findall(r'(\w+)[=:]([\d.]+\s*\w*)', rest)
        
        return {
            'stage': stage,
            'metrics': pairs[:2],  # Take first 2 metrics
            'status': '✓' if 'success' in msg.lower() or 'ok' in msg.lower() else ''
        }
    
    def _format_columns(self, time_str: str, parts: dict) -> str:
        """Format as aligned columns."""
        w = self.COLUMN_WIDTHS
        
        stage = parts['stage'][:w[1]].ljust(w[1])
        
        # Format metrics
        metrics_str = ""
        for i, (key, val) in enumerate(parts['metrics']):
            metric = f"{key}={val}"
            metrics_str += metric[:w[2+i]].ljust(w[2+i])
        
        status = parts.get('status', '')
        
        return f"{time_str} {stage} {metrics_str}{status}"


class CompactTableFormatter:
    """
    Format dense data tables for console output.
    
    Tufte principle: "Table-ink ratio" - minimize borders, maximize data.
    
    Example (lightcurve data):
        MJD        r      g      i     S/N   
        ──────────────────────────────────────
        60523.45  18.34  19.12  18.87  12.5 
        60524.51  18.31  19.08  18.84  13.2 
        60525.60  18.28  19.05  18.82  11.8 
    """
    
    @staticmethod
    def format_table(
        headers: list[str],
        rows: list[list],
        align: str = 'right',
        precision: int = 2
    ) -> str:
        """
        Format a compact data table.
        
        Parameters
        ----------
        headers : list of column headers
        rows : list of row data (each row is a list matching headers)
        align : 'left', 'right', or 'center' for numeric columns
        precision : decimal places for float values
        """
        if not rows:
            return ""
        
        # Convert all values to strings, format floats
        str_rows = []
        for row in rows:
            str_row = []
            for val in row:
                if isinstance(val, float):
                    str_row.append(f"{val:.{precision}f}")
                else:
                    str_row.append(str(val))
            str_rows.append(str_row)
        
        # Calculate column widths (max of header and data)
        widths = []
        for i, header in enumerate(headers):
            header_width = len(str(header))
            data_width = max(len(row[i]) for row in str_rows) if str_rows else 0
            widths.append(max(header_width, data_width) + 1)  # +1 for spacing
        
        # Build output
        lines = []
        
        # Header row
        header_line = "".join(h[:w-1].ljust(w) for h, w in zip(headers, widths))
        lines.append(header_line.rstrip())
        
        # Minimal separator (single line, light weight)
        sep = "─" * sum(widths)
        lines.append(sep[:sum(widths)])
        
        # Data rows
        for row in str_rows:
            if align == 'right':
                line = "".join(r[:w-1].rjust(w) for r, w in zip(row, widths))
            else:
                line = "".join(r[:w-1].ljust(w) for r, w in zip(row, widths))
            lines.append(line.rstrip())
        
        return "\n".join(lines)


def format_progress_bar(
    current: int,
    total: int,
    width: int = 30,
    suffix: str = ""
) -> str:
    """
    Minimal progress bar with high data-ink ratio.
    
    Uses block characters for density: ▓▓▓▓▓▓▓░░░ 70% (7/10)
    
    Example:
        ▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░░░░░░░░ 42% (5/12)  SN2024pba_r
    """
    if total == 0:
        return ""
    
    filled = int(width * current / total)
    bar = "▓" * filled + "░" * (width - filled)
    pct = int(100 * current / total)
    
    return f"{bar} {pct}% ({current}/{total}){suffix}"


def format_metric_line(
    label: str,
    value: float,
    unit: str = "",
    precision: int = 2,
    width: int = 50
) -> str:
    """
    Single metric line with right-aligned value (easy scanning).
    
    Example:
        Seeing FWHM .................................... 3.52 px
        Zeropoint ...................................... 25.34 mag
    """
    val_str = f"{value:.{precision}f}"
    if unit:
        val_str += f" {unit}"
    
    # Dots to guide eye from label to value
    label_part = f"{label} "
    available = width - len(label_part) - len(val_str)
    
    if available > 0:
        return f"{label_part}{'.' * available}{val_str}"
    else:
        return f"{label_part}{val_str}"


# Convenience function for dense summary blocks
def format_execution_summary(
    image_name: str,
    metrics: dict[str, tuple],
    width: int = 70
) -> str:
    """
    Dense execution summary block.
    
    Parameters
    ----------
    image_name : str
        Base filename (truncated if needed)
    metrics : dict
        Dict of metric_name -> (value, unit, precision)
    width : int
        Total width of block
    
    Example:
        ┌─ SN2024pba_ZTF_r.fits ─────────────────────────────────────┐
        │  Photometry    m=18.34±0.03  S/N=12.5  β=2.1              │
        │  Calibration   zp=25.34±0.02  n_cal=47  σ_zp=0.02          │
        │  Quality       FWHM=3.5px  ellip=0.08  θ=12°               │
        └────────────────────────────────────────────────────────────┘
    """
    lines = []
    
    # Header with filename
    name = os.path.basename(image_name)
    if len(name) > width - 6:
        name = "..." + name[-(width-9):]
    
    lines.append(f"┌─ {name}{'─' * (width - len(name) - 4)}┐")
    
    # Group metrics by category
    categories = {
        'Photometry': ['mag', 'mag_err', 'snr', 'beta', 'flux'],
        'Calibration': ['zp', 'zp_err', 'n_cal', 'zp_sigma'],
        'Quality': ['fwhm', 'ellipticity', 'theta', 'background'],
    }
    
    for cat_name, cat_metrics in categories.items():
        cat_values = []
        for m in cat_metrics:
            if m in metrics:
                val, unit, prec = metrics[m]
                if isinstance(val, float):
                    cat_values.append(f"{m}={val:.{prec}f}{unit}")
                else:
                    cat_values.append(f"{m}={val}{unit}")
        
        if cat_values:
            values_str = "  ".join(cat_values)
            line = f"│  {cat_name:<12} {values_str:<{width-18}}│"
            lines.append(line[:width])
    
    lines.append(f"└{'─' * (width - 2)}┘")
    
    return "\n".join(lines)
