"""CSV gaze-data preprocessing: parse a CSV upload into a DataFrame plus
the metadata (column mapping, normalization flag) that ``EventDetection``
needs to interpret it.

Used by the screen-based (stationary camera) upload path. The head-mounted
path lives in :mod:`preprocess_headmounted`.
"""

import io

import pandas as pd


# Candidate column names — first match (case-insensitive) wins.
_TIMESTAMP_CANDIDATES = ("timestamp", "time", "t")
_X_CANDIDATES = ("x", "gaze_x", "gazex", "pos_x", "posx")
_Y_CANDIDATES = ("y", "gaze_y", "gazey", "pos_y", "posy")
_FLOW_X_CANDIDATES = ("flow_x", "optical_flow_x", "of_x", "optic_flow_x")
_FLOW_Y_CANDIDATES = ("flow_y", "optical_flow_y", "of_y", "optic_flow_y")
_VIDEO_TS_CANDIDATES = ("video_timestamp", "video_time", "frame_timestamp")


def detect_delimiter(content) -> str:
    """Return the delimiter used in the first line of *content*.

    Args:
        content: CSV text as ``str`` or ``bytes`` (only the first line is read).
    """
    if isinstance(content, bytes):
        content = content.decode("utf-8", errors="replace")
    first_line = content.splitlines()[0] if content else ""
    for delimiter in ("|", ";", ",", "\t", " "):
        if delimiter in first_line:
            return delimiter
    return ","


def detect_column_mapping(df: pd.DataFrame) -> dict:
    """Map standard names (``x``/``y``/``timestamp`` etc.) to actual columns.

    Returns a dict with at least ``x``, ``y``, ``timestamp`` when those are
    found. Optional keys: ``flow_x``, ``flow_y``, ``video_timestamp`` (used
    by the head-mounted head-motion compensation pipeline).
    """
    columns_lower = {col.lower(): col for col in df.columns}
    mapping: dict = {}

    def _first_match(candidates):
        for c in candidates:
            if c in columns_lower:
                return columns_lower[c]
        return None

    for key, candidates in (
        ("timestamp", _TIMESTAMP_CANDIDATES),
        ("x", _X_CANDIDATES),
        ("y", _Y_CANDIDATES),
        ("flow_x", _FLOW_X_CANDIDATES),
        ("flow_y", _FLOW_Y_CANDIDATES),
        ("video_timestamp", _VIDEO_TS_CANDIDATES),
    ):
        match = _first_match(candidates)
        if match is not None:
            mapping[key] = match

    return mapping


def detect_if_normalized(df: pd.DataFrame, x_col: str, y_col: str) -> bool:
    """True when both x and y look like normalized [0, 1] coordinates.

    Heuristic: if either column has a max value above 2 the data is treated
    as pixel coordinates. Defaults to ``True`` when the columns are empty.
    """
    x_values = df[x_col].dropna()
    y_values = df[y_col].dropna()
    if len(x_values) == 0 or len(y_values) == 0:
        return True
    return not (x_values.max() > 2 or y_values.max() > 2)


def load_csv_gaze_data(csv_content):
    """Parse a CSV upload into ``(DataFrame, column_mapping, is_normalized)``.

    Args:
        csv_content: Raw bytes (or string) of the uploaded CSV file.

    Raises:
        ValueError: When the required ``x``/``y``/``timestamp`` columns
            cannot be found.
    """
    delimiter = detect_delimiter(csv_content)
    df = pd.read_csv(io.BytesIO(csv_content) if isinstance(csv_content, bytes)
                     else io.StringIO(csv_content), sep=delimiter)

    mapping = detect_column_mapping(df)
    for required in ("x", "y", "timestamp"):
        if required not in mapping:
            raise ValueError(
                f"Could not detect '{required}' column. "
                f"Found columns: {list(df.columns)}"
            )

    is_normalized = detect_if_normalized(df, mapping["x"], mapping["y"])
    return df, mapping, is_normalized
