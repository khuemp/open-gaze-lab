"""Shared HTML rendering for the Plotly-based visualizations.

The plotters return the HTML as a string rather than only writing a file, so
the browser build can hand it straight to an iframe without touching a
filesystem. Writing to *output_path* stays available for native use.
"""

from pathlib import Path

import plotly.io as pio


def render_figure_html(fig, config, output_path=None):
    """Render *fig* to a standalone HTML document.

    Args:
        fig: A ``plotly.graph_objs.Figure``.
        config: Plotly ``config`` dict (toolbar/export options).
        output_path: When given, the HTML is also written to this path.

    Returns:
        The HTML document as a string, with plotly.js inlined so the page is
        self-contained.
    """
    html = pio.to_html(fig, config=config, full_html=True, include_plotlyjs=True)

    if output_path:
        Path(output_path).write_text(html, encoding="utf-8")

    return html
