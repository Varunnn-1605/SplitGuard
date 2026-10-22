"""Report serialisation (detection logic lives in ``splitguard.detectors``)."""

from splitguard.reporting.json_report import (
    load_json_report,
    render_summary,
    report_to_json,
    write_json_report,
)

__all__ = ["load_json_report", "render_summary", "report_to_json", "write_json_report"]
