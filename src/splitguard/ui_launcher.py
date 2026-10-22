"""``splitguard-ui`` launcher.

Importable without Streamlit installed: it only imports Streamlit when run,
and prints an install hint otherwise. By default it

* binds to ``localhost`` (Streamlit's own default listens on all interfaces),
* disables Streamlit's anonymous usage statistics,
* caps uploads at the engine's default file-size limit, and
* runs without a file watcher.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from splitguard.config import Limits

APP_PATH = Path(__file__).with_name("ui_app.py")
PRIMARY_COLOR = "#2f5d8a"


def streamlit_args(address: str, port: int, headless: bool) -> list[str]:
    """Arguments passed to ``streamlit run`` (exposed for testing)."""
    max_mb = max(1, Limits().max_file_bytes // (1024 * 1024))
    return [
        "run",
        str(APP_PATH),
        "--browser.gatherUsageStats",
        "false",
        "--server.address",
        address,
        "--server.port",
        str(port),
        "--server.maxUploadSize",
        str(max_mb),
        "--server.fileWatcherType",
        "none",
        "--server.headless",
        "true" if headless else "false",
        # Hide developer options such as the "Deploy" (publish to cloud) button.
        "--client.toolbarMode",
        "viewer",
        # Neutral accent: Streamlit's default red reads as "danger" on filters and buttons.
        "--theme.primaryColor",
        PRIMARY_COLOR,
    ]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="splitguard-ui", description="Launch the SplitGuard web UI locally."
    )
    parser.add_argument(
        "--address", default="localhost", help="interface to bind (default: localhost)"
    )
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--headless", action="store_true", help="do not open a browser window")
    args = parser.parse_args(argv)
    try:
        from streamlit.web import cli as stcli
    except ImportError:
        print(
            'The SplitGuard UI needs Streamlit. Install it with:\n    pip install "splitguard[ui]"',
            file=sys.stderr,
        )
        return 2
    sys.argv = ["streamlit", *streamlit_args(args.address, args.port, args.headless)]
    try:
        stcli.main()
    except SystemExit as exc:
        return int(exc.code or 0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
