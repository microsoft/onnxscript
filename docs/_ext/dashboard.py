"""Sphinx extension: generate the confidence dashboard after build.

Hooks into ``build-finished`` to run ``scripts/generate_dashboard.py``
and place the output in the build directory under ``/dashboard/``.
Also creates a redirect from the old ``/docs/`` path to the new root.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

from sphinx.application import Sphinx

logger = logging.getLogger(__name__)

_REDIRECT_HTML = """\
<!DOCTYPE html>
<html><head>\
<meta http-equiv="refresh" content="0; url=../">\
<title>Redirecting...</title></head>
<body><a href="../">Click here</a></body></html>
"""


def generate_dashboard(app: Sphinx, exception: Exception | None) -> None:
    """Generate dashboard HTML into the build output directory."""
    if exception is not None:
        # Don't generate dashboard if Sphinx build failed
        return

    repo_root = Path(app.srcdir).parent
    script = repo_root / "scripts" / "generate_dashboard.py"

    if not script.exists():
        logger.warning("Dashboard script not found: %s", script)
        return

    outdir = Path(app.outdir)
    dashboard_dir = outdir / "dashboard"
    dashboard_dir.mkdir(parents=True, exist_ok=True)

    # Determine current git commit for display in the dashboard
    commit = _git_short_hash(repo_root)

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--output",
            str(dashboard_dir / "index.html"),
            "--commit",
            commit,
        ],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Dashboard generation failed:\n{result.stderr}")

    if result.stdout:
        logger.info(result.stdout.strip())

    # Add redirect from old /docs/ path to root
    docs_redirect_dir = outdir / "docs"
    docs_redirect_dir.mkdir(parents=True, exist_ok=True)
    (docs_redirect_dir / "index.html").write_text(_REDIRECT_HTML)


def _git_short_hash(repo_root: Path) -> str:
    """Return the short git commit hash, or 'unknown' on failure."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def setup(app: Sphinx) -> dict[str, Any]:
    app.connect("build-finished", generate_dashboard)
    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
