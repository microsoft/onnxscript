"""Sphinx extension: generate feature flags documentation at build time.

Hooks into ``builder-inited`` to generate ``docs/feature-flags.md`` from
the ``_Flags`` dataclass.  This extension is conditional — it only runs
when the flags module exists (the feature-flags script lives on a
separate PR branch and may not be merged yet).
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

from sphinx.application import Sphinx

logger = logging.getLogger(__name__)


def generate_flags_docs(app: Sphinx) -> None:
    """Generate feature-flags.md if the generator script exists."""
    docs_dir = Path(app.srcdir)

    # Check both possible locations for the script
    script = docs_dir / "_generate_flags_docs.py"
    if not script.exists():
        # Fallback to the scripts/ directory (legacy location)
        script = docs_dir.parent / "scripts" / "generate_flags_docs.py"

    if not script.exists():
        # Not an error — feature-flags PR may not be merged yet
        logger.debug("Flags docs generator not found; skipping")
        return

    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(docs_dir.parent),
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Feature flags doc generation failed:\n{result.stderr}")

    if result.stdout:
        logger.info(result.stdout.strip())


def setup(app: Sphinx) -> dict[str, Any]:
    app.connect("builder-inited", generate_flags_docs)
    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
