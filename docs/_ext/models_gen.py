"""Sphinx extension: generate model documentation pages at build time.

Hooks into ``builder-inited`` to run the existing model page generator
(``docs/_generate_models.py``) before Sphinx processes source files.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

from sphinx.application import Sphinx

logger = logging.getLogger(__name__)


def generate_model_pages(app: Sphinx) -> None:
    """Generate model .md pages from the registry."""
    docs_dir = Path(app.srcdir)
    script = docs_dir / "_generate_models.py"

    if not script.exists():
        logger.warning("Model generation script not found: %s", script)
        return

    # Run as subprocess to avoid polluting Sphinx's import state.
    # The script handles its own sys.path manipulation.
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(docs_dir.parent),
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Model page generation failed:\n{result.stderr}")

    if result.stdout:
        logger.info(result.stdout.strip())


def setup(app: Sphinx) -> dict[str, Any]:
    app.connect("builder-inited", generate_model_pages)
    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
