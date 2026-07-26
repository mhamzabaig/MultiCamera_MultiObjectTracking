"""Import-order regression tests.

A circular import between `mcmot.core` and the feature packages was only visible
depending on which module a process imported first: importing `mcmot.core.device`
triggered `core/__init__`, which imported the container, which imported `mcmot.reid`,
which imported `mcmot.core.device` again. Everything worked as long as something
imported the container first, so the whole test suite and every script passed while
the cycle sat there.

These tests import each module in a clean interpreter, one at a time, so no earlier
import can mask a cycle.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

MODULES = [
    "mcmot",
    "mcmot.core",
    "mcmot.core.settings",
    "mcmot.core.logging",
    "mcmot.core.errors",
    "mcmot.core.device",
    "mcmot.domain",
    "mcmot.storage",
    "mcmot.detection",
    "mcmot.tracking",
    "mcmot.reid",
    "mcmot.api",
    "mcmot.container",
    "mcmot.main",
]


@pytest.mark.parametrize("module", MODULES)
def test_module_imports_in_a_clean_interpreter(module: str) -> None:
    """Each module must import on its own, with nothing imported before it."""
    result = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, f"importing {module} first failed:\n{result.stderr[-2000:]}"


def test_core_does_not_depend_on_feature_packages() -> None:
    """`core` must stay a leaf.

    Asserted by import rather than by reading source: after importing only
    `mcmot.core`, none of the feature packages should have been pulled in. That
    one-way dependency is what keeps the package graph acyclic as phases are added.
    """
    code = (
        "import sys, mcmot.core; "
        "leaked = [m for m in sys.modules "
        "if m.startswith(('mcmot.storage', 'mcmot.detection', "
        "'mcmot.tracking', 'mcmot.reid', 'mcmot.api', 'mcmot.container'))]; "
        "print(','.join(sorted(leaked)))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=180
    )

    assert result.returncode == 0, result.stderr[-2000:]
    leaked = [name for name in result.stdout.strip().split(",") if name]
    assert not leaked, f"mcmot.core pulled in feature packages: {leaked}"
