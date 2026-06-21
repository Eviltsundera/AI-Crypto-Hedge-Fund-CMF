"""Repository-relative path helpers."""

from pathlib import Path


def project_root() -> Path:
    """Return the repository root based on the installed source layout."""
    return Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    """Return the project data directory."""
    return project_root() / "data"


def reports_dir() -> Path:
    """Return the project reports directory."""
    return project_root() / "reports"
