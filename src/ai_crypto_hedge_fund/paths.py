"""Repository-relative path helpers."""

from pathlib import Path


def project_root() -> Path:
    """Return the repository root based on the installed source layout."""
    return Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    """Return the project data directory."""
    return project_root() / "data"


def sample_data_dir() -> Path:
    """Return the committed smoke-test data directory."""
    return data_dir() / "sample"


def external_data_dir() -> Path:
    """Return the ignored directory for large external data bundles."""
    return data_dir() / "external"


def reports_dir() -> Path:
    """Return the project reports directory."""
    return project_root() / "reports"
