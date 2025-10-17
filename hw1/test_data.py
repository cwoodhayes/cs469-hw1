"""
pytest tests for data.py
"""

from pathlib import Path

import pytest

from hw1.data import Dataset

REPO_ROOT = Path(__file__).parent.parent


def test_from_dataset_directory():
    p = REPO_ROOT / "data/ds1"
    ds = Dataset.from_dataset_directory(p)

    assert ds.barcodes.shape == (20, 2)
