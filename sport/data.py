#!usr/bin/env python
"""Loads datasets"""

from datasets import load_dataset


def load_data() -> None:
    """Loads data from datasets"""
    _ = load_dataset("MCG-NJU/MultiSports")
    return


if __name__ == "__main__":
    load_data()
