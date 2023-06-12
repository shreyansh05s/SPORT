#!/usr/bin/env python
"""Main module of the project."""

import argparse
from sport.data import load_data


def main() -> None:
    """Main entry point of the project."""
    parser = argparse.ArgumentParser(
        description="Sports Object Recognition And Tracking"
    )
    parser.add_argument("--version", action="version", version="%(prog)s 0.1")

    _ = parser.parse_args()

    print("Hello World!")

    if _:
        load_data()

    return


if __name__ == "__main__":
    main()
