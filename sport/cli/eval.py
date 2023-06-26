#!usr/env/bin python
"""Runs Evaluation on the Model."""

import argparse


def eval(args: argparse.Namespace) -> None:
    
    pass


def add_eval_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add arguments to the parser."""
    parser.add_argument("batch_size", help="Batch size", type=int, default=4)

    return parser
