#!usr/bin/env python
"""Main module of the project."""

import os
import argparse

import torch
from sport.cli import eval, infer, demo, add_eval_args, add_infer_args
from sport import get_project_dir

def main() -> None:
    """Main entry point of the project."""

    # Instantiate the parser
    parser = argparse.ArgumentParser(
        description="Sports Object Recognition And Tracking",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="cmd", help="Choose a command")
    # subparser_train = subparsers.add_parser("train", help="Train the model")
    subparser_eval = subparsers.add_parser("eval", help="Evaluate the model")
    subparser_infer = subparsers.add_parser("infer", help="Infer the model")
    subparser_demo = subparsers.add_parser("demo", help="Run the demo")

    subparser_eval = add_eval_args(add_common_args(subparser_eval))
    subparser_infer = add_infer_args(add_common_args(subparser_infer))
    subparser_demo = add_common_args(subparser_demo)

    args = parser.parse_args()
    
    args.dataset_dir = os.path.join(get_project_dir(), args.dataset_dir, "dataset")
    
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.cmd == "eval":
        eval(args)
    elif args.cmd == "infer":
        infer(args)
    elif args.cmd == "demo":
        demo(args)
    else:
        raise ValueError(f"Unknown command: {args.cmd}")

    return


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add arguments to the parser."""
    parser.add_argument(
        "-D",
        "--dataset_dir",
        help="Path to dataset directory",
        type=str,
        default="dataset/sportsmot_publish",
    )
    parser.add_argument(
        "-m",
        "--model",
        help="Object Detection Model name",
        type=str,
        default="DETR",
    )
    parser.add_argument(
        "-t",
        "--tracker",
        help="Object Tracking Model name",
        type=str,
        choices=["DeepSORT"],
        default="DeepSORT",
    )
    parser.add_argument(
        "-P",
        "--pretrained_model",
        help="Pretrained model",
        type=str,
        default=None,
    )    
    parser.add_argument("--num_labels", help="Number of labels", type=int, default=None)
    parser.add_argument("--num_workers", help="Number of workers", type=int, default=4)
    parser.add_argument("--epochs", help="Number of epochs", type=int, default=10)
    parser.add_argument("--lr", help="Learning rate", type=float, default=1e-4)
    parser.add_argument("--model_dir", help="Model directory", type=str, default=None)

    #############################
    # DeepSORT specific arguments
    #############################
    parser.add_argument("--max_age", help="Maximum age", type=int, default=10)
    parser.add_argument("--n_init", help="Number of initial frames", type=int, default=3)
    parser.add_argument("--nn_budget", help="NN budget", type=int, default=100)
    #############################
    

    return parser


if __name__ == "__main__":
    main()
