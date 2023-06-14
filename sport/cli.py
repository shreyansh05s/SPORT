#!/usr/bin/env python
"""Main module of the project."""

import argparse
from sport.MultiSports import SportsMOTDataset
from torch.utils.data import DataLoader


def main() -> None:
    """Main entry point of the project."""
    parser = argparse.ArgumentParser(
        description="Sports Object Recognition And Tracking"
    )
    parser.add_argument("--version", action="version", version="%(prog)s 0.1")
    parser.add_argument(
        "-D",
        "--dataset_dir",
        help="Path to dataset directory",
        type=str,
        default="./sport/dataset/sportsmot_publish/dataset",
    )

    args = parser.parse_args()

    # Instantiate the dataset
    train_dataset = SportsMOTDataset(
        root_dir=args.dataset_dir,
        data_type="train",
        sequence_length=8,
    )
    val_dataset = SportsMOTDataset(
        root_dir=args.dataset_dir,
        data_type="val",
        sequence_length=8,
    )

    # Prepare the data loaders
    _ = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    _ = DataLoader(val_dataset, batch_size=4, shuffle=True, num_workers=4)

    print("length of train_loader: ", len(train_dataset))
    print(train_dataset[0])
    return


if __name__ == "__main__":
    main()
