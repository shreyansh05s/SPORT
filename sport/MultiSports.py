#!/usr/bin/env python3
"""SportsMOTDataset class"""

import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class SportsMOTDataset(Dataset):
    """
    SportsMOTDataset class

    Args:
        root_dir (str): root directory of the dataset
        data_type (str): train or val
        sequence_length (int): length of the sequence
    """

    def __init__(self, root_dir, data_type="train", sequence_length=16) -> None:
        # use the absolute path of the root directory
        self.root_dir = os.path.join(root_dir, data_type)
        self.video_names = os.listdir(self.root_dir)
        self.sequence_length = sequence_length

        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(
                    (224, 224)
                ),  # Or whatever size is required by your model
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # calculate the length of total sequences
        self.total_sequences = [
            len(os.listdir(os.path.join(self.root_dir, video_name, "img1")))
            - self.sequence_length
            + 1
            for video_name in self.video_names
        ]
        self.sequence_idx = np.cumsum(self.total_sequences)

        return

    def __len__(self) -> int:
        return sum(self.total_sequences)

    def __getitem__(self, idx):
        # find the video name and the frame number
        video_idx = np.searchsorted(self.sequence_idx, idx + 1)
        video_name = self.video_names[video_idx]
        frame_idx = idx - self.sequence_idx[video_idx - 1] if video_idx > 0 else idx

        video_dir = os.path.join(self.root_dir, video_name, "img1")
        gt_path = os.path.join(self.root_dir, video_name, "gt", "gt.txt")

        gt_df = pd.read_csv(
            gt_path,
            header=None,
            names=[
                "frame",
                "id",
                "bb_left",
                "bb_top",
                "bb_width",
                "bb_height",
                "conf",
                "x",
                "y",
                "z",
            ],
        )

        video_files = sorted(os.listdir(video_dir))
        video_files = [
            os.path.join(video_dir, vf) for vf in video_files if vf.endswith(".jpg")
        ]

        sequence_files = video_files[frame_idx : frame_idx + self.sequence_length]
        sequence: np.ndarray = []
        label_sequence: torch.Tensor = []

        for seq_idx, sequence_file in enumerate(sequence_files):
            # load image from file and convert to tensor
            img = Image.open(sequence_file)
            img = np.array(img)

            img = self.transform(img)
            sequence.append(img)

            # Getting corresponding labels
            frame_num = (
                frame_idx + seq_idx + 1
            )  # Assuming that the frame numbers start from 1
            label_df = gt_df[gt_df["frame"] == frame_num]
            seq = label_df[
                ["id", "bb_left", "bb_top", "bb_width", "bb_height", "conf"]
            ].values.tolist()
            label_sequence.append(torch.tensor(seq))

        sequence = torch.stack(
            sequence, dim=0
        )  # Convert list of tensors to a single 4D tensor
        # label_sequence = np.array(label_sequence)  # Convert list of arrays to a single 3D array

        return sequence, label_sequence
