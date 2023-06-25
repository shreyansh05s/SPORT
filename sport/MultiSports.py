#!/usr/bin/env python3
"""SportsMOTDataset class"""

import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from transformers import AutoImageProcessor


class SportsMOTDataset(Dataset):
    """
    SportsMOTDataset class

    Args:
        root_dir (str): root directory of the dataset
        data_type (str): train or val
        sequence_length (int): length of the sequence
    """

    def __init__(
        self, root_dir, data_type="train", sequence_length=16, image_processor=None
    ) -> None:
        #############################
        # TODO: Absolutize the path  #
        #############################
        # use the absolute path of the root directory
        self.root_dir = os.path.join(root_dir, data_type)
        self.video_names = os.listdir(self.root_dir)
        self.sequence_length = sequence_length
        self.image_processor = image_processor

        if image_processor is None:
            self.transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [transforms.ToPILImage(), self.process_image]
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

    def process_image(self, image):
        inputs = self.image_processor(image, return_tensors="pt")
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        return inputs

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

        results = {
            "pixel_values": torch.tensor([]),
            "pixel_mask": torch.tensor([]),
            "labels": [],
            # "boxes": torch.tensor([]),
            "id": torch.tensor([], dtype=torch.int64),
            "image": None,
            "video_name": video_name,
            # save string paths for visualization
            "image_path": [os.path.join(video_dir, vf).__str__() for vf in sequence_files], 
        }

        for seq_idx, sequence_file in enumerate(sequence_files):
            # load image from file and convert to tensor
            img = Image.open(sequence_file)
            img = np.array(img)
            
            if seq_idx == 0:
                results["image"] = np.array(img)
            else:
                results["image"] = np.concatenate((results["image"], img), axis=0)

            inputs = self.transform(img)

            # Getting corresponding labels
            frame_num = (
                frame_idx + seq_idx + 1
            )  # Assuming that the frame numbers start from 1
            label_df = gt_df[gt_df["frame"] == frame_num]

            boxes = torch.tensor(
                label_df[
                    ["bb_left", "bb_top", "bb_width", "bb_height"]
                ].values.tolist(),
                dtype=torch.float32,
            )

            # labels for DETR (List[Dict] of len `(batch_size,)`, *optional*):
            class_labels = torch.ones((boxes.shape[0],), dtype=torch.long)

            labels = [{"class_labels": class_labels, "boxes": boxes}]

            # instead of returning list return a dictionary with concatenated tensors
            results["pixel_values"] = torch.cat(
                (results["pixel_values"], inputs["pixel_values"])
            )
            results["pixel_mask"] = torch.cat(
                (results["pixel_mask"], inputs["pixel_mask"])
            )
            results["id"] = torch.cat((results["id"], torch.tensor(label_df["id"].values, dtype=torch.int64)))
            results["labels"] += labels

        return results
