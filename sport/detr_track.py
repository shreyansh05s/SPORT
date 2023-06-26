#!/usr/bin/env python3

from scipy.optimize import linear_sum_assignment
from torch.nn.functional import mse_loss
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import torch
import os
import torch.nn as nn
from PIL import Image
import numpy as np
from tqdm.auto import tqdm
from transformers import (
    DetrConfig,
    DetrModel,
    DetrForObjectDetection,
    AutoImageProcessor,
    DetrFeatureExtractor,
)
from sport.MultiSports import SportsMOTDataset


class DetrForTracking(nn.Module):
    def __init__(self) -> None:
        super(DetrForTracking, self).__init__()

        # change the config to only have 2 class
        self.config: DetrConfig = DetrConfig.from_pretrained("facebook/detr-resnet-50")
        self.config.num_labels = 1

        # instead use th object detection model
        self.detr_model: DetrForObjectDetection = (
            DetrForObjectDetection.from_pretrained(
                "facebook/detr-resnet-50",
                config=self.config,
                ignore_mismatched_sizes=True,
            )
        )

        # freeze the model except the last layer
        if True:
            for param in self.detr_model.parameters():
                param.requires_grad = False
            for param in self.detr_model.class_labels_classifier.parameters():
                param.requires_grad = True
            for param in self.detr_model.bbox_predictor.parameters():
                param.requires_grad = True

        return

    def forward(self, img, prev_features=None) -> torch.Tensor:
        # Object detection
        outputs = self.detr_model(**img)

        return outputs


def train(
    model: DetrForTracking,
    dataloader: SportsMOTDataset,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler,
    device: torch.device,
):
    model.train()
    total_loss = 0

    # tqdm progress bar
    tqdm_bar = tqdm(dataloader, desc="Training", total=len(dataloader))

    for inputs in tqdm_bar:
        labels = inputs["labels"]
        # ids = [i.to(device) for i in inputs["id"]]

        # labels is a List[Dict[Tensor]] where the size of the list is the batch size
        # now we need to put all the tensors to device
        labels = [{k: v.to(device) for k, v in l.items()} for l in labels]

        inputs = {
            k: v.to(device)
            for k, v in inputs.items()
            if k in ["pixel_values", "pixel_mask"]
        }

        inputs["labels"] = labels

        prev_features = None

        prev_features = prev_features.to(device) if prev_features is not None else None

        # Forward pass
        outputs = model(inputs, prev_features)

        loss = outputs.loss

        # Compute the loss
        # loss = model.compute_loss(similarity_scores, labels, pred_boxes, true_boxes)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        tqdm_bar.set_description(f"Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    return avg_loss


def collate_fn(batch):
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "pixel_mask": torch.stack([x["pixel_mask"] for x in batch]),
        #########
        # TODO: change the labels so that it can handle multiple labels and flatten the boxes
        #########
        "labels": [x["labels"][0] for x in batch],
        # "boxes": [x["boxes"] for x in batch],
        # "id": [x["id"] for x in batch],
        "image": np.array([x["image"] for x in batch]),
    }


def main():
    # Initialize the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Image processor for DETR
    image_processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")

    dataset_dir = "dataset/sportsmot_publish/dataset"

    dataset_dir = os.path.join(os.path.dirname(__file__), dataset_dir)

    # Load data from MultiSports dataset
    # Instantiate the dataset
    train_dataset = SportsMOTDataset(
        root_dir=dataset_dir,
        data_type="train",
        sequence_length=1,
        image_processor=image_processor,
    )
    val_dataset = SportsMOTDataset(
        root_dir=dataset_dir,
        data_type="val",
        sequence_length=1,
        image_processor=image_processor,
    )

    # Instantiate the dataloader
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=collate_fn
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=4, shuffle=False, num_workers=4, collate_fn=collate_fn
    )

    # Initialize the model
    model: DetrForTracking = DetrForTracking()
    model.to(device)

    # Initialize the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    # scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.95)

    epochs = 5

    for i in range(epochs):
        # Train the model
        train(model, train_dataloader, optimizer, scheduler, device)

        # Evaluate the model
        # evaluate(model, val_dataloader, device)

        # Save the model
        torch.save(model.state_dict(), f"models/detr_{i}.pth")

    return


if __name__ == "__main__":
    main()
