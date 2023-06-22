#!/usr/bin/env python3

from scipy.optimize import linear_sum_assignment
from torch.nn.functional import mse_loss
import torch
import os
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm
from transformers import (
    DetrConfig,
    DetrModel,
    DetrForObjectDetection,
    ViTModel,
    AutoImageProcessor,
    DetrFeatureExtractor,
)
from sport.MultiSports import SportsMOTDataset


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.model.eval()
        self.fc = nn.Linear(256, 128)

    def forward(self, x1, x2):
        x1 = self.model(x1)[0]
        x2 = self.model(x2)[0]
        x1 = self.fc(x1)
        x2 = self.fc(x2)
        return torch.cosine_similarity(x1, x2, dim=-1)

class DetrForTracking(nn.Module):
    def __init__(self):
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

        # self.siamese_net = SiameseNetwork()
        # self.bbox_predictor = MLPPredictionHead(
        #     input_dim=256, hidden_dim=256, output_dim=4, num_layers=3
        # )

    def forward(self, img, prev_features=None):
        # Object detection
        outputs = self.detr_model(**img)

        logits = outputs.logits
        pred_boxes = outputs.pred_boxes
        encoder_last_hidden_state = outputs.encoder_last_hidden_state
        logits = outputs.logits
        loss = outputs.loss

        # Object matching
        similarity_scores = None
        # if prev_features is not None:
        #     similarity_scores = self.siamese_net(
        #         prev_features, encoder_last_hidden_state
        #     )

        return similarity_scores, loss

    def compute_loss(self, similarity_scores, labels, pred_boxes, true_boxes):
        # Compute the Siamese network loss
        loss_siamese = 0
        if similarity_scores is not None:
            loss_fn_siamese = nn.BCEWithLogitsLoss()
            loss_siamese = loss_fn_siamese(similarity_scores, labels)

        # compute DETR loss
        matcher = HungarianMatcher(cost_class=1, cost_bbox=5, cost_giou=2)
        loss_boxes = 0
        if pred_boxes is not None:
            loss_fn_boxes = DetrLoss(
                num_classes=1,
                matcher=matcher,
                weight_dict={"loss_ce": 1, "loss_bbox": 5, "loss_giou": 2},
                eos_coef=0.1,
                losses=["labels", "boxes", "cardinality"],
            )
            loss_boxes = loss_fn_boxes(pred_boxes, true_boxes)

        # Combine the losses
        loss = loss_siamese + loss_boxes
        return loss


def train(
    model: DetrForTracking,
    dataloader: SportsMOTDataset,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
):
    model.train()
    total_loss = 0
    
    # tqdm progress bar
    tqdm_bar = tqdm(dataloader, desc="Training", total=len(dataloader))

    for i in range(epochs):
        
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
            similarity_scores, loss = model(inputs, prev_features)

            # Compute the loss
            # loss = model.compute_loss(similarity_scores, labels, pred_boxes, true_boxes)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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
        train_dataset, batch_size=8, shuffle=True, num_workers=4, collate_fn=collate_fn
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=2, shuffle=False, num_workers=4, collate_fn=collate_fn
    )

    # Initialize the model
    model: DetrForTracking = DetrForTracking()
    model.to(device)

    # Initialize the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Train the model
    train(model, train_dataloader, optimizer, device, epochs=5)

    return


if __name__ == "__main__":
    main()
