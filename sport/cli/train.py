#!usr/bin/env python
"""Trains an Object Detection model."""

import os
import argparse
import tqdm.auto as tqdm

import torch
import numpy as np
from sport import SportsMOTDataset
from sport.detector import ObjectDetectionModel, get_pretrained


def train(args: argparse.Namespace) -> None:
    # get pretrained model
    args.pretrained_model = get_pretrained(args)
    args.num_labels = 1

    # load dataset
    print("Loading dataset...")
    train_dataset = SportsMOTDataset(args, data_type="train", sequence_length=1)
    val_dataset = SportsMOTDataset(args, data_type="val", sequence_length=1)

    # data loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # Load Object Detector
    print("Loading Object Detector...")
    model = ObjectDetectionModel(args, train=True).to(args.device)

    # optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma
    )

    # progress bar
    tqdm_bar = tqdm.tqdm(range(args.num_epochs), desc="Training", total=args.num_epochs)

    for epoch in tqdm_bar:
        # train for one epoch
        model = train_one_epoch(
            model, optimizer, scheduler, train_loader, args.device, epoch, args, tqdm_bar
        )

        # evaluate on the validation dataset
        # evaluate(model, val_loader, args.device, epoch, args, tqdm_bar)

        # save the model
        if not os.path.exists("models"): os.mkdir("models")
        torch.save(model.state_dict(), os.path.join("models", f"{args.model}-epoch-{epoch}.pth"))

    return


def train_one_epoch(
    model: ObjectDetectionModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    epoch: int,
    args: argparse.Namespace,
    tqdm_bar: tqdm.tqdm,
) -> ObjectDetectionModel:
    # set model to training mode
    model.train()

    # progress bar
    tqdm_bar.set_description(f"Epoch {epoch}, Training")

    # iterate over the dataset
    for i, inputs in enumerate(data_loader):
        # forward pass
        outputs, _ = model(inputs)

        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        tqdm_bar.set_postfix(loss=loss.item(), percent=i / len(data_loader), lr=scheduler.get_last_lr()[0])

    return model

def evaluate(
    model: ObjectDetectionModel,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    epoch: int,
    args: argparse.Namespace,
    tqdm_bar: tqdm.tqdm,
) -> None:
    import evaluate
    # Evaluator
    ##############################
    # TODO: Add evaluation metrics
    ##############################
    module = evaluate.load("ybelkada/cocoevaluate",)
    
    # set model to evaluation mode
    model.eval()

    # progress bar
    tqdm_bar.set_description(f"Epoch {epoch}, Evaluating")
    
    with torch.no_grad():
        # iterate over the dataset
        for i, inputs in enumerate(data_loader):
            # forward pass
            outputs, results = model(inputs, train=False)
            
            labels = inputs["labels"]
            
            module.evaluate(predictions=results, labels=labels)
            
            tqdm_bar.set_postfix(loss=outputs.loss.item(), percent=i / len(data_loader))
    
    # print results
    results = module.compute()
    print(results)
    return 


def add_train_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size to use for training.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=20,
        help="Number of epochs to train the model for.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for the optimizer.",
    )
    parser.add_argument(
        "--lr_step_size",
        type=int,
        default=10000,
        help="Step size for the scheduler.",
    )
    parser.add_argument(
        "--lr_gamma",
        type=float,
        default=0.95,
        help="Gamma for the scheduler.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay for the optimizer.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device to use for training.",
    )
    parser.add_argument(
        "--video_name", help="Video name", type=str, default=None
    )
    return parser

def collate_fn(batch):
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "pixel_mask": torch.stack([x["pixel_mask"] for x in batch]),
        "labels": [x["labels"][0] for x in batch],
        # "boxes": [x["boxes"] for x in batch],
        # "id": [x["id"] for x in batch],
        # "image": np.array([x["image"] for x in batch]),
    }
