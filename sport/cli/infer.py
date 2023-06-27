#!usr/bin/env python
"""Runs inference on a video."""

import os
import argparse
from tqdm.auto import tqdm

import torch
import numpy as np
from sport import SportsMOTDataset
from sport.detector import ObjectDetectionModel, get_pretrained
from sport.tracker import ObjectTrackingModel


def infer(args: argparse.Namespace) -> None:
    # get pretrained model
    args.pretrained_model = get_pretrained(args)
    
    if args.all:
        args.video_name = None
    
    # load dataset
    print("Loading dataset...")
    val_dataset = SportsMOTDataset(
        args, data_type="val", sequence_length=1
    )

    # data loader
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
    object_detector = ObjectDetectionModel(args, train=False).to(args.device)

    # Load Object Tracker
    print("Loading Object Tracker...")
    tracker = ObjectTrackingModel(args)

    # progress bar
    tqdm_bar = tqdm(val_loader, desc="Inference", total=len(val_loader))

    all_results = []
    
    video_name = None
    
    for i, inputs in enumerate(tqdm_bar):
        if i == 0:
            video_name = inputs["video_name"][0]
        elif video_name != inputs["video_name"][0]:
            save_results(all_results, video_name)
            all_results = []
            video_name = inputs["video_name"][0]
        
        # forward pass
        _, pred_boxes = object_detector(inputs, train=False)

        # track objects
        tracked_objects = tracker.track(
            pred_boxes, frame=inputs["image"][0], frame_id=i+1
        )
                
        all_results.extend(tracked_objects)
    
    # create results directory
    if not os.path.exists("results"): os.mkdir("results")
    
    # save the results
    if len(all_results) > 0:
        save_results(all_results, video_name)

    return

def save_results(results, video_name) -> None:
    """Save the results."""	
    with open(os.path.join("results", f"{video_name}.txt"), "w") as f:
        f.write("\n".join([" ".join([str(x) for x in y]) for y in results]))
    
    return
    

def add_infer_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add arguments to the parser."""
    parser.add_argument("--batch_size", help="Batch size", type=int, default=1)
    parser.add_argument(
        "--video_name", help="Video name", type=str, default="v_2QhNRucNC7E_c017"
    )
    parser.add_argument(
        "--all", help="Run inference on all videos", action="store_true"
    )

    return parser


def collate_fn(batch):
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "pixel_mask": torch.stack([x["pixel_mask"] for x in batch]),
        # "labels": [x["labels"][0] for x in batch],
        # "id": [x["id"] for x in batch],
        "image": np.array([x["image"] for x in batch]),
        "video_name": [x["video_name"] for x in batch],
        # "image_path": [x["image_path"] for x in batch],
    }
