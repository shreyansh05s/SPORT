#!usr/bin/env python
"""Runs a Streamlit app to visualize the results of the model."""

import os
import time
import random
import argparse
import numpy as np
import streamlit as st
from tqdm.auto import tqdm

import cv2
import torch
from sport import get_project_dir
from sport import SportsMOTDataset, get_pretrained
from sport.cli.main import add_common_args
from sport.cli.infer import add_infer_args
from sport.detector import ObjectDetectionModel
from sport.tracker import ObjectTrackingModel
from sport.detector import models


# cache the model and dataset for stremalit app
def load_model(_args):
    # Load Object Detector
    print("Loading Object Detector...")
    object_detector = ObjectDetectionModel(args, train=False).to(args.device)

    # Load Object Tracker
    print("Loading Object Tracker...")
    args.max_age = 10
    tracker = ObjectTrackingModel(args)

    return object_detector, tracker

def load_dataset(_args):
    # load dataset
    print("Loading dataset...")
    val_dataset = SportsMOTDataset(args, data_type="val", sequence_length=1)

    # data loader
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    return val_loader


# create a streamlit app to visualize the results
def demo(args: argparse.Namespace) -> None:
    st.title("Sports Object Recognition And Tracking")
    st.write("This is a demo of the Sports Object Recognition And Tracking project.")

    # model selection
    model = st.selectbox("Select Model", models.keys())
    args.model = model
    
    # drop down menu to select the video type
    video_type = st.selectbox("Select Video Type", ["football", "basketball", "volleyball"])
    
    video_names = open(os.path.join(get_project_dir(), args.splits_dir, f"{video_type}.txt")).read().splitlines()
    
    validation_videos = open(os.path.join(get_project_dir(), args.splits_dir, "val.txt")).read().splitlines()
    
    # filter out the videos that are not in the validation set
    video_names = [video_name for video_name in video_names if video_name in validation_videos]

    # select a video randomly
    video_name = random.choice(video_names)
    args.video_name = video_name
    
    # add button to start demo
    start_demo = st.button("Start Demo")

    if start_demo:
        # get pretrained model
        args.pretrained_model = get_pretrained(args)
        
        # load model
        object_detector, tracker = load_model(args)
        
        # load dataset
        val_loader = load_dataset(args)

        # progress bar
        tqdm_bar = tqdm(val_loader, desc="Demo", total=len(val_loader))
                
        imageLocation = st.empty()
        
        for i, inputs in enumerate(tqdm_bar):
            # forward pass
            _, pred_boxes = object_detector(inputs)
            
            # track objects
            tracked_objects = tracker.track(
                pred_boxes, frame=inputs["image"][0], frame_id=i + 1
            )
            
            # create bbox on the image
            bbox_image = cv2.imread(inputs["image_path"][0][0])
            
            for obj in tracked_objects:
                _, track_id, x, y, w, h, *_ = obj
                cv2.rectangle(
                    bbox_image,
                    (int(x), int(y)),
                    (int(w), int(h)),
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    bbox_image,
                    str(track_id),
                    (int(x), int(y)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
            
            # display the results
            imageLocation.image(bbox_image[..., ::-1], caption=f"Frame {i+1}", use_column_width=True)

    return


def add_demo_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add arguments to the parser."""

    return parser


def collate_fn(batch):
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "pixel_mask": torch.stack([x["pixel_mask"] for x in batch]),
        "image": np.array([x["image"] for x in batch]),
        "image_path": [x["image_path"] for x in batch],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_demo_args(add_infer_args(add_common_args(parser)))

    args = parser.parse_args()
    
    args.splits_dir = os.path.join(get_project_dir(), args.dataset_dir, "splits_txt")

    args.dataset_dir = os.path.join(get_project_dir(), args.dataset_dir, "dataset")

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    demo(args)
