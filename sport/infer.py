#!usr/bin/env python3
"""Inference script for SportsMOT dataset
The script is used to generate the inference results for SportsMOT dataset.
Each video is processed sequentially.
DETR model is used to generate the predictions,
And DeepSORT is used to generate the tracking results
"""

import os
import cv2
import torch
import numpy as np
from PIL import Image
import torch.nn as nn
from tqdm.auto import tqdm
from sport.MultiSports import SportsMOTDataset
from deep_sort_realtime.deepsort_tracker import DeepSort
from transformers import AutoImageProcessor, DetrForObjectDetection


def collate_fn(batch):
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "pixel_mask": torch.stack([x["pixel_mask"] for x in batch]),
        "labels": [x["labels"][0] for x in batch],
        "id": [x["id"] for x in batch],
        "image": np.array([x["image"] for x in batch]),
        "video_name": [x["video_name"] for x in batch],
        "image_path": [x["image_path"] for x in batch],
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

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn
    )
    
    # Load the DETR model
    object_detector = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50"
    ).to(device)

    # load the DeepSORT tracker
    tracker = DeepSort(max_age=70)

    # tqdm progress bar
    tqdm_bar = tqdm(train_dataloader, desc="Training", total=len(train_dataloader))

    # reset tracker when video changes
    video_idx = None
    
    count = 0

    for inputs in tqdm_bar:
        if not video_idx:
            video_idx = inputs["video_name"][0]
        elif video_idx != inputs["video_name"][0]:
            tracker = DeepSort(max_age=70)
            video_idx = inputs["video_name"][0]

        object_detector.eval()

        # Forward pass
        outputs = object_detector(
            pixel_values=inputs["pixel_values"].to(device),
            pixel_mask=inputs["pixel_mask"].to(device),
        )

        # load image
        image = Image.fromarray(inputs["image"][0])
        
        # convert ouputs to coco format
        target_sizes = torch.tensor([image.size[::-1]])

        results = image_processor.post_process_object_detection(
            outputs, threshold=0.9, target_sizes=target_sizes
        )[0]
        
        pred_boxes = results['boxes']
        confidences = results['scores']
        classes = results['labels']
        
        # change format of predboxes to be compatible with the deepsort tracker
        pred_boxes = [
            (b.detach().cpu().numpy().tolist(), c.detach().cpu().numpy().tolist(), str(l.detach().cpu().numpy().tolist()))
            for b, c, l in zip(pred_boxes, confidences, classes)
        ]

        # DeepSORT tracker where frame takes a numpy array
        tracks = tracker.update_tracks(pred_boxes, frame=inputs["image"][0])
        
        track_to_ltrb = []

        # print(pred_boxes)
        
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()

            track_to_ltrb.append((track_id, ltrb))
        
        if len(track_to_ltrb) > 0:
            # draw boxes on image with track id

            bbox_image = cv2.imread(inputs["image_path"][0][0])
            for track_id, ltrb in track_to_ltrb:
                cv2.rectangle(bbox_image, (int(ltrb[0]), int(ltrb[1])), (int(ltrb[2])-int(ltrb[0]),  int(ltrb[3])-int(ltrb[1])), (0, 255, 0), 2)
                cv2.putText(bbox_image, str(track_id), (int(ltrb[0]), int(ltrb[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            cv2.imwrite("outputs/{}.jpg".format(count), bbox_image)
        
        count += 1
        if count == 10:
            quit()


if __name__ == "__main__":
    main()
