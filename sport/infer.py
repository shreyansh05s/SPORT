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
from transformers import AutoImageProcessor, DetrForObjectDetection, DetrConfig
from detr_track import DetrForTracking


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


def main(draw=False):
    # TrackEval directory
    trackeval_dir = os.path.join(os.path.dirname(__file__), "TrackEval", "data", "res", "sportsmot-train", "tracker_to_eval", "data")
    
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

    # load config
    # object_detector = DetrForTracking().to(device)

    # object_detector.load_state_dict(torch.load("models/detr_0.pth"))

    # Load the DETR model
    object_detector = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50"
    ).to(device)

    # load the DeepSORT tracker
    tracker = DeepSort(max_age=10, n_init=0)

    # tqdm progress bar
    tqdm_bar = tqdm(train_dataloader, desc="Training", total=len(train_dataloader))

    # reset tracker when video changes
    video_idx = None

    count = 0
    
    frame_count = 1
    
    object_detector.eval()
    
    for inputs in tqdm_bar:
        
        if video_idx==None:
            print(inputs["video_name"][0])
            video_idx = inputs["video_name"][0]
            eval_results = []
        elif video_idx != inputs["video_name"][0]:
            
            # save the results
            with open(os.path.join(trackeval_dir, f"{video_idx}.txt"), "w") as f:
                f.write("\n".join([" ".join([str(x) for x in y]) for y in eval_results]))
            
            tracker = DeepSort(max_age=10, n_init=0)
            video_idx = inputs["video_name"][0]
            eval_results = []
        
        # Forward pass
        outputs = object_detector(
            pixel_values=inputs["pixel_values"].to(device),
            pixel_mask=inputs["pixel_mask"].to(device),
        )

        # convert outputs to coco format
        target_sizes = torch.tensor([(720, 1280)])

        results = image_processor.post_process_object_detection(
            outputs, threshold=0.9, target_sizes=target_sizes
        )[0]
        
        # filter results by label for person
        label_mask = results["labels"] == 1
                
        pred_boxes = results['boxes'][label_mask]
        confidences = results['scores'][label_mask]
        classes = results['labels'][label_mask]

        # change format of predboxes to be compatible with the deepsort tracker
        pred_boxes = [
            (
                b.detach().cpu().numpy(),
                c.detach().cpu().numpy(),
                str(l.detach().cpu().numpy()),
            )
            for b, c, l in zip(pred_boxes, confidences, classes)
        ]
        
        # print(pred_boxes)

        # DeepSORT tracker where frame takes a numpy array
        tracks = tracker.update_tracks(pred_boxes, frame=inputs["image"][0])

        track_to_ltrb = []

        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()

            track_to_ltrb.append((str(int(track_id)-1), ltrb))

        if len(track_to_ltrb) > 0:
            # draw boxes on image with track id
            if draw:
                bbox_image = cv2.imread(inputs["image_path"][0][0])
            for track_id, ltrb in track_to_ltrb:
                if draw:
                    cv2.rectangle(
                        bbox_image,
                        (int(ltrb[0]), int(ltrb[1])),
                        (int(ltrb[2]) - int(ltrb[0]), int(ltrb[3]) - int(ltrb[1])),
                        (0, 255, 0),
                        2,
                    )
                    cv2.putText(
                        bbox_image,
                        str(track_id),
                        (int(ltrb[0]), int(ltrb[1])),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
                
                eval_results.append([frame_count, track_id, int(ltrb[0]), int(ltrb[1]), int(ltrb[2]) - int(ltrb[0]), int(ltrb[3]) - int(ltrb[1]), -1, -1, -1, -1])
            if draw:
                cv2.imwrite("./outputs2/{}.jpg".format(count), bbox_image)
            
        else:
            # save boxes with track id as -1
            for box in pred_boxes:
                eval_results.append([frame_count, -1, int(box[0][0]), int(box[0][1]), int(box[0][2]), int(box[0][3]), -1, -1, -1, -1])
        
        count += 1
        frame_count += 1
    if len(eval_results) > 0:
        print("Saving results for video {}".format(video_idx))
        with open(os.path.join(trackeval_dir, f"{video_idx}.txt"), "w") as f:
            f.write("\n".join([" ".join([str(x) for x in y]) for y in eval_results]))


if __name__ == "__main__":
    main()
