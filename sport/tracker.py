#!usr/bin/env python
"""Defines Models for Object Tracking."""

import torch
from deep_sort_realtime.deepsort_tracker import DeepSort


class ObjectTrackingModel:
    """Object Tracking Model."""

    def __init__(self, args) -> None:
        """Initialize the model."""

        # Load the model
        if args.tracker == "DeepSORT":
            self.tracker = DeepSort(
                max_age=args.max_age,
                n_init=args.n_init,
                nn_budget=args.nn_budget,
                embedder="clip_ViT-B/16",
            )
        else:
            raise ValueError(f"Unknown tracker: {args.tracker}")

        return

    def track(self, boxes, frame, frame_id) -> list:
        """Update the tracks."""
        tracks = self.tracker.update_tracks(boxes, frame=frame)
        
        results = []
        
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()
            ltrb[2] = ltrb[2] - ltrb[0]
            ltrb[3] = ltrb[3] - ltrb[1]
            
            results.append([frame_id, track_id, *ltrb, -1, -1, -1, -1])
        
        return results
