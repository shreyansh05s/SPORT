#!usr/bin/env python
"""Defines Models for Object Detection and Tracking."""

import torch
from transformers import AutoConfig, AutoModelForObjectDetection, AutoImageProcessor


models = {
    "DETR": {
        "default": "facebook/detr-resnet-50",
        "pretrained": ["facebook/detr-resnet-50", "facebook/detr-resnet-101"],
        "threshold": 0.95
    },
    "ConditionalDETR": {
        "default": "microsoft/conditional-detr-resnet-50",
        "pretrained": ["microsoft/conditional-detr-resnet-50"],
        "threshold": 0.6
    },
    
    ############################
    # TODO: Add YOLOS model
    # Unlike DETR based models, YOLOS image procesor is different
    # "YOLO": {
    #     "default": "hustvl/yolos-base",
    #     "pretrained": ["hustvl/yolos-base"],
    #     "threshold": 0.9
    # }
    
    ############################
    # Out of Memory issues for GPU
    # "DeformableDETR": {
    #     "default": "SenseTime/deformable-detr",
    #     "pretrained": ["SenseTime/deformable-detr"],
    #     "threshold": 0.8
    # },
    # "DETA": {
    #     "default": "jozhang97/deta-swin-large",
    #     "pretrained": ["jozhang97/deta-swin-large"],
    #     "threshold": 0.5
    # }
}


class ObjectDetectionModel(torch.nn.Module):
    """Object Detection Model."""

    def __init__(self, args, train=False) -> None:
        """Initialize the model."""
        super().__init__()

        self.train = train
        self.args = args
        
        self.threshold = models[args.model]["threshold"]

        # Load the config
        self.config = AutoConfig.from_pretrained(args.pretrained_model)

        if args.num_labels:
            self.config.num_labels = args.num_labels

        # Load the model
        self.model = AutoModelForObjectDetection.from_pretrained(
            args.pretrained_model,
            config=self.config,
            ignore_mismatched_sizes=True,
        )

        if args.model_dir:
            self.model.load_state_dict(torch.load(args.model_dir))

        self.inputs_list = ["pixel_values", "pixel_mask", "labels"]

        if not train:
            self.model.eval()
            self.inputs_list = ["pixel_values", "pixel_mask"]

            self.image_processor = AutoImageProcessor.from_pretrained(
                args.pretrained_model, config=self.config
            )

        return

    def forward(self, x):
        """Forward pass of the model."""

        inputs = {
            k: v.to(self.args.device) for k, v in x.items() if k in self.inputs_list
        }

        outputs = self.model(**inputs)

        # post process if eval mode
        if not self.train:
            coco_data = self.image_processor.post_process_object_detection(
                outputs, threshold=self.threshold, target_sizes=torch.tensor([(720, 1280)])
            )[0]

            # filter results by label for person
            label_mask = coco_data["labels"] == 1

            pred_boxes = coco_data["boxes"][label_mask]
            confidences = coco_data["scores"][label_mask]
            classes = coco_data["labels"][label_mask]

            # change format of predboxes to be compatible with the tracker
            result = [
                (
                    b.detach().cpu().numpy(),
                    c.detach().cpu().numpy(),
                    str(l.detach().cpu().numpy()),
                )
                for b, c, l in zip(pred_boxes, confidences, classes)
            ]
            return outputs, result

        return outputs, None


def get_pretrained(args):
    """Get pretrained model."""
    if args.model not in models:
        raise ValueError(f"Model {args.model} not found.")

    if args.pretrained_model:
        if args.pretrained_model not in models[args.model]["pretrained"]:
            raise ValueError(f"Pretrained model {args.pretrained_model} not found.")
        return args.pretrained_model

    return models[args.model]["default"]
