#!usr/bin/env python
"""DETR based transformer for object detection."""

import cv2
import torch
from PIL import Image
from transformers import AutoTokenizer, DetrConfig, DetrModel, AutoImageProcessor, DetrForObjectDetection


def main():
    
    # Load the model
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
    model.eval()
    
    # Load the image processor
    image_processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
    
    # Load Image
    img_name = "dataset/sportsmot_publish/dataset/test/v_1UDUODIBSsc_c615/img1/000001.jpg"
    image = Image.open(img_name)
    
    # Load the image
    inputs = image_processor(image, return_tensors="pt", )

    #Run the model
    outputs = model(**inputs)
    
    # convert outputs (bounding boxes and class logits) to COCO API
    target_sizes = torch.tensor([image.size[::-1]])
    print(target_sizes)

    results = image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[
        0
    ]
    
    # print(results.keys())
    
    cv_img = cv2.imread(img_name)

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        
        cv2.rectangle(cv_img, (int(box[0]), int(box[1])), (int(box[2]),  int(box[3])), (0, 255, 0), 2)

        print(
            f"Detected {model.config.id2label[label.item()]} with confidence "
            f"{round(score.item(), 3)} at location {box}"
        )
    
    cv2.imwrite("DETR_test.jpg", cv_img)
    
    return

if __name__ == "__main__":
    main()