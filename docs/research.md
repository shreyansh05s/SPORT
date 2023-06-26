# All Findings and Research will be documented here for tracking 

## Exploration
<!-- Add checbox for items -->
- [x] [Video MAE](
    https://huggingface.co/docs/transformers/v4.30.0/en/model_doc/videomae#transformers.VideoMAEForVideoClassification
    )
- [] [Mix Former](
    https://github.com/MCG-NJU/MixFormerV2
)

- [] [Transformer Trajectory](
    https://huggingface.co/docs/transformers/model_doc/trajectory_transformer
)

## Dataset
SPORTS-MOT
Link : https://huggingface.co/datasets/MCG-NJU/MultiSports

## Notes
- Video MAE and Timesformer 
- First setup object detection for RWKV and then implement tracking
- Another wild experiment: Video MAE pretrained model and then decode the output to get the object detection and trackid

<!-- RNN Transformer -->
- Encode and Concatenate For RNN
    - Encode the frame with VIT 
    - Concatenate the encoded frame with its trajectory
    - Feed the concatenated vector to RNN
    - Decode RNN output to predict the bounding box and track id

<!-- Without RNN only Transformer -->
- Transformer
    - Encode the trajectory 
        - Spatial Encoding: 
            - Use the bounding box coordinates to get the center of the bounding box
            - Use the center coordinates to get the spatial encoding
            - (Optional) Use the normalized track id for spatial encoding
                - Additonally, represent the trace for n frames to show motion
            - (Optional) Heatmap Encoding
            - Create an encoder for Spatial Encoding to get the new location of the object in the next frame
            - Then use the new location to match the object with track id
        - Sequence Encoding:
            - Use the trajectory seperated by specical token to get the sequence encoding

<!-- One issue with providing spatial encoding is that the model will not be able to generalize to new objects -->
<!-- One solution is to pretrain DETR on the dataset and then use the pretrained model for Object Detection -->
- Pretrain DETR on the dataset
    - Use the pretrained DETR model to get the object detection
    - Use the object detection to setup above Transformer model
    - Add a decoder to DETR or Decoder of objects to create spatial encoding for the new frame
    - Use the spatial encoding to match the object with track id

https://huggingface.co/docs/transformers/tasks/object_detection#preprocess-the-data


https://github.com/MCG-NJU/SportsMOT/tree/main/codes/evaluation/TrackEval