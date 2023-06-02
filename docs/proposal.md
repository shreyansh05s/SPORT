# Transfer Learning for Enhanced Object Recognition in Sports

## Introduction:
This proposal outlines a research project that aims to advance object recognition in sports through the combined use of pretraining and fine-tuning techniques. The project focuses on exploring the effectiveness of pretrained models on various sports/object detection datasets and subsequently fine-tuning them on an object detection dataset, with the potential to improve accuracy in player tracking or ball tracking tasks.

## Methodology:
The project consists of two key phases. In the pretraining phase, state-of-the-art object detection models like Faster R-CNN or YOLO will be pretrained on existing datasets from multiple sports. This will enable the models to develop a comprehensive understanding of player-related or ball-related features and motion patterns. The fine-tuning phase involves employing transfer learning techniques to adapt and specialize the pretrained models for the unseen sport. This adaptation will allow the models to accurately detect and track players or balls within the specific dynamics and contexts of the target sport.

## Related work:
SportsMot (https://arxiv.org/abs/2304.05170) is a dataset that has annotations for players across 4 different games. Many architectures exist for this particular dataset and as a part of this project I would be exploring and taking inspiration from this.

## Expected Outcome:
-	Try to implement and compare a few existing benchmarks for selected dataset
-	Create a novel architecture for object detection in sports
-	Automatic annotation of objects in sports
