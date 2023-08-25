# Project Name: Basketball Tracking System

## Overview:
The project entails the development of a comprehensive basketball tracking system capable of accurately identifying and tracking players, referees, the ball, and the basket within video frames. By automating the generation of gameplay statistics and enabling real-time coaching insights, this system seeks to elevate basketball coaching quality and competitiveness in Singapore schools.

## Data Sources:
- [NBA Play by Play Video](https://www.nba.com/stats/help/videostatus/)
- [Labelled Basketball Images](https://universe.roboflow.com/aidatasets-qwszk/datasets-iztee/dataset/4)

## Data Specifications:
- 5,792 x 30 second play-by-play basketball videos in mp4 format
- 3,960 labeled basketball images in jpeg format

## Problem Statement:
To enhance basketball coaching and interest within Singapore schools, the project aims to automate gameplay statistics generation through player tracking and possession determination. Accurate tracking provides insights into player performance, aids in strategic coaching, improves player competitiveness, and contributes to the growth of Singapore’s basketball landscape.

## Objectives:
1. Accurate detection of basketball players, referees, ball, and basket.
2. Precise tracking of movement for all detected entities.
3. Correct identification of the player in possession of the ball.

## Models Utilized:
1. YOLOv8n for object detection.
2. BYTETrack for object tracking.

## Project Structure:

### 0_Scrape_Video:
- Contains code for scraping NBA official website videos.
- Attributions to authors: [Dekun Wu](https://jackwu502.github.io/), [He Zhao](https://joehezhao.github.io/), Xingce Bao, [Richard P. Wildes](http://www.cse.yorku.ca/~wildes/).

### 1_Train_Validation_Data:
- Training and validation data for the model.
- Contains folders and files for images and videos.
- Includes data splitting configuration in 'data_directory.yaml'.

### 2_Code:
- Jupyter notebook scripts for dataset preparation, object detection/tracking, and model evaluation.

### 3_Training_Validation_Metric_Results:
- Contains evaluation metric results for YOLO detection across iterations.

### 4_Presentation:
- Contains the presentation deck, along with used videos and images.

## Process:
1. Scrape NBA play-by-play video data using the provided script.
2. Extract a single random frame from each video for YOLO model training.
3. Utilize an auto-learning YOLO detection model to label images iteratively.
4. Apply YOLO model to video frames and pass detections through BYTETrack for tracking.
5. Determine player possession based on ball proximity.

## Conclusion:
The project successfully achieved player detection, tracking, and possession determination objectives. The final YOLOv8n model reached an mAP50-95% of 52.2%. While tracking exhibited some challenges due to occlusion, further optimization of BYTETrack parameters and feature extraction may enhance accuracy.

## Next Steps:
- Increase data augmentation for improved detection accuracy.
- Develop a submodel to enhance re-identification within BYTETrack.
- Fine-tune BYTETrack's hyperparameters.
- Implement shot counting and a mobile app interface.
- Apply perspective transformation for 2D court mapping.

Feel free to explore the code and presentation decks in the respective folders.





