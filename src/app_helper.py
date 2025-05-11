\
import cv2
import numpy as np
# import tensorflow.compat.v1 as tf
from flask import Flask, request, Response, jsonify, send_from_directory, abort
import os
# Import MediaPipe
import mediapipe as mp
from .config import shooting_result
import sys
from sys import platform
import argparse
# Set Matplotlib backend before importing pyplot
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
# Import the modified utils with YOLO support
from .utils import detect_shot, detect_image, detect_API, yolo_init
from statistics import mean
# tf.disable_v2_behavior()

def getVideoStream(video_path):
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose 
    pose_estimator = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2, # 0, 1, or 2. Higher = more accurate but slower
        enable_segmentation=False, # Not needed for this use case
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7) 

    # Try to initialize the YOLO model first, fall back to TensorFlow if it fails
    try:
        # Initialize YOLO model
        model = yolo_init()
        print("Using YOLO model for detection")
    except Exception as e:
        print(f"Failed to initialize YOLO model: {e}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    trace = np.full((int(height), int(width), 3), 255, np.uint8)

    fig = plt.figure()
    fig.add_subplot(111)  # Create the axes just once
    # objects to store detection status (remain the same)
    previous = {
        'ball': np.array([0, 0]),  # x, y
        'hoop': np.array([0, 0, 0, 0]),  # xmin, ymax, xmax, ymin
        'hoop_height': 0
    }
    during_shooting = {
        'isShooting': False,
        'balls_during_shooting': [],
        'release_angle_list': [],
        'release_point': []
    }
    shooting_pose = {
        'ball_in_hand': False,
        'elbow_angle': 370, # Keep initial value high
        'knee_angle': 370,  # Keep initial value high
        'ballInHand_frames': 0,
        'elbow_angle_list': [],
        'knee_angle_list': [],
        'ballInHand_frames_list': []
    }
    shot_result = {
        'displayFrames': 0,
        'release_displayFrames': 0,
        'judgement': ""
    }

 
    # Using YOLO model
    frame_count = 0
    while True:
        ret, img = cap.read()
        if ret == False:
            break
        frame_count += 1
        if frame_count % 2 != 0:
            continue
        
        detection, trace = detect_shot(img, trace, width, height, model, 
                                    previous=previous, during_shooting=during_shooting, 
                                    shot_result=shot_result, fig=fig, pose_estimator=pose_estimator, 
                                    shooting_pose=shooting_pose)

        detection = cv2.resize(detection, (0, 0), fx=0.83, fy=0.83) # Keep resize if needed
        frame = cv2.imencode('.jpg', detection)[1].tobytes()
        result = (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        yield result
        

    # Release resources
    cap.release()
    pose_estimator.close() # Close the MediaPipe pose estimator
    plt.close(fig) # Close the matplotlib figure
    print("Video processing finished.")

    # --- Calculate Average Results (Check for empty lists) ---
    if shooting_pose['elbow_angle_list']:
        shooting_result['avg_elbow_angle'] = round(mean(shooting_pose['elbow_angle_list']), 2)
    else:
        shooting_result['avg_elbow_angle'] = 0 # Default value
        
    if shooting_pose['knee_angle_list']:
        shooting_result['avg_knee_angle'] = round(mean(shooting_pose['knee_angle_list']), 2)
    else:
        shooting_result['avg_knee_angle'] = 0 # Default value
        
    if during_shooting['release_angle_list']:
        shooting_result['avg_release_angle'] = round(mean(during_shooting['release_angle_list']), 2)
    else:
        shooting_result['avg_release_angle'] = 0 # Default value
        
    if shooting_pose['ballInHand_frames_list']:
        if fps > 0: # Avoid division by zero
            shooting_result['avg_ballInHand_time'] = round(mean(shooting_pose['ballInHand_frames_list']) * (4 / fps), 2) # Assuming skip factor 4
        else:
             shooting_result['avg_ballInHand_time'] = 0
    else:
        shooting_result['avg_ballInHand_time'] = 0 # Default value

    print("avg elbow:", shooting_result['avg_elbow_angle'])
    print("avg knee:", shooting_result['avg_knee_angle'])
    print("avg release:", shooting_result['avg_release_angle'])
    print("avg time:", shooting_result['avg_ballInHand_time'])

    # Save trajectory plot only if shots were detected
    if during_shooting['release_angle_list'] or shooting_result['attempts'] > 0:
        # Set the title on the axes object, not using plt directly
        ax = fig.gca()
        ax.set_title("Trajectory Fitting")
        ax.set_ylim(bottom=0, top=height)
        
        trajectory_path = os.path.join(
            os.getcwd(), "static/detections/trajectory_fitting.jpg")
        try:
            fig.savefig(trajectory_path)
        except Exception as e:
            print(f"Error saving trajectory figure: {e}")
    
    trace_path = os.path.join(os.getcwd(), "static/detections/basketball_trace.jpg")
    cv2.imwrite(trace_path, trace)

def get_image(image_path, img_name, response):
    output_path = './static/detections/'
    # reading the images & apply detection 
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error reading image: {image_path}")
        return # Or handle error appropriately

    filename = img_name
    # detect_image will automatically use YOLO if available or fall back to TensorFlow
    detection = detect_image(image, response)

    cv2.imwrite(output_path + '{}' .format(filename), detection)
    print('output saved to: {}'.format(output_path + '{}'.format(filename)))

def detectionAPI(response, image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error reading image: {image_path}")
        return # Or handle error appropriately
        
    # detect_API will automatically use YOLO if available or fall back to TensorFlow
    detect_API(response, image)
