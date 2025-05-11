
import time
from absl import app, logging
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
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Initialize MediaPipe solutions (do this once)
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def yolo_init():
    """
    Initialize the YOLO model for basketball shot detection.
    Returns a ShotDetector instance that provides the same interface as TensorFlow.
    """
    try:
        from basketball_shot_detector_model.shot_detector import ShotDetector
        # Initialize the model with the best.pt file
        detector = ShotDetector()
        return detector
    except ImportError as e:
        print("Error while loading YOLO model: ", e)
        
model = yolo_init()

def fit_func(x, a, b, c):
    return a*(x ** 2) + b * x + c

def trajectory_fit(balls, height, width, shotJudgement, fig):
    # Don't clear the figure - we want to accumulate trajectories
    # Get the current axes or create one if none exists
    ax = fig.gca()
    
    x = [ball[0] for ball in balls]
    y = [height - ball[1] for ball in balls]

    try:
        params = curve_fit(fit_func, x, y)
        [a, b, c] = params[0]
    except:
        print("fitting error")
        a = 0
        b = 0
        c = 0
    x_pos = np.arange(0, width, 1)
    y_pos = [(a * (x_val ** 2)) + (b * x_val) + c for x_val in x_pos]

    if(shotJudgement == "MISS"):
        ax.plot(x, y, 'ro')
        ax.plot(x_pos, y_pos, linestyle='-', color='red', alpha=0.4, linewidth=5)
    else:
        ax.plot(x, y, 'go')
        ax.plot(x_pos, y_pos, linestyle='-', color='green', alpha=0.4, linewidth=5)

def distance(x, y):
    x = np.array(x)
    y = np.array(y)
    return np.linalg.norm(x - y)

def calculateAngle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    ba = a - b
    bc = c - b
    
    # Check for zero vectors to avoid division by zero
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba == 0 or norm_bc == 0:
        return 0.0 # Or handle as an error/undefined case

    cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    
    # Clamp cosine_angle to [-1, 1] to avoid floating point errors outside domain
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    
    angle = np.arccos(cosine_angle)
    return round(np.degrees(angle), 2)

# get angles from MediaPipe landmarks
def getAngleFromLandmarks(landmarks, image_width, image_height):
    # Default values if landmarks are not detected or points are missing
    elbowAngle = 0.0
    kneeAngle = 0.0
    elbowCoord = np.array([0, 0])
    kneeCoord = np.array([0, 0])
    headCoord = np.array([0,0])
    handCoord = np.array([0,0]) # Using right wrist

    if landmarks:
        lm = landmarks.landmark
        
        # Helper to get pixel coordinates and check visibility
        def get_coords(landmark_index):
            if len(lm) > landmark_index and lm[landmark_index].visibility > 0.5: # Check visibility threshold
                 return np.array([int(lm[landmark_index].x * image_width), int(lm[landmark_index].y * image_height)])
            return None

        # Right elbow angle: 12 (shoulder), 14 (elbow), 16 (wrist)
        r_shoulder = get_coords(mp_pose.PoseLandmark.RIGHT_SHOULDER)
        r_elbow = get_coords(mp_pose.PoseLandmark.RIGHT_ELBOW)
        r_wrist = get_coords(mp_pose.PoseLandmark.RIGHT_WRIST)
        
        if r_shoulder is not None and r_elbow is not None and r_wrist is not None:
            elbowAngle = calculateAngle(r_shoulder, r_elbow, r_wrist)
            elbowCoord = r_elbow # Store elbow pixel coordinates

        # Right knee angle: 24 (hip), 26 (knee), 28 (ankle)
        r_hip = get_coords(mp_pose.PoseLandmark.RIGHT_HIP)
        r_knee = get_coords(mp_pose.PoseLandmark.RIGHT_KNEE)
        r_ankle = get_coords(mp_pose.PoseLandmark.RIGHT_ANKLE)

        if r_hip is not None and r_knee is not None and r_ankle is not None:
            kneeAngle = calculateAngle(r_hip, r_knee, r_ankle)
            kneeCoord = r_knee # Store knee pixel coordinates

        # Head and Hand (Right Wrist) coordinates for proximity checks
        head = get_coords(mp_pose.PoseLandmark.NOSE) # Using Nose as proxy for head center
        if head is not None:
            headCoord = head
            
        if r_wrist is not None:
            handCoord = r_wrist # Use right wrist as hand coordinate

    return elbowAngle, kneeAngle, elbowCoord, kneeCoord, headCoord, handCoord

# Modified detect_shot function to use MediaPipe
def detect_shot(frame, trace, width, height, model, image_tensor=None, boxes=None, scores=None, classes=None, num_detections=None, previous=None, during_shooting=None, shot_result=None, fig=None, pose_estimator=None, shooting_pose=None): 
    """
    Detect shots in a frame.
    
    Args:
        frame (numpy.ndarray): Input image frame in RGB format
        ...
    """
    global shooting_result

    if(shot_result['displayFrames'] > 0):
        shot_result['displayFrames'] -= 1
    if(shot_result['release_displayFrames'] > 0):
        shot_result['release_displayFrames'] -= 1
    if(shooting_pose['ball_in_hand']):
        shooting_pose['ballInHand_frames'] += 1

    # --- MediaPipe Pose Detection ---
    results = pose_estimator.process(frame)
    
    # Extract landmarks and calculate angles
    elbowAngle, kneeAngle, elbowCoord, kneeCoord, headCoord, handCoord = getAngleFromLandmarks(results.pose_landmarks, width, height)
    
    # Draw the pose annotation on the image.
    frame.flags.writeable = True # Make original frame writeable for drawing
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    
    # --- Object Detection (Basketball & Hoop) --- 
    boxes, scores, classes, num_detections = model.run(frame)

    # displaying joint angle and release angle - adjusted coordinates slightly if needed
    # Add checks to only draw if angle was calculated (coord != [0,0])
    if not np.array_equal(elbowCoord, [0,0]):
         cv2.putText(frame, 'Elbow: ' + str(elbowAngle) + ' deg',
                    (elbowCoord[0] + 15, elbowCoord[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (102, 255, 0), 2) # Adjusted font size/pos
    if not np.array_equal(kneeCoord, [0,0]):
        cv2.putText(frame, 'Knee: ' + str(kneeAngle) + ' deg',
                    (kneeCoord[0] + 15, kneeCoord[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (102, 255, 0), 2) # Adjusted font size/pos
    if(shot_result['release_displayFrames'] > 0 and len(during_shooting['release_angle_list']) > 0): # Check list not empty
        cv2.putText(frame, 'Release: ' + str(during_shooting['release_angle_list'][-1]) + ' deg',
                    (during_shooting['release_point'][0] - 80, during_shooting['release_point'][1] + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (102, 255, 255), 2) # Adjusted font size

    for i, box in enumerate(boxes[0]):
        if (scores[0][i] > 0.5):
            ymin = int((box[0] * height))
            xmin = int((box[1] * width))
            ymax = int((box[2] * height))
            xmax = int((box[3] * width))
            xCoor = int(np.mean([xmin, xmax]))
            yCoor = int(np.mean([ymin, ymax]))
            ballCenter = np.array([xCoor, yCoor])

            # Basketball (class 1) and check distance from head (using MediaPipe headCoord)
            if(classes[0][i] == 1 and (not np.array_equal(headCoord, [0,0])) and distance(headCoord, ballCenter) > 30):

                # recording shooting pose - check distance from hand (using MediaPipe handCoord)
                if(not np.array_equal(handCoord, [0,0]) and distance(ballCenter, handCoord) < 120): # Adjusted distance maybe needed
                    shooting_pose['ball_in_hand'] = True
                    # Only update if angle was actually calculated this frame
                    if elbowAngle > 0: 
                         shooting_pose['elbow_angle'] = min(shooting_pose['elbow_angle'], elbowAngle) if shooting_pose['elbow_angle'] != 370 else elbowAngle
                    if kneeAngle > 0:
                         shooting_pose['knee_angle'] = min(shooting_pose['knee_angle'], kneeAngle) if shooting_pose['knee_angle'] != 370 else kneeAngle
                else:
                    # Update angle lists only when ball leaves hand
                    if shooting_pose['ball_in_hand']:
                        if shooting_pose['elbow_angle'] != 370: # Check if angle was updated
                             shooting_pose['elbow_angle_list'].append(shooting_pose['elbow_angle'])
                        if shooting_pose['knee_angle'] != 370: # Check if angle was updated
                             shooting_pose['knee_angle_list'].append(shooting_pose['knee_angle'])
                        shooting_pose['ballInHand_frames_list'].append(shooting_pose['ballInHand_frames'])
                        # Reset for next possession
                        shooting_pose['elbow_angle'] = 370
                        shooting_pose['knee_angle'] = 370
                        shooting_pose['ballInHand_frames'] = 0
                        
                    shooting_pose['ball_in_hand'] = False

                # During Shooting (ball y-coordinate is above hoop height)
                if(ymin < (previous['hoop_height'])):
                    if(not during_shooting['isShooting']):
                        during_shooting['isShooting'] = True

                    during_shooting['balls_during_shooting'].append(
                        [xCoor, yCoor])

                    #calculating release angle
                    if(len(during_shooting['balls_during_shooting']) == 2):
                        first_shooting_point = during_shooting['balls_during_shooting'][0]
                        # Calculate release angle relative to horizontal
                        p1 = np.array(first_shooting_point)
                        p2 = np.array(during_shooting['balls_during_shooting'][1])
                        delta_y = p1[1] - p2[1] # y decreases upwards in image coords
                        delta_x = p2[0] - p1[0]
                        release_angle = np.degrees(np.arctan2(delta_y, delta_x))
                        # Adjust angle to be 0-90 range typical for release angle definition
                        if release_angle < 0: release_angle += 180 # Handle angles in 2nd quadrant if needed based on definition
                        if release_angle > 90: release_angle = 180 - release_angle # Mirror angles > 90

                        during_shooting['release_angle_list'].append(round(release_angle, 2))
                        during_shooting['release_point'] = first_shooting_point
                        shot_result['release_displayFrames'] = 30
                        print("release angle:", release_angle)

                    #draw purple circle for ball during shot
                    cv2.circle(img=frame, center=(xCoor, yCoor), radius=7,
                               color=(235, 103, 193), thickness=3)
                    cv2.circle(img=trace, center=(xCoor, yCoor), radius=7,
                               color=(235, 103, 193), thickness=3)

                # Ball detected below hoop level after potentially being above it
                elif(ymin >= (previous['hoop_height'] - 30) and (distance(ballCenter, previous['ball']) < 100)):
                    # If we were tracking a shot (isShooting == True), judge it now
                    if(during_shooting['isShooting']):
                        # Assume SCORE if x-coordinate is within hoop boundaries, else MISS
                        if(xCoor >= previous['hoop'][0] and xCoor <= previous['hoop'][2]):
                            shooting_result['attempts'] += 1
                            shooting_result['made'] += 1
                            shot_result['displayFrames'] = 10
                            shot_result['judgement'] = "SCORE"
                            print("SCORE")
                            # draw green trace for made shot
                            if len(during_shooting['balls_during_shooting']) > 0:
                                trajectory_fit(during_shooting['balls_during_shooting'], height, width, shot_result['judgement'], fig)
                                points = np.asarray(during_shooting['balls_during_shooting'], dtype=np.int32)
                                cv2.polylines(trace, [points], False, color=(82, 168, 50), thickness=2, lineType=cv2.LINE_AA)
                                for ballCoor in during_shooting['balls_during_shooting']:
                                    cv2.circle(img=trace, center=(ballCoor[0], ballCoor[1]), radius=10, color=(82, 168, 50), thickness=-1)
                        else: # Miss
                            shooting_result['attempts'] += 1
                            shooting_result['miss'] += 1
                            shot_result['displayFrames'] = 10
                            shot_result['judgement'] = "MISS"
                            print("miss")
                            # draw red trace for missed shot
                            if len(during_shooting['balls_during_shooting']) > 0:
                                trajectory_fit(during_shooting['balls_during_shooting'], height, width, shot_result['judgement'], fig)
                                points = np.asarray(during_shooting['balls_during_shooting'], dtype=np.int32)
                                cv2.polylines(trace, [points], color=(0, 0, 255), isClosed=False, thickness=2, lineType=cv2.LINE_AA)
                                for ballCoor in during_shooting['balls_during_shooting']:
                                    cv2.circle(img=trace, center=(ballCoor[0], ballCoor[1]), radius=10, color=(0, 0, 255), thickness=-1)

                        # Reset shooting state regardless of make/miss
                        during_shooting['balls_during_shooting'].clear()
                        during_shooting['isShooting'] = False

                    # Ball detected near hoop level, but not part of a shot OR shot just ended. Draw Blue.
                    cv2.circle(img=frame, center=(xCoor, yCoor), radius=7,
                               color=(255, 117, 20), thickness=-1)
                    cv2.circle(img=trace, center=(xCoor, yCoor), radius=7,
                               color=(255, 117, 20), thickness=-1)
                    previous['ball'] = ballCenter # Update last known ball position

                # Ball detected far below hoop - normal state
                else:
                     # Update angle lists when ball is not near hand and not shooting
                    if shooting_pose['ball_in_hand']:
                        if shooting_pose['elbow_angle'] != 370: # Check if angle was updated
                             shooting_pose['elbow_angle_list'].append(shooting_pose['elbow_angle'])
                        if shooting_pose['knee_angle'] != 370: # Check if angle was updated
                             shooting_pose['knee_angle_list'].append(shooting_pose['knee_angle'])
                        shooting_pose['ballInHand_frames_list'].append(shooting_pose['ballInHand_frames'])
                        # Reset for next possession
                        shooting_pose['elbow_angle'] = 370
                        shooting_pose['knee_angle'] = 370
                        shooting_pose['ballInHand_frames'] = 0
                    shooting_pose['ball_in_hand'] = False
                    
                    # draw blue circle for normal ball state
                    cv2.circle(img=frame, center=(xCoor, yCoor), radius=7,
                               color=(255, 117, 20), thickness=-1)
                    cv2.circle(img=trace, center=(xCoor, yCoor), radius=7,
                               color=(255, 117, 20), thickness=-1)
                    previous['ball'] = ballCenter # Update last known ball position

            # Hoop Detection (class 0)
            elif (classes[0][i] == 0):
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                # Update hoop position if this detection is higher (smaller ymin) than the last known one
                if(ymin < previous['hoop'][3] or previous['hoop'][3] == 0):
                    previous['hoop'] = np.array([xmin, ymax, xmax, ymin])
                    previous['hoop_height'] = int(np.mean([ymin, ymax]))

    # Display shot judgement text on frame
    if(shot_result['displayFrames']):
        cv2.putText(frame, shot_result['judgement'],
                    (int(width/2) - 50, 100), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 6)

    return frame, trace


def detect_image(img, response):
    """Process a single image for detection using the appropriate model.
    
    This function can use either the TensorFlow model or the YOLO model,
    depending on which one has been initialized.
    
    Args:
        img (numpy.ndarray): Input image
        response (list): List to append detection results to
    
    Returns:
        numpy.ndarray: Annotated image with detections
    """
    # Get model (assuming it has been initialized)
    model = yolo_init()
    height, width = img.shape[:2]
    boxes, scores, classes, num_detections = model.run(img)


    for i, box in enumerate(boxes[0]):
        # print("detect")
        if (scores[0][i] > 0.5):
            # valid_detections += 1
            ymin = int((box[0] * height))
            xmin = int((box[1] * width))
            ymax = int((box[2] * height))
            xmax = int((box[3] * width))
            xCoor = int(np.mean([xmin, xmax]))
            yCoor = int(np.mean([ymin, ymax]))
            if(classes[0][i] == 1):  # basketball
                cv2.circle(img=img, center=(xCoor, yCoor), radius=25,
                            color=(255, 0, 0), thickness=-1)
                cv2.putText(img, "BALL", (xCoor - 50, yCoor - 50),
                            cv2.FONT_HERSHEY_COMPLEX, 3, (255, 0, 0), 8)
                print("add basketball")
                response.append({
                    'class': 'Basketball',
                    'detection_detail': {
                        'confidence': float("{:.5f}".format(scores[0][i])),
                        'center_coordinate': {'x': xCoor, 'y': yCoor},
                        'box_boundary': {'x_min': xmin, 'x_max': xmax, 'y_min': ymin, 'y_max': ymax}
                    }
                })
            if(classes[0][i] == 2):  # Rim
                cv2.rectangle(img, (xmin, ymax),
                                (xmax, ymin), (48, 124, 255), 10)
                cv2.putText(img, "HOOP", (xCoor - 65, yCoor - 65),
                            cv2.FONT_HERSHEY_COMPLEX, 3, (48, 124, 255), 8)
                print("add hoop")
                response.append({
                    'class': 'Hoop',
                    'detection_detail': {
                        'confidence': float("{:.5f}".format(scores[0][i])),
                        'center_coordinate': {'x': xCoor, 'y': yCoor},
                        'box_boundary': {'x_min': xmin, 'x_max': xmax, 'y_min': ymin, 'y_max': ymax}
                    }
                })
    
    # if(valid_detections < 2):
    #     for i in range(2):
    #         response.append({
    #             'class': 'Not Found',
    #             'detection_detail': {
    #                 'confidence': 0.0,
    #                 'center_coordinate': {'x': 0, 'y': 0},
    #                 'box_boundary': {'x_min': 0, 'x_max': 0, 'y_min': 0, 'y_max': 0}
    #             }
    #         })
            
    return img

def detect_API(response, img):
    height, width = img.shape[:2]
    model = yolo_init()
    boxes, scores, classes, num_detections = model.run(img)
    

    for i, box in enumerate(boxes[0]):
        if (scores[0][i] > 0.5):
            ymin = int((box[0] * height))
            xmin = int((box[1] * width))
            ymax = int((box[2] * height))
            xmax = int((box[3] * width))

        if(classes[0][i] == 1):
            response.append({
                "label": "basketball", 
                "confidence": round(float(scores[0][i]), 2), 
                "coordinates": [xmin, ymin, xmax, ymax]
            })
        elif (classes[0][i] == 0):
            response.append({
                "label": "hoop", 
                "confidence": round(float(scores[0][i]), 2), 
                "coordinates": [xmin, ymin, xmax, ymax]
            })

