# Avi Shah - Basketball Shot Detector/Tracker - July 2023
# Refactored for compatibility with AI Basketball Analysis project

from ultralytics import YOLO
import cv2
import numpy as np
import torch
import math
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import in_hoop_region, get_device


class ShotDetector:
    def __init__(self, model_path=None):
        """
        Initialize the YOLOv8-based basketball shot detector.
        
        Args:
            model_path (str, optional): Path to the YOLO model. 
                                       If None, will use 'best.pt' in the same directory.
        """
        # Use provided model path or default to 'best.pt' in the same directory
        if model_path is None:
            # Try to find the model in the same directory as this file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(current_dir, "best.pt")
            
        self.model = YOLO(model_path)
        self.class_names = ['Basketball', 'Basketball Hoop']
        self.device = get_device()
        
        # Map YOLO class indices to match TensorFlow model indices
        # TensorFlow model: 0=Hoop, 1=Basketball, 2=Person
        # YOLO model: 0=Basketball, 1=Basketball Hoop
        self.class_map = {
            0: 1,  # Basketball -> 1
            1: 0   # Basketball Hoop -> 0
        }
        
        # Confidence threshold for detections
        self.conf_threshold = 0.5

    def predict(self, frame):
        """
        Run prediction on a single frame and return the YOLO results directly.
        Useful for debugging or when raw YOLO output is needed.
        
        Args:
            frame (numpy.ndarray): Input image
            
        Returns:
            ultralytics.yolo.engine.results.Results: YOLO detection results
        """
        return self.model(frame, stream=True, device=self.device)
        
    def run(self, frame):
        """
        Process a frame and return detection results in TensorFlow format.
        
        Args:
            frame (numpy.ndarray): Input image frame
            
        Returns:
            tuple: (boxes, scores, classes, num_detections) in TensorFlow model format
                - boxes: numpy array of shape [1, N, 4] with bounding boxes in [y1, x1, y2, x2] order
                - scores: numpy array of shape [1, N] with confidence scores
                - classes: numpy array of shape [1, N] with class indices
                - num_detections: Number of valid detections
        """
        # Get image dimensions
        height, width = frame.shape[:2]
        
        # Run YOLO detection
        results = self.model(frame, stream=True, device=self.device, verbose=False)
        
        # Initialize empty lists to store detections
        all_boxes = []
        all_scores = []
        all_classes = []
        
        # Process the results
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Get confidence score
                conf = float(box.conf[0])
                
                # Skip low confidence detections
                if conf < self.conf_threshold:
                    continue
                    
                # Get class index
                cls = int(box.cls[0])
                
                # Map YOLO class to TensorFlow class
                tf_cls = self.class_map.get(cls, 2)  # Default to 2 (person) if unknown class
                
                # Get bounding box in [x1, y1, x2, y2] format (absolute coordinates)
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Convert to TensorFlow format [y1, x1, y2, x2] (normalized coordinates)
                tf_box = [
                    float(y1) / height,  # y1
                    float(x1) / width,   # x1
                    float(y2) / height,  # y2
                    float(x2) / width    # x2
                ]
                
                all_boxes.append(tf_box)
                all_scores.append(conf)
                all_classes.append(tf_cls)
        
        # Calculate number of valid detections
        num_detections = len(all_boxes)
        
        # Ensure there's always at least one box (even if empty)
        if num_detections == 0:
            all_boxes.append([0.0, 0.0, 0.0, 0.0])
            all_scores.append(0.0)
            all_classes.append(0)
            num_detections = 0
        
        # Convert lists to numpy arrays and add batch dimension [1, num_detections, ...]
        boxes_np = np.array([all_boxes], dtype=np.float32)
        scores_np = np.array([all_scores], dtype=np.float32)
        classes_np = np.array([all_classes], dtype=np.float32)
        num_detections_np = np.array([num_detections], dtype=np.int32)
        
        return boxes_np, scores_np, classes_np, num_detections_np
    
    def detect_image(self, img, response=None):
        """
        Process a single image and optionally update a response list.
        Compatible with the original detect_image function.
        
        Args:
            img (numpy.ndarray): Input image
            response (list, optional): List to append detection results to
            
        Returns:
            numpy.ndarray: Annotated image
        """
        if response is None:
            response = []
            
        boxes, scores, classes, num_detections = self.run(img)
        height, width = img.shape[:2]
        
        output = img.copy()
        
        valid_detections = 0
        
        for i in range(int(num_detections[0])):
            if scores[0][i] > 0.5:
                valid_detections += 1
                
                # Convert normalized coordinates to pixel coordinates
                box = boxes[0][i]
                ymin = int(box[0] * height)
                xmin = int(box[1] * width)
                ymax = int(box[2] * height)
                xmax = int(box[3] * width)
                
                # Calculate center coordinates
                xCoor = int((xmin + xmax) / 2)
                yCoor = int((ymin + ymax) / 2)
                
                class_id = int(classes[0][i])
                
                if class_id == 1:  # Basketball
                    cv2.circle(img=output, center=(xCoor, yCoor), radius=25,
                               color=(255, 0, 0), thickness=-1)
                    cv2.putText(output, "BALL", (xCoor - 50, yCoor - 50),
                                cv2.FONT_HERSHEY_COMPLEX, 3, (255, 0, 0), 8)
                    
                    response.append({
                        'class': 'Basketball',
                        'detection_detail': {
                            'confidence': float("{:.5f}".format(scores[0][i])),
                            'center_coordinate': {'x': xCoor, 'y': yCoor},
                            'box_boundary': {'x_min': xmin, 'x_max': xmax, 'y_min': ymin, 'y_max': ymax}
                        }
                    })
                    
                elif class_id == 0:  # Hoop (changed from 2 to 0 to match TensorFlow model mapping)
                    cv2.rectangle(output, (xmin, ymax),
                                  (xmax, ymin), (48, 124, 255), 10)
                    cv2.putText(output, "HOOP", (xCoor - 65, yCoor - 65),
                                cv2.FONT_HERSHEY_COMPLEX, 3, (48, 124, 255), 8)
                    
                    response.append({
                        'class': 'Hoop',
                        'detection_detail': {
                            'confidence': float("{:.5f}".format(scores[0][i])),
                            'center_coordinate': {'x': xCoor, 'y': yCoor},
                            'box_boundary': {'x_min': xmin, 'x_max': xmax, 'y_min': ymin, 'y_max': ymax}
                        }
                    })
        
        # If not enough detections, pad the response with empty entries
        if valid_detections < 2:
            for i in range(2 - valid_detections):
                response.append({
                    'class': 'Not Found',
                    'detection_detail': {
                        'confidence': 0.0,
                        'center_coordinate': {'x': 0, 'y': 0},
                        'box_boundary': {'x_min': 0, 'x_max': 0, 'y_min': 0, 'y_max': 0}
                    }
                })
                
        return output
    
    def detect_API(self, response, img):
        """
        Process a single image for API responses.
        Compatible with the original detect_API function.
        
        Args:
            response (list): List to append detection results to
            img (numpy.ndarray): Input image
        """
        boxes, scores, classes, num_detections = self.run(img)
        height, width = img.shape[:2]
        
        for i in range(int(num_detections[0])):
            if scores[0][i] > 0.5:
                # Convert normalized coordinates to pixel coordinates
                box = boxes[0][i]
                ymin = int(box[0] * height)
                xmin = int(box[1] * width)
                ymax = int(box[2] * height)
                xmax = int(box[3] * width)
                
                class_id = int(classes[0][i])
                
                if class_id == 1:  # Basketball
                    response.append({
                        "label": "basketball", 
                        "confidence": round(float(scores[0][i]), 2), 
                        "coordinates": [xmin, ymin, xmax, ymax]
                    })
                elif class_id == 0:  # Hoop
                    response.append({
                        "label": "hoop", 
                        "confidence": round(float(scores[0][i]), 2), 
                        "coordinates": [xmin, ymin, xmax, ymax]
                    })
                # elif class_id == 2:  
                #     response.append({
                #         "label": "person", 
                #         "confidence": round(float(scores[0][i]), 2), 
                #         "coordinates": [xmin, ymin, xmax, ymax]
                #     })


# Legacy code below - kept for reference but not active
if __name__ == "__main__":
    # Example usage - show video with detections
    detector = ShotDetector()
    video_path = r"D:\AI-basketball-analysis\basketball_shot_detector_model\video_test_5.mp4"
    cap = cv2.VideoCapture(video_path)
    
    # Output video configuration (set save_output to True to save video)
    save_output = True
    output_path = "output_detection.mp4"
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
    else:
        # Get video properties for output writer
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Initialize video writer if saving output
        out = None
        if save_output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'XVID' for .avi
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"Saving output to {output_path}")
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"End of video - processed {frame_count} frames")
                break
                
            frame_count += 1
            
            # Run detection on the RGB frame
            boxes, scores, classes, num_detections = detector.run(frame)
            
            # Draw annotations on the frame
            for i in range(int(num_detections[0])):
                if scores[0][i] > 0.5:
                    # Convert normalized coordinates to pixel coordinates
                    box = boxes[0][i]
                    ymin = int(box[0] * height)
                    xmin = int(box[1] * width)
                    ymax = int(box[2] * height)
                    xmax = int(box[3] * width)
                    
                    # Calculate center coordinates
                    xCoor = int((xmin + xmax) / 2)
                    yCoor = int((ymin + ymax) / 2)
                    
                    class_id = int(classes[0][i])
                    
                    if class_id == 1:  # Basketball
                        cv2.circle(img=frame, center=(xCoor, yCoor), radius=25,
                                  color=(255, 0, 0), thickness=2)
                        cv2.putText(frame, "Basketball", (xCoor - 40, yCoor - 15),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    
                    elif class_id == 0:  # Hoop
                        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax),
                                     (48, 124, 255), 2)
                        cv2.putText(frame, "Hoop", (xCoor - 25, yCoor - 15),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (48, 124, 255), 2)
            
            # Add frame number to the top left
            cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Write the frame to output video if saving
            if save_output and out is not None:
                out.write(frame)
            
            # Show the frame with detections
            cv2.imshow("Basketball Shot Detection", frame)
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Use 1ms delay for faster processing
                break
        
        # Release resources
        cap.release()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()
        print("Video processing finished")

