import cv2
import numpy as np
from datetime import datetime
from typing import Dict, List


class VehicleDetector:
    def __init__(self, model_path='yolov3.weights', config_path='yolov3.cfg', confidence_threshold=0.5):
        """
        Initialize the Vehicle Detector with YOLO model.
        
        :param model_path: Path to pre-trained YOLO weights
        :param config_path: Path to YOLO configuration file
        :param confidence_threshold: Minimum confidence for detection
        """
        # Load YOLO
        self.net = cv2.dnn.readNet(model_path, config_path)
        
        # Load class names
        with open('coco.names', 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
        
        # Vehicle types to track
        self.vehicle_types = ['car', 'truck', 'bus', 'motorcycle', 'bicycle']
        
        # Confidence threshold
        self.confidence_threshold = confidence_threshold
    
    def detect_vehicles(self, frame):
        """
        Detect vehicles in a single frame.
        
        :param frame: Input video frame
        :return: List of detected vehicle information
        """
        height, width, _ = frame.shape
        
        # Create blob from frame
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        
        # Set input to the network
        self.net.setInput(blob)
        
        # Get output layer names
        output_layers_names = self.net.getUnconnectedOutLayersNames()
        
        # Forward pass through the network
        layer_outputs = self.net.forward(output_layers_names)
        
        # Lists to store detected vehicles
        vehicles = []
        
        # Process detection results
        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                # Get class name
                class_name = self.classes[class_id]
                
                # Filter for vehicles with high confidence
                if confidence > self.confidence_threshold and class_name in self.vehicle_types:
                    # Object detected is a vehicle
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    vehicles.append({
                        'type': class_name,
                        'bbox': [x, y, w, h],
                        'confidence': float(confidence)
                    })
        
        return vehicles
    
    def draw_vehicle_detections(self, frame, vehicles):
        """
        Draw bounding boxes and labels for detected vehicles.
        
        :param frame: Input frame
        :param vehicles: List of detected vehicles
        :return: Frame with vehicle detections drawn
        """
        for vehicle in vehicles:
            x, y, w, h = vehicle['bbox']
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{vehicle['type']} ({vehicle['confidence']:.2f})", 
                        (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame