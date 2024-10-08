import cv2
import numpy as np
from datetime import datetime
from typing import Dict, List


class CameraData:
    def __init__(self, camera_id: int, timestamp: datetime, vehicle_count: int, vehicle_types: Dict[str, int], traffic_density: float):
        self.camera_id = camera_id
        self.timestamp = timestamp
        self.vehicle_count = vehicle_count
        self.vehicle_types = vehicle_types
        self.traffic_density = traffic_density
        
        
        
class Camera:
    def __init__(self, id: int, resolution: str, frame_rate: float, camera_source: str):
       
        self.id = id
        self.resolution = resolution
        self.frame_rate = frame_rate
        self.camera_source = camera_source  
        
      
        self.capture = cv2.VideoCapture(camera_source)
      
        width, height = map(int, resolution.split('x'))
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.capture.set(cv2.CAP_PROP_FPS, frame_rate)
        
        
    def capture_frame(self):
        """
        Captures a frame from the camera feed.
        :return: Captured frame (image) or None if unable to capture.
        """
        ret, frame = self.capture.read()
        if ret:
            return frame
        else:
            print("Failed to capture frame")
            return None

    def detect_vehicles(self, frame) -> int:
    
        # Example using dummy data: Assuming vehicle count is 10
        vehicle_count = 10
        print(f"Detected {vehicle_count} vehicles.")
        return vehicle_count

    def estimate_traffic_density(self, vehicle_count: int, frame) -> float:
      
        frame_area = frame.shape[0] * frame.shape[1]  # height * width
        density = vehicle_count / frame_area
        print(f"Estimated traffic density: {density}")
        return density

    def classify_vehicle_types(self, frame) -> Dict[str, int]:
      
        # Placeholder
        vehicle_types = {
            'car': 8,
            'truck': 1,
            'bus': 1
        }
        print(f"Classified vehicle types: {vehicle_types}")
        return vehicle_types

    def process_camera_feed(self) -> CameraData:
        
        frame = self.capture_frame()
        if frame is None:
            return None 

        
        vehicle_count = self.detect_vehicles(frame)


        traffic_density = self.estimate_traffic_density(vehicle_count, frame)


        vehicle_types = self.classify_vehicle_types(frame)

        camera_data = CameraData(
            camera_id=self.id,
            timestamp=datetime.now(),
            vehicle_count=vehicle_count,
            vehicle_types=vehicle_types,
            traffic_density=traffic_density
        )

        return camera_data

    def release(self):
      
        self.capture.release()
        cv2.destroyAllWindows()
        
        
        
               
if __name__ == "__main__":
    camera = Camera(id=1, resolution='1920x1080', frame_rate=30, camera_source=0)  # 0 for default webcam

    try:
        while True:
            camera_data = camera.process_camera_feed()
            if camera_data:
                print(f"Camera ID: {camera_data.camera_id}")
                print(f"Timestamp: {camera_data.timestamp}")
                print(f"Vehicle Count: {camera_data.vehicle_count}")
                print(f"Vehicle Types: {camera_data.vehicle_types}")
                print(f"Traffic Density: {camera_data.traffic_density}")

            # Break the loop with 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        camera.release()