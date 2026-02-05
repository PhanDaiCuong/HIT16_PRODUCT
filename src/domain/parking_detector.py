import os
import logging
from typing import List, Dict, Tuple, Optional
import numpy as np
import cv2
from ultralytics import YOLO


try:
    from ..utils.configs import (
        CONFIDENCE_THRESHOLD as DEFAULT_CONFIDENCE,
        FRAME_SKIP as DEFAULT_FRAME_SKIP,
        DEVICE as DEFAULT_DEVICE,
        IMAGE_SIZE as DEFAULT_IMAGE_SIZE,
        MODEL_PATH as DEFAULT_MODEL_PATH
    )
    CONFIG_AVAILABLE = True
except ImportError:
    
    DEFAULT_CONFIDENCE = 0.5
    DEFAULT_FRAME_SKIP = 5
    DEFAULT_DEVICE = "cpu"
    DEFAULT_IMAGE_SIZE = 640
    DEFAULT_MODEL_PATH = "models/best.pt"
    CONFIG_AVAILABLE = False

logger = logging.getLogger(__name__)

if CONFIG_AVAILABLE:
    logger.info("Loaded config from utils.configs")
else:
    logger.warning("Config not found, using default values")


class ParkingDetector:
    def __init__(
        self,
        polygons: List[Dict],  # ✅ Required - phải có
        model_path: str = DEFAULT_MODEL_PATH,
        confidence_threshold: float = DEFAULT_CONFIDENCE,
        frame_skip: int = DEFAULT_FRAME_SKIP,
        device: str = DEFAULT_DEVICE,
        image_size: int = DEFAULT_IMAGE_SIZE
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        if not polygons or len(polygons)==0:
            raise ValueError("Polygons list cannot be empty")
        if not 0<=confidence_threshold<=1:
            raise ValueError(f"Confidence threshold must be between 0 and 1, got {confidence_threshold}")
        if not isinstance(frame_skip, int) or frame_skip < 0:
            raise ValueError(f"Frame skip must be non-negative integer, got {frame_skip}")
        if not device in ["cuda", "cpu", "mps"]:
            raise ValueError(f"Device must be 'cuda', 'cpu' or 'mps', got {device}")
        if not isinstance(image_size, int) or not (320 <= image_size <= 1920):
            raise ValueError(f"Image size must be integer between 320-1920 pixels, got {image_size}")
        
        self.polygons =polygons
        self.model_path=model_path
        self.confidence_threshold=confidence_threshold
        self.frame_skip=frame_skip
        self.device=device
        self.image_size=image_size
        
        logger.info(f"loading model from {model_path}")
        try:
            self.model=YOLO(model_path)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
        logger.info(
            f"ParkingDetector initialized:\n"
            f"  - {len(polygons)} parking spots\n"
            f"  - Confidence: {confidence_threshold}\n"
            f"  - Device: {device}\n"
            f"  - Image size: {image_size}"
        )
    def detect_vehicles(self,image:np.ndarray)->List[Dict]:
        if image is None:
            logger.error("Image is None")
            return []
        if not isinstance(image,np.ndarray):
            logger.error(f"Image must be numpy array, got {type(image)}")
            return []
        if image.size==0:
            logger.error("img is empty")
            return []
        logger.debug(f"Running YOLO detection on image shape: {image.shape}")
        try:
            results=self.model(
                image,
                verbose=False,
                device=self.device,
                imgsz=self.image_size,
                conf=self.confidence_threshold,     
            )
        except Exception as e:
            logger.error(f"failed to run Yolo:{e}")
            return []
        detections=[]
        try:
            for result in results:
                boxes=result.boxes
                if boxes is None or len(boxes)==0:
                    continue
                for box in boxes:
                    x1,y1,x2,y2=box.xyxy[0].cpu().numpy()
                    confidence=float(box.conf[0])
                    class_id=int(box.cls[0])
                    class_name=self.model.names.get(class_id, f"class_{class_id}")
                    detections.append({
                        'bbox':[float(x1),float(y1),float(x2),float(y2)],
                        'confidence':confidence,
                        'class_id':class_id,
                        'class_name':class_name
                    })
        except Exception as e:
            logger.error(f"failed to process detections: {e}")
            return []
        logger.debug(f"Found {len(detections)} vehicles")
        return detections
    def point_in_polygon(
        self,
        point: Tuple[float,float],
        polygon_points: List[List[float]]
    ) -> bool:
        polygon = np.array(polygon_points,dtype=np.int32)
        result = cv2.pointPolygonTest(polygon,point,False)
        return result>=0

    def check_polygon_occupancy(self,detections:List[Dict],polygon:Dict)->bool:
        polygon_points=polygon['points']

        for detection in detections:
            bbox=detection['bbox']
            x1,y1,x2,y2=bbox
            x_center=(x1 + x2)/2
            y_center=(y1+y2)/2
            if self.point_in_polygon((x_center,y_center),polygon_points):
                logger.debug(
                    f"Polygon{polygon.get('id','?',)} occupied by"
                f"{detection['class_name']}({detection['confidence']:.2f})"
                )
                return True
        return False
    def detect(self,image:np.ndarray)->dict:

        logger.info(f"Starting detection on image: {image.shape}")

        detections=self.detect_vehicles(image)
        logger.info(f"Detected {len(detections)} vehicles")

        spots=[]
        occupied_count=0

        for polygon in self.polygons:
            is_occupied=self.check_polygon_occupancy(detections,polygon)
            spots.append({
                'id':polygon.get('id',len(spots)+1),
                'is_occupied':is_occupied,
                'polygon':polygon['points']
            })
            if is_occupied:
                occupied_count+=1
        total_spots=len(self.polygons)
        vacant_count=total_spots-occupied_count
        occupancy_rate=(occupied_count/total_spots)*100 if total_spots>0 else 0

        logger.info(
            f"Detection completed:{occupied_count}/{total_spots} occupied"
            f"({occupancy_rate:.1f}%)"
        )
        return{
            'spots':spots,
            'occupied-count':occupied_count,
            'vacant-count':vacant_count,
            'occupancy-rate':occupancy_rate
        }
    def detect_video(self, video_path: str, skip_frames: int = None):
        if skip_frames is None:
            skip_frames = self.frame_skip
        logger.info(f"Processing video: {video_path} (skip={skip_frames})")
        cap = None
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"failed to open video: {video_path}")
                raise ValueError(f"failed to open video: {video_path}")
            logger.info("video opened!")
            frame_count = 0
            processed_count = 0
            consecutive_errors = 0
            max_errors = 10
            
            while cap.isOpened():
                try:
                    ret, frame = cap.read()
                    if not ret:
                        logger.debug(f"end of video at frame: {frame_count}")
                        break
                    consecutive_errors=0
                    if frame_count % (skip_frames+1)==0:
                        try:
                            result=self.detect(frame)
                            result['frame_number']=frame_count
                            yield result
                            processed_count+=1
                        except Exception as e:
                            logger.warning(f"failed to process frame {frame_count}:{e}")
                    frame_count+=1
                except Exception as e:
                    consecutive_errors+=1
                    logger.warning(
                        f"error reading frame {frame_count}:{e}"
                        f"({consecutive_errors}/{max_errors} errors)"
                    )
                    if consecutive_errors>=max_errors:
                        logger.error("too many errors,stopping video processing")
                        break
                    frame_count+=1
                    continue
        except Exception as e:
            logger.error(f"unexpected error: {e}")
            raise
        finally:
            if cap is not None:
                cap.release()
                logger.info("video released")
            logger.info(
                f"video processing completed: {processed_count} frames processed "
                f"({frame_count} total frames)"
            )
