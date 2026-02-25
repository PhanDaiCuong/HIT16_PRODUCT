import os
import logging
from typing import List, Dict, Tuple, Optional
import numpy as np
import cv2
from ultralytics import YOLO


from ..utils.configs import (
    CONFIDENCE_THRESHOLD as DEFAULT_CONFIDENCE,
    CAR_CONFIDENCE_THRESHOLD as DEFAULT_CAR_CONFIDENCE,
    FREE_CONFIDENCE_THRESHOLD as DEFAULT_FREE_CONFIDENCE,
    GENERAL_CONFIDENCE_THRESHOLD as DEFAULT_GENERAL_CONFIDENCE,
    FRAME_SKIP as DEFAULT_FRAME_SKIP,
    DEVICE as DEFAULT_DEVICE,
    IMAGE_SIZE as DEFAULT_IMAGE_SIZE,
    MODEL_PATH as DEFAULT_MODEL_PATH
)

logger = logging.getLogger(__name__)

_MODEL_CACHE = {}

def get_or_load_model(model_path: str, device: str = "cpu") -> YOLO:
    cache_key = f"{model_path}_{device}"
  
    if cache_key in _MODEL_CACHE:
        logger.info(f"Using cached model from {model_path}")
        return _MODEL_CACHE[cache_key]
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    
    logger.info(f"Loading new model from {model_path}")
    try:
        model = YOLO(model_path)
        _MODEL_CACHE[cache_key] = model
        logger.info(f"Model loaded and cached successfully (key: {cache_key})")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

def clear_model_cache():
    global _MODEL_CACHE
    _MODEL_CACHE.clear()
    logger.info("Model cache cleared")

class ParkingDetector:
    def __init__(
        self,
        polygons: List[Dict], 
        model_path: str = DEFAULT_MODEL_PATH,
        confidence_threshold: float = DEFAULT_CONFIDENCE,  
        car_confidence: Optional[float] = None,
        free_confidence: Optional[float] = None,
        general_confidence: Optional[float] = None,
        frame_skip: int = DEFAULT_FRAME_SKIP,
        device: str = DEFAULT_DEVICE,
        image_size: int = DEFAULT_IMAGE_SIZE
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        if not polygons or len(polygons) == 0:
            raise ValueError("Polygons list cannot be empty")
        if not isinstance(frame_skip, int) or frame_skip < 0:
            raise ValueError(f"Frame skip must be non-negative integer, got {frame_skip}")
        if not device in ["cuda", "cpu"]:
            raise ValueError(f"Device must be 'cuda' or 'cpu', got {device}")
        if not isinstance(image_size, int) or not (320 <= image_size <= 1920):
            raise ValueError(f"Image size must be integer between 320-1920 pixels, got {image_size}")
        
        
        self.car_confidence = car_confidence if car_confidence is not None else DEFAULT_CAR_CONFIDENCE
        self.free_confidence = free_confidence if free_confidence is not None else DEFAULT_FREE_CONFIDENCE
        self.general_confidence = general_confidence if general_confidence is not None else DEFAULT_GENERAL_CONFIDENCE
        
        
        if not 0 <= self.car_confidence <= 1:
            raise ValueError(f"Car confidence must be between 0 and 1, got {self.car_confidence}")
        if not 0 <= self.free_confidence <= 1:
            raise ValueError(f"Free confidence must be between 0 and 1, got {self.free_confidence}")
        if not 0 <= self.general_confidence <= 1:
            raise ValueError(f"General confidence must be between 0 and 1, got {self.general_confidence}")
        
        self.polygons = polygons
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold  
        self.frame_skip = frame_skip
        self.device = device
        self.image_size = image_size
        
        self.model = get_or_load_model(model_path, device)
        
        # --- AUTO-SCALING ---
        self.original_polygons = [p.copy() for p in polygons]
        self.design_resolution = self._estimate_design_resolution()
        self.current_polygons = self.original_polygons
        self.current_resolution = self.design_resolution

        logger.info(
            f"ParkingDetector initialized:\n"
            f"  - {len(polygons)} parking spots\n"
            f"  - Car confidence: {self.car_confidence}\n"
            f"  - Free confidence: {self.free_confidence}\n"
            f"  - General confidence: {self.general_confidence}\n"
            f"  - Device: {device}\n"
            f"  - Image size: {image_size}\n"
            f"  - Estimated Design Resolution: {self.design_resolution}"
        )

    def _estimate_design_resolution(self) -> Tuple[int, int]:
        """
        Ước lượng độ phân giải gốc mà các polygon này được vẽ bên trên.
        Hệ thống sẽ chọn độ phân giải tiêu chuẩn NHỎ NHẤT mà vẫn chứa được hết các điểm.
        """
        max_x = 0
        max_y = 0
        for poly in self.original_polygons:
            for p in poly['points']:
                max_x = max(max_x, p[0])
                max_y = max(max_y, p[1])
        
        # Danh sách các độ phân giải phổ biến (W, H)
        standards = [
            (640, 360),   # nHD
            (640, 480),   # VGA
            (800, 600),   # SVGA
            (1024, 768),  # XGA
            (1280, 720),  # HD
            (1920, 1080), # FHD
            (2560, 1440), # 2K
            (3840, 2160)  # 4K
        ]

        for w, h in standards:
            if max_x <= w and max_y <= h:
                return (w, h)
        
        # Nếu vượt quá các chuẩn trên, lấy max + margin
        return (int(max_x + 20), int(max_y + 20))

    def _rescale_polygons(self, new_resolution: Tuple[int, int]):
        """Căng chỉnh lại tọa độ polygon để khớp với độ phân giải mới."""
        if new_resolution == self.current_resolution:
            return
        
        # Tránh chia cho 0
        base_w = max(1, self.design_resolution[0])
        base_h = max(1, self.design_resolution[1])
        
        scale_x = new_resolution[0] / base_w
        scale_y = new_resolution[1] / base_h
        
        logger.info(f"Auto-rescaling polygons: {self.design_resolution} -> {new_resolution} (Scale: {scale_x:.2f}x, {scale_y:.2f}x)")
        
        new_polygons = []
        for poly in self.original_polygons:
            new_poly = poly.copy()
            # Quan trọng: tạo list mới để không ghi đè vào original_polygons
            new_poly['points'] = [[p[0] * scale_x, p[1] * scale_y] for p in poly['points']]
            new_polygons.append(new_poly)
            
        self.current_polygons = new_polygons
        self.current_resolution = new_resolution
        self.polygons = new_polygons
    def detect_objects(self, image: np.ndarray) -> Dict[str, List[Dict]]:
        
        if image is None:
            logger.error("Image is None")
            return {'cars': [], 'free_spots': []}
        if not isinstance(image, np.ndarray):
            logger.error(f"Image must be numpy array, got {type(image)}")
            return {'cars': [], 'free_spots': []}
        if image.size == 0:
            logger.error("Image is empty")
            return {'cars': [], 'free_spots': []}
        
        logger.debug(f"Running YOLO detection on image shape: {image.shape}")
        try:
            
            results = self.model(
                image,
                verbose=False,
                device=self.device,
                imgsz=self.image_size,
                conf=self.general_confidence,  
                iou=0.7,  
            )
        except Exception as e:
            logger.error(f"Failed to run YOLO: {e}")
            return {'cars': [], 'free_spots': []}
        
        cars = []
        free_spots = []
        filtered_count = {'car': 0, 'free': 0}
        
        try:
            for result in results:
                boxes = result.boxes
                if boxes is None or len(boxes) == 0:
                    continue
                    
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = self.model.names.get(class_id, f"class_{class_id}")
                    
                    
                    if class_name == 'car':
                        if confidence < self.car_confidence:
                            filtered_count['car'] += 1
                            continue  
                    elif class_name == 'free':
                        if confidence < self.free_confidence:
                            filtered_count['free'] += 1
                            continue  
                    else:
                        
                        continue
                    
                    detection = {
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'center': [(float(x1) + float(x2)) / 2, (float(y1) + float(y2)) / 2],
                        'confidence': confidence,
                        'class_id': class_id,
                        'class_name': class_name
                    }
                    
                    
                    if class_name == 'car':
                        cars.append(detection)
                    elif class_name == 'free':
                        free_spots.append(detection)
                        
        except Exception as e:
            logger.error(f"Failed to process detections: {e}")
            return {'cars': [], 'free_spots': []}
        
        logger.debug(
            f"Found {len(cars)} cars and {len(free_spots)} free spots "
            f"(filtered: {filtered_count['car']} cars, {filtered_count['free']} free spots)"
        )
        return {'cars': cars, 'free_spots': free_spots}
    def point_in_polygon(
        self,
        point: Tuple[float,float],
        polygon_points: List[List[float]]
    ) -> bool:
        polygon = np.array(polygon_points,dtype=np.int32)
        result = cv2.pointPolygonTest(polygon,point,False)
        return result>=0

    def check_polygon_occupancy(self, detections: Dict[str, List[Dict]], polygon: Dict) -> Dict:
       
        polygon_points = polygon['points']
        polygon_id = polygon.get('id', '?')
        for car in detections['cars']:
            x_center, y_center = car['center']
            if self.point_in_polygon((x_center, y_center), polygon_points):
                logger.debug(
                    f"Polygon {polygon_id} occupied by car "
                    f"(confidence: {car['confidence']:.2f})"
                )
                return {
                    'is_occupied': True,
                    'status': 'occupied',
                    'detected_object': car,
                    'detection_type': 'car'
                }
        for free_spot in detections['free_spots']:
            x_center, y_center = free_spot['center']
            if self.point_in_polygon((x_center, y_center), polygon_points):
                logger.debug(
                    f"Polygon {polygon_id} detected as free "
                    f"(confidence: {free_spot['confidence']:.2f})"
                )
                return {
                    'is_occupied': False,
                    'status': 'free',
                    'detected_object': free_spot,
                    'detection_type': 'free'
                }
        logger.debug(f"Polygon {polygon_id}: no detection")
        return {
            'is_occupied': False,
            'status': 'unknown',
            'detected_object': None,
            'detection_type': None
        }
    def detect(self, image: np.ndarray) -> dict:
        if image is None:
            return {'spots': [], 'summary': {}}
            
        h, w = image.shape[:2]
        self._rescale_polygons((w, h))

        logger.info(f"Starting detection on image: {image.shape}")
 
        detections = self.detect_objects(image)
        logger.info(
            f"Detected {len(detections['cars'])} cars and "
            f"{len(detections['free_spots'])} free spots"
        )
        
        spots = []
        occupied_count = 0
        free_count = 0
        unknown_count = 0
        
        for polygon in self.polygons:
            occupancy_info = self.check_polygon_occupancy(detections, polygon)
            
            spot_data = {
                'id': polygon.get('id', len(spots) + 1),
                'polygon': polygon['points'],
                'is_occupied': occupancy_info['is_occupied'],
                'status': occupancy_info['status'],
                'detection_type': occupancy_info['detection_type']
            }
   
            if occupancy_info['detected_object']:
                spot_data['detected_object'] = {
                    'bbox': occupancy_info['detected_object']['bbox'],
                    'confidence': occupancy_info['detected_object']['confidence'],
                    'class_name': occupancy_info['detected_object']['class_name']
                }
            
            spots.append(spot_data)
            if occupancy_info['status'] == 'occupied':
                occupied_count += 1
            elif occupancy_info['status'] == 'free':
                free_count += 1
            else:
                unknown_count += 1
        
        total_spots = len(self.polygons)
        vacant_count = free_count + unknown_count
        occupancy_rate = (occupied_count / total_spots * 100) if total_spots > 0 else 0
        
        logger.info(
            f"Detection completed: {occupied_count} occupied, "
            f"{free_count} free, {unknown_count} unknown "
            f"({occupancy_rate:.1f}% occupancy)"
        )
        
        return {
            'spots': spots,
            'summary': {
                'total_spots': total_spots,
                'occupied_count': occupied_count,
                'free_count': free_count,
                'unknown_count': unknown_count,
                'vacant_count': vacant_count,
                'occupancy_rate': round(occupancy_rate, 2)
            },
            'detections': {
                'cars': detections['cars'],
                'free_spots': detections['free_spots']
            }
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
