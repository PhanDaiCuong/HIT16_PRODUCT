
import base64
from typing import List, Optional, Dict

import cv2
import numpy as np
from fastapi import HTTPException, status
from pydantic import BaseModel, Field, validator
class ImageDetectionRequest(BaseModel):
   
    image: str = Field(..., description="Base64 encoded image string")
    
    class Config:
        schema_extra = {
            "example": {
                "image": "iVBORw0KGgoAAAANS..."
            }
        }

class DetectedObject(BaseModel):
    
    bbox: List[float] = Field(..., description="Bounding box [x1, y1, x2, y2]")
    confidence: float = Field(..., description="Detection confidence", ge=0.0, le=1.0)
    class_name: str = Field(..., description="'car' hoặc 'free'")
    
    @validator('class_name')
    def validate_class_name(cls, v):
        if v not in ["car", "free"]:
            raise ValueError(f"Class name must be 'car' or 'free', got '{v}'")
        return v

class ParkingSpot(BaseModel):
    
    id: int = Field(..., description="ID của parking spot")
    is_occupied: bool = Field(..., description="Spot có xe hay không")
    status: str = Field(..., description="'occupied', 'free', hoặc 'unknown'")
    polygon: List[List[float]] = Field(..., description="Polygon points")
    detection_type: Optional[str] = Field(None, description="'car' hoặc 'free'")
    detected_object: Optional[DetectedObject] = Field(None, description="Object info (nếu có)")
    
    @validator('status')
    def validate_status(cls, v):
        if v not in ["occupied", "free", "unknown"]:
            raise ValueError(f"Status must be 'occupied', 'free', or 'unknown'")
        return v


class DetectionSummary(BaseModel):
    total_spots: int = Field(..., description="Tổng số parking spots")
    occupied_count: int = Field(..., description="Số spots có xe")
    free_count: int = Field(..., description="Số spots trống (model detect)")
    unknown_count: int = Field(..., description="Số spots không detect được")
    vacant_count: int = Field(..., description="Tổng spots trống (free + unknown)")
    occupancy_rate: float = Field(..., description="% lấp đầy", ge=0.0, le=100.0)


class DetectionResponse(BaseModel):
    spots: List[ParkingSpot] = Field(..., description="Tất cả parking spots")
    summary: DetectionSummary = Field(..., description="Summary statistics")
    detections: Optional[Dict] = Field(None, description="Raw data (optional)")
class PolygonConfig(BaseModel):
   
    id: int
    points: List[List[float]]


class DetectionConfig(BaseModel):
    car_confidence: float = 0.40
    free_confidence: float = 0.25
    general_confidence: float = 0.25
    device: str = "cpu"
    image_size: int = 640


class DetectRequest(BaseModel):
    """Request body gửi lên để phát hiện bãi đỗ xe từ ảnh."""
    image: str = Field(..., description="Base64 string (có hoặc không có data URI prefix)")
    polygon_id: Optional[str] = Field(default=None, description="Tên file polygon (không kèm .json)")
    config: Optional[DetectionConfig] = Field(default=None, description="Cấu hình confidence (tuỳ chọn)")

    def to_numpy(self) -> np.ndarray:
        """Decode base64 image → numpy BGR array. Raise HTTPException nếu lỗi."""
        b64 = self.image.split(",", 1)[-1] if "," in self.image else self.image
        try:
            image_bytes = base64.b64decode(b64)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Không thể giải mã base64: {exc}",
            )
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Không đọc được ảnh. Hãy đảm bảo đây là JPEG/PNG hợp lệ.",
            )
        return img



# ========================== VIDEO SCHEMAS ==========================

class FrameDetectionResult(BaseModel):
    """Kết quả phát hiện cho một frame trong video."""
    frame_number: int = Field(..., description="Số thứ tự frame (0-indexed)")
    summary: DetectionSummary = Field(..., description="Thống kê parking spots trong frame này")
    spots: List[ParkingSpot] = Field(default_factory=list, description="Trạng thái từng ô trong frame")
    annotated_frame_b64: Optional[str] = Field(
        None,
        description="Frame JPEG đã vẽ polygon, encode base64 (chỉ có khi return_frames=true)"
    )


class VideoDetectionResponse(BaseModel):
    """Kết quả tổng hợp sau khi xử lý toàn bộ video."""
    total_frames_processed: int = Field(..., description="Số frame đã được detect")
    total_frames_read: int = Field(..., description="Tổng số frame đã đọc (gồm cả frame bỏ qua)")
    frames: List[FrameDetectionResult] = Field(..., description="Kết quả từng frame")
    overall_summary: DetectionSummary = Field(..., description="Thống kê trung bình toàn video")
