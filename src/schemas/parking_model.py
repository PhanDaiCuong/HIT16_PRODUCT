
from typing import List, Optional, Dict
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
