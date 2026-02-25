"""
parking_detect.py
FastAPI router cho parking detection API.
Logic thực thi được đặt trong các utils module:
  - utils.polygon_utils → load_polygons()
  - utils.draw_utils    → annotate_frame()
  - utils.video_utils   → mjpeg_generator()
"""
import logging
import os
import tempfile
import uuid
from typing import Dict, List

from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile, status
from starlette.responses import StreamingResponse

from ..domain.parking_detector import ParkingDetector
from ..schemas.parking_model import DetectRequest, DetectionConfig, DetectionResponse
from ..utils.configs import POLYGON_PATH, POLYGONS_DIR
from ..utils.polygon_utils import load_polygons
from ..utils.video_utils import mjpeg_generator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/parking", tags=["Parking Detection"])

# Session store: session_id → đường dẫn file video tạm
_VIDEO_SESSIONS: Dict[str, str] = {}


# ========================== PRIVATE HELPERS ==========================

def _get_polygons(polygon_id: str = None) -> List[dict]:
    """Wrapper load polygon, đổi exception thuần → HTTPException."""
    path = POLYGON_PATH
    if polygon_id:
        path = os.path.join(POLYGONS_DIR, f"{polygon_id}.json")
    
    try:
        return load_polygons(path)
    except FileNotFoundError:
        # Fallback to default if custom not found
        if polygon_id:
            logger.warning(f"Polygon {polygon_id} not found, falling back to default.")
            return load_polygons(POLYGON_PATH)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Default polygon file not found.")
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))


def _make_detector(request: Request, polygons: List[dict], cfg: DetectionConfig) -> ParkingDetector:
    """Tạo ParkingDetector từ model đã pre-load trong app.state."""
    if getattr(request.app.state, "model", None) is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model YOLO chưa được load. Kiểm tra MODEL_PATH và restart server.",
        )
    try:
        return ParkingDetector(
            polygons=polygons,
            model_path=request.app.state.model_path,
            car_confidence=cfg.car_confidence,
            free_confidence=cfg.free_confidence,
            general_confidence=cfg.general_confidence,
            device=request.app.state.device,
            image_size=cfg.image_size,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))


async def _save_upload_to_temp(video: UploadFile) -> str:
    """Lưu UploadFile ra file tạm, trả về đường dẫn."""
    suffix = os.path.splitext(video.filename or "video.mp4")[1] or ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        tmp.write(await video.read())
        tmp.flush()
        tmp.close()
    except Exception as exc:
        tmp.close()
        os.unlink(tmp.name)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Không đọc được video: {exc}")
    return tmp.name


# ========================== ENDPOINTS ==========================

@router.post("/detect", response_model=DetectionResponse, summary="Phát hiện xe từ ảnh (base64)")
async def detect_parking(body: DetectRequest, request: Request):
    """Gửi ảnh base64 → nhận kết quả từng ô đỗ xe. Polygon tự load từ server."""
    detector = _make_detector(request, _get_polygons(body.polygon_id), body.config or DetectionConfig())
    try:
        result = detector.detect(body.to_numpy())
    except Exception as exc:
        logger.exception(f"Lỗi detection ảnh: {exc}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))

    s = result["summary"]
    logger.info(f"detect (area={body.polygon_id}): {s['occupied_count']} occupied, {s['free_count']} free")
    return result


@router.get("/polygons", summary="Danh sách các file polygon có sẵn")
async def list_polygons():
    """Trả về danh sách các ID khu vực (tên file json trong data/polygons)."""
    if not os.path.exists(POLYGONS_DIR):
        return []
    files = [f.replace(".json", "") for f in os.listdir(POLYGONS_DIR) if f.endswith(".json")]
    return sorted(files)


@router.post("/session/upload", summary="Upload video, nhận session_id để stream")
async def upload_video_session(
    video: UploadFile = File(...),
    polygon_id: str = Form(default=None)
):
    """Upload video → trả session_id. Dùng session_id để gọi GET /session/{id}/stream."""
    session_id = str(uuid.uuid4())
    tmp_path   = await _save_upload_to_temp(video)
    _VIDEO_SESSIONS[session_id] = {
        "path": tmp_path,
        "polygon_id": polygon_id
    }
    logger.info(f"Session {session_id}: {video.filename} (area={polygon_id}) → {tmp_path}")
    return {
        "session_id": session_id,
        "filename":   video.filename,
        "polygon_id": polygon_id,
        "stream_url": f"/api/v1/parking/session/{session_id}/stream",
    }


@router.get(
    "/session/{session_id}/stream",
    summary="Stream MJPEG từ video đã upload (dùng <img> tag)",
    response_class=StreamingResponse,
)
async def stream_session(
    session_id: str,
    request: Request,
    car_confidence:     float = Query(default=0.40),
    free_confidence:    float = Query(default=0.25),
    general_confidence: float = Query(default=0.25),
    skip_frames:        int   = Query(default=2, description="Bỏ qua N frame giữa mỗi lần detect"),
):
    """Trỏ <img src="..."> tới URL này → browser render MJPEG annotated real-time."""
    if session_id not in _VIDEO_SESSIONS:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail="Session không tồn tại hoặc đã hết hạn.")
    
    session_data = _VIDEO_SESSIONS[session_id]
    video_path = session_data["path"]
    polygon_id = session_data["polygon_id"]

    if not os.path.exists(video_path):
        _VIDEO_SESSIONS.pop(session_id, None)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail="File video không còn tồn tại.")

    cfg      = DetectionConfig(car_confidence=car_confidence,
                               free_confidence=free_confidence,
                               general_confidence=general_confidence)
    detector = _make_detector(request, _get_polygons(polygon_id), cfg)

    def _generator_with_cleanup():
        try:
            yield from mjpeg_generator(video_path, detector, skip_frames)
        finally:
            _VIDEO_SESSIONS.pop(session_id, None)
            try:
                os.unlink(video_path)
                logger.info(f"Đã xoá temp file, session={session_id}")
            except Exception:
                pass

    return StreamingResponse(
        _generator_with_cleanup(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache"},
    )


@router.post(
    "/detect/stream",
    summary="Upload + stream MJPEG trong 1 request (POST)",
    response_class=StreamingResponse,
)
async def stream_video_post(
    request: Request,
    video:              UploadFile = File(...),
    car_confidence:     float      = Form(default=0.40),
    free_confidence:    float      = Form(default=0.25),
    general_confidence: float      = Form(default=0.25),
    skip_frames:        int        = Form(default=2),
):
    """Dùng khi muốn upload + stream trong cùng 1 request (không cần session)."""
    cfg      = DetectionConfig(car_confidence=car_confidence,
                               free_confidence=free_confidence,
                               general_confidence=general_confidence)
    detector = _make_detector(request, _get_polygons(), cfg)
    tmp_path = await _save_upload_to_temp(video)

    async def _cleanup():
        try:
            for chunk in mjpeg_generator(tmp_path, detector, skip_frames):
                yield chunk
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    return StreamingResponse(
        _cleanup(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.get("/health", summary="Kiểm tra trạng thái service")
async def health_check(request: Request):
    return {
        "status":          "ok",
        "model_loaded":    getattr(request.app.state, "model", None) is not None,
        "device":          getattr(request.app.state, "device", "unknown"),
        "polygon_file":    POLYGON_PATH,
        "active_sessions": len(_VIDEO_SESSIONS),
    }