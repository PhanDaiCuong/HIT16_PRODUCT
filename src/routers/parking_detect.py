import json
import logging
import os
import tempfile
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile, status
from starlette.responses import StreamingResponse

from ..domain.parking_detector import ParkingDetector
from ..schemas.parking_model import (
    DetectRequest,
    DetectionConfig,
    DetectionResponse,
)
from ..utils.configs import POLYGON_PATH

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/parking", tags=["Parking Detection"])

# In-memory session store: session_id → temp video file path
_VIDEO_SESSIONS: Dict[str, str] = {}


# ========================== HELPER ==========================

def _load_polygons() -> List[dict]:
    """Load danh sách polygon từ file cấu hình server, tự thêm id nếu thiếu."""
    path = Path(POLYGON_PATH)
    if not path.exists():
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Không tìm thấy file polygon: {POLYGON_PATH}",
        )
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for i, poly in enumerate(data):
            if "id" not in poly:
                poly["id"] = i + 1
        return data
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi đọc file polygon: {exc}",
        )


def _make_detector(request: Request, polygons: List[dict], cfg: DetectionConfig) -> ParkingDetector:
    model = getattr(request.app.state, "model", None)
    if model is None:
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


def _mjpeg_generator(video_path: str, detector: ParkingDetector, skip: int = 2):
    """Generator yield từng frame MJPEG đã annotate polygon màu."""
    COLOR = {"occupied": (0, 0, 255), "free": (0, 255, 0), "unknown": (0, 255, 255)}

    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % (skip + 1) != 0:
                continue

            try:
                result = detector.detect(frame)
                for spot in result["spots"]:
                    polygon = np.array(spot["polygon"], np.int32)
                    color = COLOR.get(spot["status"], (128, 128, 128))
                    cv2.polylines(frame, [polygon], True, color, 2)
                    x, y = polygon[0]
                    cv2.putText(frame, f'#{spot["id"]} {spot["status"]}',
                                (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

                s = result["summary"]
                cv2.putText(frame,
                            f"Occupied:{s['occupied_count']} Free:{s['free_count']} Unknown:{s['unknown_count']}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            except Exception as exc:
                logger.warning(f"Frame {frame_count} lỗi: {exc}")

            _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + jpeg.tobytes()
                + b"\r\n"
            )
    finally:
        cap.release()


# ========================== IMAGE ENDPOINT ==========================

@router.post("/detect", response_model=DetectionResponse, summary="Phát hiện xe từ ảnh (base64)")
async def detect_parking(body: DetectRequest, request: Request):
    """Chỉ cần gửi `image` base64. Polygon tự load từ server."""
    polygons = _load_polygons()
    cfg = body.config or DetectionConfig()
    detector = _make_detector(request, polygons, cfg)

    try:
        result = detector.detect(body.to_numpy())
    except Exception as exc:
        logger.exception(f"Lỗi detection ảnh: {exc}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))

    s = result["summary"]
    logger.info(f"detect: {s['occupied_count']} occupied, {s['free_count']} free")
    return result


# ========================== SESSION UPLOAD ==========================

@router.post(
    "/session/upload",
    summary="Upload video, nhận session_id để stream",
    description=(
        "Upload video lên server. Server lưu tạm và trả về `session_id`. "
        "Dùng session_id này để gọi `GET /session/{session_id}/stream`."
    ),
)
async def upload_video_session(
    video: UploadFile = File(..., description="File video (mp4, avi, mov, webm)"),
):
    session_id = str(uuid.uuid4())
    suffix = os.path.splitext(video.filename or "video.mp4")[1] or ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        tmp.write(await video.read())
        tmp.flush()
        tmp.close()
        _VIDEO_SESSIONS[session_id] = tmp.name
        logger.info(f"Session upload: {video.filename} → session={session_id}, path={tmp.name}")
    except Exception as exc:
        tmp.close()
        os.unlink(tmp.name)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Không đọc được video: {exc}",
        )

    return {
        "session_id": session_id,
        "filename": video.filename,
        "stream_url": f"/api/v1/parking/session/{session_id}/stream",
    }


# ========================== SESSION STREAM (GET — browser native MJPEG) ==========================

@router.get(
    "/session/{session_id}/stream",
    summary="Stream MJPEG từ video đã upload (dùng img tag)",
    description=(
        "Sau khi upload xong qua `/session/upload`, trỏ thẻ `<img>` tới URL này. "
        "Browser sẽ hiển thị từng frame annotate real-time, mượt mà không lag."
    ),
    response_class=StreamingResponse,
)
async def stream_session(
    session_id: str,
    request: Request,
    car_confidence: float = Query(default=0.40),
    free_confidence: float = Query(default=0.25),
    general_confidence: float = Query(default=0.25),
    skip_frames: int = Query(default=2, description="Bỏ qua N frame"),
):
    if session_id not in _VIDEO_SESSIONS:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session không tồn tại hoặc đã hết hạn.")

    video_path = _VIDEO_SESSIONS[session_id]
    if not os.path.exists(video_path):
        _VIDEO_SESSIONS.pop(session_id, None)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File video không còn tồn tại.")

    polygons = _load_polygons()
    cfg = DetectionConfig(
        car_confidence=car_confidence,
        free_confidence=free_confidence,
        general_confidence=general_confidence,
    )
    detector = _make_detector(request, polygons, cfg)

    def generator():
        try:
            yield from _mjpeg_generator(video_path, detector, skip_frames)
        finally:
            # Dọn dẹp sau khi stream xong hoặc client ngắt kết nối
            _VIDEO_SESSIONS.pop(session_id, None)
            try:
                os.unlink(video_path)
                logger.info(f"Đã xoá temp file session={session_id}")
            except Exception:
                pass

    return StreamingResponse(
        generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache"},
    )




@router.post(
    "/detect/stream",
    summary="Stream video detection (MJPEG, POST)",
    response_class=StreamingResponse,
)
async def stream_video(
    request: Request,
    video: UploadFile = File(...),
    car_confidence: float = Form(default=0.40),
    free_confidence: float = Form(default=0.25),
    general_confidence: float = Form(default=0.25),
    skip_frames: int = Form(default=2),
):
    """Upload + stream trong 1 request. Dùng khi không cần smooth playback qua img tag."""
    polygons = _load_polygons()
    cfg = DetectionConfig(
        car_confidence=car_confidence,
        free_confidence=free_confidence,
        general_confidence=general_confidence,
    )
    detector = _make_detector(request, polygons, cfg)

    suffix = os.path.splitext(video.filename or "video.mp4")[1] or ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        tmp.write(await video.read())
        tmp.flush()
        tmp.close()
    except Exception as exc:
        tmp.close()
        os.unlink(tmp.name)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))

    async def cleanup_generator():
        try:
            for chunk in _mjpeg_generator(tmp.name, detector, skip_frames):
                yield chunk
        finally:
            try:
                os.unlink(tmp.name)
            except Exception:
                pass

    return StreamingResponse(
        cleanup_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.get("/health", summary="Kiểm tra trạng thái service")
async def health_check(request: Request):
    return {
        "status": "ok",
        "model_loaded": getattr(request.app.state, "model", None) is not None,
        "device": getattr(request.app.state, "device", "unknown"),
        "polygon_file": POLYGON_PATH,
        "active_sessions": len(_VIDEO_SESSIONS),
    }