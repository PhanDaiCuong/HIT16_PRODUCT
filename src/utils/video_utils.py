
import logging
from typing import Generator, Optional, Union

import cv2
import numpy as np

from .draw_utils import annotate_frame

logger = logging.getLogger(__name__)




def open_video(source: Union[int, str]) -> cv2.VideoCapture:
    """Mở nguồn video từ file path hoặc webcam index."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"Không thể mở video: {source}")
    return cap


def read_frame(cap: cv2.VideoCapture) -> Optional[np.ndarray]:
    """Đọc 1 frame; trả None nếu hết video hoặc lỗi."""
    if cap is None:
        return None
    ok, frame = cap.read()
    return frame if ok else None


def release_video(cap: Optional[cv2.VideoCapture]) -> None:
    """Giải phóng VideoCapture an toàn."""
    if cap is not None:
        cap.release()




def mjpeg_generator(
    video_path: str,
    detector,
    skip: int = 2,
    jpeg_quality: int = 85,
) -> Generator[bytes, None, None]:
    """
    Generator yield các MJPEG chunk đã được annotate.

    Args:
        video_path:    Đường dẫn file video tạm.
        detector:      ParkingDetector instance.
        skip:          Bỏ qua N frame giữa mỗi lần detect (giảm tải CPU).
        jpeg_quality:  Chất lượng JPEG encode (0-100).

    Yields:
        bytes: MJPEG multipart chunk (header + JPEG data).
    """
    cap         = open_video(video_path)
    frame_index = 0

    try:
        while cap.isOpened():
            frame = read_frame(cap)
            if frame is None:
                break

            frame_index += 1
            if frame_index % (skip + 1) != 0:
                continue

            try:
                result = detector.detect(frame)
                frame  = annotate_frame(frame, result["spots"], result["summary"])
            except Exception as exc:
                logger.warning(f"[mjpeg_generator] Frame {frame_index} lỗi: {exc}")

            _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + jpeg.tobytes()
                + b"\r\n"
            )
    finally:
        release_video(cap)
