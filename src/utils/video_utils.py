import cv2
import numpy as np
from typing import Optional, Union


def open_video(source: Union[int, str]) -> cv2.VideoCapture:
    """
    Mở nguồn video từ file hoặc webcam.

    Trả về:
        Đối tượng cv2.VideoCapture đã được mở.
    ValueError: Nếu không thể mở nguồn video.
    """
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"Unable to open video source: {source}")
    return capture

def read_frame(capture: cv2.VideoCapture) -> Optional[np.ndarray]:
    """
    Đọc một khung hình từ nguồn video.

    Trả về:
        Khung hình dưới dạng mảng NumPy, hoặc None nếu không đọc được.
    """
    if capture is None:
        return None

    success, frame = capture.read()
    if not success:
        return None

    return frame


def release_video(capture: Optional[cv2.VideoCapture]) -> None:
    """
    Giải phóng tài nguyên video.

    Tham số:
        capture: Đối tượng cv2.VideoCapture cần giải phóng.
    """
    if capture is not None:
        capture.release()
