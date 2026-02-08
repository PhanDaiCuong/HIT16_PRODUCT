import cv2
import numpy as np
from typing import Optional

def load_image(image_path: str, flags: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    """
    Đọc ảnh từ ổ đĩa bằng thư viện OpenCV.

    Trả về:
        Ảnh dưới dạng mảng np, hoặc None nếu không đọc được ảnh.
    """
    return cv2.imread(image_path, flags)

def resize_image(image: np.ndarray, width: int, height: int, interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
    """
    Thay đổi kích thước ảnh theo chiều rộng và chiều cao cho trước.

    Trả về:
        Ảnh đã được thay đổi kích thước dưới dạng mảng np.

    Ngoại lệ:
        ValueError: Nếu ảnh đầu vào là None.
    """
    if image is None:
        raise ValueError("image is None")

    return cv2.resize(image, (width, height), interpolation=interpolation)
