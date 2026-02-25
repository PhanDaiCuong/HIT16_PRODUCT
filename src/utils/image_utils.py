import base64
from typing import Optional

import cv2
import numpy as np


def load_image(image_path: str, flags: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    return cv2.imread(image_path, flags)


def resize_image(image: np.ndarray, width: int, height: int,
                 interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
    if image is None:
        raise ValueError("image is None")
    return cv2.resize(image, (width, height), interpolation=interpolation)


def base64_to_numpy(b64_str: str) -> np.ndarray:
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    raw   = base64.b64decode(b64_str)
    nparr = np.frombuffer(raw, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Không decode được ảnh từ base64.")
    return img


def numpy_to_base64(image: np.ndarray, ext: str = ".jpg", quality: int = 90) -> str:
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if ext in (".jpg", ".jpeg") else []
    ok, buf = cv2.imencode(ext, image, params)
    if not ok:
        raise ValueError(f"Không encode được ảnh sang {ext}.")
    return base64.b64encode(buf.tobytes()).decode("utf-8")