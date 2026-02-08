import cv2
import numpy as np
from typing import Iterable, Tuple


def draw_parking_boxes(
    image: np.ndarray,
    boxes: Iterable[Tuple[int, int, int, int]],
    occupied: Iterable[bool],
    *,
    thickness: int = 2
) -> np.ndarray:
    """
    Vẽ khung cho các ô đỗ xe.
    - Ô có xe: màu đỏ
    - Ô trống: màu xanh
    """
    for box, is_occupied in zip(boxes, occupied):
        x1, y1, x2, y2 = map(int, box)

        color = (0, 0, 255) if is_occupied else (0, 255, 0)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    return image
