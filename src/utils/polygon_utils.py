"""
polygon_utils.py
Tiện ích load và parse dữ liệu polygon bãi đỗ xe.
"""
import json
from pathlib import Path
from typing import List


def load_polygons(path: str) -> List[dict]:
    """
    Load danh sách polygon từ file JSON.

    File JSON có cấu trúc: list of {"points": [[x,y], ...]}.
    Tự thêm `id` (1-based) nếu thiếu.

    Args:
        path: Đường dẫn tới file JSON chứa polygon.

    Returns:
        List[dict] mỗi phần tử có key `id` và `points`.

    Raises:
        FileNotFoundError: Nếu file không tồn tại.
        ValueError:        Nếu nội dung JSON không hợp lệ.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Không tìm thấy file polygon: {path}")
    try:
        with open(p, "r", encoding="utf-8") as f:
            data: list = json.load(f)
        for i, poly in enumerate(data):
            if "id" not in poly:
                poly["id"] = i + 1
        return data
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError(f"File polygon không hợp lệ: {exc}") from exc
