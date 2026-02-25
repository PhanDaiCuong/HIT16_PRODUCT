import json
from pathlib import Path
from typing import List


def load_polygons(path: str) -> List[dict]:
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