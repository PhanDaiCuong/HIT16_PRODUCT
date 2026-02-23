"""
draw_utils.py
Các hàm vẽ annotation lên frame ảnh / video.
"""
import cv2
import numpy as np

# ── Bảng màu BGR theo trạng thái ──
SPOT_COLORS: dict = {
    "occupied": (40,  40, 220),   # đỏ
    "free":     (50, 205,  70),   # xanh lá
    "unknown":  (20, 190, 230),   # cyan
}


def draw_spot_fills(frame: np.ndarray, spots: list, alpha: float = 0.22) -> None:
    """
    Vẽ semi-transparent fill cho tất cả spots (1 lần addWeighted).
    Thao tác in-place lên frame.
    """
    overlay = frame.copy()
    for spot in spots:
        polygon = np.array(spot["polygon"], np.int32)
        color   = SPOT_COLORS.get(spot["status"], (120, 120, 120))
        cv2.fillPoly(overlay, [polygon], color)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)


def draw_spot_borders_and_badges(frame: np.ndarray, spots: list) -> None:
    """
    Vẽ viền polygon anti-aliased + badge #ID căn giữa mỗi ô.
    Thao tác in-place lên frame.
    """
    font, scale, thick = cv2.FONT_HERSHEY_DUPLEX, 0.38, 1
    pad = 4

    for spot in spots:
        polygon = np.array(spot["polygon"], np.int32)
        color   = SPOT_COLORS.get(spot["status"], (120, 120, 120))

        # Viền
        cv2.polylines(frame, [polygon], True, color, 2, cv2.LINE_AA)

        # Tâm polygon → đặt badge
        cx = int(np.mean(polygon[:, 0]))
        cy = int(np.mean(polygon[:, 1]))

        label = f"#{spot['id']}"
        (tw, th), _ = cv2.getTextSize(label, font, scale, thick)

        bx1, by1 = cx - tw // 2 - pad, cy - th - pad
        bx2, by2 = cx + tw // 2 + pad, cy + pad

        # Background tối + viền màu
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (12, 12, 20), -1)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 1, cv2.LINE_AA)

        # Chữ trắng
        cv2.putText(frame, label, (cx - tw // 2, cy - 2),
                    font, scale, (240, 240, 240), thick, cv2.LINE_AA)


def draw_hud_bar(frame: np.ndarray, summary: dict, hud_h: int = 46) -> None:
    """
    Vẽ HUD bar bán trong suốt phía trên frame với thống kê trực tiếp.
    Thao tác in-place lên frame.
    """
    h, w = frame.shape[:2]

    # Nền HUD
    hud = frame.copy()
    cv2.rectangle(hud, (0, 0), (w, hud_h), (10, 12, 22), -1)
    cv2.addWeighted(hud, 0.82, frame, 0.18, 0, frame)
    cv2.line(frame, (0, hud_h), (w, hud_h), (50, 90, 140), 1)

    total    = summary.get("total_spots", 0)
    occupied = summary.get("occupied_count", 0)
    free     = summary.get("free_count", 0)
    unknown  = summary.get("unknown_count", 0)
    rate     = summary.get("occupancy_rate", 0.0)

    font = cv2.FONT_HERSHEY_SIMPLEX

    # Logo
    cv2.putText(frame, "PARKVISION AI",
                (12, 30), font, 0.52, (99, 179, 237), 1, cv2.LINE_AA)
    cv2.line(frame, (175, 10), (175, 38), (50, 90, 140), 1)

    # Các chỉ số
    items = [
        (f"TOTAL  {total}",       (180, 180, 180)),
        (f"OCCUPIED  {occupied}", ( 80,  80, 220)),
        (f"FREE  {free}",         ( 60, 205,  70)),
        (f"UNKNOWN  {unknown}",   ( 30, 185, 225)),
        (f"RATE  {rate:.0f}%",    (220, 185,  60)),
    ]
    x = 190
    for text, color in items:
        cv2.putText(frame, text, (x, 30), font, 0.42, color, 1, cv2.LINE_AA)
        tw, _ = cv2.getTextSize(text, font, 0.42, 1)
        x += tw[0] + 22
        if x < w - 30:
            cv2.line(frame, (x - 11, 14), (x - 11, 34), (40, 60, 90), 1)


def annotate_frame(frame: np.ndarray, spots: list, summary: dict) -> np.ndarray:
    """
    Hàm tổng hợp: vẽ đầy đủ fill + viền + badge + HUD bar lên 1 frame.

    Args:
        frame:   Ảnh BGR (numpy array).
        spots:   Danh sách spot dict (polygon, id, status).
        summary: Dict thống kê (occupied_count, free_count, ...).

    Returns:
        Frame đã annotate (trả lại cùng object, thao tác in-place).
    """
    draw_spot_fills(frame, spots)
    draw_spot_borders_and_badges(frame, spots)
    draw_hud_bar(frame, summary)
    return frame
