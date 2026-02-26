import cv2
import numpy as np

SPOT_COLORS: dict = {
    "occupied": (40,  40, 220),   
    "free":     (50, 205,  70),   
    "unknown":  (20, 190, 230),   
}

def draw_spot_fills(frame: np.ndarray, spots: list, alpha: float = 0.22) -> None:
    overlay = frame.copy()
    for spot in spots:
        polygon = np.array(spot["polygon"], np.int32)
        color   = SPOT_COLORS.get(spot["status"], (120, 120, 120))
        cv2.fillPoly(overlay, [polygon], color)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)

def draw_spot_borders_and_badges(frame: np.ndarray, spots: list) -> None:
    h, w = frame.shape[:2]
    s_factor = w / 1280.0
    
    font = cv2.FONT_HERSHEY_DUPLEX
    scale = max(0.25, 0.38 * s_factor)
    thick = max(1, int(1 * s_factor))
    pad = max(2, int(4 * s_factor))
    line_thick = max(1, int(2 * s_factor))

    for spot in spots:
        polygon = np.array(spot["polygon"], np.int32)
        color   = SPOT_COLORS.get(spot["status"], (120, 120, 120))

        cv2.polylines(frame, [polygon], True, color, line_thick, cv2.LINE_AA)

        cx = int(np.mean(polygon[:, 0]))
        cy = int(np.mean(polygon[:, 1]))

        label = f"#{spot['id']}"
        (tw, th), _ = cv2.getTextSize(label, font, scale, thick)

        bx1, by1 = cx - tw // 2 - pad, cy - th - pad
        bx2, by2 = cx + tw // 2 + pad, cy + pad

        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (12, 12, 20), -1)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 1, cv2.LINE_AA)

        cv2.putText(frame, label, (cx - tw // 2, cy - 2),
                    font, scale, (240, 240, 240), thick, cv2.LINE_AA)

def draw_hud_bar(frame: np.ndarray, summary: dict) -> None:
    h, w = frame.shape[:2]
    s_factor = w / 1280.0
    hud_h = int(46 * s_factor) if w > 640 else 32
    
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
    f_scale = max(0.35, 0.45 * s_factor)
    f_thick = 1
    y_pos = int(hud_h * 0.65)

    logo_text = "PARKVISION AI"
    cv2.putText(frame, logo_text, (12, y_pos), font, f_scale * 1.2, (99, 179, 237), f_thick, cv2.LINE_AA)
    
    lw, _ = cv2.getTextSize(logo_text, font, f_scale * 1.2, f_thick)
    sep_x = 12 + lw[0] + 15
    cv2.line(frame, (sep_x, int(hud_h*0.2)), (sep_x, int(hud_h*0.8)), (50, 90, 140), 1)

    items = [
        (f"TOTAL {total}",       (180, 180, 180)),
        (f"OCCUPIED {occupied}", ( 80,  80, 220)),
        (f"FREE {free}",         ( 60, 205,  70)),
        (f"UNKNOWN {unknown}",   ( 30, 185, 225)),
        (f"RATE {rate:.0f}%",    (220, 185,  60)),
    ]
    
    x = sep_x + 15
    for text, color in items:
        cv2.putText(frame, text, (x, y_pos), font, f_scale, color, f_thick, cv2.LINE_AA)
        tw, _ = cv2.getTextSize(text, font, f_scale, f_thick)
        x += tw[0] + int(20 * s_factor)
        if x < w - 20:
            cv2.line(frame, (x - 10, int(hud_h*0.3)), (x - 10, int(hud_h*0.7)), (40, 60, 90), 1)

def annotate_frame(frame: np.ndarray, spots: list, summary: dict) -> np.ndarray:
    draw_spot_fills(frame, spots)
    draw_spot_borders_and_badges(frame, spots)
    draw_hud_bar(frame, summary)
    return frame