import streamlit as st
import numpy as np
import tempfile
import os
import json
import cv2
from pathlib import Path

from src.utils.image_utils import load_image
from src.domain.parking_detector import ParkingDetector


# ================= UI =================

st.set_page_config(page_title="Parking Management Sytstem", layout="wide")
st.title("HỆ THỐNG NHẬN DIỆN CHỖ ĐỖ XE")

# ================= PATH =================

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "best.pt"
POLYGON_PATH = ROOT / "polygons.json"

# ================= DRAW FUNCTION =================

def draw_spots(frame, spots):
    """
    Vẽ polygon từng ô đỗ xe dựa trên trạng thái.
    """
    for spot in spots:
        polygon = np.array(spot["polygon"], np.int32)

        if spot["status"] == "occupied":
            color = (0, 0, 255)      # đỏ
        elif spot["status"] == "free":
            color = (0, 255, 0)      # xanh
        else:
            color = (0, 255, 255)    # vàng (unknown)

        cv2.polylines(frame, [polygon], True, color, 2)

        # ghi ID ô đỗ
        x, y = polygon[0]
        cv2.putText(frame, f'ID {spot["id"]}', (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return frame

# ================= LOAD MODEL =================

@st.cache_resource
def load_detector():

    if not MODEL_PATH.exists():
        st.error(f"Không tìm thấy model: {MODEL_PATH}")
        st.stop()

    if not POLYGON_PATH.exists():
        st.error("Thiếu file polygons.json (file đánh dấu vị trí ô đỗ)")
        st.stop()

    # đọc polygons
    with open(POLYGON_PATH, "r", encoding="utf-8") as f:
        polygons = json.load(f)

    detector = ParkingDetector(
        polygons=polygons,
        model_path=str(MODEL_PATH)
    )

    return detector


detector = load_detector()

# ================= MODE =================

mode = st.radio(
    "Chọn dữ liệu đầu vào:",
    ["Ảnh", "Video"],
    horizontal=True
)

# =========================================================
# IMAGE MODE
# =========================================================

if mode == "Ảnh":

    uploaded_image = st.file_uploader(
        "Tải ảnh bãi đỗ xe",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_image is not None:

        # lưu file tạm
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(uploaded_image.getbuffer())
            temp_path = tmp.name

        image = load_image(temp_path)

        if image is None:
            st.error("Không đọc được ảnh.")
        else:

            # ===== DETECT =====
            result = detector.detect(image)

            # ===== DRAW =====
            annotated = draw_spots(image.copy(), result["spots"])

            # ===== HIỂN THỊ =====
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Ảnh gốc")
                st.image(image, channels="BGR", use_container_width=True)

            with col2:
                st.subheader("Kết quả nhận diện")
                st.image(annotated, channels="BGR", use_container_width=True)

            # ===== THỐNG KÊ =====
            summary = result["summary"]

            total_spaces = summary["total_spots"]
            occupied_spaces = summary["occupied_count"]
            empty_spaces = summary["vacant_count"]
            unknown_spaces = summary["unknown_count"]

            st.divider()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Tổng số chỗ", total_spaces)
            c2.metric("Có xe", occupied_spaces)
            c3.metric("Chỗ trống", empty_spaces)
            c4.metric("Không chắc chắn", unknown_spaces)
        try:
            os.remove(temp_path)
        except:
            pass



# =========================================================
# VIDEO MODE
# =========================================================

elif mode == "Video":

    uploaded_video = st.file_uploader(
        "Tải video bãi đỗ xe",
        type=["mp4", "avi", "mov", "webm"]
    )

    if uploaded_video is not None:

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_video.getbuffer())
            temp_path = tmp.name

        st.info("Đang xử lý video... (có thể mất 10-30s tùy độ dài)")

        frame_area = st.empty()
        info_area = st.empty()
        stats_area = st.empty()

        try:
            for result in detector.detect_video(temp_path):

                frame = result["frame"]
                annotated = draw_spots(frame, result["spots"])

                frame_area.image(annotated, channels="BGR", use_container_width=True)

                summary = result["summary"]

                stats_area.markdown(f"""
**Frame:** {result["frame_number"]}  
**Tổng chỗ:** {summary["total_spots"]}  
**Có xe:** {summary["occupied_count"]}  
**Trống:** {summary["vacant_count"]}  
**Unknown:** {summary["unknown_count"]}  
**Tỉ lệ lấp đầy:** {summary["occupancy_rate"]:.2f}%
""")

        except Exception as e:
            st.error(f"Lỗi khi xử lý video: {e}")

        os.remove(temp_path)
        st.success("Xử lý video hoàn tất!")
