import cv2
import numpy as np
import requests
import streamlit as st
import base64
import streamlit.components.v1 as components

# =====================================================================
# CẤU HÌNH HỆ THỐNG
# =====================================================================
API_BASE = "http://localhost:8000/api/v1/parking"

st.set_page_config(
    page_title="ParkVision AI",
    page_icon="🅿️",
    layout="wide"
)
# --- CSS ---
st.markdown("""
<style>
    /* Nhúng font chữ Inter */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp p, .stApp span, .stApp label, .stApp input, .stApp div[data-baseweb="select"] {
        font-size: 1.2rem !important; 
    }

    /* Nền */
    .stApp { 
        background: linear-gradient(135deg, #0a0e1a 0%, #0d1528 50%, #0a1020 100%); 
        color: #e2e8f0; 
    }
    [data-testid="stSidebar"] { 
        background: linear-gradient(180deg, #0f1923 0%, #0a1420 100%);
        border-right: 1px solid rgba(99,179,237,0.15); 
    }
            
    /* Màu chữ đồng bộ chung */
    h1, h2, h3, h4, h5, h6, p, label, span { 
        color: #e2e8f0 !important; 
    } 
    [data-testid="stSidebar"] *{ 
        color: #cbd5e0 !important; 
    }
    details summary { 
        color: #90cdf4 !important; 
        font-weight: 600 !important; 
    }
            
    /* BOX TIÊU ĐỀ CHÍNH */
    .hero-box {
        background: rgba(15, 23, 42, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 2.5rem 2rem;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
    }
    .hero-badge {
        display: inline-block;
        border: 1px solid rgba(79, 209, 197, 0.3);
        background: rgba(79, 209, 197, 0.1);
        color: #4fd1c5 !important;
        padding: 6px 16px;
        border-radius: 50px;
        font-size: 1rem;
        font-weight: 600;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 1.2rem;
    }
    .hero-title-wrapper {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 16px;
        margin-bottom: 0.8rem;
    }
    .hero-logo {
        width: 45px;
        height: 45px;
        background-color: #4fd1c5;
        border-radius: 12px;
        box-shadow: 0 0 20px rgba(79, 209, 197, 0.3);
    }
    .hero-title {
        font-size: 3rem !important;
        font-weight: 700 !important;
        color: #4fd1c5 !important;
        margin: 0 !important;
        line-height: 1 !important;
    }
    .hero-subtitle {
        color: #94a3b8 !important;
        font-size: 1rem;
        font-weight: 400;
        margin: 0;
    }
            
    /* SIDEBAR */
    [data-testid="stSidebar"] .stTextInput,
    [data-testid="stSidebar"] .stSlider,
    [data-testid="stSidebar"] .stSelectbox {
        margin-bottom: 3rem !important; 
    }

    /* ── KHU VỰC KÉO THẢ FILE ── */
    [data-testid="stFileUploader"] { 
        border: 2px dashed rgba(99,179,237,0.3) !important; 
        background: rgba(99,179,237,0.05) !important;
        border-radius: 12px !important; 
        padding: 2rem !important;
    }
            
    /* Chữ bên trong vùng thả file */
    div[data-testid="stFileUploader"] div,
    div[data-testid="stFileUploader"] span,
    div[data-testid="stFileUploader"] small,
    div[data-testid="stFileUploader"] p {
        color: #94a3b8 !important;
        font-weight: 500 !important;
    }

    /* Nút bấm */
    .stButton > button {
        background: linear-gradient(135deg, #f8fafc, #e2e8f0) !important;
        border: 1px solid #cbd5e0 !important;
        border-radius: 8px !important;
    }
    .stButton > button, .stButton > button * {
        color: #0f172a !important;
        font-weight: 800 !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #e2e8f0, #cbd5e0) !important;
        border-color: #94a3b8 !important;
    }

    /* Đổi màu thanh trượt (Slider) */
    [data-testid="stSlider"] > div > div > div { background: #3182ce !important; }
    
    /* Ô Selectbox */
    .stSelectbox select { 
        background: rgba(255,255,255,0.05) !important; 
        border: 1px solid rgba(99,179,237,0.2) !important; 
        color: white !important; 
        border-radius: 8px !important;
    }

    /* Ô nhập văn bản TextInput */
    .stTextInput input {
        background-color: #ffffff !important;
        color: #0f172a !important;
        font-weight: 600 !important;
        border: 1px solid rgba(99,179,237,0.2) !important;
        border-radius: 8px !important;
        box-shadow: inset 0 1px 3px rgba(0,0,0,0.1) !important;
    }

    /* Ẩn menu */
    #MainMenu, footer, header { visibility: hidden; }
    hr { border-color: rgba(255,255,255,0.07) !important; }
            
    /* ── KHU VỰC CHI TIẾT Ô ĐỖ XE ── */
    .spot-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
        gap: 0.8rem;
        margin-top: 1rem;
    }
    .spot-item {
        background: rgba(255,255,255,0.04);
        border-radius: 10px;
        padding: 0.7rem 0.9rem;
        border-left: 4px solid;
        font-size: 0.85rem;
        font-weight: 500;
        display: flex;
        align-items: center;
        gap: 0.6rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
    }
    .spot-item.occupied { border-color: #e53e3e; color: #fc8181; }
    .spot-item.free     { border-color: #38a169; color: #68d391; }
    .spot-item.unknown  { border-color: #ed8936; color: #f6ad55; }
</style>
""", unsafe_allow_html=True)

# =====================================================================
# HELPER FUNCTIONS
# =====================================================================
def numpy_to_base64(image_np: np.ndarray) -> str:
    """Chuyển đổi ảnh Numpy Array sang Base64 để gửi qua API"""
    _, buffer = cv2.imencode('.jpg', image_np)
    return base64.b64encode(buffer).decode('utf-8')

def call_image_api(image_b64: str, conf_params: dict) -> dict:
    """Gọi API phát hiện chỗ đỗ xe từ ảnh"""
    payload = {
        "image": image_b64,
        "config": conf_params
    }
    r = requests.post(f"{API_BASE}/detect", json=payload, timeout=60)
    r.raise_for_status()
    return r.json()

def draw_spots(frame: np.ndarray, spots: list) -> np.ndarray:
    """Vẽ các đa giác và trạng thái lên ảnh"""
    # Mã màu: BGR
    COLORS = {
        "occupied": (40, 40, 220),  # Đỏ
        "free": (50, 205, 70),      # Xanh lá
        "unknown": (20, 190, 230),  # Vàng/Cam
    }
    
    # Tạo overlay để làm hiệu ứng trong suốt (transparent)
    overlay = frame.copy()
    for spot in spots:
        polygon = np.array(spot["polygon"], np.int32)
        color = COLORS.get(spot["status"], (120, 120, 120))
        cv2.fillPoly(overlay, [polygon], color)
    
    # Trộn ảnh overlay với ảnh gốc
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

    # Vẽ viền và ID
    for spot in spots:
        polygon = np.array(spot["polygon"], np.int32)
        color = COLORS.get(spot["status"], (120, 120, 120))
        cv2.polylines(frame, [polygon], isClosed=True, color=color, thickness=2)

        # Tính toán điểm chính giữa để ghi chữ
        cx = int(np.mean(polygon[:, 0]))
        cy = int(np.mean(polygon[:, 1]))
        label = f"#{spot['id']}"
        
        cv2.putText(frame, label, (cx - 15, cy + 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return frame

# =====================================================================
# GIAO DIỆN CHÍNH (UI)
# =====================================================================
st.markdown("""
    <div class="hero-box">
        <div class="hero-badge">⚡ AI-POWERED • REAL-TIME DETECTION</div>
        <div class="hero-title-wrapper">
            <div class="hero-logo"></div>
            <h1 class="hero-title">ParkVision AI</h1>
        </div>
        <p class="hero-subtitle">Hệ thống nhận diện bãi đỗ xe thông minh – YOLO · Computer Vision</p>
    </div>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.header("Cấu hình hệ thống")
    api_url = st.text_input("API Base URL", value=API_BASE)
    
    st.subheader("Ngưỡng tin cậy ")
    car_conf = st.slider("🚗 Phát hiện Xe ", 0.0, 1.0, 0.40, 0.05)
    free_conf = st.slider("🟢 Phát hiện Chỗ trống", 0.0, 1.0, 0.25, 0.05)
    
    st.subheader("⚙️ Phần cứng")
    device = st.selectbox("💻 Device", ["cpu", "cuda"])
    skip_frames = st.slider("⏭️ Bỏ qua N frame (Video)", 0, 15, 3)
    
    # Nút kiểm tra trạng thái API
    if st.button("Kiểm tra kết nối API"):
        try:
            r = requests.get(f"{api_url}/health", timeout=3)
            if r.status_code == 200:
                st.success("✅ Kết nối Server thành công!")
            else:
                st.warning("⚠️ Server phản hồi nhưng có lỗi.")
        except:
            st.error("❌ Không thể kết nối đến Server.")

# Tổng hợp config để truyền đi
current_config = {
    "car_confidence": car_conf,
    "free_confidence": free_conf,
    "general_confidence": 0.25,
    "device": device,
    "image_size": 640
}

# --- TÙY CHỌN CHẾ ĐỘ ---
mode = st.radio("Chọn nguồn cấp dữ liệu:", ["Phát hiện từ Ảnh", "Phát hiện từ Video"], horizontal=True)
st.divider()

# =====================================================================
# CHẾ ĐỘ 1: XỬ LÝ ẢNH
# =====================================================================
if mode == "Phát hiện từ Ảnh":
    uploaded_file = st.file_uploader("📂 Tải ảnh lên (JPG / PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        # Đọc ảnh thành numpy array
        image_bytes = uploaded_file.getvalue()
        image_np = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        
        # Gọi API
        with st.spinner("🔍 Đang phân tích ảnh qua AI Server..."):
            try:
                image_b64 = numpy_to_base64(image_np)
                result = call_image_api(image_b64, current_config)
                
                # Vẽ box lên ảnh
                annotated_img = draw_spots(image_np.copy(), result["spots"])
                
                # Hiển thị 2 ảnh cạnh nhau
                col1, col2 = st.columns(2)
                with col1:
                    st.image(image_np, channels="BGR", caption="Ảnh gốc")
                with col2:
                    st.image(annotated_img, channels="BGR", caption="Ảnh kết quả AI")

                # Hiển thị thống kê bằng components có sẵn của Streamlit
                st.subheader("📊 Thống kê bãi đỗ")
                s = result["summary"]
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Tổng chỗ", s['total_spots'])
                m2.metric("Có xe 🔴", s['occupied_count'])
                m3.metric("Chỗ trống 🟢", s['free_count'])
                m4.metric("Tỷ lệ lấp đầy", f"{s['occupancy_rate']:.0f}%")

                # Bảng chi tiết từng ô đỗ xe
                # Bảng chi tiết từng ô đỗ xe
                with st.expander("📋 Chi tiết từng ô đỗ xe", expanded=True):
                    items_html = ""
                    for spot in result["spots"]:
                        status = spot["status"]
                        # Xác định Icon
                        icon = "🔴" if status == "occupied" else "🟢" if status == "free" else "🟡"
                        
                        # Lấy độ tin cậy (nếu có)
                        conf = ""
                        if spot.get("detected_object"):
                            conf = f' <span style="font-size: 0.75rem; opacity: 0.7;">— {spot["detected_object"]["confidence"]:.0%}</span>'
                        
                        # Nối chuỗi HTML cực ngắn dùng class CSS
                        items_html += f'<div class="spot-item {status}">{icon} <span>Ô #{spot["id"]}{conf}</span></div>'

                    # In toàn bộ lưới ra màn hình
                    st.markdown(f'<div class="spot-grid">{items_html}</div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"❌ Lỗi xử lý API: {e}")

# =====================================================================
# CHẾ ĐỘ 2: XỬ LÝ VIDEO
# =====================================================================
elif mode == "Phát hiện từ Video":
    uploaded_video = st.file_uploader("📂 Tải video lên (MP4 / AVI / WEBM)", type=["mp4", "avi", "webm"])

    if uploaded_video:
        if st.button("▶️ Bắt đầu phân tích Video"):
            with st.spinner("📤 Đang gửi video lên server..."):
                try:
                    # Gửi file video
                    files = {"video": (uploaded_video.name, uploaded_video.getvalue(), "video/mp4")}
                    r = requests.post(f"{API_BASE}/session/upload", files=files)
                    r.raise_for_status()
                    
                    sid = r.json()["session_id"]
                    st.session_state["stream_sid"] = sid
                    st.success("✅ Upload thành công!")
                except Exception as e:
                    st.error(f"❌ Lỗi tải video lên Server: {e}")

    # Nếu có session id đang chạy thì hiển thị luồng stream
    if "stream_sid" in st.session_state:
        sid = st.session_state["stream_sid"]
        st.subheader("🎞️ Live Stream - AI Detection")
        
        # URL stream MJPEG từ API Server
        stream_url = f"{API_BASE}/session/{sid}/stream?car_confidence={car_conf}&free_confidence={free_conf}"
        
        # Sử dụng thẻ img HTML cơ bản nhất để hứng luồng MJPEG
        components.html(f"""
            <img src="{stream_url}" style="width:100%; border: 2px solid #ccc; border-radius: 10px;">
        """, height=600)

        if st.button("🛑 Dừng Video & Xoá Session"):
            del st.session_state["stream_sid"]
            st.rerun()