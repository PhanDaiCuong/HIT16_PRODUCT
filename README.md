# 🅿️ ParkVision AI: Hệ Thống Giám Sát Bãi Đỗ Xe Thông Minh Thời Gian Thực

Một hệ thống quản lý bãi đỗ xe dựa trên AI, có khả năng mở rộng và được xây dựng với kiến trúc hiện đại, thiết kế để phát hiện phương tiện và phân tích việc sử dụng chỗ đỗ hiệu quả bằng YOLOv8.

## 🚀 Tính Năng Nổi Bật

- **Deep Learning**: Phát hiện xe và ô đỗ độ chính xác cao với YOLOv8.
- **Trực Quan Hóa**: Hiển thị trạng thái ô đỗ (Đỏ/Xanh) phong cách HUD.
- **Cân Chỉnh Động**: Tự động khớp vùng đỗ với mọi độ phân giải video.
- **Giao Diện Kép**: Kết hợp linh hoạt giữa API và Dashboard.
- **Tối Ưu**: Xử lý hiệu suất cao, độ trễ thấp trên CPU/GPU.

## 📁 Cấu Trúc Dự Án

```
HIT16_PRODUCT/
├── data/                    # (Cục bộ) Lưu trữ video và tọa độ ô đỗ
├── models/                  # (Cục bộ) Chứa file weights .pt của YOLO
├── scripts/                 # (Cục bộ) Các script hỗ trợ/tiện ích
├── src/                     # Mã nguồn chính (Được đẩy lên GitHub)
│   ├── app_streamlit.py     # Giao diện giám sát (Streamlit)
│   ├── main.py              # Điểm khởi đầu API Backend (FastAPI)
│   ├── domain/              # Logic nghiệp vụ cốt lõi
│   ├── routers/             # Định nghĩa các tuyến API
│   ├── schemas/             # Các mô hình dữ liệu Pydantic
│   ├── utils/               # Các hàm tiện ích dùng chung
│   └── visualization/       # Logic vẽ và hiển thị HUD
├── .gitignore               # Quy tắc bỏ qua của Git
├── requirements.txt         # Danh sách thư viện phụ thuộc
└── README.md                # Tệp này
```

## 🛠️ Cài Đặt

```bash
# Clone repository
git clone https://github.com/yourusername/HIT16_PRODUCT.git
cd HIT16_PRODUCT

# Tạo và kích hoạt môi trường ảo (Khuyến nghị)
python -m venv venv
# Windows
.\venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# Cài đặt các thư viện cần thiết
pip install -r requirements.txt
```

## 📖 Hướng Dẫn Sử Dụng

### Khởi Chạy Hệ Thống

1. **Khởi chạy Backend API**:

   ```bash
   uvicorn src.main:app --reload
   ```

2. **Khởi chạy Dashboard**:
   ```bash
   streamlit run src/app_streamlit.py
   ```

### Các Endpoint API (Tham khảo nhanh)

| Phương thức | Endpoint  | Mô tả                                   |
| ----------- | --------- | --------------------------------------- |
| GET         | `/health` | Kiểm tra trạng thái hệ thống và mô hình |
| POST        | `/detect` | Xử lý hình ảnh để phát hiện chỗ đỗ      |
| GET         | `/stream` | Luồng video MJPEG thời gian thực        |

## 📦 Thư Viện Chính

- **Framework**: FastAPI (Backend) / Streamlit (Frontend)
- **AI Engine**: Ultralytics (YOLOv8)
- **Computer Vision**: OpenCV
- **Xử lý dữ liệu**: Pydantic, NumPy

## 📄 Giấy Phép

Dự án này được cấp phép theo Giấy phép MIT.

---

**Lưu ý**: ParkVision AI được thiết kế để biến bất kỳ camera CCTV tiêu chuẩn nào thành một cảm biến đỗ xe thông minh mà không cần lắp đặt phần cứng tốn kém.
