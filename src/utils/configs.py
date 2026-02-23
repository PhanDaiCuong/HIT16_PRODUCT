import torch

# ========================== ĐƯỜNG DẪN MODEL ==========================
# Đường dẫn đến model YOLOv8 đã được train để phát hiện xe và chỗ trống
MODEL_PATH = "models/best.pt"

# Đường dẫn đến file polygon định nghĩa các ô đỗ xe
POLYGON_PATH = "data/polygons.json"


# ========================== CÁC THAM SỐ CẤU HÌNH ==========================

# Số frame bỏ qua trước khi xử lý video (giảm tải tính toán)
FRAME_SKIP = 5

# Ngưỡng độ tin cậy chung (legacy, để tương thích ngược)
CONFIDENCE_THRESHOLD = 0.5

# Ngưỡng độ tin cậy riêng cho từng loại đối tượng
CAR_CONFIDENCE_THRESHOLD = 0.5      # Ngưỡng tin cậy để phát hiện xe (class 'car')
FREE_CONFIDENCE_THRESHOLD = 0.5     # Ngưỡng tin cậy để phát hiện chỗ trống (class 'free')
GENERAL_CONFIDENCE_THRESHOLD = 0.3  # Ngưỡng tin cậy tổng quát khi chạy YOLO (lọc ban đầu)

# Ngưỡng IoU (Intersection Over Union) cho NMS (Non-Maximum Suppression)
IOU_THRESHOLD = 0.7

# Chọn thiết bị: Dùng GPU nếu có, nếu không sẽ dùng CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Kích thước ảnh đầu vào cho model YOLO (kích thước vuông)
IMAGE_SIZE = 640
