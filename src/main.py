import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.domain.parking_detector import get_or_load_model
from src.routers import parking_router
from src.utils.configs import DEVICE, MODEL_PATH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"[Startup] Đang load model từ '{MODEL_PATH}' trên device '{DEVICE}'...")
    try:
        model = get_or_load_model(MODEL_PATH, DEVICE)
        app.state.model = model
        app.state.model_path = MODEL_PATH
        app.state.device = DEVICE
        logger.info("[Startup] Model đã sẵn sàng!")
    except FileNotFoundError:
        logger.warning(
            f"[Startup] Không tìm thấy model tại '{MODEL_PATH}'. "
            "API vẫn chạy nhưng /detect sẽ báo lỗi 503."
        )
        app.state.model = None
        app.state.model_path = MODEL_PATH
        app.state.device = DEVICE

    yield

    logger.info("[Shutdown] Server đang tắt.")

app = FastAPI(
    title="Parking Detection API",
    description=(
        "API phát hiện xe và chỗ đậu xe trong bãi đỗ sử dụng YOLOv8. "
        "Nhận ảnh base64 + polygon định nghĩa các ô → trả về trạng thái từng ô."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(parking_router, prefix="/api/v1")

@app.get("/", tags=["Root"])
async def root():
    model_status = "ready" if getattr(app.state, "model", None) is not None else "not loaded"
    return {
        "message": "Parking Detection API đang chạy",
        "model_status": model_status,
        "docs": "/docs",
    }