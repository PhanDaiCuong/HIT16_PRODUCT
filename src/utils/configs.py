import torch

MODEL_PATH = "models/best.pt"

POLYGON_PATH = "data/polygons/area_1.json"
POLYGONS_DIR = "data/polygons"

FRAME_SKIP = 5

CONFIDENCE_THRESHOLD = 0.5

CAR_CONFIDENCE_THRESHOLD = 0.5
FREE_CONFIDENCE_THRESHOLD = 0.5
GENERAL_CONFIDENCE_THRESHOLD = 0.3

IOU_THRESHOLD = 0.7

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_SIZE = 640