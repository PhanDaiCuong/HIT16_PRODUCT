# HIT16_PRODUCT

## 🚗 Smart Parking Management System

A computer vision-based parking lot management system using YOLOv11 and Ultralytics to detect and monitor parking space occupancy in real-time.

## 📋 Features

- **Real-time parking space detection** using YOLO object detection
- **Video processing** for parking lot surveillance
- **Polygon-based zone definition** for precise parking spot tracking
- **Occupancy status visualization** with OpenCV

## 🗂️ Project Structure

```
HIT16_PRODUCT/
├── models/
│   └── best.pt              # Trained YOLO model weights
├── bounding_boxes.json      # Original parking spot polygons
├── polygons.json            # Parking zone definitions
├── test.py                  # Main test script
└── README.md                # This file
```

## 🔧 Requirements

```bash
pip install ultralytics opencv-python numpy matplotlib
```

## 🚀 Usage

### Running the Parking Manager

1. **Prepare your video file**: Place your parking lot video in the project directory
2. **Update paths in `test.py`**: Modify the video path to point to your test video
3. **Run the script**:

```bash
python test.py
```

### Configuration

Edit the following variables in `test.py`:

```python
video_path = "./test_video.webm"      # Your parking lot video
model_path = "./models/best.pt"       # YOLO model (already set)
json_file = "./polygons.json"         # Parking zone definitions
```

### Controls

- **Q**: Quit the video playback

## 📊 Model Information

- **Model**: YOLOv11 (custom trained)
- **Model file**: `models/best.pt` (20.3 MB)
- **Training dataset**: PKLot dataset
- **Purpose**: Vehicle detection in parking spaces

## 🎯 How It Works

1. The system loads a pre-trained YOLO model
2. Video frames are processed sequentially (with frame skipping for performance)
3. Each parking spot is defined by a polygon in `polygons.json`
4. The model detects vehicles within these polygons
5. Real-time visualization shows occupied/vacant spots

## 📝 Notes

- Frame skipping logic skips frames divisible by 2, 3, or 5 for better performance
- The system uses `ParkingManagement` from Ultralytics Solutions
- Labels are hidden by default (`show_labels=False`)

## 👨‍💻 Development

To customize parking zones, edit `polygons.json` with your own polygon coordinates. Each parking spot is defined by 4 corner points:

```json
{
    "points": [
        [x1, y1],  // top-left
        [x2, y2],  // top-right
        [x3, y3],  // bottom-right
        [x4, y4]   // bottom-left
    ]
}
```

## 📄 License

This project is part of HIT16 coursework.
