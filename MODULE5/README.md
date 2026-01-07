# YOLO11 Video Tracking API

A FastAPI-based REST API for person detection and tracking in videos using YOLO11 with BoT-SORT tracker.

## Features

- **YOLO11 Detection**: Uses the latest YOLO11 model for accurate person detection
- **BoT-SORT Tracking**: Built-in BoT-SORT tracker for consistent person tracking across frames
- **Configurable Confidence**: Set minimum confidence threshold via API parameter
- **JSON Output**: Frame-by-frame detection coordinates with tracking IDs

## Installation

```bash
# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Start the Server

```bash
python main.py
```

Or using uvicorn directly:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`

### API Endpoints

#### Health Check
```bash
GET /health
```

#### Track Video (Main Endpoint)
```bash
POST /track
```

**Parameters:**
- `video` (required): Video file (mp4, avi, mov, mkv, webm, flv, wmv)
- `confidence` (optional): Minimum confidence threshold (0.0-1.0, default: 0.5)

### Example Request

Using `curl`:

```bash
curl -X POST "http://localhost:8000/track?confidence=0.6" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "video=@/path/to/your/video.mp4"
```

Using Python `requests`:

```python
import requests

url = "http://localhost:8000/track"
params = {"confidence": 0.6}  # Optional, default is 0.5

with open("video.mp4", "rb") as video_file:
    files = {"video": ("video.mp4", video_file, "video/mp4")}
    response = requests.post(url, files=files, params=params)

result = response.json()
print(result)
```

### Example Response

```json
{
  "status": "success",
  "video_name": "sample.mp4",
  "confidence_threshold": 0.6,
  "total_frames": 150,
  "results": {
    "video_metadata": {
      "fps": 30.0,
      "total_frames": 150,
      "resolution": {"width": 1920, "height": 1080},
      "duration_sec": 5.0
    },
    "tracking_summary": {
      "unique_persons_tracked": 3,
      "unique_track_ids": [1, 2, 3]
    },
    "frames": [
      {
        "frame_number": 0,
        "timestamp_sec": 0.0,
        "detections_count": 2,
        "detections": [
          {
            "track_id": 1,
            "confidence": 0.8521,
            "bbox": {
              "x1": 100.5,
              "y1": 200.3,
              "x2": 250.8,
              "y2": 500.1
            },
            "bbox_center": {
              "x": 175.65,
              "y": 350.2
            },
            "bbox_dimensions": {
              "width": 150.3,
              "height": 299.8
            }
          }
        ]
      }
    ]
  }
}
```

## API Documentation

Once the server is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Configuration

You can modify the following in `main.py`:

- `MODEL_PATH`: Change YOLO11 model variant (yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt)
- `DEFAULT_CONFIDENCE`: Default confidence threshold when not specified in request

## Model Variants

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| yolo11n.pt | Nano | Fastest | Good |
| yolo11s.pt | Small | Fast | Better |
| yolo11m.pt | Medium | Moderate | High |
| yolo11l.pt | Large | Slower | Higher |
| yolo11x.pt | XLarge | Slowest | Best |
