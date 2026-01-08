# YOLO11 Video Tracking API with Analytics

A FastAPI-based REST API for person detection, tracking, and analytics in videos using YOLO11 with BoT-SORT tracker.

## Features

- **YOLO11 Detection**: Uses the latest YOLO11 model for accurate person detection
- **BoT-SORT Tracking**: Built-in BoT-SORT tracker for consistent person tracking across frames
- **Configurable Confidence**: Set minimum confidence threshold via API parameter
- **ROI Filtering**: Exclude edge detections for cleaner results
- **Analytics Suite**:
  - **Overcrowding Alerts**: Detect when person count exceeds threshold
  - **Trend Analysis**: Min/max/average person count over time
  - **Presence Heat Maps**: Where people stand/exist
  - **Movement Heat Maps**: Where people move through
  - **Time-Based Heat Maps**: Cumulative heat with time buckets

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

---

## API Endpoints

### 1. `GET /health` - Health Check

Returns server status and model information.

```bash
curl http://localhost:8000/health
```

---

### 2. `POST /track` - Basic Tracking

Frame-by-frame person detection with tracking IDs.

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `video` | file | required | Video file (mp4, avi, mov, etc.) |
| `confidence` | float | 0.5 | Minimum detection confidence (0.0-1.0) |

**Example Request:**
```bash
curl -X POST "http://localhost:8000/track?confidence=0.5" \
  -F "video=@video.mp4"
```

**Response:**
```json
{
  "status": "success",
  "video_name": "video.mp4",
  "confidence_threshold": 0.5,
  "total_frames": 150,
  "results": {
    "video_metadata": {
      "fps": 30.0,
      "total_frames": 150,
      "resolution": {"width": 1920, "height": 1080},
      "duration_sec": 5.0
    },
    "tracking_summary": {
      "unique_persons_tracked": 5,
      "unique_track_ids": [1, 2, 3, 4, 5]
    },
    "frames": [
      {
        "frame_number": 0,
        "timestamp_sec": 0.0,
        "detections_count": 3,
        "detections": [
          {
            "track_id": 1,
            "confidence": 0.85,
            "bbox": {"x1": 100, "y1": 200, "x2": 250, "y2": 500},
            "bbox_center": {"x": 175, "y": 350},
            "bbox_dimensions": {"width": 150, "height": 300}
          }
        ]
      }
    ]
  }
}
```

---

### 3. `POST /analyze` - Full Analytics

Complete tracking plus all analytics (overcrowding, trends, heatmaps).

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `video` | file | required | Video file |
| `confidence` | float | 0.5 | Confidence threshold |
| `max_persons` | int | 10 | Overcrowding threshold |
| `bucket_seconds` | float | 1.0 | Time bucket for heatmaps |
| `heatmap_scale` | float | 0.1 | Heatmap resolution scale (0.01-1.0) |

**Example Request:**
```bash
curl -X POST "http://localhost:8000/analyze?max_persons=5&confidence=0.6" \
  -F "video=@video.mp4"
```

**Response:** Same as `/track` plus:
```json
{
  "analytics": {
    "overcrowding_alerts": {
      "threshold": 5,
      "total_alerts": 2,
      "total_overcrowding_duration_sec": 5.5,
      "events": [...]
    },
    "trend_analysis": {
      "min_persons": 0,
      "max_persons": 12,
      "average_persons": 4.5,
      "std_deviation": 2.3,
      "person_count_timeline": [...]
    },
    "presence_heatmap": {
      "heatmap": [[0, 50, 100, ...], ...],
      "resolution": {"width": 192, "height": 108},
      "blur_size": 27,
      "roi_margins": {"top": 50, "bottom": 30}
    },
    "movement_heatmap": {...},
    "time_based_heatmap": {...}
  }
}
```

---

### 4. `POST /analyze/overcrowding` - Overcrowding Alerts

Detect when person count exceeds a threshold.

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `video` | file | required | Video file |
| `confidence` | float | 0.5 | Confidence threshold |
| `max_persons` | int | 10 | Maximum persons before alert |

**Example Request:**
```bash
curl -X POST "http://localhost:8000/analyze/overcrowding?max_persons=5" \
  -F "video=@video.mp4"
```

**Response:**
```json
{
  "status": "success",
  "video_name": "video.mp4",
  "video_metadata": {...},
  "tracking_summary": {...},
  "overcrowding_alerts": {
    "threshold": 5,
    "total_alerts": 2,
    "total_overcrowding_duration_sec": 5.5,
    "events": [
      {
        "start_frame": 45,
        "end_frame": 120,
        "start_time_sec": 1.5,
        "end_time_sec": 4.0,
        "duration_sec": 2.5,
        "peak_count": 8,
        "peak_frame": 78
      }
    ]
  }
}
```

---

### 5. `POST /analyze/trends` - Trend Analysis

Get person count statistics over time.

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `video` | file | required | Video file |
| `confidence` | float | 0.5 | Confidence threshold |

**Example Request:**
```bash
curl -X POST "http://localhost:8000/analyze/trends" \
  -F "video=@video.mp4"
```

**Response:**
```json
{
  "status": "success",
  "video_name": "video.mp4",
  "video_metadata": {...},
  "tracking_summary": {...},
  "trend_analysis": {
    "min_persons": 0,
    "max_persons": 12,
    "average_persons": 4.5,
    "std_deviation": 2.3,
    "total_frames_analyzed": 150,
    "person_count_timeline": [
      {"frame": 0, "time_sec": 0.0, "count": 3},
      {"frame": 30, "time_sec": 1.0, "count": 5},
      {"frame": 60, "time_sec": 2.0, "count": 8}
    ]
  }
}
```

---

### 6. `POST /analyze/heatmaps` - Heat Maps

Generate presence, movement, and time-based heat maps.

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `video` | file | required | Video file |
| `confidence` | float | 0.5 | Confidence threshold |
| `bucket_seconds` | float | 1.0 | Time bucket size |
| `heatmap_scale` | float | 0.1 | Resolution scale |

**Example Request:**
```bash
curl -X POST "http://localhost:8000/analyze/heatmaps?bucket_seconds=2.0" \
  -F "video=@video.mp4"
```

**Response:**
```json
{
  "status": "success",
  "video_name": "video.mp4",
  "video_metadata": {...},
  "tracking_summary": {...},
  "heatmaps": {
    "presence": {
      "heatmap": [[0, 50, 100, ...], ...],
      "heatmap_raw": [[0, 5, 10, ...], ...],
      "resolution": {"width": 192, "height": 108},
      "original_resolution": {"width": 1920, "height": 1080},
      "scale_factor": 0.1,
      "blur_size": 27,
      "roi_margins": {"top": 50, "bottom": 30},
      "max_heat_value": 450
    },
    "movement": {
      "heatmap": [[...], ...],
      "resolution": {...},
      "blur_size": 27,
      "roi_margins": {...}
    },
    "time_based": {
      "bucket_seconds": 2.0,
      "total_buckets": 3,
      "cumulative_heatmap": [[...], ...],
      "time_buckets": [
        {
          "bucket_index": 0,
          "start_time_sec": 0.0,
          "end_time_sec": 2.0,
          "frame_count": 60,
          "total_detections": 180,
          "heatmap": [[...], ...]
        }
      ]
    }
  }
}
```

---

## ROI Filtering

All analytics endpoints apply Region of Interest (ROI) filtering to exclude edge detections:

| Margin | Default Value | Description |
|--------|---------------|-------------|
| Top | 50 pixels | Exclude from top edge |
| Bottom | 30 pixels | Exclude from bottom edge |
| Left | 40 pixels | Exclude from left edge |
| Right | 40 pixels | Exclude from right edge |

Detections with centers outside the ROI are excluded from heatmap calculations.

---

## Test Client

The included `test_client.py` provides visualization capabilities:

### Basic Tracking
```bash
python test_client.py --video video.mp4
```

### Full Analytics with Graphs
```bash
python test_client.py --video video.mp4 --analyze --max-persons 5
```

### Heatmap Overlay on Video
```bash
python test_client.py --video video.mp4 --heatmap --heatmap-type presence
python test_client.py --video video.mp4 --heatmap --heatmap-type movement
python test_client.py --video video.mp4 --heatmap --heatmap-type cumulative
```

### All Options
```bash
python test_client.py --video video.mp4 \
  --analyze \
  --max-persons 5 \
  --heatmap \
  --heatmap-type movement \
  --show-trends \
  --show-heatmap-static \
  --save-json
```

### Test Client Options

| Option | Description |
|--------|-------------|
| `--video`, `-v` | Path to input video (required) |
| `--confidence`, `-c` | Confidence threshold (default: 0.5) |
| `--analyze` | Use full analytics endpoint |
| `--max-persons` | Overcrowding threshold (default: 10) |
| `--heatmap` | Overlay heatmap on video |
| `--heatmap-type` | Type: presence, movement, cumulative |
| `--show-trends` | Display trend analysis graph |
| `--show-heatmap-static` | Display static heatmap image |
| `--no-display` | Don't show video window |
| `--save-json` | Save results to JSON file |
| `--output`, `-o` | Output video path |

---

## API Documentation (Swagger)

Once the server is running, visit:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

---

## Configuration

Modify constants in `main.py`:

| Constant | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `yolo11l.pt` | YOLO11 model variant |
| `DEFAULT_CONFIDENCE` | `0.5` | Default confidence threshold |
| `DEFAULT_MAX_PERSONS` | `10` | Default overcrowding threshold |
| `DEFAULT_BUCKET_SECONDS` | `1.0` | Default time bucket size |
| `DEFAULT_HEATMAP_SCALE` | `0.1` | Default heatmap scale factor |

## Model Variants

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| yolo11n.pt | Nano | Fastest | Good |
| yolo11s.pt | Small | Fast | Better |
| yolo11m.pt | Medium | Moderate | High |
| yolo11l.pt | Large | Slower | Higher |
| yolo11x.pt | XLarge | Slowest | Best |

---

## File Structure

```
MODULE5/
├── main.py           # FastAPI server with all endpoints
├── analytics.py      # Analytics computation functions
├── test_client.py    # Test client with visualizations
├── requirements.txt  # Python dependencies
├── README.md         # This file
└── output/           # Generated output files
```

---

## Python Usage Example

```python
import requests

# Send video to analyze endpoint
url = "http://localhost:8000/analyze"
params = {
    "confidence": 0.6,
    "max_persons": 5,
    "bucket_seconds": 1.0
}

with open("video.mp4", "rb") as f:
    files = {"video": ("video.mp4", f, "video/mp4")}
    response = requests.post(url, files=files, params=params)

result = response.json()

# Access tracking data
frames = result["results"]["frames"]
unique_persons = result["results"]["tracking_summary"]["unique_persons_tracked"]

# Access analytics
overcrowding = result["analytics"]["overcrowding_alerts"]
trends = result["analytics"]["trend_analysis"]
presence_heatmap = result["analytics"]["presence_heatmap"]["heatmap"]
```
