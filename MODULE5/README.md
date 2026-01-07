# YOLO11 Video Tracking API with Analytics

A FastAPI-based REST API for person detection, tracking, and analytics in videos using YOLO11 with BoT-SORT tracker.

## Features

- **YOLO11 Detection**: Uses the latest YOLO11 model for accurate person detection
- **BoT-SORT Tracking**: Built-in BoT-SORT tracker for consistent person tracking across frames
- **Configurable Confidence**: Set minimum confidence threshold via API parameter
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

### Health Check
```bash
GET /health
```

### Basic Tracking
```bash
POST /track
```
Returns frame-by-frame detection coordinates with tracking IDs.

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `video` | file | required | Video file (mp4, avi, mov, etc.) |
| `confidence` | float | 0.5 | Minimum confidence threshold (0.0-1.0) |

---

### Full Analytics
```bash
POST /analyze
```
Returns tracking data plus all analytics (overcrowding, trends, heatmaps).

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `video` | file | required | Video file |
| `confidence` | float | 0.5 | Confidence threshold |
| `max_persons` | int | 10 | Overcrowding threshold |
| `bucket_seconds` | float | 1.0 | Time bucket size for heatmaps |
| `heatmap_scale` | float | 0.1 | Heatmap resolution scale (0.01-1.0) |

---

### Overcrowding Analysis
```bash
POST /analyze/overcrowding
```
Returns overcrowding alerts with time ranges, duration, and peak counts.

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `video` | file | required | Video file |
| `confidence` | float | 0.5 | Confidence threshold |
| `max_persons` | int | 10 | Maximum persons before alert |

**Example Response:**
```json
{
  "overcrowding_alerts": {
    "threshold": 10,
    "total_alerts": 2,
    "total_overcrowding_duration_sec": 5.5,
    "events": [
      {
        "start_frame": 45,
        "end_frame": 120,
        "start_time_sec": 1.5,
        "end_time_sec": 4.0,
        "duration_sec": 2.5,
        "peak_count": 15,
        "peak_frame": 78
      }
    ]
  }
}
```

---

### Trend Analysis
```bash
POST /analyze/trends
```
Returns person count statistics over time.

**Example Response:**
```json
{
  "trend_analysis": {
    "min_persons": 0,
    "max_persons": 12,
    "average_persons": 4.5,
    "std_deviation": 2.3,
    "total_frames_analyzed": 150,
    "person_count_timeline": [
      {"frame": 0, "time_sec": 0.0, "count": 3},
      {"frame": 30, "time_sec": 1.0, "count": 5}
    ]
  }
}
```

---

### Heat Maps
```bash
POST /analyze/heatmaps
```
Returns presence, movement, and time-based heat maps.

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `video` | file | required | Video file |
| `confidence` | float | 0.5 | Confidence threshold |
| `bucket_seconds` | float | 1.0 | Time bucket size |
| `heatmap_scale` | float | 0.1 | Resolution scale |

**Example Response:**
```json
{
  "heatmaps": {
    "presence": {
      "heatmap": [[0, 5, 10, ...], ...],
      "resolution": {"width": 192, "height": 108},
      "original_resolution": {"width": 1920, "height": 1080},
      "max_heat_value": 450
    },
    "movement": { ... },
    "time_based": {
      "cumulative_heatmap": [[...], ...],
      "time_buckets": [
        {
          "bucket_index": 0,
          "start_time_sec": 0.0,
          "end_time_sec": 1.0,
          "total_detections": 45
        }
      ]
    }
  }
}
```

---

## Test Client

The included `test_client.py` provides visualization capabilities:

### Basic Tracking
```bash
python test_client.py --video sample.mp4
```

### Full Analytics with Graphs
```bash
python test_client.py --video sample.mp4 --analyze --max-persons 5
```

### Heatmap Overlay
```bash
python test_client.py --video sample.mp4 --heatmap --heatmap-type presence
python test_client.py --video sample.mp4 --heatmap --heatmap-type movement
```

### All Options
```bash
python test_client.py --video sample.mp4 \
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
| `--show-overcrowding` | Display overcrowding graph |
| `--show-heatmap-static` | Display static heatmap image |
| `--no-display` | Don't show video window |
| `--save-json` | Save results to JSON file |
| `--output`, `-o` | Output video path |

---

## API Documentation

Once the server is running, visit:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

---

## Configuration

Modify in `main.py`:

| Constant | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `yolo11m.pt` | YOLO11 model variant |
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
