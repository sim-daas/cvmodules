"""
FastAPI Video Inference API with YOLO11 and BoT-SORT Tracker

This module provides a POST endpoint to upload a video file and receive
frame-by-frame person detection coordinates with tracking IDs in JSON format.
"""

import os
import tempfile
import shutil
from typing import Optional

import cv2
from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.responses import JSONResponse
from ultralytics import YOLO

# Initialize FastAPI app
app = FastAPI(
    title="YOLO11 Video Tracking API",
    description="API for person detection and tracking using YOLO11 with BoT-SORT tracker",
    version="1.0.0",
)

# Load YOLO11 model globally for efficiency
# The model is loaded once at startup and reused for all requests
MODEL_PATH = "yolo11m.pt"  # You can change to yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt for better accuracy
model = YOLO(MODEL_PATH)

# Default confidence threshold
DEFAULT_CONFIDENCE = 0.5

# Person class ID in COCO dataset
PERSON_CLASS_ID = 0


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "YOLO11 Video Tracking API",
        "endpoints": {
            "/track": "POST - Upload video for person tracking",
            "/health": "GET - Health check",
        },
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model": MODEL_PATH}


@app.post("/track")
async def track_video(
    video: UploadFile = File(..., description="Video file to process"),
    confidence: Optional[float] = Query(
        default=DEFAULT_CONFIDENCE,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for detections (0.0 to 1.0)",
    ),
):
    """
    Process an uploaded video with YOLO11 person detection and BoT-SORT tracking.

    Args:
        video: Video file (mp4, avi, mov, mkv, etc.)
        confidence: Minimum confidence threshold for detections (default: 0.5)

    Returns:
        JSON with frame-by-frame detection coordinates and tracking IDs
    """
    # Validate file type
    allowed_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}
    file_ext = os.path.splitext(video.filename)[1].lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed types: {', '.join(allowed_extensions)}",
        )

    # Create a temporary file to store the uploaded video
    temp_dir = tempfile.mkdtemp()
    temp_video_path = os.path.join(temp_dir, video.filename)

    try:
        # Save uploaded video to temporary file
        with open(temp_video_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)

        # Process video and get tracking results
        tracking_results = process_video(temp_video_path, confidence)

        return JSONResponse(
            content={
                "status": "success",
                "video_name": video.filename,
                "confidence_threshold": confidence,
                "total_frames": len(tracking_results["frames"]),
                "results": tracking_results,
            }
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing video: {str(e)}",
        )

    finally:
        # Cleanup temporary files
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def process_video(video_path: str, confidence: float) -> dict:
    """
    Process video with YOLO11 and BoT-SORT tracker.

    Args:
        video_path: Path to the video file
        confidence: Minimum confidence threshold

    Returns:
        Dictionary containing frame-by-frame detection results
    """
    # Open video capture
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    # Get video metadata
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Store all frame results
    frames_data = []
    frame_number = 0
    unique_track_ids = set()

    # Process each frame
    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            break

        # Run YOLO11 tracking with BoT-SORT (default tracker)
        # persist=True maintains track IDs across frames
        # classes=[0] filters to only detect persons
        results = model.track(
            frame,
            persist=True,
            conf=confidence,
            classes=[PERSON_CLASS_ID],  # Only detect persons
            tracker="botsort.yaml",  # Explicitly use BoT-SORT tracker
            verbose=False,  # Suppress console output
        )

        # Extract detections from this frame
        frame_detections = []

        if results and len(results) > 0:
            result = results[0]

            # Check if there are any boxes detected
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes

                # Get tracking IDs (may be None if tracking hasn't assigned IDs yet)
                track_ids = boxes.id
                if track_ids is not None:
                    track_ids = track_ids.cpu().numpy().astype(int)
                else:
                    track_ids = [None] * len(boxes)

                # Get bounding boxes (xyxy format: x1, y1, x2, y2)
                xyxy_boxes = boxes.xyxy.cpu().numpy()

                # Get confidence scores
                confidences = boxes.conf.cpu().numpy()

                # Process each detection
                for i, (box, conf, track_id) in enumerate(
                    zip(xyxy_boxes, confidences, track_ids)
                ):
                    x1, y1, x2, y2 = box.tolist()

                    detection = {
                        "track_id": int(track_id) if track_id is not None else None,
                        "confidence": round(float(conf), 4),
                        "bbox": {
                            "x1": round(x1, 2),
                            "y1": round(y1, 2),
                            "x2": round(x2, 2),
                            "y2": round(y2, 2),
                        },
                        "bbox_center": {
                            "x": round((x1 + x2) / 2, 2),
                            "y": round((y1 + y2) / 2, 2),
                        },
                        "bbox_dimensions": {
                            "width": round(x2 - x1, 2),
                            "height": round(y2 - y1, 2),
                        },
                    }

                    frame_detections.append(detection)

                    # Track unique IDs
                    if track_id is not None:
                        unique_track_ids.add(int(track_id))

        # Store frame data
        frame_data = {
            "frame_number": frame_number,
            "timestamp_sec": round(frame_number / fps, 3) if fps > 0 else 0,
            "detections_count": len(frame_detections),
            "detections": frame_detections,
        }

        frames_data.append(frame_data)
        frame_number += 1

    # Release video capture
    cap.release()

    # Build final results
    results_dict = {
        "video_metadata": {
            "fps": round(fps, 2),
            "total_frames": total_frames,
            "resolution": {"width": width, "height": height},
            "duration_sec": round(total_frames / fps, 2) if fps > 0 else 0,
        },
        "tracking_summary": {
            "unique_persons_tracked": len(unique_track_ids),
            "unique_track_ids": sorted(list(unique_track_ids)),
        },
        "frames": frames_data,
    }

    return results_dict


if __name__ == "__main__":
    import uvicorn

    # Run the server
    uvicorn.run(app, host="0.0.0.0", port=8000)
