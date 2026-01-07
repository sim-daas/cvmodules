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

from analytics import (
    compute_overcrowding_alerts,
    compute_trend_analysis,
    compute_presence_heatmap,
    compute_movement_heatmap,
    compute_time_based_heatmap,
    compute_all_analytics,
)

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

# Default analytics parameters
DEFAULT_MAX_PERSONS = 10
DEFAULT_BUCKET_SECONDS = 1.0
DEFAULT_HEATMAP_SCALE = 0.1


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "YOLO11 Video Tracking API with Analytics",
        "endpoints": {
            "/track": "POST - Basic video tracking (returns frame-by-frame detections)",
            "/analyze": "POST - Full analytics (tracking + overcrowding + trends + heatmaps)",
            "/analyze/overcrowding": "POST - Overcrowding alerts only",
            "/analyze/trends": "POST - Trend analysis only",
            "/analyze/heatmaps": "POST - Heat maps only (presence + movement + time-based)",
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


# =============================================================================
# Analytics Endpoints
# =============================================================================


async def _process_video_for_analytics(
    video: UploadFile,
    confidence: float,
) -> tuple:
    """
    Helper function to process video and return tracking results with temp path handling.

    Returns:
        Tuple of (tracking_results, video_filename)
    """
    allowed_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}
    file_ext = os.path.splitext(video.filename)[1].lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed types: {', '.join(allowed_extensions)}",
        )

    temp_dir = tempfile.mkdtemp()
    temp_video_path = os.path.join(temp_dir, video.filename)

    try:
        with open(temp_video_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)

        tracking_results = process_video(temp_video_path, confidence)
        return tracking_results, video.filename

    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


@app.post("/analyze")
async def analyze_video(
    video: UploadFile = File(..., description="Video file to process"),
    confidence: Optional[float] = Query(
        default=DEFAULT_CONFIDENCE,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for detections",
    ),
    max_persons: Optional[int] = Query(
        default=DEFAULT_MAX_PERSONS,
        ge=1,
        description="Maximum persons before overcrowding alert",
    ),
    bucket_seconds: Optional[float] = Query(
        default=DEFAULT_BUCKET_SECONDS,
        ge=0.1,
        description="Time bucket size in seconds for time-based heatmap",
    ),
    heatmap_scale: Optional[float] = Query(
        default=DEFAULT_HEATMAP_SCALE,
        ge=0.01,
        le=1.0,
        description="Scale factor for heatmap resolution (0.1 = 10% of original)",
    ),
):
    """
    Full analytics endpoint: tracking + overcrowding + trends + heatmaps.

    Returns all tracking data plus comprehensive analytics.
    """
    try:
        tracking_results, video_filename = await _process_video_for_analytics(
            video, confidence
        )

        frames_data = tracking_results.get("frames", [])
        video_metadata = tracking_results.get("video_metadata", {})

        analytics = compute_all_analytics(
            frames_data=frames_data,
            video_metadata=video_metadata,
            max_persons=max_persons,
            bucket_seconds=bucket_seconds,
            heatmap_scale=heatmap_scale,
        )

        return JSONResponse(
            content={
                "status": "success",
                "video_name": video_filename,
                "confidence_threshold": confidence,
                "total_frames": len(frames_data),
                "results": tracking_results,
                "analytics": analytics,
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error analyzing video: {str(e)}",
        )


@app.post("/analyze/overcrowding")
async def analyze_overcrowding(
    video: UploadFile = File(..., description="Video file to process"),
    confidence: Optional[float] = Query(
        default=DEFAULT_CONFIDENCE,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for detections",
    ),
    max_persons: Optional[int] = Query(
        default=DEFAULT_MAX_PERSONS,
        ge=1,
        description="Maximum persons before overcrowding alert",
    ),
):
    """
    Overcrowding analysis endpoint.

    Returns tracking data with overcrowding alerts including:
    - Time ranges (start/end timestamps)
    - Duration of each event
    - Peak count during overcrowding
    """
    try:
        tracking_results, video_filename = await _process_video_for_analytics(
            video, confidence
        )

        frames_data = tracking_results.get("frames", [])
        video_metadata = tracking_results.get("video_metadata", {})
        fps = video_metadata.get("fps", 30)

        overcrowding_alerts = compute_overcrowding_alerts(
            frames_data=frames_data,
            max_persons=max_persons,
            fps=fps,
        )

        return JSONResponse(
            content={
                "status": "success",
                "video_name": video_filename,
                "confidence_threshold": confidence,
                "total_frames": len(frames_data),
                "video_metadata": video_metadata,
                "tracking_summary": tracking_results.get("tracking_summary", {}),
                "overcrowding_alerts": overcrowding_alerts,
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error analyzing overcrowding: {str(e)}",
        )


@app.post("/analyze/trends")
async def analyze_trends(
    video: UploadFile = File(..., description="Video file to process"),
    confidence: Optional[float] = Query(
        default=DEFAULT_CONFIDENCE,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for detections",
    ),
):
    """
    Trend analysis endpoint.

    Returns person count statistics over time:
    - Min/max/average person count
    - Standard deviation
    - Person count timeline (sampled for efficiency)
    """
    try:
        tracking_results, video_filename = await _process_video_for_analytics(
            video, confidence
        )

        frames_data = tracking_results.get("frames", [])
        video_metadata = tracking_results.get("video_metadata", {})
        fps = video_metadata.get("fps", 30)

        trend_analysis = compute_trend_analysis(
            frames_data=frames_data,
            fps=fps,
        )

        return JSONResponse(
            content={
                "status": "success",
                "video_name": video_filename,
                "confidence_threshold": confidence,
                "total_frames": len(frames_data),
                "video_metadata": video_metadata,
                "tracking_summary": tracking_results.get("tracking_summary", {}),
                "trend_analysis": trend_analysis,
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error analyzing trends: {str(e)}",
        )


@app.post("/analyze/heatmaps")
async def analyze_heatmaps(
    video: UploadFile = File(..., description="Video file to process"),
    confidence: Optional[float] = Query(
        default=DEFAULT_CONFIDENCE,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for detections",
    ),
    bucket_seconds: Optional[float] = Query(
        default=DEFAULT_BUCKET_SECONDS,
        ge=0.1,
        description="Time bucket size in seconds for time-based heatmap",
    ),
    heatmap_scale: Optional[float] = Query(
        default=DEFAULT_HEATMAP_SCALE,
        ge=0.01,
        le=1.0,
        description="Scale factor for heatmap resolution",
    ),
):
    """
    Heat maps analysis endpoint.

    Returns:
    - Presence heatmap (where people stand)
    - Movement heatmap (where people move through)
    - Time-based cumulative heatmap with bucket metadata
    """
    try:
        tracking_results, video_filename = await _process_video_for_analytics(
            video, confidence
        )

        frames_data = tracking_results.get("frames", [])
        video_metadata = tracking_results.get("video_metadata", {})
        fps = video_metadata.get("fps", 30)
        width = video_metadata.get("resolution", {}).get("width", 1920)
        height = video_metadata.get("resolution", {}).get("height", 1080)

        presence_heatmap = compute_presence_heatmap(
            frames_data=frames_data,
            width=width,
            height=height,
            scale_factor=heatmap_scale,
        )

        movement_heatmap = compute_movement_heatmap(
            frames_data=frames_data,
            width=width,
            height=height,
            scale_factor=heatmap_scale,
        )

        time_based_heatmap = compute_time_based_heatmap(
            frames_data=frames_data,
            width=width,
            height=height,
            fps=fps,
            bucket_seconds=bucket_seconds,
            scale_factor=heatmap_scale,
        )

        return JSONResponse(
            content={
                "status": "success",
                "video_name": video_filename,
                "confidence_threshold": confidence,
                "total_frames": len(frames_data),
                "video_metadata": video_metadata,
                "tracking_summary": tracking_results.get("tracking_summary", {}),
                "heatmaps": {
                    "presence": presence_heatmap,
                    "movement": movement_heatmap,
                    "time_based": time_based_heatmap,
                },
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating heatmaps: {str(e)}",
        )


if __name__ == "__main__":
    import uvicorn

    # Run the server
    uvicorn.run(app, host="0.0.0.0", port=8000)

