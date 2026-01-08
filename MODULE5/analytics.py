"""
Analytics Module for YOLO11 Video Tracking API

This module contains functions for computing advanced analytics from tracking data:
- Overcrowding alerts
- Trend analysis
- Presence and movement heat maps
- Time-based heat visualization
"""

from typing import Optional
import numpy as np
import cv2


# Default ROI margins (pixels to exclude from edges)
DEFAULT_ROI_MARGIN_TOP = 50
DEFAULT_ROI_MARGIN_BOTTOM = 30
DEFAULT_ROI_MARGIN_LEFT = 40
DEFAULT_ROI_MARGIN_RIGHT = 40

# Default heatmap blur kernel size (must be odd number)
DEFAULT_HEATMAP_BLUR_SIZE = 27


def is_detection_in_roi(
    detection: dict,
    frame_width: int,
    frame_height: int,
    margin_top: int = DEFAULT_ROI_MARGIN_TOP,
    margin_bottom: int = DEFAULT_ROI_MARGIN_BOTTOM,
    margin_left: int = DEFAULT_ROI_MARGIN_LEFT,
    margin_right: int = DEFAULT_ROI_MARGIN_RIGHT,
) -> bool:
    """
    Check if a detection's center is within the valid ROI.

    Args:
        detection: Detection dictionary with bbox_center
        frame_width: Video frame width
        frame_height: Video frame height
        margin_top: Pixels to exclude from top
        margin_bottom: Pixels to exclude from bottom
        margin_left: Pixels to exclude from left
        margin_right: Pixels to exclude from right

    Returns:
        True if detection center is within ROI
    """
    bbox_center = detection.get("bbox_center", {})
    x = bbox_center.get("x", 0)
    y = bbox_center.get("y", 0)

    # Define ROI boundaries
    roi_left = margin_left
    roi_right = frame_width - margin_right
    roi_top = margin_top
    roi_bottom = frame_height - margin_bottom

    return roi_left <= x <= roi_right and roi_top <= y <= roi_bottom


def apply_heatmap_blur(heatmap: np.ndarray, blur_size: int = DEFAULT_HEATMAP_BLUR_SIZE) -> np.ndarray:
    """
    Apply Gaussian blur to inflate/spread the heatmap for better visualization.

    Args:
        heatmap: 2D numpy array with heat values
        blur_size: Size of the Gaussian kernel (must be odd)

    Returns:
        Blurred heatmap
    """
    # Ensure blur_size is odd
    if blur_size % 2 == 0:
        blur_size += 1

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(heatmap.astype(np.float32), (blur_size, blur_size), 0)

    return blurred


def compute_overcrowding_alerts(
    frames_data: list,
    max_persons: int,
    fps: float,
) -> dict:
    """
    Compute overcrowding alerts from frame-by-frame tracking data.

    Args:
        frames_data: List of frame dictionaries with detection counts
        max_persons: Maximum allowed persons before alert
        fps: Video frames per second

    Returns:
        Dictionary containing overcrowding alert information
    """
    events = []
    current_event = None

    for frame in frames_data:
        frame_number = frame.get("frame_number", 0)
        count = frame.get("detections_count", 0)
        timestamp = frame.get("timestamp_sec", frame_number / fps if fps > 0 else 0)

        if count > max_persons:
            # Overcrowding detected
            if current_event is None:
                # Start new overcrowding event
                current_event = {
                    "start_frame": frame_number,
                    "start_time_sec": round(timestamp, 3),
                    "peak_count": count,
                    "peak_frame": frame_number,
                }
            else:
                # Continue existing event, update peak if necessary
                if count > current_event["peak_count"]:
                    current_event["peak_count"] = count
                    current_event["peak_frame"] = frame_number
        else:
            # No overcrowding
            if current_event is not None:
                # End the current event
                prev_frame = frames_data[frame_number - 1] if frame_number > 0 else frame
                current_event["end_frame"] = prev_frame.get("frame_number", frame_number - 1)
                current_event["end_time_sec"] = round(
                    prev_frame.get("timestamp_sec", (frame_number - 1) / fps if fps > 0 else 0), 3
                )
                current_event["duration_sec"] = round(
                    current_event["end_time_sec"] - current_event["start_time_sec"], 3
                )
                events.append(current_event)
                current_event = None

    # Handle case where video ends during overcrowding
    if current_event is not None:
        last_frame = frames_data[-1] if frames_data else {"frame_number": 0, "timestamp_sec": 0}
        current_event["end_frame"] = last_frame.get("frame_number", 0)
        current_event["end_time_sec"] = round(last_frame.get("timestamp_sec", 0), 3)
        current_event["duration_sec"] = round(
            current_event["end_time_sec"] - current_event["start_time_sec"], 3
        )
        events.append(current_event)

    return {
        "threshold": max_persons,
        "total_alerts": len(events),
        "total_overcrowding_duration_sec": round(sum(e["duration_sec"] for e in events), 3),
        "events": events,
    }


def compute_trend_analysis(frames_data: list, fps: float) -> dict:
    """
    Compute trend analysis for person count over time.

    Args:
        frames_data: List of frame dictionaries with detection counts
        fps: Video frames per second

    Returns:
        Dictionary containing trend analysis data
    """
    if not frames_data:
        return {
            "min_persons": 0,
            "max_persons": 0,
            "average_persons": 0.0,
            "std_deviation": 0.0,
            "person_count_timeline": [],
        }

    # Extract counts
    counts = [frame.get("detections_count", 0) for frame in frames_data]

    # Compute statistics
    min_count = min(counts)
    max_count = max(counts)
    avg_count = sum(counts) / len(counts)
    
    # Standard deviation
    variance = sum((c - avg_count) ** 2 for c in counts) / len(counts)
    std_dev = variance ** 0.5

    # Build timeline (sample every N frames to reduce data size for long videos)
    sample_interval = max(1, len(frames_data) // 500)  # Max ~500 data points
    
    timeline = []
    for i, frame in enumerate(frames_data):
        if i % sample_interval == 0 or i == len(frames_data) - 1:
            timeline.append({
                "frame": frame.get("frame_number", i),
                "time_sec": round(frame.get("timestamp_sec", i / fps if fps > 0 else 0), 3),
                "count": frame.get("detections_count", 0),
            })

    return {
        "min_persons": min_count,
        "max_persons": max_count,
        "average_persons": round(avg_count, 2),
        "std_deviation": round(std_dev, 2),
        "total_frames_analyzed": len(frames_data),
        "person_count_timeline": timeline,
    }


def compute_presence_heatmap(
    frames_data: list,
    width: int,
    height: int,
    scale_factor: float = 0.1,
    blur_size: int = DEFAULT_HEATMAP_BLUR_SIZE,
    roi_margin_top: int = DEFAULT_ROI_MARGIN_TOP,
    roi_margin_bottom: int = DEFAULT_ROI_MARGIN_BOTTOM,
) -> dict:
    """
    Compute presence heat map showing where people stand/exist.

    Uses bounding box centers accumulated over all frames.

    Args:
        frames_data: List of frame dictionaries with detections
        width: Video width in pixels
        height: Video height in pixels
        scale_factor: Scale factor to reduce heatmap size (0.1 = 10% of original)
        blur_size: Gaussian blur kernel size for heatmap inflation
        roi_margin_top: Pixels to exclude from top edge
        roi_margin_bottom: Pixels to exclude from bottom edge

    Returns:
        Dictionary containing presence heat map data
    """
    # Create scaled heatmap for memory efficiency
    scaled_width = max(1, int(width * scale_factor))
    scaled_height = max(1, int(height * scale_factor))
    
    heatmap = np.zeros((scaled_height, scaled_width), dtype=np.int32)

    for frame in frames_data:
        detections = frame.get("detections", [])
        for detection in detections:
            # Filter by ROI
            if not is_detection_in_roi(detection, width, height, 
                                        roi_margin_top, roi_margin_bottom):
                continue

            bbox_center = detection.get("bbox_center", {})
            x = bbox_center.get("x", 0)
            y = bbox_center.get("y", 0)

            # Scale coordinates to heatmap size
            hx = int(x * scale_factor)
            hy = int(y * scale_factor)

            # Ensure within bounds
            hx = max(0, min(hx, scaled_width - 1))
            hy = max(0, min(hy, scaled_height - 1))

            # Increment heat at this location
            heatmap[hy, hx] += 1

    # Apply Gaussian blur to inflate the heatmap
    heatmap_blurred = apply_heatmap_blur(heatmap, blur_size)

    # Normalize to 0-255 for visualization
    max_val = heatmap_blurred.max()
    if max_val > 0:
        heatmap_normalized = (heatmap_blurred / max_val * 255).astype(np.uint8)
    else:
        heatmap_normalized = heatmap_blurred.astype(np.uint8)

    return {
        "heatmap": heatmap_normalized.tolist(),
        "heatmap_raw": heatmap.tolist(),
        "resolution": {
            "width": scaled_width,
            "height": scaled_height,
        },
        "original_resolution": {
            "width": width,
            "height": height,
        },
        "scale_factor": scale_factor,
        "blur_size": blur_size,
        "roi_margins": {
            "top": roi_margin_top,
            "bottom": roi_margin_bottom,
        },
        "max_heat_value": int(max_val),
    }


def compute_movement_heatmap(
    frames_data: list,
    width: int,
    height: int,
    scale_factor: float = 0.1,
    blur_size: int = DEFAULT_HEATMAP_BLUR_SIZE,
    roi_margin_top: int = DEFAULT_ROI_MARGIN_TOP,
    roi_margin_bottom: int = DEFAULT_ROI_MARGIN_BOTTOM,
) -> dict:
    """
    Compute movement heat map showing where people move through.

    Tracks center point movement between consecutive frames for each tracked ID.

    Args:
        frames_data: List of frame dictionaries with detections
        width: Video width in pixels
        height: Video height in pixels
        scale_factor: Scale factor to reduce heatmap size
        blur_size: Gaussian blur kernel size for heatmap inflation
        roi_margin_top: Pixels to exclude from top edge
        roi_margin_bottom: Pixels to exclude from bottom edge

    Returns:
        Dictionary containing movement heat map data
    """
    scaled_width = max(1, int(width * scale_factor))
    scaled_height = max(1, int(height * scale_factor))
    
    heatmap = np.zeros((scaled_height, scaled_width), dtype=np.int32)

    # Track previous positions for each ID
    prev_positions = {}

    for frame in frames_data:
        detections = frame.get("detections", [])
        current_positions = {}

        for detection in detections:
            # Filter by ROI
            if not is_detection_in_roi(detection, width, height,
                                        roi_margin_top, roi_margin_bottom):
                continue

            track_id = detection.get("track_id")
            if track_id is None:
                continue

            bbox_center = detection.get("bbox_center", {})
            x = bbox_center.get("x", 0)
            y = bbox_center.get("y", 0)

            current_positions[track_id] = (x, y)

            # If we have a previous position for this ID, draw line between them
            if track_id in prev_positions:
                prev_x, prev_y = prev_positions[track_id]
                
                # Use Bresenham-like approach to mark all points along movement path
                points = _get_line_points(
                    int(prev_x * scale_factor),
                    int(prev_y * scale_factor),
                    int(x * scale_factor),
                    int(y * scale_factor),
                )
                
                for px, py in points:
                    px = max(0, min(px, scaled_width - 1))
                    py = max(0, min(py, scaled_height - 1))
                    heatmap[py, px] += 1

        prev_positions = current_positions

    # Apply Gaussian blur to inflate the heatmap
    heatmap_blurred = apply_heatmap_blur(heatmap, blur_size)

    # Normalize to 0-255
    max_val = heatmap_blurred.max()
    if max_val > 0:
        heatmap_normalized = (heatmap_blurred / max_val * 255).astype(np.uint8)
    else:
        heatmap_normalized = heatmap_blurred.astype(np.uint8)

    return {
        "heatmap": heatmap_normalized.tolist(),
        "heatmap_raw": heatmap.tolist(),
        "resolution": {
            "width": scaled_width,
            "height": scaled_height,
        },
        "original_resolution": {
            "width": width,
            "height": height,
        },
        "scale_factor": scale_factor,
        "blur_size": blur_size,
        "roi_margins": {
            "top": roi_margin_top,
            "bottom": roi_margin_bottom,
        },
        "max_heat_value": int(max_val),
    }


def _get_line_points(x0: int, y0: int, x1: int, y1: int) -> list:
    """
    Get all points along a line using Bresenham's algorithm.

    Args:
        x0, y0: Start point
        x1, y1: End point

    Returns:
        List of (x, y) tuples
    """
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

    return points


def compute_time_based_heatmap(
    frames_data: list,
    width: int,
    height: int,
    fps: float,
    bucket_seconds: float = 1.0,
    scale_factor: float = 0.1,
    blur_size: int = DEFAULT_HEATMAP_BLUR_SIZE,
    roi_margin_top: int = DEFAULT_ROI_MARGIN_TOP,
    roi_margin_bottom: int = DEFAULT_ROI_MARGIN_BOTTOM,
) -> dict:
    """
    Compute time-based heat map with presence data aggregated by time buckets.

    Args:
        frames_data: List of frame dictionaries with detections
        width: Video width in pixels
        height: Video height in pixels
        fps: Video frames per second
        bucket_seconds: Time bucket size in seconds
        scale_factor: Scale factor to reduce heatmap size
        blur_size: Gaussian blur kernel size for heatmap inflation
        roi_margin_top: Pixels to exclude from top edge
        roi_margin_bottom: Pixels to exclude from bottom edge

    Returns:
        Dictionary containing cumulative heat map with time metadata
    """
    scaled_width = max(1, int(width * scale_factor))
    scaled_height = max(1, int(height * scale_factor))
    
    # Group frames by time bucket
    buckets = {}
    
    for frame in frames_data:
        timestamp = frame.get("timestamp_sec", 0)
        bucket_index = int(timestamp / bucket_seconds)
        
        if bucket_index not in buckets:
            buckets[bucket_index] = []
        buckets[bucket_index].append(frame)

    # Compute heat map for each bucket
    time_buckets_data = []
    cumulative_heatmap = np.zeros((scaled_height, scaled_width), dtype=np.int32)

    for bucket_index in sorted(buckets.keys()):
        bucket_frames = buckets[bucket_index]
        bucket_heatmap = np.zeros((scaled_height, scaled_width), dtype=np.int32)
        
        for frame in bucket_frames:
            detections = frame.get("detections", [])
            for detection in detections:
                # Filter by ROI
                if not is_detection_in_roi(detection, width, height,
                                            roi_margin_top, roi_margin_bottom):
                    continue

                bbox_center = detection.get("bbox_center", {})
                x = bbox_center.get("x", 0)
                y = bbox_center.get("y", 0)

                hx = int(x * scale_factor)
                hy = int(y * scale_factor)
                hx = max(0, min(hx, scaled_width - 1))
                hy = max(0, min(hy, scaled_height - 1))

                bucket_heatmap[hy, hx] += 1
                cumulative_heatmap[hy, hx] += 1

        # Apply blur and normalize bucket heatmap
        bucket_blurred = apply_heatmap_blur(bucket_heatmap, blur_size)
        bucket_max = bucket_blurred.max()
        if bucket_max > 0:
            bucket_normalized = (bucket_blurred / bucket_max * 255).astype(np.uint8)
        else:
            bucket_normalized = bucket_blurred.astype(np.uint8)

        time_buckets_data.append({
            "bucket_index": bucket_index,
            "start_time_sec": round(bucket_index * bucket_seconds, 3),
            "end_time_sec": round((bucket_index + 1) * bucket_seconds, 3),
            "frame_count": len(bucket_frames),
            "total_detections": sum(f.get("detections_count", 0) for f in bucket_frames),
            "heatmap": bucket_normalized.tolist(),
        })

    # Apply blur and normalize cumulative heatmap
    cumulative_blurred = apply_heatmap_blur(cumulative_heatmap, blur_size)
    max_val = cumulative_blurred.max()
    if max_val > 0:
        cumulative_normalized = (cumulative_blurred / max_val * 255).astype(np.uint8)
    else:
        cumulative_normalized = cumulative_blurred.astype(np.uint8)

    return {
        "bucket_seconds": bucket_seconds,
        "total_buckets": len(time_buckets_data),
        "resolution": {
            "width": scaled_width,
            "height": scaled_height,
        },
        "original_resolution": {
            "width": width,
            "height": height,
        },
        "scale_factor": scale_factor,
        "cumulative_heatmap": cumulative_normalized.tolist(),
        "cumulative_heatmap_raw": cumulative_heatmap.tolist(),
        "max_heat_value": int(max_val),
        "time_buckets": time_buckets_data,
    }


def compute_all_analytics(
    frames_data: list,
    video_metadata: dict,
    max_persons: int = 10,
    bucket_seconds: float = 1.0,
    heatmap_scale: float = 0.1,
) -> dict:
    """
    Compute all analytics from tracking data.

    Args:
        frames_data: List of frame dictionaries with detections
        video_metadata: Dictionary with video info (fps, width, height)
        max_persons: Overcrowding threshold
        bucket_seconds: Time bucket size for time-based heatmap
        heatmap_scale: Scale factor for heatmaps

    Returns:
        Dictionary containing all analytics results
    """
    fps = video_metadata.get("fps", 30)
    width = video_metadata.get("resolution", {}).get("width", 1920)
    height = video_metadata.get("resolution", {}).get("height", 1080)

    return {
        "overcrowding_alerts": compute_overcrowding_alerts(frames_data, max_persons, fps),
        "trend_analysis": compute_trend_analysis(frames_data, fps),
        "presence_heatmap": compute_presence_heatmap(frames_data, width, height, heatmap_scale),
        "movement_heatmap": compute_movement_heatmap(frames_data, width, height, heatmap_scale),
        "time_based_heatmap": compute_time_based_heatmap(
            frames_data, width, height, fps, bucket_seconds, heatmap_scale
        ),
    }
