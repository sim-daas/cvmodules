"""
Test Client for YOLO11 Video Tracking API with Analytics

This script simulates a client sending a video to the FastAPI server,
receives tracking results and analytics, and visualizes them.

Usage:
    # Basic tracking
    python test_client.py --video path/to/video.mp4 --confidence 0.5

    # Full analytics with visualizations
    python test_client.py --video path/to/video.mp4 --analyze --max-persons 5

    # Heat map overlay on video
    python test_client.py --video path/to/video.mp4 --heatmap
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import requests

# Check if matplotlib is available
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠️  matplotlib not installed. Graphs will not be displayed.")
    print("   Install with: pip install matplotlib")


# Default settings
DEFAULT_API_URL = "http://localhost:8000"
DEFAULT_CONFIDENCE = 0.5
DEFAULT_MAX_PERSONS = 10
DEFAULT_BUCKET_SECONDS = 1.0
DEFAULT_HEATMAP_SCALE = 0.1
DEFAULT_OUTPUT_DIR = "output"

# Visualization colors (BGR format)
BOX_COLOR = (0, 255, 0)  # Green
TEXT_COLOR = (255, 255, 255)  # White
TEXT_BG_COLOR = (0, 255, 0)  # Green background for text
FRAME_INFO_COLOR = (0, 255, 255)  # Yellow for frame info
OVERCROWDING_COLOR = (0, 0, 255)  # Red for overcrowding


def send_video_to_api(
    video_path: str,
    api_url: str = DEFAULT_API_URL,
    confidence: float = DEFAULT_CONFIDENCE,
    endpoint: str = "/track",
    extra_params: dict = None,
) -> dict:
    """
    Send a video file to the tracking/analytics API and return the results.

    Args:
        video_path: Path to the video file
        api_url: Base URL of the API server
        confidence: Minimum confidence threshold (0.0-1.0)
        endpoint: API endpoint to call (e.g., /track, /analyze)
        extra_params: Additional query parameters

    Returns:
        Dictionary containing the API response
    """
    full_endpoint = f"{api_url}{endpoint}"
    params = {"confidence": confidence}
    if extra_params:
        params.update(extra_params)

    print(f"📤 Sending video to API: {full_endpoint}")
    print(f"   Video: {video_path}")
    print(f"   Confidence threshold: {confidence}")
    if extra_params:
        for key, value in extra_params.items():
            print(f"   {key}: {value}")

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    try:
        with open(video_path, "rb") as video_file:
            files = {"video": (os.path.basename(video_path), video_file, "video/mp4")}
            response = requests.post(full_endpoint, files=files, params=params, timeout=600)

        response.raise_for_status()
        result = response.json()

        print(f"✅ API response received successfully!")
        print(f"   Total frames processed: {result.get('total_frames', 'N/A')}")

        return result

    except requests.exceptions.ConnectionError:
        print(f"❌ Error: Could not connect to API at {api_url}")
        print("   Make sure the server is running: python main.py")
        sys.exit(1)
    except requests.exceptions.Timeout:
        print("❌ Error: Request timed out. The video might be too large.")
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        print(f"❌ HTTP Error: {e}")
        print(f"   Response: {response.text}")
        sys.exit(1)


def display_overcrowding_graph(analytics_results: dict, max_persons: int):
    """
    Display an overcrowding graph with time on X-axis and person count on Y-axis.

    Args:
        analytics_results: Dictionary containing analytics data
        max_persons: Overcrowding threshold
    """
    if not MATPLOTLIB_AVAILABLE:
        print("⚠️  Cannot display overcrowding graph: matplotlib not installed")
        return

    # Get trend analysis data
    trend_data = analytics_results.get("trend_analysis", {})
    timeline = trend_data.get("person_count_timeline", [])

    if not timeline:
        print("⚠️  No timeline data available for overcrowding graph")
        return

    # Extract data for plotting
    times = [point["time_sec"] for point in timeline]
    counts = [point["count"] for point in timeline]

    # Get overcrowding events
    overcrowding = analytics_results.get("overcrowding_alerts", {})
    events = overcrowding.get("events", [])

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot person count over time
    ax.plot(times, counts, 'b-', linewidth=1.5, label='Person Count')

    # Draw threshold line
    ax.axhline(y=max_persons, color='r', linestyle='--', linewidth=2, 
               label=f'Threshold ({max_persons})')

    # Highlight overcrowding events
    for event in events:
        start_time = event.get("start_time_sec", 0)
        end_time = event.get("end_time_sec", 0)
        peak_count = event.get("peak_count", 0)

        # Shade the overcrowding region
        ax.axvspan(start_time, end_time, alpha=0.3, color='red', 
                   label='_nolegend_')

        # Mark peak point
        # Find the closest time point to mark the peak
        for i, t in enumerate(times):
            if start_time <= t <= end_time and counts[i] == peak_count:
                ax.scatter([t], [peak_count], color='red', s=100, zorder=5,
                          marker='^', label='_nolegend_')
                ax.annotate(f'Peak: {peak_count}', (t, peak_count),
                           textcoords="offset points", xytext=(0, 10),
                           ha='center', fontsize=9, color='red')
                break

    # Labels and title
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Person Count', fontsize=12)
    ax.set_title('Overcrowding Analysis', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Add statistics text
    stats_text = (f"Total Alerts: {overcrowding.get('total_alerts', 0)}\n"
                  f"Total Duration: {overcrowding.get('total_overcrowding_duration_sec', 0):.1f}s\n"
                  f"Avg: {trend_data.get('average_persons', 0):.1f} | "
                  f"Max: {trend_data.get('max_persons', 0)}")
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.show()
    print("📊 Overcrowding graph displayed!")


def display_trend_graph(analytics_results: dict):
    """
    Display a trend graph showing person count over time with statistics.

    Args:
        analytics_results: Dictionary containing analytics data
    """
    if not MATPLOTLIB_AVAILABLE:
        print("⚠️  Cannot display trend graph: matplotlib not installed")
        return

    trend_data = analytics_results.get("trend_analysis", {})
    timeline = trend_data.get("person_count_timeline", [])

    if not timeline:
        print("⚠️  No timeline data available for trend graph")
        return

    times = [point["time_sec"] for point in timeline]
    counts = [point["count"] for point in timeline]

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot person count
    ax.plot(times, counts, 'b-', linewidth=1.5, label='Person Count')
    ax.fill_between(times, counts, alpha=0.3)

    # Draw statistics lines
    avg = trend_data.get("average_persons", 0)
    min_val = trend_data.get("min_persons", 0)
    max_val = trend_data.get("max_persons", 0)

    ax.axhline(y=avg, color='green', linestyle='--', linewidth=1.5, 
               label=f'Average ({avg:.1f})')
    ax.axhline(y=min_val, color='orange', linestyle=':', linewidth=1, 
               label=f'Min ({min_val})')
    ax.axhline(y=max_val, color='red', linestyle=':', linewidth=1, 
               label=f'Max ({max_val})')

    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Person Count', fontsize=12)
    ax.set_title('Person Count Trend Analysis', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    print("📊 Trend graph displayed!")


def create_heatmap_colormap(heatmap_2d: np.ndarray, colormap=cv2.COLORMAP_JET) -> np.ndarray:
    """
    Convert a 2D heatmap array to a colored image.

    Args:
        heatmap_2d: 2D numpy array with heat values (0-255)
        colormap: OpenCV colormap to use

    Returns:
        BGR color image
    """
    # Ensure it's uint8
    heatmap_uint8 = np.array(heatmap_2d, dtype=np.uint8)

    # Apply colormap
    colored = cv2.applyColorMap(heatmap_uint8, colormap)

    return colored


def overlay_heatmap_on_video(
    video_path: str,
    heatmap_data: dict,
    heatmap_type: str = "presence",
    output_path: str = None,
    display: bool = True,
    alpha: float = 0.4,
    playback_speed: float = 1.0,
) -> str:
    """
    Overlay a heatmap on the video frames.

    Args:
        video_path: Path to the original video file
        heatmap_data: Dictionary containing heatmap data from API
        heatmap_type: Type of heatmap to overlay ("presence", "movement", "cumulative")
        output_path: Path to save the output video
        display: Whether to display the video
        alpha: Transparency of the heatmap overlay (0.0-1.0)
        playback_speed: Playback speed multiplier

    Returns:
        Path to saved output video (or None)
    """
    # Select the right heatmap
    if heatmap_type == "presence":
        heatmap_info = heatmap_data.get("presence", {})
    elif heatmap_type == "movement":
        heatmap_info = heatmap_data.get("movement", {})
    elif heatmap_type == "cumulative":
        heatmap_info = heatmap_data.get("time_based", {})
        heatmap_info = {
            "heatmap": heatmap_info.get("cumulative_heatmap", []),
            "resolution": heatmap_info.get("resolution", {}),
            "original_resolution": heatmap_info.get("original_resolution", {}),
        }
    else:
        print(f"⚠️  Unknown heatmap type: {heatmap_type}")
        return None

    heatmap_2d = np.array(heatmap_info.get("heatmap", []), dtype=np.uint8)
    
    if heatmap_2d.size == 0:
        print("⚠️  No heatmap data available")
        return None

    # Get original resolution
    orig_width = heatmap_info.get("original_resolution", {}).get("width", 1920)
    orig_height = heatmap_info.get("original_resolution", {}).get("height", 1080)

    # Create colored heatmap
    colored_heatmap = create_heatmap_colormap(heatmap_2d)

    # Resize to original video resolution
    colored_heatmap_resized = cv2.resize(colored_heatmap, (orig_width, orig_height))

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\n🔥 Overlaying {heatmap_type} heatmap on video...")
    print(f"   Resolution: {width}x{height}")
    print(f"   Heatmap alpha: {alpha}")

    # Resize heatmap if video dimensions don't match
    if colored_heatmap_resized.shape[:2] != (height, width):
        colored_heatmap_resized = cv2.resize(colored_heatmap_resized, (width, height))

    # Setup writer
    writer = None
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_number = 0
    wait_time = int(1000 / (fps * playback_speed)) if fps > 0 else 33

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Blend heatmap with frame
        blended = cv2.addWeighted(frame, 1 - alpha, colored_heatmap_resized, alpha, 0)

        # Add heatmap type label
        label = f"Heatmap: {heatmap_type.capitalize()}"
        cv2.putText(blended, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 
                    (255, 255, 255), 2)

        if writer:
            writer.write(blended)

        if display:
            cv2.imshow(f"Heatmap Overlay - {heatmap_type}", blended)
            key = cv2.waitKey(wait_time) & 0xFF
            if key == ord("q"):
                print("\n⏹️  Playback stopped by user")
                break
            elif key == ord(" "):
                cv2.waitKey(0)

        frame_number += 1
        if frame_number % 30 == 0:
            progress = (frame_number / total_frames) * 100
            print(f"   Progress: {progress:.1f}%", end="\r")

    cap.release()
    if writer:
        writer.release()
    if display:
        cv2.destroyAllWindows()

    print(f"\n✅ Heatmap overlay complete!")
    return output_path


def display_heatmap_static(heatmap_data: dict, heatmap_type: str = "presence"):
    """
    Display a static heatmap image using matplotlib.

    Args:
        heatmap_data: Dictionary containing heatmap data
        heatmap_type: Type of heatmap to display
    """
    if not MATPLOTLIB_AVAILABLE:
        print("⚠️  Cannot display heatmap: matplotlib not installed")
        return

    if heatmap_type == "presence":
        heatmap_info = heatmap_data.get("presence", {})
        title = "Presence Heatmap (Where People Stand)"
    elif heatmap_type == "movement":
        heatmap_info = heatmap_data.get("movement", {})
        title = "Movement Heatmap (Where People Move)"
    elif heatmap_type == "cumulative":
        heatmap_info = heatmap_data.get("time_based", {})
        heatmap_info["heatmap"] = heatmap_info.get("cumulative_heatmap", [])
        title = "Cumulative Time-Based Heatmap"
    else:
        print(f"⚠️  Unknown heatmap type: {heatmap_type}")
        return

    heatmap_2d = np.array(heatmap_info.get("heatmap", []))
    
    if heatmap_2d.size == 0:
        print(f"⚠️  No {heatmap_type} heatmap data available")
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(heatmap_2d, cmap='jet', aspect='auto')
    plt.colorbar(im, ax=ax, label='Heat Intensity')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')

    # Add metadata
    max_heat = heatmap_info.get("max_heat_value", 0)
    resolution = heatmap_info.get("resolution", {})
    orig_res = heatmap_info.get("original_resolution", {})
    
    info_text = (f"Max Heat: {max_heat}\n"
                f"Heatmap Size: {resolution.get('width', 0)}x{resolution.get('height', 0)}\n"
                f"Original: {orig_res.get('width', 0)}x{orig_res.get('height', 0)}")
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.show()
    print(f"📊 {heatmap_type.capitalize()} heatmap displayed!")


def overlay_tracking_results(
    video_path: str,
    tracking_results: dict,
    output_path: str = None,
    display: bool = True,
    playback_speed: float = 1.0,
    overcrowding_threshold: int = None,
) -> str:
    """
    Overlay tracking results on the video and optionally display/save it.

    Args:
        video_path: Path to the original video file
        tracking_results: Dictionary containing frame-by-frame tracking data
        output_path: Path to save the annotated video (None to skip saving)
        display: Whether to display the video in a window
        playback_speed: Playback speed multiplier (1.0 = normal, 2.0 = 2x speed)
        overcrowding_threshold: If set, highlight frames exceeding this count

    Returns:
        Path to the saved output video (or None if not saved)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\n🎬 Processing video overlay...")
    print(f"   Resolution: {width}x{height}")
    print(f"   FPS: {fps:.2f}")
    print(f"   Total frames: {total_frames}")
    if overcrowding_threshold:
        print(f"   Overcrowding threshold: {overcrowding_threshold}")

    writer = None
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"   Output: {output_path}")

    # Get frames data - handle both /track and /analyze response formats
    if "results" in tracking_results:
        frames_data = tracking_results.get("results", {}).get("frames", [])
        tracking_summary = tracking_results.get("results", {}).get("tracking_summary", {})
    else:
        frames_data = tracking_results.get("frames", [])
        tracking_summary = tracking_results.get("tracking_summary", {})

    frame_lookup = {f["frame_number"]: f for f in frames_data}

    frame_number = 0
    wait_time = int(1000 / (fps * playback_speed)) if fps > 0 else 33

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_data = frame_lookup.get(frame_number, {"detections": []})
        detections = frame_data.get("detections", [])
        timestamp = frame_data.get("timestamp_sec", 0)
        detection_count = len(detections)

        # Check for overcrowding
        is_overcrowded = (overcrowding_threshold and 
                          detection_count > overcrowding_threshold)

        # Draw red border if overcrowded
        if is_overcrowded:
            cv2.rectangle(frame, (0, 0), (width-1, height-1), OVERCROWDING_COLOR, 10)

        # Draw each detection
        for detection in detections:
            bbox = detection.get("bbox", {})
            track_id = detection.get("track_id")
            confidence = detection.get("confidence", 0)

            x1 = int(bbox.get("x1", 0))
            y1 = int(bbox.get("y1", 0))
            x2 = int(bbox.get("x2", 0))
            y2 = int(bbox.get("y2", 0))

            box_color = OVERCROWDING_COLOR if is_overcrowded else BOX_COLOR
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

            if track_id is not None:
                label = f"ID:{track_id} ({confidence:.2f})"
            else:
                label = f"({confidence:.2f})"

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(
                label, font, font_scale, thickness
            )

            bg_color = OVERCROWDING_COLOR if is_overcrowded else TEXT_BG_COLOR
            cv2.rectangle(
                frame,
                (x1, y1 - text_height - 10),
                (x1 + text_width + 10, y1),
                bg_color,
                -1,
            )

            cv2.putText(
                frame,
                label,
                (x1 + 5, y1 - 5),
                font,
                font_scale,
                TEXT_COLOR,
                thickness,
            )

        # Draw frame info
        info_text = f"Frame: {frame_number}/{total_frames} | Time: {timestamp:.2f}s | Count: {detection_count}"
        if is_overcrowded:
            info_text += " | ⚠️ OVERCROWDED"
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    FRAME_INFO_COLOR, 2)

        unique_count = tracking_summary.get("unique_persons_tracked", 0)
        summary_text = f"Unique: {unique_count}"
        text_size = cv2.getTextSize(summary_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.putText(frame, summary_text, (width - text_size[0] - 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, FRAME_INFO_COLOR, 2)

        if writer:
            writer.write(frame)

        if display:
            cv2.imshow("YOLO11 Tracking Results", frame)
            key = cv2.waitKey(wait_time) & 0xFF
            if key == ord("q"):
                print("\n⏹️  Playback stopped by user")
                break
            elif key == ord(" "):
                print("⏸️  Paused. Press any key to continue...")
                cv2.waitKey(0)

        frame_number += 1

        if frame_number % 30 == 0:
            progress = (frame_number / total_frames) * 100
            print(f"   Progress: {progress:.1f}% ({frame_number}/{total_frames})", end="\r")

    cap.release()
    if writer:
        writer.release()
    if display:
        cv2.destroyAllWindows()

    print(f"\n✅ Overlay complete! Processed {frame_number} frames.")
    return output_path


def save_json_results(results: dict, output_path: str) -> str:
    """Save the results to a JSON file."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"💾 JSON results saved to: {output_path}")
    return output_path


def main():
    """Main entry point for the test client."""
    parser = argparse.ArgumentParser(
        description="Test client for YOLO11 Video Tracking API with Analytics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic tracking
  python test_client.py --video sample.mp4

  # Full analytics with overcrowding graph
  python test_client.py --video sample.mp4 --analyze --max-persons 5

  # Heatmap overlay on video
  python test_client.py --video sample.mp4 --heatmap --heatmap-type presence

  # All analytics without video display
  python test_client.py --video sample.mp4 --analyze --no-display --save-json
        """,
    )

    parser.add_argument(
        "--video", "-v", required=True, help="Path to the input video file"
    )
    parser.add_argument(
        "--confidence", "-c", type=float, default=DEFAULT_CONFIDENCE,
        help=f"Confidence threshold (0.0-1.0, default: {DEFAULT_CONFIDENCE})"
    )
    parser.add_argument(
        "--api-url", "-u", default=DEFAULT_API_URL,
        help=f"API server URL (default: {DEFAULT_API_URL})"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output video file path (default: output/<video_name>_tracked.mp4)"
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="Don't display the video (only save to file)"
    )
    parser.add_argument(
        "--speed", "-s", type=float, default=1.0,
        help="Playback speed multiplier (default: 1.0)"
    )
    parser.add_argument(
        "--save-json", action="store_true",
        help="Save JSON results to a file"
    )

    # Analytics options
    analytics_group = parser.add_argument_group("Analytics Options")
    analytics_group.add_argument(
        "--analyze", action="store_true",
        help="Use /analyze endpoint for full analytics"
    )
    analytics_group.add_argument(
        "--max-persons", type=int, default=DEFAULT_MAX_PERSONS,
        help=f"Overcrowding threshold (default: {DEFAULT_MAX_PERSONS})"
    )
    analytics_group.add_argument(
        "--bucket-seconds", type=float, default=DEFAULT_BUCKET_SECONDS,
        help=f"Time bucket size for heatmaps (default: {DEFAULT_BUCKET_SECONDS})"
    )
    analytics_group.add_argument(
        "--show-trends", action="store_true",
        help="Display trend analysis graph"
    )
    analytics_group.add_argument(
        "--show-overcrowding", action="store_true",
        help="Display overcrowding analysis graph"
    )

    # Heatmap options
    heatmap_group = parser.add_argument_group("Heatmap Options")
    heatmap_group.add_argument(
        "--heatmap", action="store_true",
        help="Overlay heatmap on video"
    )
    heatmap_group.add_argument(
        "--heatmap-type", choices=["presence", "movement", "cumulative"],
        default="presence", help="Type of heatmap to overlay (default: presence)"
    )
    heatmap_group.add_argument(
        "--heatmap-alpha", type=float, default=0.4,
        help="Heatmap overlay transparency (0.0-1.0, default: 0.4)"
    )
    heatmap_group.add_argument(
        "--show-heatmap-static", action="store_true",
        help="Display static heatmap image"
    )

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.video):
        print(f"❌ Error: Video file not found: {args.video}")
        sys.exit(1)

    if not 0.0 <= args.confidence <= 1.0:
        print(f"❌ Error: Confidence must be between 0.0 and 1.0")
        sys.exit(1)

    # Generate default output path
    if args.output is None:
        video_name = Path(args.video).stem
        suffix = "_analyzed" if args.analyze else "_tracked"
        args.output = os.path.join(DEFAULT_OUTPUT_DIR, f"{video_name}{suffix}.mp4")

    print("=" * 60)
    print("🎯 YOLO11 Video Tracking API - Test Client")
    print("=" * 60)

    # Determine endpoint and parameters
    if args.analyze or args.heatmap:
        endpoint = "/analyze"
        extra_params = {
            "max_persons": args.max_persons,
            "bucket_seconds": args.bucket_seconds,
        }
    else:
        endpoint = "/track"
        extra_params = None

    # Send video to API
    results = send_video_to_api(
        args.video, args.api_url, args.confidence, endpoint, extra_params
    )

    # Save JSON if requested
    if args.save_json:
        json_path = os.path.join(
            DEFAULT_OUTPUT_DIR, f"{Path(args.video).stem}_results.json"
        )
        save_json_results(results, json_path)

    # Display analytics graphs
    analytics = results.get("analytics", {})
    
    if args.analyze or args.show_overcrowding:
        if analytics:
            print("\n" + "-" * 60)
            display_overcrowding_graph(analytics, args.max_persons)
        else:
            print("⚠️  No analytics data available for overcrowding graph")

    if args.show_trends and analytics:
        display_trend_graph(analytics)

    # Display static heatmaps
    if args.show_heatmap_static and analytics:
        heatmaps = {
            "presence": analytics.get("presence_heatmap", {}),
            "movement": analytics.get("movement_heatmap", {}),
            "time_based": analytics.get("time_based_heatmap", {}),
        }
        display_heatmap_static(heatmaps, args.heatmap_type)

    # Overlay heatmap on video
    if args.heatmap and analytics:
        print("\n" + "-" * 60)
        heatmaps = {
            "presence": analytics.get("presence_heatmap", {}),
            "movement": analytics.get("movement_heatmap", {}),
            "time_based": analytics.get("time_based_heatmap", {}),
        }
        heatmap_output = args.output.replace(".mp4", f"_heatmap_{args.heatmap_type}.mp4")
        overlay_heatmap_on_video(
            video_path=args.video,
            heatmap_data=heatmaps,
            heatmap_type=args.heatmap_type,
            output_path=heatmap_output,
            display=not args.no_display,
            alpha=args.heatmap_alpha,
            playback_speed=args.speed,
        )

    # Overlay tracking results on video
    print("\n" + "-" * 60)
    overcrowding_threshold = args.max_persons if args.analyze else None
    overlay_tracking_results(
        video_path=args.video,
        tracking_results=results,
        output_path=args.output,
        display=not args.no_display,
        playback_speed=args.speed,
        overcrowding_threshold=overcrowding_threshold,
    )

    print("\n" + "=" * 60)
    print("🎉 Test completed successfully!")
    if args.output:
        print(f"   📹 Output video: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
