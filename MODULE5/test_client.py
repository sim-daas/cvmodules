"""
Test Client for YOLO11 Video Tracking API

This script simulates a client sending a video to the FastAPI server,
receives tracking results, and overlays them on the video for visualization.

Usage:
    python test_client.py --video path/to/video.mp4 --confidence 0.5
    python test_client.py --video path/to/video.mp4 --output output_tracked.mp4
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import requests


# Default settings
DEFAULT_API_URL = "http://localhost:8000"
DEFAULT_CONFIDENCE = 0.5
DEFAULT_OUTPUT_DIR = "output"

# Visualization colors (BGR format)
BOX_COLOR = (0, 255, 0)  # Green
TEXT_COLOR = (255, 255, 255)  # White
TEXT_BG_COLOR = (0, 255, 0)  # Green background for text
FRAME_INFO_COLOR = (0, 255, 255)  # Yellow for frame info


def send_video_to_api(
    video_path: str, api_url: str = DEFAULT_API_URL, confidence: float = DEFAULT_CONFIDENCE
) -> dict:
    """
    Send a video file to the tracking API and return the results.

    Args:
        video_path: Path to the video file
        api_url: Base URL of the API server
        confidence: Minimum confidence threshold (0.0-1.0)

    Returns:
        Dictionary containing the API response with tracking results
    """
    endpoint = f"{api_url}/track"
    params = {"confidence": confidence}

    print(f"📤 Sending video to API: {endpoint}")
    print(f"   Video: {video_path}")
    print(f"   Confidence threshold: {confidence}")

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    try:
        with open(video_path, "rb") as video_file:
            files = {"video": (os.path.basename(video_path), video_file, "video/mp4")}
            response = requests.post(endpoint, files=files, params=params, timeout=300)

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


def overlay_tracking_results(
    video_path: str,
    tracking_results: dict,
    output_path: str = None,
    display: bool = True,
    playback_speed: float = 1.0,
) -> str:
    """
    Overlay tracking results on the video and optionally display/save it.

    Args:
        video_path: Path to the original video file
        tracking_results: Dictionary containing frame-by-frame tracking data
        output_path: Path to save the annotated video (None to skip saving)
        display: Whether to display the video in a window
        playback_speed: Playback speed multiplier (1.0 = normal, 2.0 = 2x speed)

    Returns:
        Path to the saved output video (or None if not saved)
    """
    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\n🎬 Processing video overlay...")
    print(f"   Resolution: {width}x{height}")
    print(f"   FPS: {fps:.2f}")
    print(f"   Total frames: {total_frames}")

    # Setup video writer if output path specified
    writer = None
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"   Output: {output_path}")

    # Get frames data from results
    frames_data = tracking_results.get("results", {}).get("frames", [])
    tracking_summary = tracking_results.get("results", {}).get("tracking_summary", {})

    # Create lookup dictionary for faster access
    frame_lookup = {f["frame_number"]: f for f in frames_data}

    frame_number = 0
    wait_time = int(1000 / (fps * playback_speed)) if fps > 0 else 33

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Get detections for this frame
        frame_data = frame_lookup.get(frame_number, {"detections": []})
        detections = frame_data.get("detections", [])
        timestamp = frame_data.get("timestamp_sec", 0)

        # Draw each detection
        for detection in detections:
            bbox = detection.get("bbox", {})
            track_id = detection.get("track_id")
            confidence = detection.get("confidence", 0)

            x1 = int(bbox.get("x1", 0))
            y1 = int(bbox.get("y1", 0))
            x2 = int(bbox.get("x2", 0))
            y2 = int(bbox.get("y2", 0))

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)

            # Prepare label text
            if track_id is not None:
                label = f"ID:{track_id} ({confidence:.2f})"
            else:
                label = f"({confidence:.2f})"

            # Calculate text size for background
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(
                label, font, font_scale, thickness
            )

            # Draw text background
            cv2.rectangle(
                frame,
                (x1, y1 - text_height - 10),
                (x1 + text_width + 10, y1),
                TEXT_BG_COLOR,
                -1,
            )

            # Draw label text
            cv2.putText(
                frame,
                label,
                (x1 + 5, y1 - 5),
                font,
                font_scale,
                TEXT_COLOR,
                thickness,
            )

        # Draw frame info overlay (top-left corner)
        info_text = f"Frame: {frame_number}/{total_frames} | Time: {timestamp:.2f}s | Detections: {len(detections)}"
        cv2.putText(
            frame,
            info_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            FRAME_INFO_COLOR,
            2,
        )

        # Draw tracking summary (top-right corner)
        unique_count = tracking_summary.get("unique_persons_tracked", 0)
        summary_text = f"Unique Persons: {unique_count}"
        text_size = cv2.getTextSize(summary_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.putText(
            frame,
            summary_text,
            (width - text_size[0] - 10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            FRAME_INFO_COLOR,
            2,
        )

        # Write frame to output video
        if writer:
            writer.write(frame)

        # Display frame
        if display:
            cv2.imshow("YOLO11 Tracking Results", frame)

            # Handle key presses
            key = cv2.waitKey(wait_time) & 0xFF
            if key == ord("q"):
                print("\n⏹️  Playback stopped by user")
                break
            elif key == ord(" "):  # Space to pause
                print("⏸️  Paused. Press any key to continue...")
                cv2.waitKey(0)

        frame_number += 1

        # Print progress
        if frame_number % 30 == 0:
            progress = (frame_number / total_frames) * 100
            print(f"   Progress: {progress:.1f}% ({frame_number}/{total_frames})", end="\r")

    # Cleanup
    cap.release()
    if writer:
        writer.release()
    if display:
        cv2.destroyAllWindows()

    print(f"\n✅ Overlay complete! Processed {frame_number} frames.")

    return output_path


def save_json_results(results: dict, output_path: str) -> str:
    """
    Save the tracking results to a JSON file.

    Args:
        results: Dictionary containing tracking results
        output_path: Path to save the JSON file

    Returns:
        Path to the saved JSON file
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"💾 JSON results saved to: {output_path}")
    return output_path


def main():
    """Main entry point for the test client."""
    parser = argparse.ArgumentParser(
        description="Test client for YOLO11 Video Tracking API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_client.py --video sample.mp4
  python test_client.py --video sample.mp4 --confidence 0.7
  python test_client.py --video sample.mp4 --output tracked_output.mp4
  python test_client.py --video sample.mp4 --no-display --output tracked.mp4
        """,
    )

    parser.add_argument(
        "--video", "-v", required=True, help="Path to the input video file"
    )
    parser.add_argument(
        "--confidence",
        "-c",
        type=float,
        default=DEFAULT_CONFIDENCE,
        help=f"Confidence threshold (0.0-1.0, default: {DEFAULT_CONFIDENCE})",
    )
    parser.add_argument(
        "--api-url",
        "-u",
        default=DEFAULT_API_URL,
        help=f"API server URL (default: {DEFAULT_API_URL})",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output video file path (default: output/<video_name>_tracked.mp4)",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Don't display the video (only save to file)",
    )
    parser.add_argument(
        "--speed",
        "-s",
        type=float,
        default=1.0,
        help="Playback speed multiplier (default: 1.0)",
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Save JSON results to a file",
    )

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.video):
        print(f"❌ Error: Video file not found: {args.video}")
        sys.exit(1)

    if not 0.0 <= args.confidence <= 1.0:
        print(f"❌ Error: Confidence must be between 0.0 and 1.0")
        sys.exit(1)

    # Generate default output path if not specified
    if args.output is None and (args.no_display or True):
        video_name = Path(args.video).stem
        args.output = os.path.join(DEFAULT_OUTPUT_DIR, f"{video_name}_tracked.mp4")

    print("=" * 60)
    print("🎯 YOLO11 Video Tracking API - Test Client")
    print("=" * 60)

    # Step 1: Send video to API
    results = send_video_to_api(args.video, args.api_url, args.confidence)

    # Step 2: Save JSON results if requested
    if args.save_json:
        json_path = os.path.join(
            DEFAULT_OUTPUT_DIR, f"{Path(args.video).stem}_results.json"
        )
        save_json_results(results, json_path)

    # Step 3: Overlay results on video
    print("\n" + "-" * 60)
    overlay_tracking_results(
        video_path=args.video,
        tracking_results=results,
        output_path=args.output,
        display=not args.no_display,
        playback_speed=args.speed,
    )

    print("\n" + "=" * 60)
    print("🎉 Test completed successfully!")
    if args.output:
        print(f"   📹 Output video: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
