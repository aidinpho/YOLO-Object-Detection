import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

import tkinter as tk
from tkinter import filedialog

# Project paths
VIDEO_DIR = Path("videos/input")
OUTPUT_DIR = Path("outputs/videos")
MODEL_PATH = "yolo11n.pt"

CONFIDENCE = 0.25
DISPLAY_WIDTH = 1100
DISPLAY_HEIGHT = 750




import tkinter as tk
from tkinter import filedialog


def get_video_source(source_type=None):
    """Select a video file or use the webcam directly."""

    if source_type == "webcam":
        return 0, "webcam"

    if source_type == "video":
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)

        try:
            video_path = filedialog.askopenfilename(
                parent=root,
                title="Select a video",
                filetypes=[
                    (
                        "Video files",
                        "*.mp4 *.avi *.mov *.mkv *.wmv *.mpeg *.mpg",
                    ),
                    ("All files", "*.*"),
                ],
            )
        finally:
            root.destroy()

        if not video_path:
            print("No video selected.")
            return None

        return video_path, "video"

    # Fallback for running detect_video.py directly
    print("\n=== YOLO Object Detection ===")
    print("1. Process a video file")
    print("2. Use the webcam")

    choice = input("Choose an option (1/2): ").strip()

    if choice == "1":
        return get_video_source("video")

    if choice == "2":
        return get_video_source("webcam")

    print("Invalid choice.")
    return None



def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    source_type = sys.argv[1] if len(sys.argv) > 1 else None
    source_info = get_video_source(source_type)

    if source_info is None:
        return

    source, source_type = source_info

    # Open the video or webcam
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("Error: Could not open the video source.")
        return

    # Read video properties
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0 or fps > 120:
        fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if width <= 0 or height <= 0:
        print("Error: Could not read video dimensions.")
        cap.release()
        return

    # Prepare output video
    output_path = OUTPUT_DIR / f"{source_type}_result.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height),
    )

    if not writer.isOpened():
        print("Error: Could not create output video.")
        cap.release()
        return

    # Load YOLO
    model = YOLO(MODEL_PATH)

    print("\nDetection started.")
    print("Press 'q' to stop and save the output video.")

    try:
        while True:
            success, frame = cap.read()

            if not success:
                print("End of video or frame could not be read.")
                break

            # Detect objects
            result = model.predict(
                source=frame,
                conf=CONFIDENCE,
                verbose=False,
            )[0]

            # Draw boxes and labels
            annotated_frame = result.plot()

            # Save the full-resolution frame
            writer.write(annotated_frame)

            # Resize only for display
            scale = min(
                DISPLAY_WIDTH / width,
                DISPLAY_HEIGHT / height,
                1.0,
            )

            display_frame = cv2.resize(
                annotated_frame,
                (
                    int(width * scale),
                    int(height * scale),
                ),
            )

            cv2.imshow(
                "YOLO Object Detection",
                display_frame,
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Stopped by user.")
                break

    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()

    print(f"\nOutput video saved to: {output_path}")


if __name__ == "__main__":
    main()
