
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

import cv2
from ultralytics import YOLO


# Load the pretrained YOLO model
model = YOLO("yolo11n.pt")


def select_image():
    """Open a Windows dialog to select an image."""

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    try:
        image_path = filedialog.askopenfilename(
            parent=root,
            title="Select an image",
            filetypes=[
                (
                    "Image files",
                    "*.jpg *.jpeg *.png *.bmp *.webp",
                ),
                ("All files", "*.*"),
            ],
        )
    finally:
        root.destroy()

    if not image_path:
        print("No image selected.")
        return None

    return Path(image_path)


def main():
    # Select an image
    image_path = select_image()

    if image_path is None:
        return

    # Define output paths
    output_dir = Path("outputs/images")
    output_path = output_dir / "result.jpg"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run object detection
    result = model.predict(
        source=str(image_path),
        conf=0.25,
        verbose=False,
    )[0]

    # Print detected objects and confidence scores
    print("\nDetected objects:")

    if result.boxes is None or len(result.boxes) == 0:
        print("No objects detected.")
    else:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = model.names[class_id]

            print(f"{class_name}: {confidence:.2f}")

    # Draw bounding boxes and labels
    annotated_image = result.plot()

    # Save the annotated image
    success = cv2.imwrite(
        str(output_path),
        annotated_image,
    )

    if not success:
        raise RuntimeError("Could not save output image.")

    print(f"\nOutput saved to: {output_path}")

    # Resize the image for display only
    display_width = 1100
    height, width = annotated_image.shape[:2]

    scale = min(
        display_width / width,
        800 / height,
        1.0,
    )

    display_image = cv2.resize(
        annotated_image,
        (
            int(width * scale),
            int(height * scale),
        ),
    )

    cv2.imshow("YOLO Image Detection", display_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
