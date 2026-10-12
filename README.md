# YOLO Object Detection

A Python-based object detection project using **YOLO11** and **OpenCV**. The project detects objects in images, video files, and real-time webcam streams using a pretrained YOLO model.

## Features

- Detect objects in images.
- Detect objects in video files.
- Perform real-time object detection using a webcam.
- Display detected objects with bounding boxes and confidence scores.
- Save annotated images and processed videos.
- Select input images and videos using a file dialog.
- Use a simple command-line menu to choose the detection mode.

## Technologies

- Python
- Ultralytics YOLO11
- OpenCV

## Requirements

- Python 3.10 or a compatible Python version supported by the installed Ultralytics release.
- A webcam is required only for webcam detection.
- Internet access may be required on the first run to download the pretrained model.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/aidinpho/YOLO-Object-Detection.git
   ```

2. Navigate to the project directory:

   ```bash
   cd YOLO-Object-Detection
   ```

3. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the main program:

```bash
python main.py
```

Choose one of the available options:

1. Detect objects in an image.
2. Detect objects in a video.
3. Detect objects using a webcam.
4. Exit the program.

You can also run the image or video detection scripts directly.

## Model

This project uses the pretrained `yolo11n.pt` model from Ultralytics. The model is designed for general-purpose object detection and supports the classes included in the COCO dataset.

The model file is excluded from the repository through `.gitignore`. It may be downloaded automatically by Ultralytics when the program runs, provided that the required network access is available.

## Output

- Annotated images are saved in `outputs/images/`.
- Processed videos are saved in `outputs/videos/`.

Input media and generated output files are excluded from Git tracking by default.

## Limitations

- Detection accuracy depends on image quality, lighting, object size, and the classes supported by the pretrained model.
- The model may miss objects or incorrectly classify similar-looking objects.
- This project uses a pretrained model and does not include custom model training.
