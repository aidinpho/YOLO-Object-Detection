import subprocess
import sys
from pathlib import Path


# Project directory
PROJECT_DIR = Path(__file__).resolve().parent

# Available detection scripts
SCRIPTS = {
    "1": ("Image Detection", "detect_image.py"),
    "2": ("Video Detection", "detect_video.py"),
}


def run_script(script_name , *args):
    """Run a Python script from the project directory."""

    script_path = PROJECT_DIR / script_name

    if not script_path.is_file():
        print(f"\nError: {script_name} was not found.")
        return

    print(f"\nStarting {script_name}...\n")

    try:
        subprocess.run(
            [sys.executable, str(script_path),*args],
            cwd=PROJECT_DIR,
            check=False,
        )
    except OSError as error:
        print(f"Could not start the script: {error}")


def main():
    while True:
        print("\n" + "=" * 36)
        print("       YOLO OBJECT DETECTION")
        print("=" * 36)
        print("1. Detect objects in an image")
        print("2. Detect objects in a video")
        print("3. Detect objects using webcam")
        print("4. Exit")
        print("=" * 36)

        choice = input("Choose an option (1-4): ").strip()

        if choice == "1":
            run_script("detect_image.py")

        elif choice == "2":
            run_script("detect_video.py", "video")

        elif choice == "3":
            run_script("detect_video.py", "webcam")

        elif choice == "4":
            print("Goodbye!")
            break

        else:
            print("Invalid option. Please choose 1, 2, 3, or 4.")


if __name__ == "__main__":
    main()
