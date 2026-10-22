"""Run initial object detection against the repository's example image."""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.object_detection import detect_objects, write_detection_json  # noqa: E402


IMAGE_PATH = PROJECT_DIR / "example.jpg"
OUTPUT_PATH = PROJECT_DIR / "example.json"


def main() -> None:
    load_dotenv(PROJECT_DIR / ".env")
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY in .env or your environment before running this script.")
    if not IMAGE_PATH.is_file():
        raise SystemExit(f"Image not found: {IMAGE_PATH}")

    result = detect_objects(IMAGE_PATH)
    write_detection_json(result, OUTPUT_PATH)
    print(OUTPUT_PATH.read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
