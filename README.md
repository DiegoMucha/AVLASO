# AVLASO
A Vision-Language-Assisted System for Object Detection, Tracking, and Spatial Memory

Idea:
Build a smart object-finding assistant using a fixed camera and a Raspberry Pi 5. The system identifies everyday objects on a desk, tracks where they move, and remembers their last confirmed visible location. A user can ask, "Where are my keys?" and receive the last known location, time, and optionally a saved image.

How it works:
1. Capture an initial desk image. Use an OpenAI vision model to identify different objects first and return compact JSON, e.g. {"objects":[{"n":"mouse","b":[100,600,180,750]}]}, with bounding boxes normalized to 0–1000.
2. Verify or refine the boxes, then register each object with a persistent ID and reference image.
3. Use local computer vision to monitor the fixed scene and track registered objects without constantly calling the API. Just track each object, dont recognize what is it since OpenAI already does it.
4. When the scene changes, identify unmatched/new objects using the vision API, then automatically register and initialize tracking for them.
5. Save meaningful position updates and last-confirmed sightings in SQLite. If an object disappears, report its last observed position rather than claiming to know where it is now.

Suggested stack: Raspberry Pi 5 + camera, OpenCV, OpenAI vision API, optional AI HAT-compatible detector, SQLite, and a simple web interface.