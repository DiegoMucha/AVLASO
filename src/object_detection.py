"""Initial object detection using an OpenAI vision model.

The public ``detect_objects`` function is intentionally independent of the
camera and persistence layers.  A pipeline can pass it either an image path or
already-encoded image bytes and receive the compact structure described in the
project README.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Literal, TypedDict

from openai import OpenAI


DEFAULT_MODEL = "gpt-5.4-mini"
DEFAULT_MAX_OBJECTS = 10
SUPPORTED_IMAGE_TYPES = {
    ".gif": "image/gif",
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}

DETECTION_PROMPT = """\
Find up to {max_objects} of the most important distinct, movable objects visible
in this scene. Rank them by usefulness for an object-finding assistant. Give
each object a short, specific name. For each object, return one tight bounding
box as [left, top, right, bottom], with coordinates normalized from 0 to 1000
relative to the full image. The origin is the top-left corner. Include only
objects you can see; omit surfaces and the background. Return fewer objects if
needed.
"""


class DetectedObject(TypedDict):
    """One named object and its normalized bounding box."""

    n: str
    b: list[int]


class DetectionResult(TypedDict):
    """Structured result returned by :func:`detect_objects`."""

    objects: list[DetectedObject]


def _objects_schema(max_objects: int) -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "objects": {
                "type": "array",
                "maxItems": max_objects,
                "items": {
                    "type": "object",
                    "properties": {
                        "n": {"type": "string"},
                        "b": {
                            "type": "array",
                            "items": {
                                "type": "integer",
                                "minimum": 0,
                                "maximum": 1000,
                            },
                            "minItems": 4,
                            "maxItems": 4,
                        },
                    },
                    "required": ["n", "b"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["objects"],
        "additionalProperties": False,
    }


def _image_data_url(
    image: str | Path | bytes,
    *,
    media_type: str | None,
) -> str:
    if isinstance(image, bytes):
        if not image:
            raise ValueError("image bytes cannot be empty")
        resolved_media_type = media_type or "image/jpeg"
        image_bytes = image
    else:
        image_path = Path(image)
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")

        inferred_media_type = SUPPORTED_IMAGE_TYPES.get(image_path.suffix.lower())
        if media_type is None and inferred_media_type is None:
            supported = ", ".join(sorted(SUPPORTED_IMAGE_TYPES))
            raise ValueError(f"Unsupported image extension; expected one of: {supported}")

        resolved_media_type = media_type or inferred_media_type
        image_bytes = image_path.read_bytes()

    if resolved_media_type not in SUPPORTED_IMAGE_TYPES.values():
        raise ValueError(f"Unsupported image media type: {resolved_media_type}")

    encoded_image = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{resolved_media_type};base64,{encoded_image}"


def _validate_result(value: object, max_objects: int) -> DetectionResult:
    if not isinstance(value, dict) or set(value) != {"objects"}:
        raise ValueError("OpenAI returned an invalid detection result")

    objects = value["objects"]
    if not isinstance(objects, list) or len(objects) > max_objects:
        raise ValueError(f"OpenAI returned more than {max_objects} objects")

    validated: list[DetectedObject] = []
    for obj in objects:
        if not isinstance(obj, dict) or set(obj) != {"n", "b"}:
            raise ValueError(f"Invalid object from OpenAI: {obj!r}")

        name = obj["n"]
        box = obj["b"]
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"Invalid object name from OpenAI: {name!r}")
        if (
            not isinstance(box, list)
            or len(box) != 4
            or not all(type(coordinate) is int and 0 <= coordinate <= 1000 for coordinate in box)
            or not (box[0] < box[2] and box[1] < box[3])
        ):
            raise ValueError(f"Invalid bounding box from OpenAI: {box!r}")

        validated.append({"n": name.strip(), "b": box.copy()})

    return {"objects": validated}


def detect_objects(
    image: str | Path | bytes,
    *,
    client: OpenAI | None = None,
    model: str = DEFAULT_MODEL,
    max_objects: int = DEFAULT_MAX_OBJECTS,
    media_type: str | None = None,
    detail: Literal["auto", "low", "high", "original"] = "high",
) -> DetectionResult:
    """Detect movable objects in an image with normalized bounding boxes.

    ``image`` may be a filesystem path or encoded PNG/JPEG/WEBP/GIF bytes.
    Byte input defaults to JPEG; pass ``media_type`` when using another format.
    The client is optional so production code can use environment credentials
    while tests or callers with custom configuration can inject their own.
    """

    if not 1 <= max_objects <= DEFAULT_MAX_OBJECTS:
        raise ValueError(f"max_objects must be between 1 and {DEFAULT_MAX_OBJECTS}")
    if not model.strip():
        raise ValueError("model cannot be empty")

    openai_client = client or OpenAI()
    response = openai_client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": DETECTION_PROMPT.format(max_objects=max_objects),
                    },
                    {
                        "type": "input_image",
                        "image_url": _image_data_url(image, media_type=media_type),
                        "detail": detail,
                    },
                ],
            }
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "detected_objects",
                "schema": _objects_schema(max_objects),
                "strict": True,
            }
        },
    )

    if response.status != "completed" or not response.output_text:
        raise RuntimeError(f"OpenAI did not return a completed detection: {response.status}")

    try:
        result = json.loads(response.output_text)
    except json.JSONDecodeError as error:
        raise ValueError("OpenAI returned malformed JSON") from error

    return _validate_result(result, max_objects)


def write_detection_json(result: DetectionResult, output_path: str | Path) -> None:
    """Write a detection result as readable UTF-8 JSON."""

    Path(output_path).write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
