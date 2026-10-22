import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.object_detection import detect_objects, write_detection_json


class FakeResponses:
    def __init__(self, response):
        self.response = response
        self.request = None

    def create(self, **kwargs):
        self.request = kwargs
        return self.response


class InitialDetectionTests(unittest.TestCase):
    def test_detect_objects_returns_validated_pipeline_result(self):
        response = SimpleNamespace(
            status="completed",
            output_text=json.dumps(
                {"objects": [{"n": " keys ", "b": [100, 200, 300, 400]}]}
            ),
        )
        responses = FakeResponses(response)
        client = SimpleNamespace(responses=responses)

        result = detect_objects(b"jpeg data", client=client, max_objects=3)

        self.assertEqual(result, {"objects": [{"n": "keys", "b": [100, 200, 300, 400]}]})
        self.assertEqual(responses.request["model"], "gpt-5.4-mini")
        self.assertIn("data:image/jpeg;base64,", responses.request["input"][0]["content"][1]["image_url"])
        self.assertEqual(
            responses.request["text"]["format"]["schema"]["properties"]["objects"]["maxItems"],
            3,
        )

    def test_detect_objects_rejects_inverted_box(self):
        response = SimpleNamespace(
            status="completed",
            output_text='{"objects":[{"n":"keys","b":[300,200,100,400]}]}',
        )
        client = SimpleNamespace(responses=FakeResponses(response))

        with self.assertRaisesRegex(ValueError, "Invalid bounding box"):
            detect_objects(b"jpeg data", client=client)

    def test_write_detection_json(self):
        result = {"objects": [{"n": "mouse", "b": [1, 2, 3, 4]}]}
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "result.json"
            write_detection_json(result, output_path)

            self.assertEqual(json.loads(output_path.read_text()), result)


if __name__ == "__main__":
    unittest.main()
