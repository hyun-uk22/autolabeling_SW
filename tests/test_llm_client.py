import unittest

from src.core.llm_client import extract_json, llm_max_tokens


class LLMClientTests(unittest.TestCase):
    def test_extract_json_accepts_markdown_fenced_json(self):
        payload = extract_json(
            """```json
{
  "segments": [
    {
      "label": "person",
      "polygon": [{"x": 0.38, "y": 0.2}, {"x": 0.5, "y": 0.2}, {"x": 0.5, "y": 0.4}],
      "confidence": 0.9
    }
  ]
}
```""",
            strict=True,
        )

        self.assertEqual(payload["segments"][0]["label"], "person")

    def test_extract_json_strict_raises_on_truncated_response(self):
        with self.assertRaises(ValueError):
            extract_json('```json\n{"segments": [{"label": "person", "polygon": [{"x": 0.38, "y":', strict=True)

    def test_segmentation_uses_larger_default_token_budget(self):
        self.assertGreaterEqual(llm_max_tokens("segmentation"), 4096)
        self.assertGreaterEqual(llm_max_tokens("object_detection"), 2048)


if __name__ == "__main__":
    unittest.main()
