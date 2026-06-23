import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from src.plugins.builtin import OCRPlugin
from src.plugins.builtin import _resolve_device
from src.plugins.registry import load_generation_plugins


class PluginRegistryTests(unittest.TestCase):
    def test_generation_plugins_are_loaded_without_config(self):
        plugins = load_generation_plugins()

        self.assertEqual(
            [plugin.plugin_name for plugin in plugins],
            ["classification", "grounding_dino", "grounded_sam2", "pose", "ocr", "tracking"],
        )
        self.assertTrue(plugins[1].supports("object_detection"))
        self.assertEqual(plugins[1].config["model"], "IDEA-Research/grounding-dino-base")
        self.assertFalse(plugins[1].supports("segmentation"))
        self.assertTrue(plugins[2].supports("segmentation"))
        self.assertFalse(plugins[2].supports("object_detection"))
        self.assertEqual(plugins[2].config["grounding_model"], "IDEA-Research/grounding-dino-base")
        self.assertEqual(plugins[2].config["sam_backend"], "ultralytics_sam2")
        self.assertEqual(plugins[2].config["sam_model"], "sam2_b.pt")
        self.assertEqual(plugins[2].config["device"], "auto")
        self.assertTrue(plugins[3].supports("pose_estimation"))
        self.assertTrue(plugins[4].supports("ocr"))
        self.assertEqual(plugins[4].config["backend"], "paddleocr")
        self.assertEqual(plugins[4].config["lang"], "korean")
        self.assertEqual(plugins[4].config["gpu"], "auto")
        self.assertTrue(plugins[5].supports("tracking"))

    def test_empty_generation_plugin_config_keeps_required_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "plugins.json"
            config.write_text('{"plugins": []}', encoding="utf-8")

            plugins = load_generation_plugins(str(config))

        self.assertEqual(
            [plugin.plugin_name for plugin in plugins],
            ["classification", "grounding_dino", "grounded_sam2", "pose", "ocr", "tracking"],
        )

    def test_generation_plugin_config_overrides_model_settings_without_disabling_chain(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "plugins.json"
            config.write_text(
                json.dumps({
                    "plugins": [{
                        "name": "grounding_dino",
                        "tasks": ["object_detection"],
                        "weight": 2.0,
                        "config": {"model": "custom-dino", "device": "cpu"},
                    }]
                }),
                encoding="utf-8",
            )

            plugins = load_generation_plugins(str(config))

        names = [plugin.plugin_name for plugin in plugins]
        grounding_dino = plugins[names.index("grounding_dino")]

        self.assertIn("grounded_sam2", names)
        self.assertEqual(grounding_dino.config["model"], "custom-dino")
        self.assertEqual(grounding_dino.config["weight"], 2.0)
        self.assertTrue(grounding_dino.supports("object_detection"))
        self.assertFalse(grounding_dino.supports("segmentation"))

    def test_candidate_labels_are_injected_into_grounding_dino(self):
        plugins = load_generation_plugins(candidate_labels=["person", "giraffe", "cup"])
        names = [plugin.plugin_name for plugin in plugins]
        grounding_dino = plugins[names.index("grounding_dino")]
        grounded_sam2 = plugins[names.index("grounded_sam2")]

        self.assertEqual(grounding_dino.config["labels"], ["person", "giraffe", "cup"])
        self.assertEqual(grounded_sam2.config["labels"], ["person", "giraffe", "cup"])

    def test_sam_can_be_configured_for_official_sam3_backend(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "plugins.json"
            config.write_text(
                json.dumps({
                    "plugins": [{
                        "name": "sam",
                        "config": {
                            "backend": "official_sam3",
                            "model": "facebook/sam3",
                            "threshold": 0.5,
                            "mask_threshold": 0.5,
                        },
                    }]
                }),
                encoding="utf-8",
            )

            plugins = load_generation_plugins(str(config))

        names = [plugin.plugin_name for plugin in plugins]
        sam = plugins[names.index("sam")]
        self.assertEqual(sam.config["backend"], "official_sam3")
        self.assertEqual(sam.config["model"], "facebook/sam3")

    def test_auto_device_prefers_cuda_and_falls_back_to_cpu(self):
        fake_cuda = types.SimpleNamespace(is_available=lambda: True)
        fake_mps = types.SimpleNamespace(is_available=lambda: False)
        fake_torch = types.SimpleNamespace(
            cuda=fake_cuda,
            backends=types.SimpleNamespace(mps=fake_mps),
        )

        with patch.dict(sys.modules, {"torch": fake_torch}):
            self.assertEqual(_resolve_device("auto"), "cuda:0")

        fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
        with patch.dict(sys.modules, {"torch": fake_torch}):
            self.assertEqual(_resolve_device("cuda:0"), "cpu")

    def test_ocr_plugin_parses_paddleocr_result_shapes(self):
        plugin = OCRPlugin({"backend": "paddleocr"})
        raw = [[[[[0, 0], [20, 0], [20, 10], [0, 10]], ("hello", 0.9)]]]

        items = list(plugin._iter_paddleocr_items(raw))

        self.assertEqual(items[0][1], "hello")
        self.assertEqual(items[0][2], 0.9)


if __name__ == "__main__":
    unittest.main()
