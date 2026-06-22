import json
import tempfile
import unittest
from pathlib import Path

from src.plugins.registry import load_generation_plugins


class PluginRegistryTests(unittest.TestCase):
    def test_generation_plugins_are_loaded_without_config(self):
        plugins = load_generation_plugins()

        self.assertEqual(
            [plugin.plugin_name for plugin in plugins],
            ["classification", "grounding_dino", "sam", "pose", "ocr", "tracking"],
        )
        self.assertTrue(plugins[1].supports("object_detection"))
        self.assertTrue(plugins[2].supports("segmentation"))
        self.assertTrue(plugins[2].supports("object_detection"))
        self.assertEqual(plugins[2].config["backend"], "ultralytics_sam2")
        self.assertEqual(plugins[2].config["model"], "sam2_b.pt")
        self.assertTrue(plugins[3].supports("pose_estimation"))
        self.assertTrue(plugins[4].supports("ocr"))
        self.assertTrue(plugins[5].supports("tracking"))

    def test_empty_generation_plugin_config_keeps_required_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "plugins.json"
            config.write_text('{"plugins": []}', encoding="utf-8")

            plugins = load_generation_plugins(str(config))

        self.assertEqual(
            [plugin.plugin_name for plugin in plugins],
            ["classification", "grounding_dino", "sam", "pose", "ocr", "tracking"],
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

        self.assertIn("sam", names)
        self.assertEqual(grounding_dino.config["model"], "custom-dino")
        self.assertEqual(grounding_dino.config["weight"], 2.0)
        self.assertTrue(grounding_dino.supports("segmentation"))

    def test_candidate_labels_are_injected_into_grounding_dino(self):
        plugins = load_generation_plugins(candidate_labels=["person", "giraffe", "cup"])
        names = [plugin.plugin_name for plugin in plugins]
        grounding_dino = plugins[names.index("grounding_dino")]
        sam = plugins[names.index("sam")]

        self.assertEqual(grounding_dino.config["labels"], ["person", "giraffe", "cup"])
        self.assertEqual(sam.config["labels"], ["person", "giraffe", "cup"])

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


if __name__ == "__main__":
    unittest.main()
