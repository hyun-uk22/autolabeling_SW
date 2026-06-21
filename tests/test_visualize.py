import os
import tempfile
import unittest

from PIL import Image

from src.core.models import BoundingBox, DetectionResult
from src.utils.visualize import normalized_box_to_pixels, visualize_boxes


class VisualizeTests(unittest.TestCase):
    def test_normalized_box_to_pixels_uses_image_width_and_height(self):
        box = BoundingBox(
            label="car",
            xmin=0.10,
            ymin=0.25,
            xmax=0.60,
            ymax=0.75,
        )

        self.assertEqual(normalized_box_to_pixels(box, 100, 80), (10, 20, 60, 60))

    def test_normalized_box_to_pixels_clamps_and_orders_coordinates(self):
        box = BoundingBox(
            label="car",
            xmin=1.20,
            ymin=-0.25,
            xmax=0.50,
            ymax=1.50,
        )

        self.assertEqual(normalized_box_to_pixels(box, 100, 80), (50, 0, 100, 80))

    def test_visualize_boxes_writes_overlay_with_expected_dimensions(self):
        with tempfile.TemporaryDirectory() as directory:
            image_path = os.path.join(directory, "sample.jpg")
            output_dir = os.path.join(directory, "visualized")
            Image.new("RGB", (100, 80), "white").save(image_path)
            result = DetectionResult(
                task_type="object_detection",
                boxes=[BoundingBox(label="car", xmin=0.10, ymin=0.25, xmax=0.60, ymax=0.75)],
                source_model="test",
                uncertainty_score=0.1,
            )

            output_path = visualize_boxes(image_path, result, output_dir)

            self.assertTrue(os.path.exists(output_path))
            with Image.open(output_path) as rendered:
                self.assertEqual(rendered.size, (100, 80))


if __name__ == "__main__":
    unittest.main()
