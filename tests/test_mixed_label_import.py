import json
import os
import tempfile
import unittest

from PIL import Image

from src.utils.label_importer import import_labels, import_labels_with_report


class MixedLabelImportTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = self.temp_dir.name
        self.label_dir = os.path.join(self.root, "labels")
        self.image_dir = os.path.join(self.root, "images")
        os.makedirs(self.label_dir)
        os.makedirs(self.image_dir)
        for name in ("image_a.jpg", "image_b.jpg"):
            Image.new("RGB", (100, 100), "white").save(os.path.join(self.image_dir, name))

    def tearDown(self):
        self.temp_dir.cleanup()

    def _write(self, relative_path, content):
        path = os.path.join(self.label_dir, relative_path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path

    def test_auto_directory_imports_and_merges_standard_formats(self):
        self._write("yolo/classes.txt", "car\ndog\n")
        self._write("yolo/image_a.txt", "0 0.300000 0.300000 0.400000 0.400000\n")
        self._write(
            "voc/image_b.xml",
            """<?xml version="1.0" encoding="utf-8"?>
<annotation>
  <filename>image_b.jpg</filename>
  <size><width>100</width><height>100</height><depth>3</depth></size>
  <object><name>dog</name><bndbox><xmin>20</xmin><ymin>20</ymin><xmax>60</xmax><ymax>60</ymax></bndbox></object>
</annotation>
""",
        )
        coco = {
            "images": [
                {"id": 1, "file_name": "image_a.jpg", "width": 100, "height": 100},
                {"id": 2, "file_name": "image_b.jpg", "width": 100, "height": 100},
            ],
            "categories": [{"id": 1, "name": "car"}, {"id": 2, "name": "dog"}],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [11, 10, 40, 40]},
                {"id": 2, "image_id": 2, "category_id": 2, "bbox": [20, 20, 40, 40]},
            ],
        }
        self._write("coco/annotations.json", json.dumps(coco))
        self._write("unknown/custom.json", json.dumps({"items": [{"shape": [1, 2, 3]}]}))

        batch = import_labels_with_report(self.label_dir, self.image_dir, source_format="auto")

        self.assertEqual([name for name, _ in batch.records], ["image_a.jpg", "image_b.jpg"])
        self.assertEqual([len(result.boxes) for _, result in batch.records], [1, 1])
        self.assertAlmostEqual(batch.records[0][1].boxes[0].xmin, 0.11)
        self.assertEqual(batch.report["formats"], {"coco": 1, "pascal_voc": 1, "yolo": 1})
        self.assertEqual(batch.report["records_before_merge"], 4)
        self.assertEqual(batch.report["records_after_merge"], 2)
        self.assertEqual(batch.report["merge"]["duplicates_removed"], 2)
        self.assertEqual(len(batch.report["skipped_files"]), 1)
        self.assertEqual(batch.report["skipped_files"][0]["reason"], "unrecognized_json_schema")

    def test_explicit_yolo_directory_keeps_existing_behavior(self):
        self._write("classes.txt", "car\n")
        self._write("image_a.txt", "0 0.300000 0.300000 0.400000 0.400000\n")
        self._write("image_b.txt", "0 0.500000 0.500000 0.200000 0.200000\n")

        records = import_labels(self.label_dir, self.image_dir, source_format="yolo")

        self.assertEqual(len(records), 2)
        self.assertEqual([result.boxes[0].label for _, result in records], ["car", "car"])


if __name__ == "__main__":
    unittest.main()
