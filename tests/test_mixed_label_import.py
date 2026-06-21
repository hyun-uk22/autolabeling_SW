import json
import os
import tempfile
import unittest

from PIL import Image

from src.utils.format_converter import LabelExportWriter
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
        self.assertEqual(batch.report["class_list"], ["car", "dog"])
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

    def test_yolo_data_yaml_takes_priority_over_classes_txt(self):
        self._write("classes.txt", "wrong0\nwrong1\nwrong2\nwrong3\n")
        self._write(
            "data.yaml",
            """path: .
train: images/train
val: images/val
names:
  0: ant
  1: bear
  2: tiger
  3: Lion
""",
        )
        self._write("image_a.txt", "3 0.300000 0.300000 0.400000 0.400000\n")

        batch = import_labels_with_report(self.label_dir, self.image_dir, source_format="yolo")

        self.assertEqual(batch.records[0][1].boxes[0].label, "Lion")
        self.assertEqual(batch.report["class_list"], ["ant", "bear", "tiger", "Lion"])
        self.assertEqual(batch.report["processed_files"][0]["class_mapping"]["status"], "found")
        self.assertTrue(batch.report["processed_files"][0]["class_mapping"]["path"].endswith("data.yaml"))

    def test_nested_yolo_source_uses_mapping_from_input_root(self):
        self._write(
            "data.yaml",
            "names: [ant, bear, tiger, Lion]\n",
        )
        self._write("yolo/image_a.txt", "3 0.300000 0.300000 0.400000 0.400000\n")

        batch = import_labels_with_report(self.label_dir, self.image_dir, source_format="auto")

        self.assertEqual(batch.records[0][1].boxes[0].label, "Lion")
        self.assertEqual(batch.report["class_list"], ["ant", "bear", "tiger", "Lion"])
        self.assertTrue(batch.report["processed_files"][0]["class_mapping"]["path"].endswith("data.yaml"))

    def test_yolo_without_mapping_reports_missing_class_mapping(self):
        self._write("image_a.txt", "3 0.300000 0.300000 0.400000 0.400000\n")

        batch = import_labels_with_report(self.label_dir, self.image_dir, source_format="yolo")

        self.assertEqual(batch.records[0][1].boxes[0].label, "3")
        self.assertEqual(batch.report["processed_files"][0]["class_mapping"]["status"], "missing")
        self.assertIn("classes.txt", "\n".join(batch.report["processed_files"][0]["class_mapping"]["searched"]))

    def test_unmapped_yolo_numeric_label_uses_matching_named_label_from_other_format(self):
        self._write("yolo/image_a.txt", "3 0.300000 0.300000 0.400000 0.400000\n")
        self._write(
            "voc/image_a.xml",
            """<?xml version="1.0" encoding="utf-8"?>
<annotation>
  <filename>image_a.jpg</filename>
  <size><width>100</width><height>100</height><depth>3</depth></size>
  <object><name>Lion</name><bndbox><xmin>10</xmin><ymin>10</ymin><xmax>50</xmax><ymax>50</ymax></bndbox></object>
</annotation>
""",
        )

        batch = import_labels_with_report(self.label_dir, self.image_dir, source_format="auto")

        self.assertEqual(len(batch.records), 1)
        self.assertEqual([box.label for box in batch.records[0][1].boxes], ["Lion"])
        self.assertEqual(batch.report["class_list"], ["Lion"])
        self.assertEqual(batch.report["merge"]["label_normalizations"][0]["numeric_label"], "3")
        self.assertEqual(batch.report["merge"]["label_normalizations"][0]["canonical_label"], "Lion")

    def test_inferred_yolo_numeric_label_mapping_applies_to_yolo_only_records(self):
        self._write("yolo/image_a.txt", "3 0.300000 0.300000 0.400000 0.400000\n")
        self._write("yolo/image_b.txt", "3 0.500000 0.500000 0.200000 0.200000\n")
        self._write(
            "voc/image_a.xml",
            """<?xml version="1.0" encoding="utf-8"?>
<annotation>
  <filename>image_a.jpg</filename>
  <size><width>100</width><height>100</height><depth>3</depth></size>
  <object><name>Lion</name><bndbox><xmin>10</xmin><ymin>10</ymin><xmax>50</xmax><ymax>50</ymax></bndbox></object>
</annotation>
""",
        )

        batch = import_labels_with_report(self.label_dir, self.image_dir, source_format="auto")

        labels_by_image = {
            image_name: [box.label for box in result.boxes]
            for image_name, result in batch.records
        }
        self.assertEqual(labels_by_image["image_a.jpg"], ["Lion"])
        self.assertEqual(labels_by_image["image_b.jpg"], ["Lion"])
        self.assertEqual(batch.report["class_list"], ["Lion"])
        self.assertEqual(
            batch.report["merge"]["global_label_normalizations"][0]["numeric_label"],
            "3",
        )

    def test_explicit_yolo_yaml_classes_path_is_supported(self):
        yaml_path = self._write(
            "custom_names.yaml",
            'names: {0: ant, 1: bear, 2: tiger, 3: Lion}\n',
        )
        self._write("classes.txt", "wrong0\nwrong1\nwrong2\nwrong3\n")
        self._write("image_a.txt", "3 0.300000 0.300000 0.400000 0.400000\n")

        records = import_labels(
            self.label_dir,
            self.image_dir,
            source_format="yolo",
            classes_path=yaml_path,
        )

        self.assertEqual(records[0][1].boxes[0].label, "Lion")

    def test_mixed_input_preserves_yolo_class_ids_on_yolo_export(self):
        Image.new("RGB", (100, 100), "white").save(os.path.join(self.image_dir, "image_c.jpg"))
        self._write("yolo/classes.txt", "ant\nbear\ntiger\nLion\n")
        self._write("yolo/image_a.txt", "3 0.300000 0.300000 0.400000 0.400000\n")
        self._write(
            "voc/image_b.xml",
            """<?xml version="1.0" encoding="utf-8"?>
<annotation>
  <filename>image_b.jpg</filename>
  <size><width>100</width><height>100</height><depth>3</depth></size>
  <object><name>Lion</name><bndbox><xmin>20</xmin><ymin>20</ymin><xmax>60</xmax><ymax>60</ymax></bndbox></object>
</annotation>
""",
        )
        coco = {
            "images": [{"id": 1, "file_name": "image_c.jpg", "width": 100, "height": 100}],
            "categories": [{"id": 7, "name": "Lion"}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 7, "bbox": [10, 10, 40, 40]}],
        }
        self._write("coco/annotations.json", json.dumps(coco))
        output_dir = os.path.join(self.root, "out")

        batch = import_labels_with_report(self.label_dir, self.image_dir, source_format="auto")
        writer = LabelExportWriter(
            output_dir,
            formats=["yolo"],
            initial_class_list=batch.report["class_list"],
        )
        for image_name, result in batch.records:
            writer.save(result, os.path.join(self.image_dir, image_name))
        writer.finalize()

        with open(os.path.join(output_dir, "classes.txt"), encoding="utf-8") as handle:
            exported_classes = handle.read().splitlines()
        with open(os.path.join(output_dir, "data.yaml"), encoding="utf-8") as handle:
            exported_yaml = handle.read()
        self.assertEqual(batch.report["class_list"], ["ant", "bear", "tiger", "Lion"])
        self.assertEqual(exported_classes, ["ant", "bear", "tiger", "Lion"])
        self.assertIn('  3: "Lion"', exported_yaml)
        for name in ("image_a.txt", "image_b.txt", "image_c.txt"):
            with self.subTest(name=name):
                with open(os.path.join(output_dir, name), encoding="utf-8") as handle:
                    line = handle.read().strip()
                self.assertTrue(line.startswith("3 "), line)


if __name__ == "__main__":
    unittest.main()
