import json
import os
import tempfile
import unittest

from PIL import Image

from src.utils.format_converter import LabelExportWriter
from src.utils.label_importer import extract_class_names_from_text, find_image_path, import_labels, import_labels_with_report
from src.utils.label_validator import validate_result


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

    def test_yolo_edge_overflow_is_clamped_without_reversing_box(self):
        self._write("classes.txt", "car\n")
        self._write("image_a.txt", "0 0.891204 0.819133 0.217593 0.339672\n")

        records = import_labels(self.label_dir, self.image_dir, source_format="yolo")
        box = records[0][1].boxes[0]

        self.assertAlmostEqual(box.xmin, 0.7824075)
        self.assertEqual(box.xmax, 1.0)
        self.assertFalse(validate_result(records[0][1]))

    def test_coco_edge_overflow_is_clamped_without_reversing_box(self):
        coco = {
            "images": [{"id": 1, "file_name": "image_a.jpg", "width": 100, "height": 100}],
            "categories": [{"id": 1, "name": "car"}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [89.1204, 81.9133, 21.7593, 33.9672]}],
        }
        coco_path = self._write("annotations.json", json.dumps(coco))

        records = import_labels(coco_path, self.image_dir, source_format="coco")
        box = records[0][1].boxes[0]

        self.assertAlmostEqual(box.xmin, 0.891204)
        self.assertEqual(box.xmax, 1.0)
        self.assertFalse(validate_result(records[0][1]))

    def test_find_image_path_searches_nested_image_directories(self):
        nested_dir = os.path.join(self.image_dir, "nested", "leaf")
        os.makedirs(nested_dir)
        nested_image = os.path.join(nested_dir, "deep_sample.jpg")
        Image.new("RGB", (80, 60), "white").save(nested_image)

        resolved = find_image_path(self.image_dir, "deep_sample.jpg")

        self.assertEqual(os.path.abspath(resolved), os.path.abspath(nested_image))

    def test_nested_image_lookup_prevents_missing_image_validation_issue(self):
        nested_dir = os.path.join(self.image_dir, "nested", "leaf")
        os.makedirs(nested_dir)
        Image.new("RGB", (80, 60), "white").save(os.path.join(nested_dir, "deep_sample.jpg"))
        self._write("deep_sample.txt", "0 0.500000 0.500000 0.200000 0.200000\n")
        self._write("classes.txt", "car\n")

        records = import_labels(self.label_dir, self.image_dir, source_format="yolo")
        image_name, result = records[0]
        image_path = find_image_path(self.image_dir, image_name)

        self.assertFalse(validate_result(result, image_path))

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

    def test_auto_import_keeps_empty_yolo_label_without_image(self):
        self._write("missing_image.txt", "")

        batch = import_labels_with_report(self.label_dir, self.image_dir, source_format="auto")

        self.assertEqual(len(batch.records), 1)
        self.assertEqual(batch.records[0][0], "missing_image.jpg")
        self.assertEqual(len(batch.records[0][1].boxes), 0)
        self.assertEqual(batch.report["sources_discovered"], 1)
        self.assertEqual(batch.report["skipped_files"], [])

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

    def test_extract_class_names_from_prompt_yaml_names_block(self):
        prompt = """Detect only these classes.
names:
  0: person
  1: giraffe
  2: dining table
  3: cup
path: d:\\datasets\\yolo
"""

        self.assertEqual(
            extract_class_names_from_text(prompt),
            ["person", "giraffe", "dining table", "cup"],
        )

    def test_extract_class_names_from_inline_prompt_yaml_names_block(self):
        prompt = "names: 0: person 1: giraffe 2: dining table 3: cup path: D:\\dataset\\images"

        self.assertEqual(
            extract_class_names_from_text(prompt),
            ["person", "giraffe", "dining table", "cup"],
        )

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

    def test_explicit_coco_directory_discovers_json_and_segments(self):
        coco = {
            "images": [{"id": 1, "file_name": "image_a.jpg", "width": 100, "height": 100}],
            "categories": [{"id": 1, "name": "object"}],
            "annotations": [{
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [10, 20, 40, 50],
                "segmentation": [[10, 20, 50, 20, 50, 70, 10, 70]],
            }],
        }
        self._write("coco/labels.json", json.dumps(coco))

        batch = import_labels_with_report(
            os.path.join(self.label_dir, "coco"),
            self.image_dir,
            source_format="coco",
        )

        self.assertEqual(len(batch.records), 1)
        self.assertEqual(batch.report["formats"], {"coco": 1})
        self.assertEqual(len(batch.records[0][1].boxes), 1)
        self.assertEqual(len(batch.records[0][1].segments), 1)

    def test_generic_json_import_accepts_bbox_key(self):
        self._write(
            "custom.json",
            json.dumps({
                "items": [
                    {
                        "image": "sample.jpg",
                        "label": "car",
                        "bbox": [0.1, 0.2, 0.5, 0.6],
                    }
                ]
            }),
        )

        batch = import_labels_with_report(self.label_dir, self.image_dir, source_format="auto")

        self.assertEqual(batch.records[0][0], "sample.jpg")
        self.assertEqual(batch.records[0][1].boxes[0].label, "car")
        self.assertEqual(batch.records[0][1].boxes[0].xmin, 0.1)
        self.assertEqual(batch.records[0][1].boxes[0].xmax, 0.5)

    def test_custom_mapping_import_uses_mapping_spec_for_unrecognized_json(self):
        self._write(
            "vendor/custom_label.json",
            json.dumps({
                "asset": {"name": "mapped.jpg", "w": 200, "h": 100},
                "instances": [
                    {"categoryText": "lion", "rectPixels": [20, 10, 60, 30]},
                ],
            }),
        )
        spec = {
            "format": "json",
            "image_name_path": "$.asset.name",
            "image_width_path": "$.asset.w",
            "image_height_path": "$.asset.h",
            "objects_path": "$.instances[*]",
            "label_path": "@.categoryText",
            "bbox_path": "@.rectPixels",
            "bbox_format": "xywh",
            "bbox_unit": "pixel",
        }

        batch = import_labels_with_report(
            os.path.join(self.label_dir, "vendor"),
            self.image_dir,
            source_format="custom_mapping",
            custom_mapping_spec=json.dumps(spec),
        )

        self.assertEqual(batch.report["formats"], {"custom_mapping": 1})
        self.assertEqual(batch.records[0][0], "mapped.jpg")
        box = batch.records[0][1].boxes[0]
        self.assertEqual(box.label, "lion")
        self.assertAlmostEqual(box.xmin, 0.1)
        self.assertAlmostEqual(box.ymin, 0.1)
        self.assertAlmostEqual(box.xmax, 0.4)
        self.assertAlmostEqual(box.ymax, 0.4)

    def test_custom_mapping_tolerates_root_paths_for_object_fields(self):
        self._write(
            "vendor/custom_label.json",
            json.dumps({
                "images": [{"file_name": "mapped.jpg", "width": 200, "height": 100}],
                "annotations": [
                    {"category_name": "lion", "bbox": [20, 10, 60, 30]},
                ],
            }),
        )
        spec = {
            "format": "json",
            "image_name_path": "$.images[0].file_name",
            "image_width_path": "$.images[0].width",
            "image_height_path": "$.images[0].height",
            "objects_path": "$.annotations[*]",
            "label_path": "$.category_name",
            "bbox_path": "$.bbox",
            "bbox_format": "xywh",
            "bbox_unit": "pixel",
        }

        batch = import_labels_with_report(
            os.path.join(self.label_dir, "vendor"),
            self.image_dir,
            source_format="custom_mapping",
            custom_mapping_spec=json.dumps(spec),
        )

        self.assertEqual(len(batch.records), 1)
        self.assertEqual(batch.records[0][0], "mapped.jpg")
        box = batch.records[0][1].boxes[0]
        self.assertEqual(box.label, "lion")
        self.assertAlmostEqual(box.xmin, 0.1)
        self.assertAlmostEqual(box.ymin, 0.1)
        self.assertAlmostEqual(box.xmax, 0.4)
        self.assertAlmostEqual(box.ymax, 0.4)


if __name__ == "__main__":
    unittest.main()
