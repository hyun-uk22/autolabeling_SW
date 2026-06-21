import unittest

from pydantic import ValidationError

from src.workflow.models import OperationPlan


class WorkflowModelTests(unittest.TestCase):
    def test_generation_mode_defaults_to_vlm_plugin(self):
        operation = OperationPlan(action="generate")

        self.assertEqual(operation.generation_mode, "vlm_plugin")

    def test_generation_mode_accepts_specialist_only(self):
        operation = OperationPlan(action="generate", generation_mode="specialist_only")

        self.assertEqual(operation.generation_mode, "specialist_only")

    def test_generation_mode_rejects_unknown_value(self):
        with self.assertRaises(ValidationError):
            OperationPlan(action="generate", generation_mode="unknown")


if __name__ == "__main__":
    unittest.main()
