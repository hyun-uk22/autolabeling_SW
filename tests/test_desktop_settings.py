import os
import tempfile
import unittest
from pathlib import Path

from src.ui.settings import ENV_FIELDS, load_user_environment, read_user_settings, save_user_settings


class DesktopSettingsTests(unittest.TestCase):
    def setUp(self):
        self.original = {key: os.environ.get(key) for key in ENV_FIELDS}

    def tearDown(self):
        for key, value in self.original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_settings_round_trip_and_environment_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            values = {key: "" for key in ENV_FIELDS}
            values.update({
                "AWS_REGION": "ap-northeast-2",
                "OPENAI_API_KEY": "key with spaces and 'quote'",
                "LOW_MODEL": "gpt-4o-mini",
                "HIGH_MODEL": "gpt-4o",
            })

            saved_path = save_user_settings(values, path)
            os.environ["OPENAI_API_KEY"] = "stale"
            load_user_environment(path)
            loaded = read_user_settings(path)

            self.assertEqual(saved_path, path)
            self.assertEqual(loaded["AWS_REGION"], "ap-northeast-2")
            self.assertEqual(loaded["OPENAI_API_KEY"], "key with spaces and 'quote'")
            self.assertEqual(os.environ["OPENAI_API_KEY"], "key with spaces and 'quote'")
            self.assertNotIn("ANTHROPIC_API_KEY", os.environ)


if __name__ == "__main__":
    unittest.main()

