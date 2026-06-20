import tempfile
import unittest
from pathlib import Path

from src.core.workspace import (
    WORKSPACE_DEFAULTS,
    load_workspace,
    relative_to_workspace,
    resolve_workspace_path,
    save_workspace,
)


class WorkspaceTests(unittest.TestCase):
    def test_workspace_is_saved_and_default_directories_are_created(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "workspace"
            config = Path(directory) / "config" / "workspace.json"

            saved = save_workspace(root, config, create_layout=True)

            self.assertEqual(load_workspace(config), saved)
            self.assertTrue((saved / WORKSPACE_DEFAULTS["images"]).is_dir())
            self.assertTrue((saved / WORKSPACE_DEFAULTS["labels"]).is_dir())
            self.assertTrue((saved / WORKSPACE_DEFAULTS["visualized"]).is_dir())
            plugin_config = saved / WORKSPACE_DEFAULTS["plugin_config"]
            self.assertTrue(plugin_config.is_file())
            self.assertEqual(plugin_config.read_text(encoding="utf-8").strip(), '{\n  "plugins": []\n}')

    def test_relative_paths_are_resolved_under_workspace(self):
        with tempfile.TemporaryDirectory() as directory:
            expected = str((Path(directory) / "data" / "raw").resolve())
            resolved = resolve_workspace_path(directory, "data/raw")

            self.assertEqual(resolved, expected)
            self.assertEqual(relative_to_workspace(directory, resolved), "data/raw")


if __name__ == "__main__":
    unittest.main()
