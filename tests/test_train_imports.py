import subprocess
import sys
import textwrap
import unittest
from pathlib import Path


class TrainImportTest(unittest.TestCase):
    def test_import_train_module_does_not_require_tensorboard(self):
        root = Path(__file__).resolve().parents[1]
        code = textwrap.dedent(
            """
            import importlib.abc
            import sys

            class BlockTensorboard(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname == "tensorboard" or fullname.startswith("tensorboard."):
                        raise ModuleNotFoundError("No module named 'tensorboard'")
                    return None

            sys.meta_path.insert(0, BlockTensorboard())
            import hawp.fsl.train
            """
        )

        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(root),
            text=True,
            capture_output=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
