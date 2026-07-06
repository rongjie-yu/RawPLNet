import unittest
from pathlib import Path


class RequirementsTest(unittest.TestCase):
    def test_tensorboard_is_declared_for_training_summary_writer(self):
        root = Path(__file__).resolve().parents[1]
        requirements = {
            line.strip().split("==", 1)[0].split(">=", 1)[0].lower()
            for line in (root / "requirement.txt").read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        }

        self.assertIn("tensorboard", requirements)


if __name__ == "__main__":
    unittest.main()
