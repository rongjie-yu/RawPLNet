import unittest

from hawp.fsl import adapt_raw, train
from hawp.fsl.utils import reached_debug_limit


class DebugLimitTest(unittest.TestCase):
    def test_none_limit_never_stops(self):
        self.assertFalse(reached_debug_limit(None, 100))

    def test_positive_limit_stops_after_limit_steps(self):
        self.assertFalse(reached_debug_limit(2, 1))
        self.assertTrue(reached_debug_limit(2, 2))

    def test_train_and_adapt_use_shared_debug_limit_helper(self):
        self.assertIs(train.reached_debug_limit, reached_debug_limit)
        self.assertIs(adapt_raw.reached_debug_limit, reached_debug_limit)


if __name__ == "__main__":
    unittest.main()
