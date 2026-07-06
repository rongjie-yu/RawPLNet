import unittest

import hawp.base.csrc as csrc


class CsrcExtensionTest(unittest.TestCase):
    def test_require_c_reports_original_extension_load_error(self):
        original_error = csrc._C_LOAD_ERROR
        try:
            csrc._C_LOAD_ERROR = RuntimeError("nvcc missing")

            with self.assertRaisesRegex(RuntimeError, "HAWP C/CUDA extension is not available"):
                csrc.require_C()
        finally:
            csrc._C_LOAD_ERROR = original_error


if __name__ == "__main__":
    unittest.main()
