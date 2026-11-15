import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class PageTest(unittest.TestCase):
    def test_committed_page_matches_results(self):
        spec = importlib.util.spec_from_file_location("build_page", ROOT / "tools" / "build_page.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual((ROOT / "docs" / "index.html").read_text(), module.render())


if __name__ == "__main__":
    unittest.main()
