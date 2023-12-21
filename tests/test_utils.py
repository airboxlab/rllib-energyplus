import sys
import unittest
from unittest.mock import patch

from rleplus.env.utils import try_import_energyplus_api


class TestUtils(unittest.TestCase):
    def setUp(self):
        # make sure we don't have any E+ installation in sys.path
        sys.path = [p for p in sys.path if "EnergyPlus" not in p]

    @patch.dict("os.environ", {"ENERGYPLUS_HOME": "/tmp/energyplus"}, clear=True)
    @patch("os.path.exists", return_value=True)
    def test_import_energyplus_home(self, *_):
        try_import_energyplus_api(do_import=False)
        self.assertEqual(sys.path[-1], "/tmp/energyplus")

    @patch.dict("os.environ", {"ENERGYPLUS_VERSION": "23-2-0"}, clear=True)
    def test_import_energyplus_version(self):
        try_import_energyplus_api()
        self.assertEqual(sys.path[-1], "/usr/local/EnergyPlus-23-2-0")
