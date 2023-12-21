import sys
import unittest
from unittest.mock import patch

from rleplus.env.utils import solve_energyplus_install_path


class TestUtils(unittest.TestCase):
    def setUp(self):
        # make sure we don't have any E+ installation in sys.path
        sys.path = [p for p in sys.path if "EnergyPlus" not in p]

    @patch.dict("os.environ", {"ENERGYPLUS_HOME": "/tmp/energyplus"}, clear=True)
    def test_import_energyplus_home(self, *_):
        path = solve_energyplus_install_path()
        self.assertEqual(path, "/tmp/energyplus")

    @patch.dict("os.environ", {"ENERGYPLUS_VERSION": "23-2-0"}, clear=True)
    def test_import_energyplus_version(self):
        path = solve_energyplus_install_path()
        self.assertEqual(path, "/usr/local/EnergyPlus-23-2-0")
