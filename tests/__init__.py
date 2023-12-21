import sys
from unittest import TextTestRunner, defaultTestLoader


def run():
    test_suite = defaultTestLoader.discover(__name__)
    result = TextTestRunner(verbosity=2).run(test_suite)
    sys.exit(not result.wasSuccessful())
