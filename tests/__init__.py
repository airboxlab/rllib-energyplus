import sys
from unittest import defaultTestLoader, TextTestRunner

test_suite = defaultTestLoader.discover("tests")
result = TextTestRunner(verbosity=2).run(test_suite)

if exit:
    sys.exit(not result.wasSuccessful())
