import unittest
import geograph2


class TestSmoke(unittest.TestCase):
    def test_version(self):
        self.assertEqual(geograph2.__version__, "0.1.0")


if __name__ == "__main__":
    unittest.main()
