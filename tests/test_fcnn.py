import unittest

from nn.fcnn import FCNN


class TestFCNN(unittest.TestCase):
    def test_something(self):
        fcnn = FCNN(3,2)
        print(fcnn.layers)
        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()