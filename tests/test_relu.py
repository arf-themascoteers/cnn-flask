import unittest

from nn.relu import ReLU


class TestReLU(unittest.TestCase):
    def test_evaluate(self):
        activation = ReLU()
        self.assertEqual(activation.evaluate(10),10)
        self.assertEqual(activation.evaluate(-10),0)


if __name__ == '__main__':
    unittest.main()
