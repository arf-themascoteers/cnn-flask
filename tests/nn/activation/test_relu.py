import unittest

from nn.activation.relu import ReLU


class TestReLU(unittest.TestCase):
    def test_evaluate(self):
        activation = ReLU()
        self.assertEqual(activation.forward(activation.forward([10, -20, 0])), [10, 0, 0])



if __name__ == '__main__':
    unittest.main()
