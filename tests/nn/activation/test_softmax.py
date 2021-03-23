import unittest

from nn.activation.softmax import SoftMax


class TestSoftMax(unittest.TestCase):
    def test_evaluate(self):
        activation = SoftMax()
        result = activation.forward([3, 4, -2])
        correct = [0.26845495065245, 0.72973621411842, 0.0018088352291383]
        for i in range(3):
            self.assertAlmostEquals(result[i],correct[i])


if __name__ == '__main__':
    unittest.main()
