import unittest

from nn.layer.loss_layer import LossLayer
from nn.layer.output_layer import OutputLayer

class TestOutputLayer(unittest.TestCase):
    def test_get_output(self):
        output_layer = OutputLayer(3, None, None)
        output = output_layer.get_output([4.8, 1.21, 2.385])
        print(output)


if __name__ == '__main__':
    unittest.main()
