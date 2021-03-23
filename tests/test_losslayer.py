import unittest

from nn.loss_layer import LossLayer

class TestLossLayer(unittest.TestCase):
    def test_get_output(self):
        loss_layer = LossLayer(None)
        loss_layer.set_expected_output([1,0,0])
        output = loss_layer.get_output([0.7,0.1,0.2])
        print(output)


if __name__ == '__main__':
    unittest.main()
