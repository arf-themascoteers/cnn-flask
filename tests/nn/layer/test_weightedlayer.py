import unittest

from nn.activation.relu import ReLU
from nn.layer.input_layer import InputLayer
from nn.layer.loss_layer import LossLayer
from nn.layer.weighted_layer import WeightedLayer


class TestWeightedLayer(unittest.TestCase):
    def test_get_output(self):
        input_layer = InputLayer(4)
        layer1 = WeightedLayer(3,ReLU(),input_layer,None)
        output = layer1.get_output([1,2,3,4])
        print(output)


if __name__ == '__main__':
    unittest.main()
