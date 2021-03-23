import unittest

from nn.activation.relu import ReLU
from nn.fcnn import FCNN


class TestFCNN(unittest.TestCase):
    def test_something(self):
        fcnn = FCNN(5, 2)
        fcnn.print()
        fcnn.print_backward()

        fcnn.add_hidden_layer(3, ReLU())
        fcnn.print()
        fcnn.print_backward()

        fcnn.add_hidden_layer(4, ReLU())
        fcnn.print()
        fcnn.print_backward()


if __name__ == '__main__':
    unittest.main()
