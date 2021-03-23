import unittest

from nn.activation.activation_factory import ActivationFactory
from nn.activation.softmax import SoftMax


class TestActivationFactory(unittest.TestCase):
    def test_evaluate(self):
        relu = ActivationFactory.createActivation("relu")
        self.assertIsNotNone(relu)

if __name__ == '__main__':
    unittest.main()
