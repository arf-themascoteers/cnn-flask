import unittest
import pandas as pd

from nn.activation.relu import ReLU
from nn.fcnn import FCNN


class TestFCNN_MNIST(unittest.TestCase):
    def test_something(self):
        fcnn = FCNN(784, 10)
        fcnn.add_hidden_layer(6, ReLU())
        fcnn.add_hidden_layer(7, ReLU())
        train_data = pd.read_csv('../data/mnist/train.csv')
        test_data = pd.read_csv('../data/mnist/test.csv')

        y_train = train_data['label'].values
        X_train = train_data.drop(columns=['label']).values / 255
        X_test = test_data.values / 255
        print(X_train)


if __name__ == '__main__':
    unittest.main()
