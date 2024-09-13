import random
import unittest

import torch
import torch.nn as nn

from deployment.observer import ObserverBase, insert_observers

from test_models import ToyFPConv


class TestObserver(unittest.TestCase):
    """Test proper observer insertion and tracing"""

    def test_observer_insertion(self):
        """Test the insertion of observers after a random number
           of target layers
        """
        n_conv = random.randint(1, 50)
        nn_ut = ToyFPConv(n=n_conv)
        new_nn = insert_observers(nn_ut, target_layers=(nn.Conv2d))
        n_observers = len(list(filter(
            lambda x: (isinstance(x, ObserverBase)),
            new_nn.modules()
            )))
        err_msg = f'{n_conv} observers are expected instead of {n_observers}'
        self.assertEqual(n_observers, n_conv, err_msg)

    def test_observer_neutrality(self):
        """Test that the output of network with/without observers is the same
        """
        n_conv = random.randint(1, 50)
        nn_ut = ToyFPConv(n=n_conv)
        dummy_inp = torch.rand((2,) + nn_ut.input_shape)
        nn_ut_out = nn_ut(dummy_inp)
        new_nn = insert_observers(nn_ut, target_layers=(nn.Conv2d))
        new_nn_out = new_nn(dummy_inp)
        self.assertTrue(torch.all(nn_ut_out == new_nn_out),
                        'Outputs are different')

    def test_observer_output(self):
        """Test that the observer measure is the correct one.
        The check is performed by comparing the observer output when the net
        input is an all-zeros tensor. In this case we expect thath each observer
        output is equal to the sum of the bias of previus convolutions.
        """
        nn_ut = ToyFPConv(n=1)
        dummy_inp = torch.zeros((2,) + nn_ut.input_shape)
        new_nn = insert_observers(nn_ut, target_layers=(nn.Conv2d))
        new_nn(dummy_inp)
        observer_output = new_nn.net.conv1_observer.max_val
        conv_bias = new_nn.net.conv1.bias
        self.assertTrue(torch.all(observer_output == conv_bias),
                        'Observed values are incorrect')
