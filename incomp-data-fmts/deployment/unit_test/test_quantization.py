import unittest

import torch

from deployment.observer import insert_observers
from deployment.quantization import build_qgraph
from models import quant_module as qm

from test_models import ToyQConv


class TestQuantization(unittest.TestCase):
    """Test proper execution of quantization operations"""

    def test_harden_weights(self):
        """Test harden-weights quantization phase i.e., the phase when
        fake-quant weights and biases are transformed in integer weights and
        biases
        """
        nn_ut = ToyQConv(conv_func=qm.QuantMultiPrecActivConv2d,
                         abits=[[7]]*3, wbits=[[2]]*3)
        nn_ut.eval()
        nn_ut.harden_weights()

        # Check that the output of each layer is integer
        rand_inp = torch.randint(0, 255, (2,)+nn_ut.input_shape)
        oup = None
        for name, module in nn_ut.named_modules():
            if isinstance(module, qm.QuantMultiPrecActivConv2d):
                with torch.no_grad():
                    if oup is not None:
                        oup = module(oup)
                    else:
                        oup = module(rand_inp)
                    is_int = torch.all(oup.int() == oup)
                msg = f'Outputs of {name} are not all integer'
                self.assertTrue(is_int, msg)

    def test_store_hardened_weights(self):
        """Test storing of hardened-weights quantization phase i.e.,
        fake-quant weights and biases are transformed and stored as integer
        weights and biases
        """
        nn_ut = ToyQConv(conv_func=qm.QuantMultiPrecActivConv2d,
                         abits=[[7]]*3, wbits=[[2]]*3)
        nn_ut.eval()
        nn_ut.harden_weights()
        nn_ut.store_hardened_weights()

        # Check that all weight and biases are represented as integers
        for name, param in nn_ut.named_parameters():
            if 'conv.weight' in name or 'conv.bias' in name:
                is_int = torch.all(param.int() == param)
                msg = f'Weights/biases of {name} are not all integer'
                self.assertTrue(is_int, msg)

    def test_qgraph_building(self):
        """Test the build phase of integer quantizated graph"""
        nn_ut = ToyQConv(conv_func=qm.QuantMultiPrecActivConv2d,
                         abits=[[7]]*3, wbits=[[2]]*3)
        dummy_inp = torch.rand((2,) + nn_ut.input_shape)

        new_nn = insert_observers(nn_ut,
                                  target_layers=(qm.QuantMultiPrecActivConv2d))
        new_nn.eval()
        new_nn.harden_weights()
        new_nn(dummy_inp)
        new_nn.store_hardened_weights()
        new_nn = build_qgraph(new_nn,
                              output_classes=2,
                              target_layers=(qm.QuantMultiPrecActivConv2d))
        # TODO: Complete
        self.assertTrue(1)
