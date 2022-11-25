import unittest
import numpy as np
from mug import utils


class TestUtils(unittest.TestCase):
    def test_slice3d(self):
        x = utils.slice3d(np.zeros([16,32,64]))
        self.assertSequenceEqual(x.shape, [32+16,64+16])

    def test_psd(self):
        utils.power_spectrum_density(np.zeros([16,32,64]))
