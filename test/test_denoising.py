import unittest
import numpy as np
from mug import denoising


class TestDenoising(unittest.TestCase):
    def test_gat(self):
        gain = 10
        offset = 1
        sigma = 1
        im = np.arange(512).reshape([512,1]) @ np.arange(512).reshape([1,512])
        noisy = gain * np.random.poisson(im) + offset + np.random.normal(0,sigma,size=im.shape)
        gat = denoising.generalized_anscombe_transform(gain, offset, sigma)
        tmp = gat.invert(gat(noisy))
        self.assertTrue(np.abs(tmp - noisy).max() < 0.1, "Inconsistent forward/backward transform.")

