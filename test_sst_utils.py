from unittest import TestCase
import numpy as np
from scipy.spatial.transform import Rotation
import torch

import sst_utils


class TestSSTUtils(TestCase):

    def test_cost_and_cost_batch_equivalence(self):

        np.random.seed(2023)
        source1 = np.random.normal(0, 1, (5, 3))
        source2 = np.random.normal(0, 1, (10, 3))
        source3 = np.random.normal(0, 1, (15, 3))
        target1 = np.random.normal(0, 1, (15, 3))
        target2 = np.random.normal(0, 1, (10, 3))
        target3 = np.random.normal(0, 1, (5, 3))

        self.assertAlmostEqual(sst_utils.cost(source1, target1), sst_utils.cost_batch(source1, target1))
        self.assertAlmostEqual(sst_utils.cost(source2, target2), sst_utils.cost_batch(source2, target2))
        self.assertAlmostEqual(sst_utils.cost(source3, target3), sst_utils.cost_batch(source3, target3))
