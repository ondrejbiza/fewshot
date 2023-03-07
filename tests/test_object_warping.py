import unittest

import numpy as np
import torch

from src import object_warping


class TestObjectWarping(unittest.TestCase):

    def test_yaw_to_rot_batch_pt(self):

        yaws_pt = torch.tensor(np.random.uniform(0, 2 * np.pi, (100, 1)).astype(np.float32), device="cpu")

        rot_batch = object_warping.yaw_to_rot_batch_pt(yaws_pt).numpy()
        rots = []
        for yaw_pt in yaws_pt:
            rots.append(object_warping.yaw_to_rot_pt(yaw_pt).numpy())
        rots = np.array(rots)
        np.testing.assert_almost_equal(rot_batch, rots)

    def test_cost_batch_pt(self):

        source = torch.tensor(np.random.normal(0, 1, (10, 3217, 3)).astype(np.float32), device="cpu")
        target = torch.tensor(np.random.normal(0, 1, (10, 1234, 3)).astype(np.float32), device="cpu")

        cost_batch = object_warping.cost_batch_pt(source, target).numpy()
        costs = []
        for ss, tt in zip(source, target):
            costs.append(object_warping.cost_pt(ss, tt).numpy())
        costs = np.array(costs)
        np.testing.assert_almost_equal(cost_batch, costs)

    def test_orthogonalize_unit_rotation(self):

        x = torch.tensor([[
            [1.0, 0., 0.],
            [0., 1.0, 0.]
        ]])

        y = object_warping.orthogonalize(x)[0].cpu().numpy()
        ref = np.eye(3)

        np.testing.assert_equal(y, ref)
