import unittest

import numpy as np

from src import utils, viz_utils
from src.object_warping import random_rots_hemisphere


class TestRandomRotsHemisphereShow(unittest.TestCase):

    def test_random_rots_hemisphere(self):

        rots = random_rots_hemisphere(1000)
        canon = utils.CanonObj.from_pickle("data/test_data/ndf_mugs_pca_8_dim.npy")

        for rot in rots:
            param = utils.ObjParam(quat=utils.rotm_to_quat(rot), latent=np.zeros(canon.n_components, dtype=np.float32))
            pcd = canon.to_transformed_pcd(param)
            viz_utils.show_pcd_pyplot(pcd)
