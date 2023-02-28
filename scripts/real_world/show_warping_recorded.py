import argparse
import os
import pickle

from src import utils
from src.real_world import perception


def main(args):

    i = 0
    while True:
        load_file = os.path.join(args.load_folder, f"{args.task}_{i}.pkl")
        if not os.path.isfile(load_file):
            break
        with open(load_file, "rb") as f:
            d = pickle.load(f)

        if args.task == "mug_tree":
            source_pcd, target_pcd = d["mug"], d["tree"]

            canon_mug = utils.CanonObj.from_pickle("data/230227_ndf_mugs_scale_large_pca_8_dim_alp_0_01.pkl")
            canon_tree = utils.CanonObj.from_pickle("data/230228_simple_trees_scale_large_pca_8_dim_alp_0_01.pkl")

            canon_mug.init_scale = 0.7
            canon_tree.init_scale = 1.

            perception.warping(
                source_pcd, target_pcd, canon_mug, canon_tree
            )
        elif args.task == "bowl_on_mug":
            source_pcd, target_pcd = d["bowl"], d["mug"]

            canon_bowl = utils.CanonObj.from_pickle("data/230227_ndf_bowls_scale_large_pca_8_dim_alp_0_01.pkl")
            canon_mug = utils.CanonObj.from_pickle("data/230227_ndf_mugs_scale_large_pca_8_dim_alp_0_01.pkl")

            canon_bowl.init_scale = 0.8
            canon_mug.init_scale = 0.7

            perception.warping(
                source_pcd, target_pcd, canon_bowl, canon_mug
            )
        elif args.task == "bottle_in_box":
            source_pcd, target_pcd = d["bottle"], d["box"]

            canon_bottle = utils.CanonObj.from_pickle("data/230227_ndf_bottles_scale_large_pca_8_dim_alp_0_01.pkl")
            canon_box = utils.CanonObj.from_pickle("data/230228_boxes_scale_large_pca_8_dim_alp_0_01.pkl")

            canon_bottle.init_scale = 1.
            canon_box.init_scale = 1.

            perception.warping(
                source_pcd, target_pcd, canon_bottle, canon_box
            )
        else:
            raise ValueError("Invalid task.")

        i += 1


parser = argparse.ArgumentParser("Find objects for a particular task, create warps.")
parser.add_argument("task", help="[mug_tree, bowl_on_mug, bottle_in_box]")
parser.add_argument("--load-folder", default="data/230228_pcds")
main(parser.parse_args())
