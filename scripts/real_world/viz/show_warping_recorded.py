import argparse
import os
import pickle

from src import utils
from src.real_world import constants, perception


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

            canon_mug = utils.CanonObj.from_pickle(constants.NDF_MUGS_PCA_PATH)
            canon_tree = utils.CanonObj.from_pickle(constants.SIMPLE_TREES_PCA_PATH)

            canon_mug.init_scale = constants.NDF_MUGS_INIT_SCALE
            canon_tree.init_scale = constants.SIMPLE_TREES_INIT_SCALE

            perception.warping(
                source_pcd, target_pcd, canon_mug, canon_tree
            )
        elif args.task == "bowl_on_mug":
            source_pcd, target_pcd = d["bowl"], d["mug"]

            canon_bowl = utils.CanonObj.from_pickle(constants.NDF_BOWLS_PCA_PATH)
            canon_mug = utils.CanonObj.from_pickle(constants.NDF_MUGS_PCA_PATH)

            canon_bowl.init_scale = constants.NDF_BOWLS_INIT_SCALE
            canon_mug.init_scale = constants.NDF_MUGS_INIT_SCALE

            perception.warping(
                source_pcd, target_pcd, canon_bowl, canon_mug
            )
        elif args.task == "bottle_in_box":
            source_pcd, target_pcd = d["bottle"], d["box"]

            canon_bottle = utils.CanonObj.from_pickle(constants.NDF_BOTTLES_PCA_PATH)
            canon_box = utils.CanonObj.from_pickle(constants.BOXES_PCA_PATH)

            canon_bottle.init_scale = constants.NDF_BOTTLES_INIT_SCALE
            canon_box.init_scale = constants.BOXES_INIT_SCALE

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
