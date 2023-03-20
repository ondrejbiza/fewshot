import argparse

import trimesh

from src.utils import convex_decomposition


def main(args):

    mesh = trimesh.load(args.path)
    save_path = args.path.split(".")
    save_path[-2] += "_decomposed"
    save_path[-1] = "obj"
    save_path = ".".join(save_path)
    convex_decomposition(mesh, save_path)


parser = argparse.ArgumentParser()
parser.add_argument("path")
main(parser.parse_args())
