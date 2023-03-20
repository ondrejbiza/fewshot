import argparse

import numpy as np
import open3d as o3d


def main(args):

    pcd = o3d.io.read_point_cloud(args.input_file)

    points = np.asarray(pcd.points)
    assert len(points) >= args.num_points, "Too many points to sample."

    indices = np.random.choice(list(range(len(points))), size=args.num_points, replace=False)
    points = points[indices]

    pcd.points = o3d.utility.Vector3dVector(points)

    output_file = args.input_file.split(".")
    output_file[-2] += "_downsample"
    output_file = ".".join(output_file)
    o3d.io.write_point_cloud(output_file, pcd)


parser = argparse.ArgumentParser()
parser.add_argument("input_file", type=str)
parser.add_argument("num_points", type=int)
args = parser.parse_args()
main(args)
