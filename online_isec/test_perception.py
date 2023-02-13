import argparse
import rospy

from online_isec.point_cloud_proxy_sync import RealsenseStructurePointCloudProxy
import online_isec.utils as isec_utils
import utils
import viz_utils


def main(args):

    max_pc_size = 2000

    rospy.init_node("moveit_plan_pick_place_plan_approach")
    pc_proxy = RealsenseStructurePointCloudProxy()
    
    cloud = pc_proxy.get_all()
    assert cloud is not None

    cloud = isec_utils.mask_workspace(cloud, (*pc_proxy.desk_center, pc_proxy.z_min))
    # viz_utils.o3d_visualize(utils.create_o3d_pointcloud(cloud))

    mug_pc, tree_pc = isec_utils.find_mug_and_tree(cloud, tall_mug_plaform=args.tall_platform, short_mug_platform=args.short_platform)
    if max_pc_size is not None:
        if len(mug_pc) > max_pc_size:
            mug_pc, _ = utils.farthest_point_sample(mug_pc, max_pc_size)
        if len(tree_pc) > max_pc_size:
            tree_pc, _ = utils.farthest_point_sample(tree_pc, max_pc_size)

    viz_utils.show_scene(
        {
            1: mug_pc,
            2: tree_pc
        },
        background=cloud
    )


parser = argparse.ArgumentParser()
parser.add_argument("-t", "--tall-platform", default=False, action="store_true")
parser.add_argument("-s", "--short-platform", default=False, action="store_true")
main(parser.parse_args())
