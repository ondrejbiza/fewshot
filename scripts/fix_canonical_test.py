import os, os.path as osp
import src.utils as utils
import src.viz_utils as viz_utils
from rndf_robot.utils import util, path_util
import numpy as np
from scipy.spatial.transform import Rotation

mug_whole_model_file = 'part_based_warp_models/whole_mug_20240501-191613_10'
mug_whole_canon_model =  utils.CanonPart.from_pickle(mug_whole_model_file)

def load_segmented_pointcloud_from_txt(
    pcl_id,
    num_points=2048,
    root="Pointnet_Pointnet2_pytorch/data/shapenetcore_partanno_segmentation_benchmark_v0_normal/03797390",
):
    fn = os.path.join(root, pcl_id + ".txt")
    cls = "Mug"
    data = np.loadtxt(fn).astype(np.float32)
    # if not self.normal_channel: #<-- ignore the normals
    point_set = data[:, 0:3]
    # else:
    # point_set = data[:, 0:6]
    seg_ids = data[:, -1].astype(np.int32)
    point_set[:, 0:3] = utils.center_pcl(point_set[:, 0:3])

    # fixed transform to align with the other mugs being used
    rotation = Rotation.from_euler("zyx", [0.0, np.pi / 2, 0.0]).as_quat()
    transform = utils.pos_quat_to_transform([0, 0, 0], rotation)
    point_set = utils.transform_pcd(point_set, transform)

    choice = np.random.choice(len(seg_ids), num_points, replace=True)
    return point_set, cls, seg_ids

def segment_mug_rack(rack_pts):
    # rack_pts = rack_mesh.vertices
    # Get the top points
    max_z = np.max(rack_pts[:, 2]) - 0.01
    max_pts = rack_pts[rack_pts[:, 2] > max_z]
    center = np.mean(max_pts, 0)

    rad = max(np.linalg.norm(max_pts - center, axis=-1)) * 5 / 4

    # rack_pts = utils.trimesh_create_verts_surface(rack_mesh, 2000)
    trunk = rack_pts[np.linalg.norm(rack_pts[:, :2] - center[:2], axis=-1) < rad]
    branch = rack_pts[np.linalg.norm(rack_pts[:, :2] - center[:2], axis=-1) > rad]
    seg_ids = np.zeros_like(rack_pts)
    seg_ids[np.linalg.norm(rack_pts[:, :2] - center[:2], axis=-1) > rad] = np.ones_like(
        seg_ids[np.linalg.norm(rack_pts[:, :2] - center[:2], axis=-1) > rad]
    )

    start_part_transforms = {
        "trunk": utils.pos_quat_to_transform(
            np.mean(trunk, 0), [0, 0, 0, 1]
        ),  # @ trans,
        "branch": utils.pos_quat_to_transform(np.mean(branch, 0), [0, 0, 0, 1]),
    }
    demo_parts = {
        "trunk": trunk,
        "branch": branch,
    }
    return demo_parts, seg_ids, start_part_transforms


def segment_mug(demo_pcl_id, trans):
    segmented_demo_mug, _, mug_seg_ids = load_segmented_pointcloud_from_txt(demo_pcl_id)

    # segmented_demo_mug = util.transform_pcd(segmented_demo_mug, trans)
    demo_cup = segmented_demo_mug[mug_seg_ids == 37]
    demo_handle = segmented_demo_mug[mug_seg_ids == 36]
    demo_mug = segmented_demo_mug

    # Hack to scale and repositions the parts, since the segmented pcls are normalized differently
    # from the raw shapenet data
    demo_cup, demo_cup_center = utils.center_pcl(demo_cup, return_centroid=True)
    demo_handle, demo_handle_center = utils.center_pcl(
        demo_handle, return_centroid=True
    )
    (
        demo_mug,
        demo_cup,
        demo_handle,
        demo_cup_center,
        demo_handle_center,
    ) = utils.scale_points_circle(
        [
            demo_mug,
            demo_cup,
            demo_handle,
            np.atleast_2d(demo_cup_center),
            np.atleast_2d(demo_handle_center),
        ],
        base_scale=0.13,
    )

    demo_cup = util.transform_pcd(
        demo_cup, utils.pos_quat_to_transform(demo_cup_center, [0, 0, 0, 1])
    )
    demo_cup = util.transform_pcd(demo_cup, trans)
    demo_handle = util.transform_pcd(
        demo_handle, utils.pos_quat_to_transform(demo_handle_center, [0, 0, 0, 1])
    )
    demo_handle = util.transform_pcd(demo_handle, trans)

    # viz_utils.show_pcds_plotly({'cup': demo_cup, 'handle':demo_handle, 'mug':pc_master_dict[pc]['demo_start_pcds'][i]})

    # TODO: Double check that this is necessary/these values actually change from the transform
    _, demo_cup_center = utils.center_pcl(demo_cup, return_centroid=True)
    _, demo_handle_center = utils.center_pcl(demo_handle, return_centroid=True)
    # _, demo_mug_center = utils.center_pcl(demo_mug, return_centroid=True)

    demo_parts = {
        "cup": demo_cup,
        "handle": demo_handle,
    }

    start_part_transforms = {
        "cup": utils.pos_quat_to_transform(demo_cup_center, [0, 0, 0, 1]),  # @ trans,
        "handle": utils.pos_quat_to_transform(demo_handle_center, [0, 0, 0, 1]),
    }
    adjusted_demo_parts = {"cup": demo_cup, "handle": demo_handle}
    return adjusted_demo_parts, -(mug_seg_ids - 37), start_part_transforms

def load_demo(n):
    demo_files = [fn for fn in sorted(os.listdir(demo_path)) if fn.endswith(".npz")]
    demos = []
    demo = np.load(demo_path + "/" + demo_files[n], mmap_mode="r", allow_pickle=True)

    start_child_pcd = demo["multi_obj_start_pcd"].item()['child']
    final_child_pcd = demo["multi_obj_final_pcd"].item()['child']
    start_parent_pcd = demo["multi_obj_start_pcd"].item()['parent']
    final_parent_pcd = demo["multi_obj_final_pcd"].item()['parent']

    start_pose_child = demo["multi_obj_start_obj_pose"].item()['child']
    start_pose_parent = demo["multi_obj_start_obj_pose"].item()['parent']
    final_pose_child = demo["multi_obj_final_obj_pose"].item()['child']
    final_pose_parent = demo["multi_obj_final_obj_pose"].item()['parent']

    demo_child_pcl_id =  demo["multi_object_ids"].item()['child']
    demo_parent_pcl_id = demo["multi_object_ids"].item()['parent']

    trans_child = utils.pos_quat_to_transform(
        start_pose_child[:3],
        start_pose_child[3:],
    )
    trans_parent = utils.pos_quat_to_transform(
        start_pose_parent[:3],
        start_pose_parent[3:],
    )

    child_demo_parts, _, child_demo_start_transforms = segment_mug(
        demo_child_pcl_id, trans_child
    )
               
    parent_demo_parts, _, parent_demo_start_transforms  = segment_mug_rack(
        start_parent_pcd#, trans_parent
    )

    return {
        'start_child_pcd': start_child_pcd, 
        'final_child_pcd': final_child_pcd,
        'start_parent_pcd': start_parent_pcd, 
        'final_parent_pcd': final_parent_pcd,
        'start_pose_child': start_pose_child,
        'final_pose_child': final_pose_child,
        'start_pose_parent': start_pose_parent, 
        'final_pose_parent': final_pose_parent,
        'child_demo_parts': child_demo_parts,
        'parent_demo_parts': parent_demo_parts,
    }


demo_path = osp.join(path_util.get_rndf_data(), "relation_demos", "release_demos/mug_on_rack_relation")


demo_data = load_demo(0)

child_part_model_files = {'cup': '/home/rthomp12/fewshot/part_based_warp_models/cup_dict_20240202-160637',
                          'handle': '/home/rthomp12/fewshot/part_based_warp_models/handle_dict_20240202-160637'}


mug_part_names = ['cup', 'handle']
mug_part_canon_models = {part: utils.CanonPart.from_pickle(child_part_model_files[part]) for part in mug_part_names}

demo_start_fig_camera = dict(
                            up=dict(x=0, y=0, z=1),
                            center = dict(x=0, y=0, z=0.05),
                            eye=dict(x=-1.7,y=-.9,z=0.2))
demo_start_fig = viz_utils.show_pcds_plotly({'mug': demo_data['start_child_pcd'], 
                                             'rack': demo_data['start_parent_pcd']} | 
                                            {'whole_mug': mug_whole_canon_model.canonical_pcl, 
                                            'cup': mug_part_canon_models['cup'].canonical_pcl,
                                            'handle':mug_part_canon_models['handle'].canonical_pcl}, 
                                             center=True,
                                             axis_visible=False,
                                             show_legend=False,
                                             camera=demo_start_fig_camera)

demo_start_fig.show()

#viz_utils.show_pcds_plotly({'whole_mug': mug_whole_canon_model.canonical_pcl}).show()
