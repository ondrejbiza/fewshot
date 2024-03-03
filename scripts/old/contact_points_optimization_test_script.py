from src.utils import get_closest_point_pairs, fit_plane_points, get_closest_point_pairs_thresh
from scipy.spatial.transform import Rotation
from src import utils, viz_utils
import numpy as np
import os
import matplotlib.pyplot as plt
import torch

from src.object_warping import ObjectSE2Batch, ObjectSE3Batch, ObjectWarpingSE2Batch, ObjectWarpingSE3Batch, WarpBatch, warp_to_pcd, warp_to_pcd_se2, warp_to_pcd_se3, warp_to_pcd_se3_hemisphere, PARAM_1
import copy as cp

inference_kwargs = {
            "train_latents": True,
            "train_scales": True,
            "train_poses": True,
        }

def load_segmented_pointcloud_from_txt(pcl_id, num_points=2048, root = 'Pointnet_Pointnet2_pytorch/data/shapenetcore_partanno_segmentation_benchmark_v0_normal/03797390'):
    fn = os.path.join(root, pcl_id + '.txt')
    cls = 'Mug'
    data = np.loadtxt(fn).astype(np.float32)
    #if not self.normal_channel:
    point_set = data[:, 0:3]
    #else:
        #point_set = data[:, 0:6]
    seg_ids = data[:, -1].astype(np.int32)
    point_set[:, 0:3] = center_pcl(point_set[:, 0:3])

    rotation = Rotation.from_euler("zyx", [0., np.pi/2, 0.]).as_quat()

    transform = utils.pos_quat_to_transform([0,0,0], rotation)
    point_set = utils.transform_pcd(point_set, transform)

    choice = np.random.choice(len(seg_ids), num_points, replace=True)
    # resample
    #point_set = point_set[choice, :]
    #seg = seg[choice]

    return point_set, cls, seg_ids

def center_pcl(pcl, return_centroid=False):
    l = pcl.shape[0]
    centroid = np.mean(pcl, axis=0)
    pcl = pcl - centroid
    if return_centroid:
        return pcl, centroid
    else:
        return pcl

#load the demo and canonical cup
source_part_names = ['cup', 'handle']
demo_pcl_ids = ['5c7c4cb503a757147dbda56eabff0c47', '187859d3c3a2fd23f54e1b6f41fdd78a', '8b780e521c906eaf95a4f7ae0be930ac', 'e9499e4a9f632725d6e865157050a80e', '6aec84952a5ffcf33f60d03e1cb068dc', 'fad118b32085f3f2c2c72e575af174cd', '85a2511c375b5b32f72755048bac3f96', '85a2511c375b5b32f72755048bac3f96']
demo_cups = []
demo_handles = []
demo_mugs = []

for demo_pcl_id in demo_pcl_ids:
    try: 
        demo_mug, _, demo_seg_ids = load_segmented_pointcloud_from_txt(demo_pcl_id)
    except:
        continue
    rotation = Rotation.from_euler("xyz", [np.pi/2., 0., np.pi/12.]).as_quat()

    demo_mug = utils.transform_pcd(demo_mug, utils.pos_quat_to_transform([0,0,0], rotation))

    demo_cup = demo_mug[demo_seg_ids==37]
    demo_handle = demo_mug[demo_seg_ids==36]
    demo_cup, demo_handle, demo_mug = utils.scale_points_circle([demo_cup, demo_handle, demo_mug], base_scale=0.125)
    demo_cups.append(demo_cup)
    demo_handles.append(demo_handle)
    demo_mugs.append(demo_mug)


canon_source_path = 'data/1234_part_based_mugs_contact_overwritten_4_dim.pkl'#'data/1234_part_based_mugs_4_dim.pkl'

cup_path = './scripts/cup_parts/m1.obj'
cup_handle_path = './scripts/handles/m1.obj'

handle_path = './scripts/handles/m4.obj'
handle_cup_path  = './scripts/cup_parts/m4.obj'

def load_obj_part(obj_path):
    mesh = utils.trimesh_load_object(obj_path)
    rotation = Rotation.from_euler("zyx", [0., 0., np.pi/2]).as_matrix()
    utils.trimesh_transform(mesh, center=False, scale=None, rotation=rotation)
    ssp = utils.trimesh_create_verts_surface(mesh, num_surface_samples=2000)
    return ssp

canon_source_parts = utils.CanonObj.from_parts_pickle(canon_source_path, source_part_names)
canon_cup, canon_cup_handle = load_obj_part(cup_path), load_obj_part(cup_handle_path)
canon_handle, canon_handle_cup = load_obj_part(handle_path), load_obj_part(handle_cup_path)

canon_cup, canon_cup_handle, canon_handle, canon_handle_cup = utils.scale_points_circle([canon_cup, canon_cup_handle, canon_handle, canon_handle_cup], base_scale=0.13)
true_canon_handle = canon_source_parts['handle'].canonical_pcd #+np.mean(canon_handle, axis=0)
true_canon_cup = canon_source_parts['cup'].canonical_pcd #+np.mean(canon_cup, axis=0)
#canon_cup, canon_cup_handle, canon_handle, canon_handle_cup, true_canon_handle, true_canon_cup = utils.scale_points_circle([canon_cup, canon_cup_handle, canon_handle, canon_handle_cup, true_canon_handle, true_canon_cup], base_scale=0.13)

# ax = plt.subplot(111, projection='3d')
# ax.scatter(canon_handle[:, 0], canon_handle[:, 1], canon_handle[:, 2], color='b')
# ax.scatter(true_canon_handle[:, 0], true_canon_handle[:, 1], true_canon_handle[:, 2], color='r')
# plt.show()

# exit(0)
canon_cup_mug = np.concatenate([canon_cup, canon_cup_handle])
canon_handle_mug = np.concatenate([canon_handle, canon_handle_cup])


def get_contact_points(cup_pcl, handle_pcl): 
    knns = get_closest_point_pairs_thresh(cup_pcl, handle_pcl, .00003)#get_closest_point_pairs(cup_pcl, handle_pcl, 11)
    #viz_utils.show_pcds_plotly({'cup': cup_pcl, 'handle':handle_pcl, 'pts_1': cup_pcl[knns[:,1 ]], 'pts_2': handle_pcl[knns[:, 0]]})
    cup_contact_points = cup_pcl[knns[:,1 ]]
    handle_contact_points = handle_pcl[knns[:, 0]]
    
    return (knns[:,1 ], knns[:, 0])

canon_contact_cup, canon_contact_cup_handle = get_contact_points(true_canon_cup, canon_cup_handle)
canon_contact_handle_cup, canon_contact_handle = get_contact_points(canon_handle_cup, true_canon_handle)

# viz_utils.show_pcds_plotly({'cup': true_canon_cup, 'handle': canon_cup_handle})
# viz_utils.show_pcds_plotly({'cup': canon_handle_cup, 'handle': true_canon_handle})

# exit(0)
def cost_batch_pt(source, target): 
    """Calculate the one-sided Chamfer distance between two batches of point clouds in pytorch."""
    # B x N x K
    diff = torch.sqrt(torch.sum(torch.square(source[:, :, None] - target[:, None, :]), dim=3))
    diff_flat = diff.view(diff.shape[0] * diff.shape[1], diff.shape[2])
    c_flat = diff_flat[list(range(len(diff_flat))), torch.argmin(diff_flat, dim=1)]
    c = c_flat.view(diff.shape[0], diff.shape[1])
    return torch.mean(c, dim=1)

def contact_constraint(source, target, source_contact_indices, canon_points, weight=10):  
    #ax = plt.subplot(111, projection='3d')  
    displayable_target = target.detach().numpy()
    displayable_canon_points = canon_points.detach().numpy()
    #print(displayable_canon_points.shape)
    displayable_source = source.detach().numpy()
    displayable_source_points = displayable_source[:, source_contact_indices, :]
    #print(displayable_source_points.shape)
    constraint = cost_batch_pt(canon_points, source[:, source_contact_indices, :]) * weight
    #print(constraint)
    # for i in range(len(target)):
    #     ax.scatter(displayable_target[i, :, 0], displayable_target[i, :, 1], displayable_target[i, :, 2], alpha=.05)
    #     ax.scatter(displayable_canon_points[i, :, 0], displayable_canon_points[i, :, 1], displayable_canon_points[i, :, 2], color='g')

    # ax.scatter(displayable_source[0, :, 0], displayable_source[0, :, 1], displayable_source[0, :, 2], color='r')
    # ax.scatter(displayable_source_points[0, :, 0], displayable_source_points[0, :, 1], displayable_source_points[0, :, 2], color='y')
    # #ax.scatter(displayable_source_points[0, :, 0], displayable_source_points[0, :, 1], displayable_source_points[0, :, 2], color='r')
    
    # plt.show()
    return cost_batch_pt(source, target) + constraint

from PIL import Image

def make_image_grid(files, save_name): 
    cols = int(np.trunc(np.sqrt(len(files))))
    rows = len(files)//cols

    images = []
    for f in files:
        images.append(Image.open(f))
        width, height = images[-1].size
        images[-1] = images[-1].crop((width*1./5., 0, width*4./5., height))

    width, height = images[0].size
    result = Image.new('RGB', (cols*width, rows*height))
    for i in range(len(images)):
        result.paste(im=images[i], box=(width*(i%cols), height*(i//cols)))
    result.save(f'./scripts/{save_name}.jpg')


do_warp = False
side_files = []
front_files = []
top_files = []
for i, (demo_cup, demo_handle, demo_mug)in enumerate(zip(demo_cups, demo_handles, demo_mugs)):
    demo_contact_cup, demo_contact_handle = get_contact_points(demo_cup, demo_handle)
    cost_function = lambda source, target, canon_points: contact_constraint(source, target, demo_contact_cup, canon_points, weight=1)

    # print(canon_contact_cup)
    # print(canon_contact_handle)
    # exit(0)
    if do_warp:
        aligned_cup = ObjectWarpingSE2Batch(
                        canon_source_parts['cup'], canon_contact_cup, demo_cup,  'cpu', cost_function=cost_function, **cp.deepcopy(PARAM_1),
                        ) 
    else:
        aligned_cup = ObjectSE2Batch(
                        canon_source_parts['cup'],canon_contact_cup, demo_cup,  'cpu', cost_function=cost_function, **cp.deepcopy(PARAM_1),
                        ) 

    aligned_cup_pcl, cost, aligned_cup_params = warp_to_pcd_se2(aligned_cup, n_angles=30, n_batches=1, inference_kwargs=inference_kwargs)

    cost_function = lambda source, target, canon_points: contact_constraint(source, target, demo_contact_handle, canon_points, weight=1)

    if do_warp:
        aligned_handle = ObjectWarpingSE3Batch(
                        canon_source_parts['handle'],canon_contact_handle, demo_handle,  'cpu', cost_function=cost_function, **cp.deepcopy(PARAM_1),
                        ) 
    else:
        aligned_handle = ObjectSE3Batch(
                        canon_source_parts['handle'],canon_contact_handle, demo_handle,  'cpu', cost_function=cost_function, **cp.deepcopy(PARAM_1),
                        ) 

    aligned_handle_pcl, cost, aligned_handle_params = warp_to_pcd_se3_hemisphere(aligned_handle, n_angles=100, n_batches=1, inference_kwargs=inference_kwargs)

    reconstructed_cup = canon_source_parts['cup'].to_transformed_pcd(aligned_cup_params)
    reconstructed_handle = canon_source_parts['handle'].to_transformed_pcd(aligned_handle_params)

    ax = plt.subplot(111, projection='3d')
    ax.set_xlim(-.25, .25)
    ax.set_ylim(-.25, .25)
    ax.set_zlim(-.25, .25)
    #ax.scatter(warped_handle[:, 0], warped_handle[:, 1], warped_handle[:, 2], color='b')
    ax.scatter(reconstructed_cup[:, 0], reconstructed_cup[:, 1], reconstructed_cup[:, 2], color='b', alpha=0.05)
    ax.scatter(reconstructed_handle[:, 0], reconstructed_handle[:, 1], reconstructed_handle[:, 2], color='b', alpha=0.05)
    #ax.scatter(aligned_cup_pcl[canon_contact_cup, 0], aligned_cup_pcl[canon_contact_cup, 1], aligned_cup_pcl[canon_contact_cup, 2], color='g')

    #ax.quiver(0, 0, -.25, canon_handle_normal[0], canon_handle_normal[1], canon_handle_normal[2], color="b")
    #ax.scatter(source[:, 0], source[:, 1], source[:, 2], color='g')
    #ax.quiver(0, 0, -.25, demo_normal[0], demo_normal[1], demo_normal[2], color="g")
    #ax.scatter(canon_aligned_demo_handle[:, 0], canon_aligned_demo_handle[:, 1], canon_aligned_demo_handle[:, 2], color='r')
    ax.scatter(demo_mug[:, 0], demo_mug[:, 1], demo_mug[:, 2], color='r', alpha=0.5)
    viz_utils.show_pcds_plotly({'recon_cup': reconstructed_cup,
                                'recon_handle':reconstructed_handle,
                                'demo_mug': demo_mug,
                                'canon_cup_contacts': reconstructed_cup[canon_contact_cup],
                                'canon_cup_handle': reconstructed_handle[canon_contact_handle],
                                'demo_cup_contacts': demo_cup[demo_contact_cup],
                                'demo_handle_contacts': demo_handle[demo_contact_handle],})

    ax.view_init(azim=0)
    print(f"Saving {i}")
    plt.savefig(f'./scripts/temp_images/side_temp_{i}.png')
    ax.view_init(elev=90, azim=0)
    plt.savefig(f'./scripts/temp_images/front_temp_{i}.png')
    ax.view_init(azim=-90)
    plt.savefig(f'./scripts/temp_images/top_temp_{i}.png')
    # print("Saved!")
    # print("")
    plt.clf()
    plt.close()
    side_files.append(f'./scripts/temp_images/side_temp_{i}.png')
    front_files.append(f'./scripts/temp_images/front_temp_{i}.png')
    top_files.append(f'./scripts/temp_images/top_temp_{i}.png')

make_image_grid(side_files, 'nowarp_side_mugs_weight_1')
make_image_grid(front_files, 'nowarp_front_mugs_weight_1')
for f in side_files: 
    os.remove(f)
for f in front_files: 
    os.remove(f)

make_image_grid(top_files, 'nowarp_top_mugs_weight_1')
for f in top_files: 
    os.remove(f)

exit(0)


do_warp = False
costs = []
side_files = []
front_files = []
top_files = []
for i, (demo_cup, demo_handle)in enumerate(zip(demo_cups, demo_handles)):
    demo_contact_cup, demo_contact_handle = get_contact_points(demo_cup, demo_handle)
    cost_function = lambda source, target, canon_points: contact_constraint(source, target, demo_contact_cup, canon_points, weight=1)

    if do_warp:
        aligned_cup = ObjectWarpingSE2Batch(
                        canon_source_parts['cup'], canon_contact_cup, demo_cup,  'cpu', cost_function=cost_function, **cp.deepcopy(PARAM_1),
                        init_scale=1)
    else: 
        aligned_cup = ObjectSE2Batch(
                        canon_source_parts['cup'], canon_contact_cup, demo_cup,  'cpu', cost_function=cost_function, **cp.deepcopy(PARAM_1),
                        init_scale=1)
    aligned_cup_pcl, cost, aligned_cup_params = warp_to_pcd_se2(aligned_cup, n_angles=30, n_batches=1, inference_kwargs=inference_kwargs)
        


    costs.append(cost)
    ax = plt.subplot(111, projection='3d')
    ax.set_xlim(-.25, .25)
    ax.set_ylim(-.25, .25)
    ax.set_zlim(-.25, .25)
    #ax.scatter(warped_handle[:, 0], warped_handle[:, 1], warped_handle[:, 2], color='b')
    ax.scatter(aligned_cup_pcl[:, 0], aligned_cup_pcl[:, 1], aligned_cup_pcl[:, 2], color='b', alpha=0.05)
    ax.scatter(aligned_cup_pcl[canon_contact_cup, 0], aligned_cup_pcl[canon_contact_cup, 1], aligned_cup_pcl[canon_contact_cup, 2], color='g', s=5)

    #ax.quiver(0, 0, -.25, canon_handle_normal[0], canon_handle_normal[1], canon_handle_normal[2], color="b")
    #ax.scatter(source[:, 0], source[:, 1], source[:, 2], color='g')
    #ax.quiver(0, 0, -.25, demo_normal[0], demo_normal[1], demo_normal[2], color="g")
    #ax.scatter(canon_aligned_demo_handle[:, 0], canon_aligned_demo_handle[:, 1], canon_aligned_demo_handle[:, 2], color='r')
    ax.scatter(demo_cup[:, 0], demo_cup[:, 1], demo_cup[:, 2], color='r')
    ax.scatter(demo_cup[demo_contact_cup, 0], demo_cup[demo_contact_cup, 1], demo_cup[demo_contact_cup, 2], color='y', s=5)
    ax.view_init(azim=0)
    print(f"Saving {i}")
    plt.savefig(f'./scripts/temp_images/side_temp_{i}.png')
    ax.view_init(elev=90, azim=0)
    plt.savefig(f'./scripts/temp_images/front_temp_{i}.png')
    ax.view_init(azim=-90)
    plt.savefig(f'./scripts/temp_images/top_temp_{i}.png')
    # print("Saved!")
    # print("")
    plt.clf()
    plt.close()
    side_files.append(f'./scripts/temp_images/side_temp_{i}.png')
    front_files.append(f'./scripts/temp_images/front_temp_{i}.png')
    top_files.append(f'./scripts/temp_images/top_temp_{i}.png')

make_image_grid(side_files, 'nowarp_side_cups_weight_1')
make_image_grid(front_files, 'nowarp_front_cups_weight_1')
for f in side_files: 
    os.remove(f)
for f in front_files: 
    os.remove(f)

make_image_grid(top_files, 'nowarp_top_cups_weight_1')
for f in top_files: 
    os.remove(f)

exit(0)

##########################################################################################

costs = []
front_files = []
side_files = []
for i, (demo_cup, demo_handle)in enumerate(zip(demo_cups, demo_handles)):
    demo_contact_cup, demo_contact_handle = get_contact_points(demo_cup, demo_handle)
    cost_function = lambda source, target, canon_points: contact_constraint(source, target, demo_contact_handle, canon_points, weight=5)

    aligned_handle = ObjectWarpingSE3Batch(
                    canon_source_parts['handle'],canon_contact_handle, demo_handle,  'cpu', cost_function=cost_function, **cp.deepcopy(PARAM_1),
                    init_scale=1) 
    aligned_handle_pcl, cost, aligned_handle_params = warp_to_pcd_se3(aligned_handle, n_angles=100, n_batches=1, inference_kwargs=inference_kwargs)
    costs.append(cost)
    ax = plt.subplot(111, projection='3d')
    ax.set_xlim(-.1, .1)
    ax.set_ylim(-0, .2)
    ax.set_zlim(-.1, .1)
    #ax.scatter(warped_handle[:, 0], warped_handle[:, 1], warped_handle[:, 2], color='b')
    ax.scatter(aligned_handle_pcl[:, 0], aligned_handle_pcl[:, 1], aligned_handle_pcl[:, 2], color='b', alpha=0.05)
    ax.scatter(aligned_handle_pcl[canon_contact_handle, 0], aligned_handle_pcl[canon_contact_handle, 1], aligned_handle_pcl[canon_contact_handle, 2], color='g', s = 5)

    #ax.quiver(0, 0, -.25, canon_handle_normal[0], canon_handle_normal[1], canon_handle_normal[2], color="b")
    #ax.scatter(source[:, 0], source[:, 1], source[:, 2], color='g')
    #ax.quiver(0, 0, -.25, demo_normal[0], demo_normal[1], demo_normal[2], color="g")
    #ax.scatter(canon_aligned_demo_handle[:, 0], canon_aligned_demo_handle[:, 1], canon_aligned_demo_handle[:, 2], color='r')
    ax.scatter(demo_handle[:, 0], demo_handle[:, 1], demo_handle[:, 2], color='r')
    ax.scatter(demo_handle[demo_contact_handle, 0], demo_handle[demo_contact_handle, 1], demo_handle[demo_contact_handle, 2], color='y', s = 5)
    ax.view_init(azim=0)
    print(f"Saving {i}")
    plt.savefig(f'./scripts/temp_images/side_temp_{i}.png')
    ax.view_init(azim=-90)
    plt.savefig(f'./scripts/temp_images/front_temp_{i}.png')
    print("Saved!")
    print("")
    plt.clf()
    plt.close()
    side_files.append(f'./scripts/temp_images/side_temp_{i}.png')
    front_files.append(f'./scripts/temp_images/front_temp_{i}.png')

make_image_grid(side_files, 'side_handles_weight_5')
make_image_grid(front_files, 'front_handles_weight_5')
for f in side_files: 
    os.remove(f)
for f in front_files: 
    os.remove(f)
print(np.mean(costs))
