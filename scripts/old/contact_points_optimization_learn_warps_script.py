from rndf_robot.config.default_eval_cfg import get_eval_cfg_defaults
from src.utils import get_closest_point_pairs, fit_plane_points, get_closest_point_pairs_thresh
from src import utils, viz_utils
from scipy.spatial.transform import Rotation
import numpy as np 
import copy as cp
import pickle
import torch


from src.object_warping import ObjectSE2Batch, ObjectSE3Batch, ObjectWarpingSE2Batch, ObjectWarpingSE3Batch, WarpBatch, warp_to_pcd, warp_to_pcd_se2, warp_to_pcd_se3, warp_to_pcd_se3_hemisphere, PARAM_1
import copy as cp

inference_kwargs = {
            "train_latents": True,
            "train_scales": True,
            "train_poses": True,
        }
def aligned_warp_gen(canon_object, canonical_index, objects, contact_points, scale_factor=1., alpha: float=2.0, visualize=False):
    source = objects[canonical_index] * scale_factor
    targets = []
    
    for obj_idx, obj in enumerate(objects):
        if obj_idx != canonical_index:
            targets.append(obj * scale_factor)
            

    contacts = []
    for obj_idx, obj in enumerate(contact_points):
        if obj_idx != canonical_index:
            print(type(obj))
            print(obj.shape)
            contacts.append(obj)

    warps = []
    costs = []


    for target_idx, (target, contact) in enumerate(zip(targets, contacts)):
        print("target {:d}".format(target_idx))

        cost_function = lambda source, target, canon_points: contact_constraint(source, target, contact, canon_points, weight=1)

        warp = ObjectSE3Batch(canon_object, contact_points[canonical_index], target,  'cpu', cost_function=cost_function, **cp.deepcopy(PARAM_1),) 
        result, _, params = warp_to_pcd_se3_hemisphere(warp, 50, n_batches=1, inference_kwargs=inference_kwargs) 
        fixed_source = utils.transform_pcd(source, utils.pos_quat_to_transform(params.position, params.quat))

        w, g = utils.cpd_transform(target, fixed_source, alpha=alpha)

        warp = np.dot(g, w)
        warp = np.hstack(warp)

        tmp = source + warp.reshape(-1, 3)
        costs.append(utils.sst_cost_batch(tmp, target))

        warps.append(warp)

        if visualize:
            viz_utils.show_pcds_plotly({
                "target": target,
                "warp": fixed_source + warp.reshape(-1, 3),
            }, center=True)

    return warps, costs

#For loading the not-shapenet objects and their respective parts
def load_object_files(name, category, scale=1.0/20.0, base_folder="./scripts"):
    obj = {"name":name, "category":category}

    if category == "mug":
        obj["parts"] = ["handle", "cup"]
        path = f"{base_folder}/mugs/" + name
    elif category == "cup":
        obj["parts"] = ["cup"]
        path = f"{base_folder}/cups/" + name
    elif category == "bowl":
        obj["parts"] = ["cup"]
        path = f"{base_folder}/bowls/" + name

    obj['obj_file'] = f'{path}.obj'
    obj['urdf'] = f'{path}.urdf'
    for part in obj["parts"]:

        if part == "cup":
            part_path = f"{base_folder}/cup_parts/{name}"
        elif part == "handle":
            part_path = f"{base_folder}/handles/{name}"

        obj[part] = f"{part_path}.obj"
    obj['scale'] = scale
    return obj


#Loads dictionaries of mugs and their parts 
def load_things_my_way():
    mug_names  = ['m1', 'm2', 'm3', 'm4', 'm5', 'm6']
    mug_files = [load_object_files(mug, 'mug') for mug in mug_names]
    
    #mug_files = ["./scripts/mugs/m1.obj", "./scripts/mugs/m2.obj", "./scripts/mugs/m3.obj", "./scripts/mugs/m4.obj", "./scripts/mugs/m5.obj", "./scripts/mugs/m6.obj"]
    mug_pcls = []
    cup_pcls = []
    handle_pcls = []

    mug_centers = [] 
    cup_centers = []
    handle_centers = []

    mug_mesh_vertices = []
    cup_mesh_vertices = []
    handle_mesh_vertices = []

    mug_mesh_faces = []
    cup_mesh_faces = []
    handle_mesh_faces = []

    mug_complete_dicts = []
    cup_complete_dicts = []
    handle_complete_dicts = []
    all_parts_dicts = []

    return mug_files

all_mugs_files = load_things_my_way()
obj_file_paths = all_mugs_files

save_path = 'data/1234_part_based_mugs_prealign_4_dim.pkl'
temp_path = 'data/saved_cups_4_dim.pkl'


canon_dict = {}
part_names = ['cup', 'handle']
rotation = Rotation.from_euler("zyx", [0., 0., np.pi/2]).as_matrix()
num_surface_samples = 10000

meshes = {}
mesh_pcls = {}
mesh_faces = {}
small_pcls = {}
hybrid_pcls = {}
centers = {}
canonical_idxs = {}


for part in part_names: 
    centers[part] = []
    translation_matrix = np.eye(4)
    
    meshes[part]  = []
    for obj_path in obj_file_paths:
        mesh = utils.trimesh_load_object(obj_path[part])
        t = mesh.centroid
        translation_matrix[:3, 3] = t
        centers[part].append(translation_matrix)
        utils.trimesh_transform(mesh, center=False, scale=None, rotation=rotation)
        meshes[part].append(mesh)

    small_surface_points = []
    surface_points = []
    mesh_points = []
    hybrid_points = []
    faces = []

    for mesh in meshes[part]:
        sp = utils.trimesh_create_verts_surface(mesh, num_surface_samples=num_surface_samples)
        ssp = utils.trimesh_create_verts_surface(mesh, num_surface_samples=2000)
        mp, f = utils.trimesh_get_vertices_and_faces(mesh)
        #ssp, sp, mp = utils.scale_points_circle([ssp, sp, mp], base_scale=0.1)
        h = np.concatenate([mp, sp])  # Order important!

        small_surface_points.append(ssp)
        surface_points.append(sp)
        mesh_points.append(mp)
        faces.append(f)
        hybrid_points.append(h)

    mesh_faces[part] = faces
    mesh_pcls[part] = mesh_points
    hybrid_pcls[part] = hybrid_points
    small_pcls[part] = small_surface_points
    canonical_idx = utils.sst_pick_canonical(hybrid_points)
    canonical_idxs[part] = canonical_idx

#hacky scaling
for i in range(len(hybrid_pcls['cup'])):
    hybrid_pcls['cup'][i], hybrid_pcls['handle'][i], small_pcls['cup'][i], small_pcls['handle'][i] = utils.scale_points_circle([hybrid_pcls['cup'][i], hybrid_pcls['handle'][i], small_pcls['cup'][i], small_pcls['handle'][i]], base_scale=0.1)


def sorter(pcl, idx):
    return np.arctan2(pcl[idx][1], pcl[idx][0])

def get_contact_points(cup_pcl, handle_pcl): 
    max_z = np.max(cup_pcl[:, 2])
    # cup_knns = np.hstack(np.argwhere(abs(cup_pcl[:,2] - max_z)<.00005))
    # cup_knns = sorted(cup_knns, key= lambda idx: sorter(cup_pcl, idx))
    # for knn in cup_knns: 
    #     print(np.arctan2(cup_pcl[knn][1], cup_pcl[knn][0]))

    # print()
    knns = get_closest_point_pairs_thresh(cup_pcl, handle_pcl, .00003)#get_closest_point_pairs(cup_pcl, handle_pcl, 11)
    #viz_utils.show_pcds_plotly({'cup': cup_pcl, 'handle':handle_pcl, 'pts_1': cup_knns,}) #'pts_2': handle_pcl[knns[:, 0]]})
    #contact_points = np.concatenate([cup_pcl[knns[:,1 ]], handle_pcl[knns[:, 0]]])
    return (knns[:,1 ], knns[:, 0])

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

contact_points = {} 
contact_points['cup'] = []
contact_points['handle'] = []
for i in range(len(obj_file_paths)):
    cup_pcl = small_pcls['cup'][i]
    handle_pcl = small_pcls['handle'][i]

    if i == canonical_idxs['cup']:
        cup_pcl = hybrid_pcls['cup'][i]
    if i == canonical_idxs['handle']:
        handle_pcl = hybrid_pcls['handle'][i]

    cup_contacts, handle_contacts = get_contact_points(cup_pcl, handle_pcl)
    contact_points['cup'].append(np.array(cup_contacts))
    contact_points['handle'].append(np.array(handle_contacts))
    small_pcls['cup'][i] = utils.center_pcl(small_pcls['cup'][i])
    small_pcls['handle'][i]= utils.center_pcl(small_pcls['handle'][i])


for part in part_names:
    # if part == 'cup':
    #     canon_dict = pickle.load(open(temp_path, 'rb'))
    #     continue

    hybrid_points = hybrid_pcls[part]
    small_surface_points = small_pcls[part]
    obj_contact_points = contact_points[part]
    canonical_idx = canonical_idxs[part]
    
    tmp_obj_points = cp.copy(small_surface_points)
    tmp_obj_points[canonical_idx] = hybrid_points[canonical_idx]

    #pre-align
    n_angles = 50
    #warp after aligning
    #viz warp process?

    
    temp_canon = utils.CanonObj(tmp_obj_points[canonical_idx], mesh_pcls[part][canonical_idx], mesh_faces[part][canonical_idx], None, None)
    # for i in range(len(tmp_obj_points)):
    #     if i == canonical_idx:
    #         continue
    #     cost_function = lambda source, target, canon_points: contact_constraint(source, target, obj_contact_points[i], canon_points, weight=1)

    #     warp = ObjectSE3Batch(temp_canon, obj_contact_points[canonical_idx], tmp_obj_points[i],  'cpu', cost_function=cost_function, **cp.deepcopy(PARAM_1),) 
    #     result, _, params= warp_to_pcd_se3_hemisphere(warp, n_angles, n_batches=1, inference_kwargs=inference_kwargs) 
    #     tmp_obj_points[i] = utils.transform_pcd(tmp_obj_points[i], utils.pos_quat_to_transform(params.position, params.quat))
    # print("DONE ALIGNING")
    # warpable_contact_points = []
    # for i in range(len(tmp_obj_points)):
    #     warpable_contact_points.append(tmp_obj_points[i][obj_contact_points[i]])

    # #contact_warps, _ = utils.warp_gen(canonical_idx, warpable_contact_points, alpha=0.01, visualize=False)
    warps, _ = aligned_warp_gen(temp_canon, canonical_idx, tmp_obj_points, obj_contact_points, alpha=0.01, visualize=True)

    for i in range(len(warps)):
        warps[i] = warps[i].reshape(-1, 3)
        # contact_warps[i] = contact_warps[i].reshape(-1, 3)
        # #overwriting
        # warps[i][obj_contact_points[canonical_idx]] = contact_warps[i]
        warps[i] = warps[i].reshape(1, -1)

# with open('temp_warps.pkl', "wb") as f:
#     pickle.dump(warps, f)


    for i in range(len(warps)):
        warps[i] = warps[i].reshape(1, -1)
        print(warps[i].shape)

    _, pca = utils.pca_transform(warps, n_dimensions=3)

    canon_dict[part] = {
        "center_transform": centers[part][canonical_idx],
        "pca": pca,
        "canonical_obj": hybrid_points[canonical_idx],
        "canonical_mesh_points": mesh_points[canonical_idx],
        "canonical_mesh_faces": faces[canonical_idx],
    }
    with open(temp_path, "wb") as f:
        pickle.dump(canon_dict, f)

with open(save_path, "wb") as f:
    pickle.dump(canon_dict, f)



## Once the warps have been learned do the testing again


