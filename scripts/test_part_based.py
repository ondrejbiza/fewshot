
import copy
from src.object_warping import WarpBatch, warp_to_pcd, PARAM_1, ObjectSE3Batch, warp_to_pcd_se3
import src.utils as utils
import src.viz_utils as viz_utils
import trimesh
import numpy as np
import open3d as o3d
from sklearn.neighbors import NearestNeighbors
import os
from scipy.spatial.transform import Rotation
from data_utils.ShapeNetDataLoader import PartNormalDataset

inference_kwargs = {
            "train_latents": True,
        }


#From Sergey Prokudin on github
def chamfer_distance(x, y, metric='l2', direction='bi'):
    """Chamfer distance between two point clouds
    Parameters
    ----------
    x: numpy array [n_points_x, n_dims]
        first point cloud
    y: numpy array [n_points_y, n_dims]
        second point cloud
    metric: string or callable, default ‘l2’
        metric to use for distance computation. Any metric from scikit-learn or scipy.spatial.distance can be used.
    direction: str
        direction of Chamfer distance.
            'y_to_x':  computes average minimal distance from every point in y to x
            'x_to_y':  computes average minimal distance from every point in x to y
            'bi': compute both
    Returns
    -------
    chamfer_dist: float
        computed bidirectional Chamfer distance:
            sum_{x_i \in x}{\min_{y_j \in y}{||x_i-y_j||**2}} + sum_{y_j \in y}{\min_{x_i \in x}{||x_i-y_j||**2}}
    """
    
    if direction=='y_to_x':
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        chamfer_dist = np.mean(min_y_to_x)
    elif direction=='x_to_y':
        y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = np.mean(min_x_to_y)
    elif direction=='bi':
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = np.mean(min_y_to_x) + np.mean(min_x_to_y)
    else:
        raise ValueError("Invalid direction type. Supported types: \'y_x\', \'x_y\', \'bi\'")
        
    return chamfer_dist

def create_transform_mat(translation, euler_rotation):
    temp_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
    T = np.eye(4)
    T[:3, :3] = temp_mesh.get_rotation_matrix_from_xyz(euler_rotation)
    T[:3, 3] = translation[:,0]
    return T

def transform_pcl(pcl, transform):
    mult_ready_pcl  = np.concatenate([pcl, np.ones((pcl.shape[0], 1))], axis=1)
    return(np.matmul(transform, mult_ready_pcl.T).T)[:, :3]

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

def load_pointcloud(f, center=True, num_surface_samples=2000): 
    mesh = trimesh.load(open(f, 'rb'), file_type='obj', force="mesh")

    surf_points, faces = trimesh.sample.sample_surface_even(
                mesh, num_surface_samples
            )
    surf_points = surf_points.T
    mesh_points = np.array(mesh.vertices).T

    points = np.concatenate([mesh_points, surf_points], axis=-1)
    centroid = np.mean(points, -1).reshape(3,1)
    if center:
        points -= centroid

    return points, mesh_points, faces, centroid


np.random.seed(2023)
trimesh.util.attach_to_log()
def load_shapenet_way(args):
    if args.objects == "ndf_mugs":
        base_dir = "data/ndf_objects_sample/mug_centered_obj_normalized"
        obj_ids = [
            "1a97f3c83016abca21d0de04f408950f", "1c9f9e25c654cbca3c71bf3f4dd78475", "1eaf8db2dd2b710c7d5b1b70ae595e60",
            "3d1754b7cb46c0ce5c8081810641ef6", "4b8b10d03552e0891898dfa8eb8eefff", "4b7888feea81219ab5f4a9188bfa0ef6",
            "5c48d471200d2bf16e8a121e6886e18d", "5d72df6bc7e93e6dd0cd466c08863ebd", "5fe74baba21bba7ca4eec1b19b3a18f8",
            "6aec84952a5ffcf33f60d03e1cb068dc"]
        obj_paths = [os.path.join(base_dir, x, "models/model_normalized.obj") for x in obj_ids]
        rotation = Rotation.from_euler("zyx", [0., 0., np.pi / 2]).as_matrix()
        num_surface_samples = 10000
    elif args.objects == "ndf_bowls":
        base_dir = "data/ndf_objects_sample/bowl_centered_obj_normalized"
        obj_ids = [
            "1b4d7803a3298f8477bdcb8816a3fac9", "1fbb9f70d081630e638b4be15b07b442",
            "2a1e9b5c0cead676b8183a4a81361b94", "2c1df84ec01cea4e525b133235812833",
            "4b32d2c623b54dd4fe296ad57d60d898", "4eefe941048189bdb8046e84ebdc62d2",
            "4fdb0bd89c490108b8c8761d8f1966ba", "5b6d840652f0050061d624c546a68fec",
            "5bb12905529c85359d3d767e1bc88d65", "7c43116dbe35797aea5000d9d3be7992"
        ]
        obj_paths = [os.path.join(base_dir, x, "models/model_normalized.obj") for x in obj_ids]
        rotation = Rotation.from_euler("zyx", [0., 0., np.pi / 2]).as_matrix()
        num_surface_samples = 10000
    elif args.objects == "ndf_bottles":
        base_dir = "data/ndf_objects/bottle_centered_obj_normalized"
        obj_ids = [
            "1ae823260851f7d9ea600d1a6d9f6e07", "1b64b36bf7ddae3d7ad11050da24bb12",
            "1cf98e5b6fff5471c8724d5673a063a6", "1d4480abe9aa45ce51a99c0e19a8a54",
            "1df41477bce9915e362078f6fc3b29f5", "1e5abf0465d97d826118a17db9de8c0",
            "1ef68777bfdb7d6ba7a07ee616e34cd7", "1ffd7113492d375593202bf99dddc268",
            "2a3e0c1cd0e9076cddf5870150a75bc", "2bbd2b37776088354e23e9314af9ae57"
        ]
        obj_paths = [os.path.join(base_dir, x, "models/model_normalized.obj") for x in obj_ids]
        rotation = Rotation.from_euler("zyx", [0., 0., np.pi / 2]).as_matrix()
        num_surface_samples = 10000
    elif args.objects == "ndf_trees":
        base_dir = "data/syn_racks_easy"
        obj_ids = [f"syn_rack_{i}.obj" for i in range(10)]
        obj_paths = [os.path.join(base_dir, obj_id) for obj_id in obj_ids]
        rotation = None
        num_surface_samples = 2000
    elif args.objects == "boxes":
        obj_paths = [f"data/boxes/train/{i}.stl" for i in range(10)]
        rotation = None
        num_surface_samples = 2000
    elif args.objects == "simple_trees":
        obj_paths = [f"data/simple_trees/train/{i}.stl" for i in range(10)]
        rotation = None
        num_surface_samples = 2000
    elif args.objects == "cuboids":
        base_dir = "data/ndf_objects/distractors/cuboids"
        obj_paths = []
        for i in range(10):
            name = "test_cuboid_smaller_{:d}.stl".format(i)
            path = os.path.join(base_dir, name)
            print(i, path)
            obj_paths.append(path)
        rotation = None
        num_surface_samples = 2000
    else:
        raise ValueError("Unknown object class.")

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

    # load object points using my method
    for mug in mug_files: 
        pcl, vertices, faces, center = load_pointcloud(mug['obj_file'])
        pcl, vertices = [pcl.T, vertices.T]#utils.scale_points_circle([pcl.T, vertices.T], 1)
        mug_pcls.append(pcl)
        mug_mesh_vertices.append(vertices)
        mug_mesh_faces.append(faces)
        mug_complete_dicts.append({"pcl":pcl, "verts":vertices, "faces":faces, 'center':center})

        pcl, vertices, faces, center = load_pointcloud(mug['cup'])
        pcl, vertices = [pcl.T, vertices.T]#utils.scale_points_circle([pcl.T, vertices.T], 1)
        cup_pcls.append(pcl)
        cup_mesh_vertices.append(vertices)
        cup_mesh_faces.append(faces)
        cup_complete_dicts.append({"pcl":pcl, "verts":vertices, "faces":faces, 'center':center})

        pcl, vertices, faces, center = load_pointcloud(mug['handle'])
        pcl, vertices = [pcl.T, vertices.T]#utils.scale_points_circle([pcl.T, vertices.T], 1)
        handle_pcls.append(pcl)
        handle_mesh_vertices.append(vertices)
        handle_mesh_faces.append(faces)
        handle_complete_dicts.append({"pcl":pcl, "verts":vertices, "faces":faces, 'center':center})
        all_parts_dict = {'mug': mug_complete_dicts[-1], 'cup': cup_complete_dicts[-1], 'handle':handle_complete_dicts[-1]}
        all_parts_dicts.append(all_parts_dict)
    return all_parts_dicts

all_parts_dicts = load_things_my_way()  

# generate training/test splits with one test per mug and the rest training
# 
train_test_splits = []
for i in range(len(all_parts_dicts)):
    train_test_splits.append((all_parts_dicts[:i] + all_parts_dicts[i+1:], [all_parts_dicts[i]]))

canonical_parts = []
warped_part_reconstructions = []
complete_reconstruction = []

chamfer_distances = []
for split in train_test_splits:
    canonical = []
    warped_part_reconstruction = []
    training = split[0] + split[0]
    test = split[1]
    for part in ['mug']:#['cup', 'handle']:
        training_pcls = [mug[part]['pcl'] for mug in training]
        test_pcls = [mug[part]['pcl'] for mug in test]

        canonical_idx = utils.sst_pick_canonical(training_pcls)

        warps, _ = utils.warp_gen(canonical_idx, training_pcls, alpha=0.01, visualize=False)
        _, pca = utils.pca_transform(warps, n_dimensions=8)

        mug_canon = utils.CanonObj(training[canonical_idx][part]['pcl'], training[canonical_idx][part]['verts'], training[canonical_idx][part]['faces'], pca)

        param_1 = copy.deepcopy(PARAM_1)
        for pcl in test_pcls:
            warp = WarpBatch(
                    mug_canon, pcl, "cpu", **param_1,
                    init_scale=1.)
            source_pcd_complete, _, source_param = warp_to_pcd(warp, inference_kwargs=inference_kwargs)
        canonical.append(mug_canon)
        warped_part_reconstruction.append(source_pcd_complete)
    canonical_parts.append(canonical)
    warped_part_reconstructions.append(warped_part_reconstruction)

    print(1)

    #mug_transform = create_transform_mat(test[0]['mug']['center'], (0,0,0))
    test_pcl = test[0]['mug']['pcl']
    print(2)
    # cup_transform = create_transform_mat(test[0]['cup']['center'], (0,0,0))
    # handle_transform = create_transform_mat(test[0]['handle']['center'], (0,0,0))
    # reconstructed_pcl = np.concatenate([transform_pcl(test[0]['cup']['pcl'], cup_transform), transform_pcl(test[0]['handle']['pcl'], handle_transform)], axis=0)
    reconstructed_pcl = warped_part_reconstruction[-1]

    print(type(test_pcl))
    print(type(reconstructed_pcl))
    print(reconstructed_pcl.shape)
    print(3)
    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(test_pcl)
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(reconstructed_pcl)

    print(4)
    threshold = 1
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, np.array([[1., 0., 0., 0.],
       [0., 1., 0., 0.],
       [0., 0., 1., 0.],
       [0., 0., 0., 1.]]),
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    source.transform(reg_p2p.transformation)
    reconstructed_pcl = np.asarray(source.points)

    complete_reconstruction.append(reconstructed_pcl)
    # visualize the results
    viz_utils.show_pcds_plotly({
        "pcd": test_pcl,
        "warp": reconstructed_pcl,
    }, center=True)

    error = chamfer_distance(test_pcl, reconstructed_pcl)
    print(error)
    chamfer_distances.append(error)
print(chamfer_distances)
exit(0)
    # visualize the results
        # viz_utils.show_pcds_plotly({
        #     "pcd": pcl,
        #     "warp": source_pcd_complete
        # }, center=False)



    # then do the other thing

