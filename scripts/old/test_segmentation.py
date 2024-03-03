import torch
import trimesh
import numpy as np
import argparse
import os
import importlib
from sklearn.cluster import DBSCAN, KMeans
from data_utils.ShapeNetDataLoader import PartNormalDataset

import matplotlib.pyplot as plt
from scripts.pointnet2_classification import GlobalSAModule, SAModule
import pointnet2_part_seg_msg as MODEL
from torch_geometric.nn import MLP, knn_interpolate
from scipy.spatial.transform import Rotation
from src import utils
import open3d as o3d

def load_pointcloud(f, center=True, num_surface_samples=2000): 
    mesh = trimesh.load(open(f, 'rb'), file_type='obj', force="mesh")
    rotation = Rotation.from_euler("zyx", [-np.pi/2., -np.pi / 2., -np.pi / 2]).as_matrix()
    utils.trimesh_transform(mesh, center=True, scale=.4, rotation=rotation)#all_mugs_dicts[4]['obj_file']

    surf_points, faces = trimesh.sample.sample_surface_even(
                mesh, num_surface_samples
            )

    surf_points = surf_points.T
    mesh_points = np.array(mesh.vertices).T
    surf_normals = mesh.face_normals[faces]

    points = np.concatenate([mesh_points, surf_points], axis=-1)

    normals = np.concatenate([mesh.vertex_normals, surf_normals], axis=0)
    centroid = np.mean(points, -1).reshape(3,1)
    if center:
        points -= centroid

    return points, mesh_points, faces, centroid, normals

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

#load mug pointclouds
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
        pcl, vertices, faces, center, normals = load_pointcloud(mug['obj_file'])
        pcl, vertices = [pcl.T, vertices.T]#utils.scale_points_circle([pcl.T, vertices.T], 1)
        mug_pcls.append(pcl)
        mug_mesh_vertices.append(vertices)
        mug_mesh_faces.append(faces)
        mug_complete_dicts.append({"pcl":pcl, "verts":vertices, "faces":faces, 'center':center, 'normals':normals})

        pcl, vertices, faces, center, normals = load_pointcloud(mug['cup'])
        pcl, vertices = [pcl.T, vertices.T]#utils.scale_points_circle([pcl.T, vertices.T], 1)
        cup_pcls.append(pcl)
        cup_mesh_vertices.append(vertices)
        cup_mesh_faces.append(faces)
        cup_complete_dicts.append({"pcl":pcl, "verts":vertices, "faces":faces, 'center':center, 'normals':normals})

        pcl, vertices, faces, center, normals = load_pointcloud(mug['handle'])
        pcl, vertices = [pcl.T, vertices.T]#utils.scale_points_circle([pcl.T, vertices.T], 1)
        handle_pcls.append(pcl)
        handle_mesh_vertices.append(vertices)
        handle_mesh_faces.append(faces)
        handle_complete_dicts.append({"pcl":pcl, "verts":vertices, "faces":faces, 'center':center, 'normals':normals})
        all_parts_dict = {'mug': mug_complete_dicts[-1], 'cup': cup_complete_dicts[-1], 'handle':handle_complete_dicts[-1]}
        all_parts_dicts.append(all_parts_dict)
    return all_parts_dicts

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y

all_parts_dicts = load_things_my_way()
#load segmentation model


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in testing')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--num_point', type=int, default=2048, help='point Number')
    parser.add_argument('--log_dir', type=str, required=True, help='experiment root')
    parser.add_argument('--normal', action='store_true', default=False, help='use normals')
    parser.add_argument('--num_votes', type=int, default=3, help='aggregate segmentation scores with voting')
    return parser.parse_args()


def visualize_segmentation(points_list, labels):
    fig = plt.figure()
    colors = []
    for label in labels:
         if label == 36:
            colors.append('red')
         else: 
            colors.append('blue') 
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points_list[1][:, 0], points_list[1][:, 1], points_list[1][:, 2])
    ax.scatter(points_list[0][:, 0], points_list[0][:, 1], points_list[0][:, 2], c=colors)
    
    plt.show()


def main(args):

    #TODO: modularize this
    #class that loads the model
    #takes in the pointcloud
    #calculates the segmentation

    root = 'Pointnet_Pointnet2_pytorch/data/shapenetcore_partanno_segmentation_benchmark_v0_normal/'
    TEST_DATASET = PartNormalDataset(root=root, npoints=args.num_point, split='test', normal_channel=args.normal)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=1, shuffle=False,)

    num_classes = 16
    num_part = 50
    seg_classes = {'Mug': [36, 37]}


    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    experiment_dir = 'Pointnet_Pointnet2_pytorch/log_dep/part_seg/' + args.log_dir

    '''MODEL LOADING'''
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    #MODEL = importlib.import_module(f'{model_name}',)
    classifier = MODEL.get_model(num_part, normal_channel=True)#.cuda()
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth', map_location=torch.device('cpu'))
    classifier.load_state_dict(checkpoint['model_state_dict'])

    seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}

    for cat in seg_classes.keys():
        for label in seg_classes[cat]:
            seg_label_to_cat[label] = cat

    classifier = classifier.eval()

    all_mug_dicts = load_things_my_way()
    for mug_dict in all_mug_dicts: 
        #points = torch.from_numpy(mug_dict['mug']['pcl']).unsqueeze(0)
        points_and_normals, _, _ = next(iter(testDataLoader))

        # o3d_pcl = o3d.geometry.PointCloud()
        # o3d_pcl.points = o3d.utility.Vector3dVector(mug_dict['mug']['pcl'])
        # o3d_pcl.estimate_normals()
        # points = torch.from_numpy(np.array(o3d_pcl.points)).unsqueeze(0)
        # print(dir(o3d_pcl))
        # normals = torch.from_numpy(np.array(o3d_pcl.normals)).unsqueeze(0)

        # # normals = torch.from_numpy(mug_dict['mug']['normals']).unsqueeze(0)
        # points_and_normals = torch.cat([points, normals], axis=-1)
        points_and_normals = points_and_normals.transpose(2, 1).type(torch.float32)
        #points_and_normals = points_and_normals[:, :3, :]
        points = points_and_normals[:, :3, :]
        vote_pool = torch.zeros(points_and_normals.size()[0], points_and_normals.size()[2], num_part)
        for _ in range(args.num_votes):
            seg_pred, _ = classifier(points_and_normals, to_categorical(torch.from_numpy(np.array([0])), num_classes))
            print(seg_pred.size())
            vote_pool += seg_pred


        seg_pred = vote_pool / args.num_votes
        cur_pred_val = seg_pred.cpu().data.numpy()
        cur_pred_val_logits = cur_pred_val
        print(cur_pred_val.shape)
        logits = cur_pred_val_logits

        print(logits[:,:, seg_classes['Mug']].shape)
        print(np.argmax(logits[:,:, seg_classes['Mug']], 2))

        cur_pred_val = np.argmax(logits[:,:, seg_classes['Mug']], axis=2) + seg_classes['Mug'][0]

        handle_center = np.mean(np.array(points.squeeze(0).T[cur_pred_val==36]), axis=0)
        cup_center = np.mean(np.array(points.squeeze(0).T[cur_pred_val==37]), axis=0)

        improved_labels = None
        print(points_and_normals.size())
        #points = points_and_normals[:, :3, :]
        handle_points = points.squeeze(0).T[cur_pred_val==36]

        init_centers = [handle_center, cup_center]


        #clustering = DBSCAN(eps=.15,).fit(handle_points)
        clustering = KMeans(2, init=np.array(init_centers)).fit(points.squeeze(0).T)

        #ms = MeanShift(seeds=cur_pred_val)

        print(set(clustering.labels_))

        max_idx = np.argmax(np.bincount(clustering.labels_+1))


        
        visualize_segmentation([points.squeeze(0).T, points.squeeze(0)], clustering.labels_+1+(36-max_idx))
        print(cur_pred_val)


if __name__ == '__main__':
    args = parse_args()
    main(args)



