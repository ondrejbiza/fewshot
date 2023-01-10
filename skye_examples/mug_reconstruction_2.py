import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import trimesh

from spatula_warping_example import (
    load_obj,
    pca_reconstruct,
    pca_transform,
    pick_canonical,
    warp_gen,
)


# Chamfer distance between two pointclouds
# Summed distance of each point in reconst to
# the closest point in target
def chamfer_distance(reconst, target):
    total = 0
    reconstruction = reconst
    for i in range(len(reconstruction)):
        point = reconstruction[i]
        distances = np.sum((point - target) ** 2, -1)
        total += np.min(distances)
    return total


# Takes two pointclouds
# Returns pairs of indices indicating the closest points
# in target to each point in reconst
def closest_points(reconst, target):
    reconstruction = reconst
    correspondences = {}
    for i in range(len(reconstruction)):
        point = reconstruction[i]
        distances = np.sum((point - target) ** 2, -1)
        correspondences[i] = np.argmin(distances)
    return correspondences


voxel_extent = 0.15  # How big the voxels are for volume sampling
num_surface_samples = 1500  # number of points to use for surface sampling


# Loads mugs pointclouds in one of three ways
# 'mesh' - gets the vertices directly from the mesh
# 'volume' - voxel centers sampled from the mesh
# 'surface' - points sampled from the surface of the mesh, plus the faces they were sampled from
def load_mugs(mug_files, method="mesh"):
    mug_points = []
    mug_faces = []
    for f in mug_files:
        mesh = trimesh.load(f, force="mesh")
        if method == "mesh":
            points = np.array(mesh.vertices).T
            faces = None
        elif method == "surface":
            points, faces = trimesh.sample.sample_surface_even(
                mesh, num_surface_samples
            )
            points = points.T
        elif method == "volume":
            points = load_obj(f, val=voxel_extent).T
            faces = None
        elif method == "hybrid":
            surf_points, faces = trimesh.sample.sample_surface_even(
                mesh, num_surface_samples
            )
            surf_points = surf_points.T
            mesh_points = np.array(mesh.vertices).T
            points = np.concatenate([mesh_points, surf_points], axis=-1)

        else:
            print(
                "Not a valid sampling method - pick 'mesh', 'surface' or 'volume' "
            )

        # Shifts pointcloud to be centered at the centroid of the mug mesh
        t = mesh.centroid
        trans_matrix = np.array(
            [[1, 0, 0, -t[0]], [0, 1, 0, -t[1]], [0, 0, 1, -t[2]], [0, 0, 0, 1]]
        )
        origin_zero_pts = np.matmul(
            trans_matrix, np.concatenate([points, np.ones((1, points.shape[1]))])
        )
        mug_points.append(origin_zero_pts[:3].T)
        mug_faces.append(faces)

    return mug_points, mug_faces


# Directly warps a source mesh to a target pointcloud
# assumes the mugs were loaded with 'mesh'
def test_mesh_warp_quality(source, target, source_mesh):
    warp = warp_gen(0, [source.T, target.T])
    warp = np.array(warp).reshape(-1, 3)

    target = target.T
    reconst = source.T + warp
    source = source.T

    # Uncomment to see the pointclouds
    # fig, axs = plt.subplots(1, 3, subplot_kw=dict(projection="3d"))
    # axs[0].scatter(source[:,0], source[:,1], source[:,2], color='red', label='Source')
    # axs[2].scatter(target[:,0], target[:,1], target[:,2], color='blue', label='Target')
    # axs[1].scatter(reconst[:,0], reconst[:,1], reconst[:,2], color='purple', label='Reconst')
    # plt.show()

    correspondences = closest_points(reconst, target)
    print(f"Chamfer dist between source and target: {chamfer_distance(source, target)}")
    print(
        f"Chamfer dist between reconstruction and target: {chamfer_distance(reconst, target)}"
    )
    source_faces = source_mesh.faces

    mesh_reconstruction = trimesh.base.Trimesh(vertices=reconst, faces=source_faces)
    mesh_reconstruction.show()


# Does ballpivot on any old pointcloud
def ball_pivot(pointcloud):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointcloud)
    pcd.estimate_normals()

    """ This refines the pointcloud normals by averaging them with their nearest neighbors
        It super breaks the mug normals bc the walls of the mug are so thin """
    # pcd.orient_normals_consistent_tangent_plane(15)

    radii = [0.6]
    reconst_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii)
    )
    o3d.visualization.draw_geometries([pcd, reconst_mesh])
    return reconst_mesh


# Creates a new mesh from points sampled on the surface of an old mesh
# by assigning each point the normal of the face it was sampled from
# then doing ball pivoting.
def cheat_point_normals_surface(mesh_file, points, faces):

    mesh = trimesh.load(mesh_file, force="mesh")
    sample_mesh = o3d.geometry.PointCloud()
    sample_mesh.points = o3d.utility.Vector3dVector(points)
    normals = []
    sample_mesh.normals = o3d.utility.Vector3dVector(mesh.face_normals[faces])

    # Draws points and normals
    # o3d.visualization.draw_geometries([new_mesh], point_show_normal=True)

    radii = [0.6]  # trial and error
    reconst_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        sample_mesh, o3d.utility.DoubleVector(radii)
    )
    # Draws reconstructed mesh
    # o3d.visualization.draw_geometries([new_mesh, reconst_mesh])
    return reconst_mesh


# Tests mesh reconstruction with SST
# Currently set up to look at one test mug at a time
def mug_tests(mug_files, sampling_method="mesh"):
    pointclouds, faces = load_mugs(mug_files, sampling_method)

    # Randomly picks training and test examples
    test_mug_index = np.random.randint(0, len(mug_files))
    test_mugs = [pointclouds[test_mug_index]]
    test_files = [mug_files[test_mug_index]]

    train_mugs = pointclouds[:test_mug_index] + pointclouds[test_mug_index + 1 :]
    train_faces = faces[:test_mug_index] + faces[test_mug_index + 1 :]
    train_files = mug_files[:test_mug_index] + mug_files[test_mug_index + 1 :]

    canonical_index = pick_canonical(train_mugs)
    warps = warp_gen(canonical_index, train_mugs)
    components, pca = pca_transform(warps)

    if sampling_method == "surface":
        source_mesh = cheat_point_normals_surface(
            train_files[canonical_index],
            train_mugs[canonical_index],
            train_faces[canonical_index],
        )
    elif sampling_method == "mesh":
        source_mesh = trimesh.load(train_files[canonical_index], force="mesh")
    elif sampling_method == "hybrid":
        source_mesh = trimesh.load(train_files[canonical_index], force="mesh")
    elif sampling_method == "volume":
        print("Not implemented")
        exit(0)

    for i in range(len(test_mugs)):
        test_pointcloud = test_mugs[i]
        test_file = test_files[i]
        target_mesh = trimesh.load(test_file, force="mesh")

        """This code just chops the test pointcloud in half """
        test_pointcloud = test_pointcloud[test_pointcloud[:,2] > 0]
        # t = target_mesh.centroid
        # trans_matrix = np.array([[1,0,0,-t[0]],[0,1,0, -t[1]], [0,0,1,-t[2]],[0,0,0,1]])
        # target_pointcloud = np.matmul(trans_matrix, np.concatenate([points, np.ones((1, points.shape[1]))]))
        # test_pointcloud = target_pointcloud[:3].T

        xlim = (-2, 2)
        ylim = (-2, 2)
        zlim = (-2, 2)

        latent_rep = pca_reconstruct(
            components, pca, train_mugs[canonical_index], test_pointcloud
        )

        fig, axs = plt.subplots(1, 3, subplot_kw=dict(projection="3d"))
        for x in axs:
            x.set_xlim(xlim)
            x.set_ylim(ylim)
            x.set_zlim(zlim)

        source = train_mugs[canonical_index]
        reconst = source + pca.inverse_transform(np.atleast_2d(latent_rep)).reshape(
            (-1, 3)
        )
        target = test_pointcloud
        axs[0].scatter(
            source[:, 0], source[:, 1], source[:, 2], color="red", label="Source"
        )
        axs[2].scatter(
            target[:, 0], target[:, 1], target[:, 2], color="blue", label="Target"
        )
        axs[1].scatter(
            reconst[:, 0], reconst[:, 1], reconst[:, 2], color="purple", label="Reconst"
        )
        plt.show()

        if sampling_method == "surface":
            new_source_mesh = trimesh.base.Trimesh(
                vertices=source_mesh.vertices, faces=source_mesh.triangles
            )
            mesh_reconstruction = trimesh.base.Trimesh(
                vertices=reconst, faces=source_mesh.triangles
            )
            source_mesh = new_source_mesh

        elif sampling_method == "volume":
            print("Not implemented")
            exit(0)

        elif sampling_method == "mesh":
            source_faces = source_mesh.faces
            mesh_reconstruction = trimesh.base.Trimesh(
                vertices=reconst, faces=source_faces
            )

        elif sampling_method == "hybrid":
            source_faces = source_mesh.faces
            mesh_reconstruction = trimesh.base.Trimesh(
                vertices=reconst[:len(source_faces)], faces=source_faces
            )


        source_mesh.show()
        mesh_reconstruction.show()

        target_mesh.show()


if __name__ == "__main__":
    # Tests mesh reconstruction with original mesh vertices
    #mug_tests(["m1.obj", "m2.obj", "m3.obj", "m4.obj", "m5.obj", "m6.obj"], "mesh")

    # Tests the mug alignment with surface sampling and ball-pivoting
    mug_tests( ['m1.obj', 'm2.obj', 'm3.obj', 'm4.obj', 'm5.obj', 'm6.obj'], 'hybrid')

    # Directly warps a mesh for comparison purposes
    # test_clouds, _ = load_mugs(['m1.obj', 'm2.obj'], 'mesh')
    # test_mesh_warp_quality(test_clouds[0].T, test_clouds[1].T, trimesh.load('m1.obj', force='mesh'))

    # ball pivoting on surface sampling
    # without the mesh normal trick
    # test_clouds, _ = load_mugs(['m1.obj', 'm2.obj'], 'surface')
    # ball_pivot(test_clouds[1])

    # ball pivoting on volume sampling
    # test_clouds, _ = load_mugs(['m1.obj', 'm2.obj'], 'volume')
    # ball_pivot(test_clouds[1])
