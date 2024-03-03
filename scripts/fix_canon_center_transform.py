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
