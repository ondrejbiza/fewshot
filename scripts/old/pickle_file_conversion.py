
#import definition of CanonicalPart
#import definition of CanonicalObject 

from scripts.generate_warps import load_all_shapenet_files, get_mesh, get_segmented_mesh, CanonPart, CanonPartMetadata
from src import utils, viz_utils
import pickle

warp_file_stamp = '20240202-160637'

#todo: generalize for other objects
object_warp_file = f'whole_mug_{warp_file_stamp}'
cup_warp_file = f'cup_{warp_file_stamp}'
handle_warp_file = f'handle_{warp_file_stamp}'

part_names = ['cup', 'handle']
part_labels = {'cup': 37, 'handle':36}

part_canonicals = {}
whole_object_canonical = pickle.load(open( object_warp_file, 'rb'))
part_canonicals['cup'] = pickle.load(open( cup_warp_file, 'rb'))
part_canonicals['handle'] = pickle.load(open( handle_warp_file, 'rb'))

#load pickle files

whole_obj = {'pcd':whole_object_canonical.canonical_pcl, 
'pca':whole_object_canonical.pca, 
'center_transform':whole_object_canonical.center_transform, 
'canonical_mesh_points':whole_object_canonical.mesh_vertices, 
'canonical_mesh_faces':whole_object_canonical.mesh_faces}


cup_obj = {'pcd':part_canonicals['cup'].canonical_pcl, 
'pca':part_canonicals['cup'].pca, 
'center_transform':part_canonicals['cup'].center_transform, 
'canonical_mesh_points':part_canonicals['cup'].mesh_vertices, 
'canonical_mesh_faces':part_canonicals['cup'].mesh_faces}

handle_obj = {'pcd':part_canonicals['handle'].canonical_pcl, 
'pca':part_canonicals['handle'].pca, 
'center_transform':part_canonicals['handle'].center_transform, 
'canonical_mesh_points':part_canonicals['handle'].mesh_vertices, 
'canonical_mesh_faces':part_canonicals['handle'].mesh_faces}


pickle.dump(whole_obj, open(f'canon_whole_mug_{warp_file_stamp}', 'wb'))
pickle.dump(cup_obj, open(f'canon_cup_{warp_file_stamp}', 'wb'))
pickle.dump(handle_obj, open(f'canon_handle_{warp_file_stamp}', 'wb'))
