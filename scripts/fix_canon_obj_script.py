#load canonical objects from pickle file
#adjust dictionary
#save them again



import numpy as np
import pickle

canon_source_path = 'data/1234_part_based_mugs_4_dim.pkl'
canon_dicts = pickle.load(open(canon_source_path, 'rb'))

cup_transform = np.array([[ 1.00000000e+00,  0.00000000e+00,  0.00000000e+00, -5.26211387e-04],
 [ 0.00000000e+00,  1.00000000e+00,  0.00000000e+00,  2.41347181e-01],
 [ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00,  2.18432279e-01],
 [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])

handle_transform = np.array([[ 1.00000000e+00,  0.00000000e+00,  0.00000000e+00, -1.15634288e-03],
 [ 0.00000000e+00,  1.00000000e+00,  0.00000000e+00,  6.79849519e-01],
 [ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00, -1.89414989e+00],
 [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])

canon_dicts['cup']['center_transform'] = cup_transform
canon_dicts['handle']['center_transform'] = handle_transform
pickle.dump(canon_dicts, open(canon_source_path, 'wb'))