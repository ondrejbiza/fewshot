from src import viz_utils, utils
import plotly
from scipy.stats import sem
import plotly.graph_objects as go
import os
import numpy as np

#get into the correct folder
comparisons = ['whole', 'part', 'rndf']

exp_type = 'bowl_on_mug'
root_dir = f'/home/rthomp12/relational_ndf/src/rndf_robot/eval_data/eval_data/exp--{exp_type}_upright_pose_new_demo-exp--release_demos/'

experiment_folders = os.listdir(root_dir) 
rndf_folders = [folder for folder in experiment_folders if 'rndf' in folder]
part_whole_folders = [folder for folder in experiment_folders if 'rndf' not in folder and folder != 'old']

by_demo = {}
by_child = {}
by_parent = {}
by_demo['rndf'] = {}
by_child['rndf'] = {}
by_parent['rndf'] = {}
by_parent_warp_file = {}
by_child_warp_file = {}

def plot_results(results_dict):
    fig = go.Figure()
    labels = list(results_dict.keys())

    fig.add_trace(go.Bar(
        name='Control',
        x=labels, y=[np.mean(results_dict[label]) for label in labels],
        error_y=dict(type='data', array=[sem(results_dict[label]) for label in labels])
    ))
    fig.update_layout(barmode='group')
    fig.show()

#rndf results
for exp_folder in rndf_folders:
    print(exp_folder)
    objects_raw = os.listdir(root_dir+exp_folder) 

    exp_folder += '/' + objects_raw[0] + '/'

    objects_raw = os.listdir(root_dir+exp_folder) 
    trial_folders = [fn for fn in objects_raw if (fn.split('_')[0] == 'trial')]

    for folder in trial_folders:
        #print(folder)
        experiment_file = root_dir + exp_folder + folder + '/success_rate_relation.npz'
        result = np.load(experiment_file, allow_pickle=True)

        parent_id = str(result['parent_id'])
        if parent_id in by_parent['rndf'].keys():
            by_parent['rndf'][parent_id].append(1 if result['place_success'] else 0)
        else:
            by_parent['rndf'][parent_id] = [1] if result['place_success'] else [0]

        child_id = str(result['child_id'])
        if child_id in by_child['rndf'].keys():
            by_child['rndf'][child_id].append(1 if result['place_success'] else 0)
        else:
            by_child['rndf'][child_id] = [1] if result['place_success'] else [0]

        demo_idx = int(result['args'].item()['demo_idx'])
        if demo_idx in by_demo['rndf'].keys():
            by_demo['rndf'][demo_idx].append(1 if result['place_success'] else 0)
        else:
           by_demo['rndf'][demo_idx] = [1] if result['place_success'] else [0]

plot_results(by_demo['rndf'])
plot_results(by_parent['rndf'])
plot_results(by_child['rndf'])

by_demo['by_parts'] = {}
by_child['by_parts'] = {}
by_parent['by_parts'] = {}
by_child_warp_file['by_parts'] = {}


by_demo['whole'] = {}
by_child['whole'] = {}
by_parent['whole'] = {}
by_parent_warp_file['whole'] = {}
by_child_warp_file['whole'] = {}


#part and whole results
for exp_folder in part_whole_folders:
    print(exp_folder)
    objects_raw = os.listdir(root_dir+exp_folder) 

    exp_folder += '/' + objects_raw[0] + '/'

    objects_raw = os.listdir(root_dir+exp_folder) 
    trial_folders = [fn for fn in objects_raw if (fn.split('_')[0] == 'trial')]

    for folder in trial_folders:
        #print(folder)
        experiment_file = root_dir + exp_folder + folder + '/parts_based_success_rate_relation.npz'
        result = np.load(experiment_file, allow_pickle=True)

        parent_id = str(result['by_parts'])
        if parent_id in by_parent['by_parts'].keys():
            by_parent['by_parts'][parent_id].append(1 if result['place_success'] else 0)
        else:
            by_parent['by_parts'][parent_id] = [1] if result['place_success'] else [0]

        child_id = str(result['child_id'])
        if child_id in by_child['by_parts'].keys():
            by_child['by_parts'][child_id].append(1 if result['place_success'] else 0)
        else:
            by_child['by_parts'][child_id] = [1] if result['place_success'] else [0]

        demo_idx = int(result['args'].item()['demo_idx'])
        if demo_idx in by_demo['by_parts'].keys():
            by_demo['by_parts'][demo_idx].append(1 if result['place_success'] else 0)
        else:
           by_demo['by_parts'][demo_idx] = [1] if result['place_success'] else [0]

        parent_warp = str(result['args'].item()['canon_source_file_stamp'])
        if parent_warp in by_parent_warp_file['by_parts'].keys():
            by_parent_warp_file['by_parts'][demo_idx].append(1 if result['place_success'] else 0)
        else:
           by_parent_warp_file['by_parts'][demo_idx] = [1] if result['place_success'] else [0]

        child_warp = str(result['args'].item()['canon_target_file_stamp'])
        if child_warp in by_child_warp_file['by_parts'].keys():
            by_child_warp_file['by_parts'][demo_idx].append(1 if result['place_success'] else 0)
        else:
           by_child_warp_file['by_parts'][demo_idx] = [1] if result['place_success'] else [0]


        experiment_file = root_dir + exp_folder + folder + '/parts_based_success_rate_relation.npz'
        result = np.load(experiment_file, allow_pickle=True)

        parent_id = str(result['whole'])
        if parent_id in by_parent['whole'].keys():
            by_parent['whole'][parent_id].append(1 if result['place_success'] else 0)
        else:
            by_parent['whole'][parent_id] = [1] if result['place_success'] else [0]

        child_id = str(result['child_id'])
        if child_id in by_child['whole'].keys():
            by_child['whole'][child_id].append(1 if result['place_success'] else 0)
        else:
            by_child['whole'][child_id] = [1] if result['place_success'] else [0]

        demo_idx = int(result['args'].item()['demo_idx'])
        if demo_idx in by_demo['whole'].keys():
            by_demo['whole'][demo_idx].append(1 if result['place_success'] else 0)
        else:
           by_demo['whole'][demo_idx] = [1] if result['place_success'] else [0]

        parent_warp = str(result['args'].item()['canon_source_file_stamp'])
        if parent_warp in by_parent_warp_file['whole'].keys():
            by_parent_warp_file['whole'][demo_idx].append(1 if result['place_success'] else 0)
        else:
           by_parent_warp_file['whole'][demo_idx] = [1] if result['place_success'] else [0]

        child_warp = str(result['args'].item()['canon_target_file_stamp'])
        if child_warp in by_child_warp_file['whole'].keys():
            by_child_warp_file['whole'][demo_idx].append(1 if result['place_success'] else 0)
        else:
           by_child_warp_file['whole'][demo_idx] = [1] if result['place_success'] else [0]

plot_results(by_demo['rndf'])
plot_results(by_parent['rndf'])
plot_results(by_child['rndf'])