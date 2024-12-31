from src import viz_utils, utils
import plotly
from scipy.stats import sem
import plotly.graph_objects as go
import os
import numpy as np
from PIL import Image, ImageDraw

#get into the correct folder
comparisons = ['whole', 'part', 'rndf']
exp_type = 'bowl_on_mug'#'mug_on_rack'#
root_dir = f'/home/rthomp12/relational_ndf/src/rndf_robot/eval_data/eval_data/exp--{exp_type}_upright_pose_new_demo-exp--release_demos/'

experiment_folders = os.listdir(root_dir) 
rndf_folders = [folder for folder in experiment_folders if 'rndf' in folder]
part_whole_folders = [folder for folder in experiment_folders if 'rndf' not in folder and folder != 'old']

by_demo = {}
by_child = {}
by_parent = {}
all_data = {}
all_data['rndf'] = []
by_demo['rndf'] = {}
by_child['rndf'] = {}
by_parent['rndf'] = {}
by_parent_warp_file = {}
by_child_warp_file = {}

def format_image(img):
    datas = img.getdata()

    newData = []
    for item in datas:
        if item[0] == 255 and item[1] == 255 and item[2] == 255:
            newData.append((255, 255, 255, 0))
        else:
            newData.append(item)
    img.putdata(newData)
    return img 



def plot_results(title, results_dicts, names, img_x_labels=False):
    fig = go.Figure()
    labels = list(set.intersection(*map(set,[result_dict.keys() for result_dict in results_dicts])))
    
    i = 0
    print(dir(plotly.colors.sequential))
    color_sequence = plotly.colors.sequential.Plotly3
    for results_dict, name in zip(results_dicts, names):
        y_data = []
        errors = []
        for label in labels:
            if label in results_dict.keys():
                y_data.append(np.mean(results_dict[label]))
                errors.append(sem(results_dict[label]))
        fig.add_trace(go.Bar(
            x=labels, y=y_data,
            error_y=dict(type='data', array=errors),
            name=name,
            marker = {'color': color_sequence[i]}
        ))
        i += 3
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=1), title=title, barmode='group', xaxis=dict(
        tickmode="array",
        ticktext=labels,
        tickvals=np.arange(0, len(labels)).astype(int)))
    
    print(fig.layout.xaxis.ticktext)
    print(fig.layout.xaxis.tickvals)
    
    
    if img_x_labels: 
        for label, val in zip(fig.layout.xaxis.ticktext, fig.layout.xaxis.tickvals):
            
            img = format_image(Image.open(f"mesh_images/mugs/{label}.png"))
            datas = np.array(img)[:,:,3]
            means = (np.min(np.transpose(np.nonzero(datas)), 0) - 300)/600
            print(means)
                
            
            print(val)
            y_pos = .23
#             if exp_type == 'bowl_on_mug':
#                 if label == 'f1c5b9bb744afd96d6e1954365b10b52':
#                     y_pos = .17
                    
            fig.add_layout_image(
                source=format_image(Image.open(f"mesh_images/mugs/{label}.png")),
                x=val + .1,
                y=y_pos + means[0]/2,
                xref="x",
                yref="y",
                xanchor="center",
                sizex=2,
                sizey=2,
            )
            fig.update_layout(xaxis={"visible":False}, yaxis={'range': [-.16,1]})
    
    fig.show()
    
    


def get_num_results(names, results_dict):
    for name in names:
        y_data = np.mean(results_dict[name])
        error = sem(results_dict[name])
        print(f'{name}: mean {y_data}, err {error}')
    
#rndf results
for exp_folder in rndf_folders:
    objects_raw = os.listdir(root_dir+exp_folder) 

    exp_folder += '/' + objects_raw[0] + '/'

    objects_raw = os.listdir(root_dir+exp_folder) 
    trial_folders = [fn for fn in objects_raw if (fn.split('_')[0] == 'trial')]

    for folder in trial_folders:
        #print(folder)
        experiment_file = root_dir + exp_folder + folder + '/success_rate_relation.npz'
        result = np.load(experiment_file, allow_pickle=True)
        
        all_data['rndf'].append(1 if result['place_success'] else 0)

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

#plot_results([by_demo['rndf']])
# plot_results([by_parent['rndf']])
# plot_results([by_child['rndf']])

by_demo['by_parts'] = {}
by_child['by_parts'] = {}
by_parent['by_parts'] = {}
by_parent_warp_file['by_parts'] = {}
by_child_warp_file['by_parts'] = {}


by_demo['whole'] = {}
by_child['whole'] = {}
by_parent['whole'] = {}
by_parent_warp_file['whole'] = {}
by_child_warp_file['whole'] = {}

all_data['whole'] = []
all_data['by_parts'] = []


#part and whole results
for exp_folder in part_whole_folders:
    # print(exp_folder)
    # objects_raw = os.listdir(root_dir+exp_folder) 

    exp_folder += '/'# + objects_raw[0] + '/'

    objects_raw = os.listdir(root_dir+exp_folder) 
    trial_folders = [fn for fn in objects_raw if (fn.split('_')[0] == 'trial')]

    for folder in trial_folders:
        #print(folder)
        try:
            experiment_file = root_dir + exp_folder + folder + '/parts_based_success_rate_relation.npz'
            result = np.load(experiment_file, allow_pickle=True)

            all_data['by_parts'].append(1 if result['place_success'] else 0)
            
            parent_id = str(result['parent_id'])
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
                by_parent_warp_file['by_parts'][parent_warp].append(1 if result['place_success'] else 0)
            else:
               by_parent_warp_file['by_parts'][parent_warp] = [1] if result['place_success'] else [0]

            child_warp = str(result['args'].item()['canon_target_file_stamp'])
            if child_warp in by_child_warp_file['by_parts'].keys():
                by_child_warp_file['by_parts'][child_warp].append(1 if result['place_success'] else 0)
            else:
               by_child_warp_file['by_parts'][child_warp] = [1] if result['place_success'] else [0]

        
            experiment_file = root_dir + exp_folder + folder + '/whole_success_rate_relation.npz'
            result = np.load(experiment_file, allow_pickle=True)
            
            all_data['whole'].append(1 if result['place_success'] else 0)

            parent_id = str(result['parent_id'])
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
                by_parent_warp_file['whole'][parent_warp].append(1 if result['place_success'] else 0)
            else:
               by_parent_warp_file['whole'][parent_warp] = [1] if result['place_success'] else [0]

            child_warp = str(result['args'].item()['canon_target_file_stamp'])
            if child_warp in by_child_warp_file['whole'].keys():
                by_child_warp_file['whole'][child_warp].append(1 if result['place_success'] else 0)
            else:
               by_child_warp_file['whole'][child_warp] = [1] if result['place_success'] else [0]
        except FileNotFoundError:
            continue

            
get_num_results(['rndf', 'whole', 'by_parts'], all_data)
                 
# plot_results("By Demo", [ by_demo['rndf'], by_demo['whole'],  by_demo['by_parts'],], [ "RNDF",  "Whole Object", "Parts Based",])
# #plot_results("By Demo", [ by_demo['by_parts'], by_demo['whole']], [ "Parts Based", "Whole Object"])

plot_results("Bowl on Mug Success Rates By Mug", [ by_parent['rndf'],  by_parent['whole'], by_parent['by_parts']], ["RNDF", "Whole Object", "Parts Based", ], True)

# plot_results("Mug on Rack Success Rates By Mug", [ by_child['rndf'],  by_child['whole'], by_child['by_parts']], ["RNDF", "Whole Object", "Parts Based", ], True)
                
# plot_results("By Child ID", [ by_child['rndf'],  by_child['whole'], by_child['by_parts'],], [ "RNDF",  "Whole Object", "Parts Based",])


#plot_results("By Warp File/Num Latents",[by_parent_warp_file['by_parts'], by_parent_warp_file['whole']], ["Parts Based", "Whole Object"])
#plot_results([by_child_warp_file['by_parts'], by_child_warp_file['whole']], ["Parts Based", "Whole Object"])
