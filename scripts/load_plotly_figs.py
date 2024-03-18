import plotly.graph_objects as go
import pickle

experiment_folder = './experiment_results/part_composed_optimization_20240314-021023/'
experiment_id = 'parent_syn_rack_9_child_e9bd4ee553eb35c1d5ccc40b510e4bd'

final_transform_fig_file = experiment_folder + experiment_id + "_final_transform_fig.pkl"
slider_fig_file = experiment_folder + experiment_id +"_slider_fig.pkl"

final_transform_fig = pickle.load(open(final_transform_fig_file, 'rb'))
slider_fig = pickle.load(open(slider_fig_file, 'rb'))

final_transform_fig.show()
slider_fig.show()