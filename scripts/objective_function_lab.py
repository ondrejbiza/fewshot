

#load the segmented pcl

#define different objective functions

def baseline_relational_descriptor():
	pass

def no_shape_regularization():
	pass

#Penalizes warps that change the length x width x height ratio substantially
def scale_preserving_objective():
	pass

#Penalizes warps that change the order of the magnitude of length x width x height by more than a little
def scale_order_preserving_objective():
	pass

def soft_relational_descriptor():
	pass

def align_max_variance_axes_descriptor():
	pass

def two_sided_chamfer_distance(): 
	pass

def semantic_alignment(): 
	pass


#take in a list of all these possible objectives 
#fits the handle pcl with each of them
#displays the results in a grid