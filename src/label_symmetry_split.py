



def optimize_y_plane(part, z_plane):
	initial_plane_normal = z_plane.coefs_
	plane_normal_param = nn.Parameter(cp.deepcopy(plane_normal), requires_grad=True)
	optimizer = optim.Adam(plane_normal_param, lr=self.lr)
#load the object and the warps

#get the planes