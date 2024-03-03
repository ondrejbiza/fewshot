from cycpd import deformable_registration as DeformableRegistration
import numpy as np
import torch

from sklearn.neighbors import NearestNeighbors

import numbers

class ConstrainedDeformableRegistration(DeformableRegistration):
    """
    Constrained deformable registration.

    Attributes
    ----------
    alpha: float (positive)
        Represents the trade-off between the goodness of maximum likelihood fit and regularization.

    beta: float(positive)
        Width of the Gaussian kernel.

    e_alpha: float (positive)
        Reliability of correspondence priors. Between 1e-8 (very reliable) and 1 (very unreliable)
    
    source_id: numpy.ndarray (int) 
        Indices for the points to be used as correspondences in the source array

    target_id: numpy.ndarray (int) 
        Indices for the points to be used as correspondences in the target array

    """

    def __init__(self, e_alpha = None, source_id = None, target_id= None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if e_alpha is not None and (not isinstance(e_alpha, numbers.Number) or e_alpha <= 0):
            raise ValueError(
                "Expected a positive value for regularization parameter e_alpha. Instead got: {}".format(e_alpha))
        
        if type(source_id) is not np.ndarray or source_id.ndim != 1:
            raise ValueError(
                "The source ids (source_id) must be a 1D numpy array of ints.")
        
        if type(target_id) is not np.ndarray or target_id.ndim != 1:
            raise ValueError(
                "The target ids (target_id) must be a 1D numpy array of ints.")

        self.e_alpha = 1e-8 if e_alpha is None else e_alpha
        self.source_id = source_id
        self.target_id = target_id
        self.P_tilde = np.zeros((self.M, self.N))
        self.P_tilde[self.source_id, self.target_id] = 1
        self.P1_tilde = np.sum(self.P_tilde, axis=1)
        self.PX_tilde = np.dot(self.P_tilde, self.X)

    def update_transform(self):
        """
        Calculate a new estimate of the deformable transformation.
        See Eq. 22 of https://arxiv.org/pdf/0905.2635.pdf.

        """
        if self.low_rank is False:
            A = np.dot(np.diag(self.P1), self.G) + \
                self.sigma2*(1/self.e_alpha)*np.dot(np.diag(self.P1_tilde), self.G) + \
                self.alpha * self.sigma2 * np.eye(self.M)
            B = self.PX - np.dot(np.diag(self.P1), self.Y) + self.sigma2*(1/self.e_alpha)*(self.PX_tilde - np.dot(np.diag(self.P1_tilde), self.Y)) 
            self.W = np.linalg.solve(A, B)

        elif self.low_rank is True:
            # Matlab code equivalent can be found here:
            # https://github.com/markeroon/matlab-computer-vision-routines/tree/master/third_party/CoherentPointDrift
            dP = np.diag(self.P1) + self.sigma2*(1/self.e_alpha)*np.diag(self.P1_tilde)
            dPQ = np.matmul(dP, self.Q)
            F = self.PX - np.dot(np.diag(self.P1), self.Y) + self.sigma2*(1/self.e_alpha)*(self.PX_tilde - np.dot(np.diag(self.P1_tilde), self.Y)) 

            self.W = 1 / (self.alpha * self.sigma2) * (F - np.matmul(dPQ, (
                np.linalg.solve((self.alpha * self.sigma2 * self.inv_S + np.matmul(self.Q.T, dPQ)),
                                (np.matmul(self.Q.T, F))))))
            QtW = np.matmul(self.Q.T, self.W)
            self.E = self.E + self.alpha / 2 * np.trace(np.matmul(QtW.T, np.matmul(self.S, QtW)))






def chamfer_distance(x, y, metric='l2', direction='y_to_x'):
    """Chamfer distance between two point clouds
    Parameters
    ----------
    x: numpy array [n_points_x, n_dims]
        first point cloud
    y: numpy array [n_points_y, n_dims]
        second point cloud
    metric: string or callable, default ‘l2’
        metric to use for distance computation. Any metric from scikit-learn or scipy.spatial.distance can be used.
    direction: str
        direction of Chamfer distance.
            'y_to_x':  computes average minimal distance from every point in y to x
            'x_to_y':  computes average minimal distance from every point in x to y
            'bi': compute both
    Returns
    -------
    chamfer_dist: float
        computed bidirectional Chamfer distance:
            sum_{x_i \in x}{\min_{y_j \in y}{||x_i-y_j||**2}} + sum_{y_j \in y}{\min_{x_i \in x}{||x_i-y_j||**2}}
    """
    
    if direction=='y_to_x':
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        chamfer_dist = np.mean(min_y_to_x)
    elif direction=='x_to_y':
        y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = np.mean(min_x_to_y)
    elif direction=='bi':
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = np.mean(min_y_to_x) + np.mean(min_x_to_y)
    else:
        raise ValueError("Invalid direction type. Supported types: \'y_x\', \'x_y\', \'bi\'")
        
    return chamfer_dist


# def cost_batch_pt(source, target): 
#     """Calculate the one-sided Chamfer distance between two batches of point clouds in pytorch."""
#     # B x N x K
#     # print(source.size())
#     # print(target.size())
#     diff = torch.sqrt(torch.sum(torch.square(source[:, :, None] - target[:, None, :]), dim=3))
#     diff_flat = diff.view(diff.shape[0] * diff.shape[1], diff.shape[2])
#     c_flat = diff_flat[list(range(len(diff_flat))), torch.argmin(diff_flat, dim=1)]
#     c = c_flat.view(diff.shape[0], diff.shape[1])
#     return torch.mean(c, dim=1)

def cost_batch_pt(source, target): 
    """Calculate the one-sided Chamfer distance between two batches of point clouds in pytorch."""
    # B x N x K
    # print(source.size())
    # print(target.size())
    diff = np.sqrt(np.sum(np.square(source[:, :, None] - target[:, :, None]), dim=3))
    diff_flat = np.view(diff.shape[0] * diff.shape[1], diff.shape[2])
    c_flat = diff_flat[list(range(len(diff_flat))), np.argmin(diff_flat, dim=1)]
    c = np.view(diff.shape[0], diff.shape[1])
    return np.mean(c, dim=1)

class ContactAwareDeformableRegistration(DeformableRegistration):
    def __init__(self, source_contacts, target_contacts, contact_weight, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.source_contacts = target_contacts
        self.target_contacts = source_contacts
        self.contact_weight = contact_weight

    def update_transform(self):
        """
        Update the transform parameters.
        """
        if self.low_rank is False:
            tic = time.time()
            A = np.dot(np.diag(self.P1), self.G) + self.alpha * self.sigma2 * np.eye(self.M)
            toc = time.time()
            self.A_times.append(toc - tic)
            tic = time.time()
            B = self.PX - np.dot(np.diag(self.P1), self.Y)
            toc = time.time()
            self.B_times.append(toc - tic)
            tic = time.time()
            self.W = np.linalg.solve(A, B)
            toc = time.time()
            self.solve_times.append(toc - tic)
        else:
            dP = np.diag(self.P1)
            dPQ = np.matmul(dP, self.Q)
            F = self.PX - np.matmul(dP, self.Y)
            print(dP.shape)
            print(self.X.shape)
            print(self.Y.shape)
            print(self.PX.shape)
            print(np.matmul(dP, self.Y).shape)
            print(F.shape)
            exit(0)

            self.W = (
                1
                / (self.alpha * self.sigma2)
                * (
                    F
                    - np.matmul(
                        dPQ,
                        (
                            np.linalg.solve(
                                (self.alpha * self.sigma2 * self.inv_S + np.matmul(self.Q.T, dPQ)),
                                (np.matmul(self.Q.T, F)),
                            )
                        ),
                    )
                )
            )
            QtW = np.matmul(self.Q.T, self.W)
            self.E = self.E + self.alpha / 2 * np.trace(np.matmul(QtW.T, np.matmul(self.S, QtW)))
            # self.err = abs((self.E - self.E_old) / self.E)

            print("data")
            print(chamfer_distance(self.Y[self.target_contacts], self.X[self.source_contacts] + np.dot(self.G, self.W)[self.source_contacts]))
            #my thing
            self.E += chamfer_distance(self.Y[self.target_contacts], self.X[self.source_contacts] + np.dot(self.G, self.W)[self.source_contacts]) * self.contact_weight
            

            self.err = abs(self.E - self.E_old)
            # The absolute difference is more conservative (does more iterations) than the line above it which
            # is calculating the normalized change in the E(L). This calculation was changed to match the matlab
            # code created for low_rank matrices.


        