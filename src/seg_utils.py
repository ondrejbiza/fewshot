import numpy as np
import matplotlib.pyplot as plt
from segment_anything import build_sam, SamPredictor 
import math
import open3d as o3d
import torch
import torchvision.transforms as transforms
import matplotlib
from PIL import Image
from pytorch_lightning import seed_everything
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from matplotlib.patches import ConnectionPatch

COLORS = {
    "Deep_Red": [1.0, 0.0, 0.0],
    "Deep_Green": [0.0, 0.5, 0.0],
    "Deep_Blue": [0.0, 0.0, 1.0],
    "Deep_Purple": [0.5, 0.0, 0.5],
    "Deep_Cyan": [0.0, 0.5, 0.5],
    "Deep_Magenta": [0.5, 0.0, 0.5],
    "Deep_Brown": [0.35, 0.16, 0.14],
    "Deep_Gray": [0.2, 0.2, 0.2],
    "Deep_Black": [0.0, 0.0, 0.0],
    "Deep_Maroon": [0.5, 0.0, 0.25],
    "Orange": [1.0, 0.5, 0.0],
    "Yellow": [1.0, 1.0, 0.0],
    "Green": [0.0, 1.0, 0.0],
    "Red": [1.0, 0.0, 0.0],
    "Blue": [0.0, 0.0, 1.0],
    "Pink": [1.0, 0.0, 1.0],
    "Teal": [0.0, 1.0, 1.0],
    "Lime": [0.5, 1.0, 0.0],
    "Gold": [1.0, 0.84, 0.0],
    "Crimson": [0.86, 0.08, 0.24],
    "Navy": [0.0, 0.0, 0.5],
    "Indigo": [0.29, 0.0, 0.51],
    "Olive": [0.5, 0.5, 0.0],
    "Coral": [1.0, 0.5, 0.31],
    "Black": [0.0, 0.0, 0.0],
    "light_gray": [0.8, 0.8, 0.8],
    "Royal_Blue":(0.1, 0.1, 0.5)
}

# Create a reverse lookup for color names
COLOR_LOOKUP = {tuple(value): key for key, value in COLORS.items()}

def trans_points_crop_to_full(points, min_x, min_y):
  """Transforms coordinates from a cropped image to a full image."""
  new_points = np.copy(points)
  new_points[:, 0] += min_x
  new_points[:, 1] += min_y
  return new_points

def trans_mask_full_to_crop(mask, min_y, max_y, min_x, max_x):
  """Crops a segmentation mask."""
  return mask[min_y: max_y, min_x: max_x]

from ipywidgets import widgets 

class ClickSegGui():
    def __init__(self):
        self.fig = None
        self.ax = None
        self.textbox = None
        self.part = None
        self.camera_name = None
        self.points = None
        self.color = None
        self.colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'pink']


    def on_click(self, event):
        ix, iy = int(event.xdata), int(event.ydata)
        print(f"Coordinates: x={ix}, y={iy}")
        self.points.append([ix, iy])

        # Plot an 'X' marker at the clicked coordinates
        self.ax.plot(ix, iy, marker='x', markersize=10, color=self.color, zorder=2)
        self.fig.canvas.draw()  # Update the figure to show the new marker

    def plot_map_with_points(self, all_image_points, gripper_points, camera_names, part_names, images):
        self.part = part_names[0]
        self.camera_name = camera_names[0]
        self.points = all_image_points[self.camera_name][self.part]
        self.color = 'red'

        buttons = widgets.RadioButtons(
            options=part_names + ['gripper'],
            disabled=False
        )
        
        display(buttons)

        def radio(value):
            part = value['new']
            if part == 'gripper':
                self.points = gripper_points[self.camera_name]
                self.color = self.colors[len(part_names)]
            else:
                self.points = all_image_points[self.camera_name][value['new']]
                self.color = self.colors[part_names.index(value['new'])]
            
            
        buttons.observe(radio, names = 'value')
        
        buttons2 = widgets.RadioButtons(
            options=camera_names,
            disabled=False
        )
        
        display(buttons2)

        def radio2(value):
            self.camera_name=value['new']
            part_points = all_image_points[self.camera_name]
            self.points = part_points[self.part]
            
            for p in part_names: 
                all_image_points[self.camera_name][p] = []
            gripper_points[self.camera_name] = []
            self.ax.imshow(images[self.camera_name], cmap=cmap, zorder=1)
            self.fig.canvas.draw()
            
        buttons2.observe(radio2, names='value')
        
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        cmap = matplotlib.colors.ListedColormap(['black', 'white'])
        #textbox = matplotlib.widgets.TextBox(ax, 'temp',)

        self.ax.imshow(images[self.camera_name], cmap=cmap, zorder=1)

        #plt.legend()
        plt.show()

        # Connect the click event
        cid = self.fig.canvas.mpl_connect('button_press_event', self.on_click)


def crop_to_bounds(min_x, max_x, min_y, max_y, height, width):
    min_x = max(0, min_x)
    min_y = max(0, min_y)
    
    max_x = min(width, max_x)
    max_y = min(height, max_y)

    return min_x, max_x, min_y, max_y

def top_k_pixels(image, K):
    """
    Given an image as a NumPy array and a number K, this function returns the K pixel coordinates (x, y)
    and their values with the highest values in the image """
    # Flatten the image array and get the indices of the top K values
    flat_image = image.flatten()
    indices = np.argpartition(flat_image, -K)[-K:]
    
    # Get the actual values
    values = flat_image[indices]
    
    # Convert flat indices back to 2D coordinates
    rows, cols = image.shape[:2] # Assuming image could potentially have more dimensions (like color channels)
    coordinates = np.array([np.unravel_index(index, (rows, cols)) for index in indices])
    
    # Combine coordinates and values into a list of tuples
    return coordinates, values

def farthest_point_sampling(mask, K):
    """
    Selects K points from a binary mask that are maximally separated from each other.
    The function begins by selecting a random point from the masked area and then
    iteratively chooses the next point that is the farthest away from all previously
    selected points. This process ensures a spread of points across the masked region.
    
    Parameters:
    mask (numpy.ndarray): A boolean array where True indicates pixels eligible for selection.
    K (int): The number of points to select.
    
    Returns:
    numpy.ndarray: An array of coordinates for the K selected points.
    """
    mask = mask.numpy()
    # Ensure the input mask is a boolean array
    mask = mask.astype(bool)

    # Extract the indices of the masked region
    masked_indices = np.argwhere(mask)

    # Select the first point randomly from the masked indices
    selected_indices = [masked_indices[np.random.choice(len(masked_indices))]]

    for _ in range(K-1):
        # Compute distances from all masked points to all selected points
        distances = np.sqrt(((masked_indices[:, None, :] - np.array(selected_indices)[None, :, :]) ** 2).sum(axis=2))

        # Get the minimum distance to the selected points for each masked point
        min_distances = distances.min(axis=1)

        # Choose the point that has the maximum of the minimum distances
        next_index = masked_indices[np.argmax(min_distances)]
        selected_indices.append(next_index)
        
    coordinates = np.array(selected_indices)
    return coordinates

class Dinov2Matcher:

  def __init__(self, repo_name="facebookresearch/dinov2", model_name="dinov2_vitb14", img_input_size=448, device="cuda"):
    self.repo_name = repo_name
    self.model_name = model_name
    self.smaller_edge_size = img_input_size
    self.device = device
    self.model = torch.hub.load(repo_or_dir=repo_name, model=model_name).to(self.device)
    self.model.eval()
    print(img_input_size)

    self.transform = transforms.Compose([
        transforms.Resize(size=img_input_size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), # imagenet defaults
      ])

  # https://github.com/facebookresearch/dinov2/blob/255861375864acdd830f99fdae3d9db65623dafe/notebooks/features.ipynb
  def prepare_image(self, rgb_image_numpy):
    image = Image.fromarray(rgb_image_numpy)
    #print(rgb_image_numpy.shape)
    image_tensor = self.transform(image)
    #print(image_tensor.shape)
    resize_scale = image.width / image_tensor.shape[2]

    # Crop image to dimensions that are a multiple of the patch size
    height, width = image_tensor.shape[1:] # C x H x W
    cropped_width, cropped_height = width - width % self.model.patch_size, height - height % self.model.patch_size # crop a bit from right and bottom parts
    image_tensor = image_tensor[:, :cropped_height, :cropped_width]

    grid_size = (cropped_height // self.model.patch_size, cropped_width // self.model.patch_size)
    return image_tensor, grid_size, resize_scale

  def prepare_mask(self, mask_image_numpy, grid_size, resize_scale):
    cropped_mask_image_numpy = mask_image_numpy[:int(grid_size[0]*self.model.patch_size*resize_scale), :int(grid_size[1]*self.model.patch_size*resize_scale)]
    image = Image.fromarray(cropped_mask_image_numpy)
    resized_mask = image.resize((grid_size[1], grid_size[0]), resample=Image.Resampling.NEAREST)
    resized_mask = np.asarray(resized_mask).flatten()
    return resized_mask

  def extract_features(self, image_tensor):
    with torch.inference_mode():
      image_batch = image_tensor.unsqueeze(0).to(self.device)
      tokens = self.model.get_intermediate_layers(image_batch)[0].squeeze()
    return tokens.cpu().numpy()

  def idx_to_source_position(self, idx, grid_size, resize_scale):
    row = (idx // grid_size[1])*self.model.patch_size*resize_scale + (self.model.patch_size * resize_scale) / 2
    col = (idx % grid_size[1])*self.model.patch_size*resize_scale + (self.model.patch_size * resize_scale) / 2
    return row, col

  def get_embedding_visualization(self, tokens, grid_size, resized_mask=None):
    seed_everything(0)
    pca = PCA(n_components=3)

    if resized_mask is not None:
      print(tokens.shape)
      tokens = tokens[resized_mask]
      print(tokens.shape)

    reduced_tokens = pca.fit_transform(tokens.astype(np.float32))

    if resized_mask is not None:
      tmp_tokens = np.zeros((*resized_mask.shape, 3), dtype=reduced_tokens.dtype)
      tmp_tokens[resized_mask] = reduced_tokens
      reduced_tokens = tmp_tokens

    reduced_tokens = reduced_tokens.reshape((*grid_size, -1))
    normalized_tokens = (reduced_tokens-np.min(reduced_tokens))/(np.max(reduced_tokens)-np.min(reduced_tokens))
    return normalized_tokens


  def visualize_cosine_values(tokens, grid_size):
    tokens = tokens.reshape((*grid_size, -1))
    normalized_tokens = (tokens-np.min(tokens))/(np.max(tokens)-np.min(tokens))
    return normalized_tokens


def top_k_pixels(image, K):
    """
    Given an image as a NumPy array and a number K, this function returns the K pixel coordinates (x, y)
    and their values with the highest values in the image """
    # Flatten the image array and get the indices of the top K values
    flat_image = image.flatten()
    indices = np.argpartition(flat_image, -K)[-K:]
    
    # Get the actual values
    values = flat_image[indices]
    
    # Convert flat indices back to 2D coordinates
    rows, cols = image.shape[:2] # Assuming image could potentially have more dimensions (like color channels)
    coordinates = np.array([np.unravel_index(index, (rows, cols)) for index in indices])
    
    # Combine coordinates and values into a list of tuples
    return coordinates, values

def farthest_point_sampling(mask, K):
    """
    Selects K points from a binary mask that are maximally separated from each other.
    The function begins by selecting a random point from the masked area and then
    iteratively chooses the next point that is the farthest away from all previously
    selected points. This process ensures a spread of points across the masked region.
    
    Parameters:
    mask (numpy.ndarray): A boolean array where True indicates pixels eligible for selection.
    K (int): The number of points to select.
    
    Returns:
    numpy.ndarray: An array of coordinates for the K selected points.
    """
    mask = mask.numpy()
    # Ensure the input mask is a boolean array
    mask = mask.astype(bool)

    # Extract the indices of the masked region
    masked_indices = np.argwhere(mask)

    # Select the first point randomly from the masked indices
    selected_indices = [masked_indices[np.random.choice(len(masked_indices))]]

    for _ in range(K-1):
        # Compute distances from all masked points to all selected points
        distances = np.sqrt(((masked_indices[:, None, :] - np.array(selected_indices)[None, :, :]) ** 2).sum(axis=2))

        # Get the minimum distance to the selected points for each masked point
        min_distances = distances.min(axis=1)

        # Choose the point that has the maximum of the minimum distances
        next_index = masked_indices[np.argmax(min_distances)]
        selected_indices.append(next_index)
        
    coordinates = np.array(selected_indices)
    return coordinates


def plot_points_on_image(image, points, point_color='r', point_marker='.'):
    # Display the image
    plt.imshow(image, cmap='gray')  # Assuming the image is grayscale for simplicity
    plt.axis('off')  # Turn off axis numbers and ticks
    
    # Plot each point
    for x, y in points:
        plt.plot(y, x, point_marker, color=point_color)  # Note: plt uses (y, x) for image coordinates
    plt.show()


def resize_coordinates(coords_resized, original_dims, resized_dims):
    """
    Maps an array of pixel coordinates from a resized image back to their corresponding
    locations in the original image using vectorized operations.

    Parameters:
    - coords_resized: An array of tuples or a NumPy array, each containing the (x, y) coordinates in the resized image.
    - original_dims: A tuple (original_width, original_height) of the original image dimensions.
    - resized_dims: A tuple (resized_width, resized_height) of the resized image dimensions.

    Returns:
    - A NumPy array of mapped coordinates in the original image.
    """
    original_width, original_height = original_dims
    resized_width, resized_height = resized_dims
    
    # Convert the list of coordinates to a NumPy array for vectorized operations
    coords_resized_np = np.array(coords_resized)
    
    # Calculate the scaling factors for width and height
    scale_x = original_width / resized_width
    scale_y = original_height / resized_height
    
    # Apply scaling to all coordinates at once
    coords_original_np = coords_resized_np * np.array([scale_x, scale_y])
    
    # Round and convert to integers
    coords_original_np = np.round(coords_original_np).astype(int)
    
    return coords_original_np


def load_sam(ckpt_filename,device):
    sam = build_sam(checkpoint=ckpt_filename)
    sam.to(device=device)
    sam_predictor = SamPredictor(sam)
    return sam_predictor


def segment(seg_model, image, input_points, input_labels, multimask_flag=False, sam_embedding=None):
    image = np.asarray(image)
    seg_model.set_image(image) #get image embedding and retain copy in model
    if sam_embedding is None:
        sam_embedding = seg_model.get_image_embedding() #get image embedding

    masks, scores, logits = seg_model.predict(
        point_coords=input_points,
        point_labels=input_labels,    # point labels: 1 is point to include, label of 0 is point to exclude 
        multimask_output=multimask_flag,
    )

    return masks, scores, sam_embedding


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)      


def rotation_matrix_from_quaternion(quaternion):
   # Step 1: Normalize the quaternion
   quaternion = quaternion / np.linalg.norm(quaternion)

   # Step 2: Extract quaternion components
   x, y, z, w = quaternion

   # Step 3: Construct rotation matrix
   R = np.array([[1 - 2 * y ** 2 - 2 * z ** 2, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
               [2 * x * y + 2 * w * z, 1 - 2 * x ** 2 - 2 * z ** 2, 2 * y * z - 2 * w * x],
               [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x ** 2 - 2 * y ** 2]])
   return R

def backproject_worldframe(pixel_x, pixel_y, pixel_depth, CX, CY, FX, FY, rotation_matrix, position):
    #Compute 3d position of pixel(i,j) in camera frame/cordinate system.
    z_RGB = pixel_depth
    x_RGB = (pixel_x - CX) * z_RGB / FX
    y_RGB = (pixel_y - CY) * z_RGB / FY   

    bad_z = z_RGB == 0 #if z_RGB is 0, the depth was 0
    if math.isnan(z_RGB):
        bad_z = True
    # transformed_xyz = np.array([x_RGB, y_RGB, z_RGB])
    transformed_xyz = np.matmul(rotation_matrix , np.array([x_RGB, y_RGB, z_RGB])) + position

    return(transformed_xyz, bad_z)

def get_mask_pixels_depth(mask,depth_img):
    mask = np.asarray(mask)
    depth_img = np.asarray(depth_img)

    boolean_mask = (mask == True)

    #get coordinates of pixels where mask is true
    mask_pixel_coords = np.argwhere(boolean_mask)

    #id array of depth values at pixels where mask is true
    depths_associated_with_mask = depth_img[boolean_mask] 

    # get average of depth values that are not zero
    if isinstance(depths_associated_with_mask,np.ndarray):
        non_zero_depths_associated_with_mask = depths_associated_with_mask[depths_associated_with_mask != 0]
        if len(non_zero_depths_associated_with_mask) == 0:
            avg_non_zero_depths_associated_with_mask = 0.0
        else:
            avg_non_zero_depths_associated_with_mask = non_zero_depths_associated_with_mask.mean()

    return mask_pixel_coords,depths_associated_with_mask,avg_non_zero_depths_associated_with_mask


def get_all_mask3dpositions(all_mask_pixels, all_mask_depths, rotation_matrix, position, CX, CY, FX, FY, diff_threshold = 0.5):
    mask3dpositions=[]
    avg_nonzero_depth = np.mean(all_mask_depths[all_mask_depths!=0])

    for mask_pixel,mask_depth in zip(all_mask_pixels,all_mask_depths):
        center_y, center_x = mask_pixel
        pixel_depth = mask_depth
        if pixel_depth == 0 or abs(pixel_depth-avg_nonzero_depth)>diff_threshold: #if depth is 0 or too different from mean of non zero depths in mask (by diff_threshold) then skip
            continue

        transformed_xyz,_ = backproject_worldframe(center_y, center_x, pixel_depth, CX, CY, FX, FY, rotation_matrix, position)
        mask3dpositions.append(transformed_xyz)

    return mask3dpositions

def segment_pointcloud(point_cloud, mask_positions_of_interest, threshold_percentage=1, kdtree_search_radius=0.5, use_colors=False):
    """
    Visualize a point cloud with a heatmap color coding, considering a threshold for points of interest.

    :param point_cloud: open3d.geometry.PointCloud object
    :param mask_positions_of_interest: dict with object names and positions of interest that will have unique colors
    :param threshold_percentage: float, percentage (0-1) of points of interest to be considered for coloring
    :return: None, displays the visualization
    """
    global COLORS
    global COLOR_LOOKUP

    num_points = np.asarray(point_cloud.points).shape[0]
    if use_colors:
            colors = np.asarray(point_cloud.colors) #Keep original colors
    else:
            colors = np.full((num_points, 3), COLORS["light_gray"]) # Base color for most points

    # Build a KDTree for efficient nearest neighbor search
    kdtree = o3d.geometry.KDTreeFlann(point_cloud)

    #Try and get matching colors for each object if object name has a color
    object_colors={}
    unique_color_values = list(COLORS.values())
    object_names = list(mask_positions_of_interest.keys())

    for name in object_names:
        name_parts = name.split("_")
        for part in name_parts:
            for color_name in COLORS:
                if part.lower() in color_name.lower():
                    color_value = COLORS[color_name]
                    print("Found color for",name," || ","assigning color: ",color_name)
                    object_colors[name] = color_value  
                    #remove color from list of available colors
                    unique_color_values.remove(color_value)
                    object_names.remove(name)
                    break
    # Assign a unique color to each remaining object
    for i, obj_name in enumerate(object_names):
        object_colors[obj_name] = unique_color_values[i % len(unique_color_values)]


    # Iterate through each object type and its positions
    for object_name, positions in mask_positions_of_interest.items():
        color_value = tuple(object_colors[object_name])
        color_name = COLOR_LOOKUP[color_value]
        
        if positions:
            # Convert list of positions to numpy array for efficient processing
            positions_array = np.vstack(positions)

            # Determine how many points to consider based on the threshold percentage
            num_points_of_interest = int(len(positions_array) * threshold_percentage)
            
            # Randomly select a subset of points based on the threshold
            if threshold_percentage < 1.0:
                indices = np.random.choice(len(positions_array), num_points_of_interest, replace=False)
                positions_array = positions_array[indices]
            
            print(f"Object:{object_name:10s}|| Color:{color_name:5s} || Considering {num_points_of_interest} out of {len(positions)} points.")

            # For efficiency, use batch query for KDTree search
            total_colored_points = 0
            for position in positions_array:
                # print(f"Position: {position}")
                [k, idx, _] = kdtree.search_radius_vector_3d(position, kdtree_search_radius) #find neighbors with distance less than kdtree_search_radius
                # [k, idx, _] = kdtree.search_knn_vector_3d(position, 1000) #find 1000 closest neighbors
                if k > 0:
                    colors[idx, :] = color_value
                    total_colored_points += k  # Sum the number of points colored
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    return point_cloud