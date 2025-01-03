import numpy as np
import matplotlib.pyplot as plt
from segment_anything import build_sam, SamPredictor 
import math
import open3d as o3d

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