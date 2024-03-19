import trimesh
import numpy as np


# Generates a mug mesh from the provided dictionary of parameters
# With specified height, radius, handle shape, and handle placement
def generate_mug(mug_params, return_parts=False):
    # Generate the cup part of the mug
    cup = generate_cup(mug_params["cup_radius"], mug_params["cup_height"])
    # Generate the handle part of the mug
    handle = generate_handle(
        mug_params["handle_radius"],
        mug_params["handle_height"],
        mug_params["handle_width"],
    )
    # Translate the handle to the correct position
    mug, cup, handle = attach_handle_to_mug(
        handle, cup, mug_params["attachment_height"]
    )
    # Return the mug mesh
    if return_parts:
        return mug, {"cup": cup, "handle": handle}
    else:
        return mug


# Takes the radius and height of a cup part and generates a mesh with those parameters
def generate_cup(radius, height):
    # Create a cylinder with the given parameters
    cup = trimesh.creation.cylinder(radius, height)
    cup_removal = trimesh.creation.cylinder(radius - 0.05, height)
    cup_removal.apply_translation([0, 0, 0.05])
    cup = trimesh.boolean.difference([cup, cup_removal])
    # Return the mesh
    return cup


# Takes the height, radius, and width of a mug handle and generates a mesh with those dimensions
# The handle is comprised of three cylinders joined to make a 'U' shape
def generate_handle(radius, height, width):
    # Create the three cylinders that make up the handle
    cylinder1 = trimesh.creation.cylinder(radius, height)
    cylinder2 = trimesh.creation.cylinder(radius, width)
    cylinder3 = trimesh.creation.cylinder(radius, width)
    # Translate the cylinders so that they are all connected

    # rotate cylinders 2 and 3 to be perpendicular to cylinder 1

    cylinder2.apply_transform(
        trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
    )
    cylinder3.apply_transform(
        trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
    )
    cylinder2.apply_translation([0, width / 2 - radius, height / 2])
    cylinder3.apply_translation([0, width / 2 - radius, -height / 2])
    # Join the three cylinders into a single mesh
    handle = trimesh.boolean.union([cylinder1, cylinder2, cylinder3])
    handle.apply_transform(
        trimesh.transformations.rotation_matrix(np.pi / 2, [0, 0, 1])
    )
    handle.apply_translation([width - radius, 0, 0])
    # Return the mesh
    return handle


# Attaches the mesh for a mug handle to a mug cup at the specified attachment height
def attach_handle_to_mug(handle_mesh, cup_mesh, attachment_height, epsilon=0.005):
    handle_bounds = handle_mesh.bounds
    cup_bounds = cup_mesh.bounds
    cup_diameter = cup_bounds[1, 0] - cup_bounds[0, 0]
    handle_width = handle_bounds[1, 0] - handle_bounds[0, 0]
    # Move the cup and mesh by half their bounds so the meshes align for attachment
    print(np.mean(np.asarray(handle_mesh.vertices), 0))
    handle_mesh.apply_translation([cup_diameter / 2 - epsilon, 0, 0])

    # Translate the handle to the attachment height
    handle_mesh.apply_translation([0, 0, cup_bounds[1, 2] - attachment_height])
    # Join the handle to the mug
    print(np.mean(np.asarray(handle_mesh.vertices), 0))
    print(" ")
    mug_mesh = trimesh.boolean.union([cup_mesh, handle_mesh])
    mesh_vertices = np.concatenate([cup_mesh.vertices, handle_mesh.vertices])

    mesh_points = np.concatenate(
        [
            trimesh.sample.sample_surface_even(cup_mesh, 1000),
            trimesh.sample.sample_surface_even(handle_mesh, 1000),
        ]
    )

    # Return the mesh
    return mug_mesh, mesh_vertices, mesh_points  # cup_mesh, handle_mesh


if __name__ == "__main__":
    # Tests to verify that mug generation code makes meshes that can be viewed
    cup_mesh = generate_cup(0.5, 1)
    handle_mesh = generate_handle(0.1, 0.5, 0.5)

    mug_mesh = attach_handle_to_mug(handle_mesh, cup_mesh, 0.5).show()
    mug_mesh.show()
