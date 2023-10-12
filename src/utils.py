import os

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

def load_image(filename):
    return nib.load(filename)

# def xyz_affine(voxel_size, origin):
#     """Create an affine transformation matrix from voxel size and origin."""
#     return np.diag([*voxel_size, 1]) * np.eye(4, dtype=np.float32) @ np.array([origin[0], origin[1], origin[2], 1])

# def plot_map(ax, affine, title=None):
#     """Plot a map of a NIfTI image.

#     Args:
#         image: A NIfTI image.
#         affine: The affine transformation matrix of the image.
#         title: The title of the plot.
#     """

#     fig, ax = plt.subplots(1, 1)
#     ax.imshow(image.get_data(), cmap='gray', origin='lower')
#     ax.set_title(title)
#     ax.set_aspect('equal')
#     ax.axis('off')
#     plt.show()

def process_output(output: str, save_dir: str = None):
    # create output directory
    if save_dir is not None:
        if not save_dir.startswith("/"):
            if not save_dir.startswith("~"):
                save_dir = f"./{save_dir}"
        os.makedirs(f"{save_dir}", exist_ok=True)

    # process output
    ext = ["png", "svg"]
    if output is None:
        output = "brainbow-output"

    output_split = output.split(".")
    if len(output_split) > 1:
        if output_split[-1] == "png":
            output = ".".join(output_split[0 : len(output_split) - 1])
            ext = ["png"]
        elif output_split[-1] == "svg":
            output = ".".join(output_split[0 : len(output_split) - 1])
            ext = ["svg"]

    return output, ext