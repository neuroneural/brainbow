""" A few helper functions, many of which come from nipy"""

import os
import warnings
import numpy as np
from scipy import ndimage


def process_output_path(output: str, save_dir: str = None):
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


def is_numlike(obj):
    """Return True if `obj` looks like a number"""
    try:
        obj + 1
    except:
        return False
    return True


def is_iterable(obj):
    """Return True if `obj` is iterable"""
    try:
        iter(obj)
    except TypeError:
        return False
    return True


def get_bounds(shape, affine):
    """Return the world-space bounds occupied by an array given an affine."""
    adim, bdim, cdim = shape
    adim -= 1
    bdim -= 1
    cdim -= 1
    # form a collection of vectors for each 8 corners of the box
    box = np.array(
        [
            [0.0, 0, 0, 1],
            [adim, 0, 0, 1],
            [0, bdim, 0, 1],
            [0, 0, cdim, 1],
            [adim, bdim, 0, 1],
            [adim, 0, cdim, 1],
            [0, bdim, cdim, 1],
            [adim, bdim, cdim, 1],
        ]
    ).T
    box = np.dot(affine, box)[:3]
    return list(zip(box.min(axis=-1), box.max(axis=-1)))


def get_mask_bounds(mask, affine):
    """Return the world-space bounds occupied by a mask given an affine.

    Notes
    -----

    The mask should have only one connect component.

    The affine should be diagonal or diagonal-permuted.
    """
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = get_bounds(mask.shape, affine)
    slices = ndimage.find_objects(mask)
    if len(slices) == 0:
        warnings.warn("empty mask", stacklevel=2)
    else:
        x_slice, y_slice, z_slice = slices[0]
        x_width, y_width, z_width = mask.shape
        xmin, xmax = (
            xmin + x_slice.start * (xmax - xmin) / x_width,
            xmin + x_slice.stop * (xmax - xmin) / x_width,
        )
        ymin, ymax = (
            ymin + y_slice.start * (ymax - ymin) / y_width,
            ymin + y_slice.stop * (ymax - ymin) / y_width,
        )
        zmin, zmax = (
            zmin + z_slice.start * (zmax - zmin) / z_width,
            zmin + z_slice.stop * (zmax - zmin) / z_width,
        )

    return xmin, xmax, ymin, ymax, zmin, zmax
