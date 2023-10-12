import sys

import json
import numpy as np
import nibabel as nib

from scipy.ndimage import affine_transform, shift
from nibabel.processing import resample_from_to, resample_to_output
from nibabel.affines import apply_affine, to_matvec, from_matvec
from scipy.ndimage import zoom


# from nibabel.viewers import OrthoSlicer3D

from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# from nipy import load_image
# from nipy.core.api import xyz_affine
# from nipy.labs.viz import plot_map, coord_transform


# from pylab import savefig, subplot, subplots_adjust, figure, rc, text


from src.colormaps import CMAPS, DSGN
from src.utils import process_output
from src.slicer import OrthoSlicer3D


def process_image(
    NIFTI,
    ANAT,
    SGN,
    output: str = None,
    save_dir: str = None,
    thr: float = 2.0,
    dpi: int = 300,
    iscale: int = 3,
):
    # create output directory (if needed) and define output extension
    output, ext = process_output(output, save_dir)

    print("Loading and processing the nifti files")

    nifti = nib.load(NIFTI)
    nifti_affine = nifti.affine
    nifti_data = nifti.get_fdata()
    # for idx in range(nifti_data.shape[-1]):

    anat = nib.load(ANAT)
    anat_affine = anat.affine
    anat_data = anat.get_fdata()

    # new_nifti_data = []
    # new_nifti_affines = []

    # for idx in range(5):
    #     single_component = nifti_data[:, :, :, idx]
    #     single_component = nib.Nifti1Image(
    #         single_component, nifti.affine, header=nifti.header
    #     )
    #     single_component = resample_from_to(single_component, anat)

    #     new_nifti_data += [single_component.get_fdata()]
    #     new_nifti_affines += [single_component.affine]

    # # # upscale nifti images
    # reference = nib.Nifti1Image(
    #     nifti_data[:, :, :, 0], nifti.affine, header=nifti.header
    # )
    # anat = resample_from_to(anat, reference)
    # anat_data = anat.get_fdata()
    # anat_affine = anat.affine

    # nifti_voxels = np.diag(to_matvec(nifti.affine)[0])
    # anat_voxels = np.diag(to_matvec(anat.affine)[0])
    # ratio = nifti_voxels / anat_voxels
    # nifti_affine = from_matvec(np.eye(3), to_matvec(nifti.affine)[1])
    # # for idx in range(nifti_data.shape[-1]):
    # for idx in range(5):
    #     single_component = nifti_data[:, :, :, idx]
    #     # print(single_component.shape)
    #     upscaled_data = zoom(single_component, zoom=ratio, order=3)
    #     # print(upscaled_data.shape)

    #     new_nifti_data += [upscaled_data]
    # nifti_data = np.stack(new_nifti_data, axis=-1)
    # print(nifti_data.shape)

    # Apply affine transform to data
    # mode = "wrap"
    # for idx in range(nifti_data.shape[-1]):
    #     component = nifti_data[:, :, :, idx]
    #     nifti_data[:, :, :, idx] = shift(
    #         input=affine_transform(
    #             input=component,
    #             matrix=np.linalg.inv(nifti_affine),
    #             mode=mode
    #         ),
    #         shift=np.array(component.shape)/2,
    #         mode=mode,
    #     )
    # anat_data = shift(
    #     input=affine_transform(
    #         input=anat_data,
    #         matrix=np.linalg.inv(anat_affine),
    #         mode=mode
    #     ),
    #     shift=np.array(anat_data.shape)/2,
    #     mode=mode,
    # )

    # inv_affine = transform.AffineTransform(np.linalg.inv(anat_affine), dimensionality=3)
    # affine = transform.AffineTransform(anat_affine,  dimensionality=3)
    # anat_data1 = transform.warp(anat_data, inverse_map=inv_affine)
    # anat_data2 = transform.warp(anat_data, inverse_map=affine)

    print(f"Anat shape: {anat_data.shape}")

    # # ### Apply affine transform to anat_data
    # reference = nib.Nifti1Image(nifti_data[:, :, :, 0], nifti.affine, header=nifti.header)
    # anat = resample_from_to(anat, reference)
    # anat_data = anat.get_fdata()
    # # whole_aff = np.linalg.inv(img2.affine).dot(img1.affine.dot(slice_aff))

    # derive a few things for plotting
    vmax = nifti_data.max()
    vmin = nifti_data.min()
    mcmap = CMAPS[SGN]
    imshow_args = {"vmax": vmax, "vmin": vmin, "cmap": mcmap}

    # n_features = nifti.shape[-1]
    n_features = 5
    n_cols = max([1, round(np.sqrt(n_features / 3))])
    # 3 here is the number of views; for the figure layout it accounts that
    # the column's width is greater than the row's height
    n_rows = np.ceil(n_features / n_cols).astype(int)

    print("Plotting components:")

    # create figure and subplots
    # fig, ax = plt.subplots(
    #     ncols=n_cols,
    #     nrows=n_rows,
    #     figsize=(iscale * n_cols, iscale * n_rows / 3),
    #     facecolor = 'black',
    # )

    # fig = plt.figure(figsize=(iscale * n_cols, iscale * n_rows / 3), facecolor = 'black')
    # DEBUGGING
    fig = plt.figure(figsize=(iscale * n_cols, iscale * n_rows))
    gs = gridspec.GridSpec(ncols=n_cols, nrows=n_rows)
    fig.subplots_adjust(
        left=0.01, right=0.99, bottom=0.01, top=0.99, wspace=0.1, hspace=0
    )

    # plot components
    # for i in tqdm(range(n_features)):
    for i in tqdm(range(5)):
        # set subplots
        # inner_gs = gridspec.GridSpecFromSubplotSpec(
        #     ncols=3,
        #     nrows=1,
        #     subplot_spec=gs[i],
        #     width_ratios=[1.2, 1, 1],
        # )
        # ax = [plt.subplot(inner_gs[j]) for j in range(9)]
        # DEBUGGING
        inner_gs = gridspec.GridSpecFromSubplotSpec(
            ncols=3,
            nrows=3,
            subplot_spec=gs[i],
            # width_ratios=[1.2, 1, 1],
            # height_ratios=[1, 1, 1]
        )
        ax0 = [plt.subplot(inner_gs[0, j]) for j in range(3)]
        ax1 = [plt.subplot(inner_gs[1, j]) for j in range(3)]
        ax2 = [plt.subplot(inner_gs[2, j]) for j in range(3)]

        # load the ICA component and filter it according to SGN
        data = nifti_data[:, :, :, i]
        data[np.sign(data) == DSGN[SGN]] = 0
        # plot component
        if max(abs(data.flatten())) > thr + 0.2:
            max_idx = np.unravel_index(np.argmax(data), data.shape)
            # max_coordinate = apply_affine(nifti_affine, max_idx)
            max_coordinate = apply_affine(nifti_affine, max_idx)
            # print(f"cut indices: {max_idx}")
            # print(f"cut coordinates: {max_coordinate}")

            masked_data = np.ma.masked_inside(
                data, v1=-thr - 0.2, v2=thr + 0.2, copy=False
            )
            # overlay_data = [
            #     masked_data[max_idx[0], :, :].T,
            #     masked_data[:, max_idx[1], :].T,
            #     masked_data[:, :, max_idx[2]].T,
            # ]

            # underlay_data = [
            #     anat_data[max_idx[0], :, :].T,
            #     anat_data[:, max_idx[1], :].T,
            #     anat_data[:, :, max_idx[2]].T,
            # ]

            OrthoSlicer3D(
                anat_data,
                anat_affine,
                axes=ax0,
                position=max_coordinate,
                underlay=True,
                imshow_args=imshow_args,
            )
            OrthoSlicer3D(
                masked_data,
                nifti_affine,
                axes=ax0,
                position=max_coordinate,
                underlay=False,
                imshow_args=imshow_args,
            )

            OrthoSlicer3D(
                anat_data,
                anat_affine,
                axes=ax1,
                position=max_coordinate,
                underlay=True,
                imshow_args=imshow_args,
            )
            OrthoSlicer3D(
                masked_data,
                nifti_affine,
                axes=ax2,
                position=max_coordinate,
                underlay=False,
                imshow_args=imshow_args,
            )
            # underlay_slicer.show()
            # for plane in range(3):

            # ax[plane].imshow(underlay_data[plane], cmap='Greys', origin='upper', extent=[0, 1, 0, 1])
            # ax[plane].imshow(overlay_data[plane], cmap=mcmap, alpha=0.5, origin='upper', extent=[0, 1, 0, 1])
            # ax[plane].axis('off')
            # ax[plane].grid(False)
            # DEBUGGING

            # slicer = OrthoSlicer3D()

            # ax0[plane].imshow(underlay_data[plane], cmap='gray', origin='lower', extent=[0, 1, 0, 1])
            # ax0[plane].imshow(overlay_data[plane], cmap=mcmap, alpha=0.5, origin='lower', extent=[0, 1, 0, 1], **imshow_args)
            # ax0[plane].axis('off')
            # ax0[plane].grid(False)

            # ax1[plane].imshow(underlay_data[plane], cmap='gray', origin='lower', extent=[0, 1, 0, 1])
            # ax1[plane].axis('off')
            # ax1[plane].grid(False)

            # ax2[plane].imshow(overlay_data[plane], cmap=mcmap, alpha=0.5, origin='lower', extent=[0, 1, 0, 1], **imshow_args)
            # ax2[plane].axis('off')
            # ax2[plane].grid(False)

            # plot_map(
            #     data,
            #     xyz_affine(nifti),
            #     anat=anat.get_data(),
            #     anat_affine=xyz_affine(anat),
            #     black_bg=True,
            #     threshold=thr,
            #     cut_coords=coord_transform(
            #         max_idx[0], max_idx[1], max_idx[2], xyz_affine(nifti)
            #     ),
            #     annotate=False,
            #     axes=ax[i],
            #     cmap=mcmap,
            #     draw_cross=False,
            #     **imshow_args,
            # )
            # text(
            #     0.0,
            #     0.95,
            #     str(i),
            #     transform=ax.transAxes,
            #     horizontalalignment="center",
            #     color=(1, 1, 1),
            # )

    # save results
    if save_dir is None:
        save_dir = ""
    else:
        save_dir = f"{dir}/"

    if "png" in ext:
        fig.savefig(
            f"{save_dir}{output}.png",
            facecolor=(0, 0, 0),
            dpi=dpi,
        )
    if "svg" in ext:
        fig.savefig(
            f"{save_dir}{output}.svg",
            facecolor=(0, 0, 0),
        )

    if save_dir != "":
        setup = {
            "nifti": NIFTI,
            "anat": ANAT,
            "sgn": SGN,
            "thr": thr,
            "dpi": dpi,
        }
        with open(f"{save_dir}setup.json", "w", encoding="utf8") as f:
            json.dump(setup, f, indent=4)

    if save_dir != "":
        print(f"Results can be found at {save_dir}")
    else:
        print("Done!")


# def parse():
if __name__ == "__main__":
    import warnings
    import argparse

    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(
        prog="brainbow",
        description="A tool for brain parcellation visualization",
    )

    parser.add_argument(
        "-n",
        "--nifti",
        type=str,
        required=True,
        help="Path to the 4D nifti to convert to images",
    )
    parser.add_argument(
        "-a",
        "--anat",
        type=str,
        required=True,
        help="Path to the anatomical image to use as underlay",
    )
    parser.add_argument(
        "-s",
        "--sign",
        type=str,
        choices=["pos", "neg", "both"],
        default="both",
        help="Show only positive components (pos), \
            only negative components (neg), \
                or both (both) (default: both)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="brainbow-output",
        help="Name of the output file(s) (default: brainbow-output.png/svg).\
            You can specify the exact extension (png or svg). If none is provided, \
                both extensions will be used.",
    )
    parser.add_argument(
        "--dir",
        type=str,
        help="(optional) Name for the results directory (can be nested).\
            If none is provided, output will be placed in the directory where brainbow is executed\n\
                If some is provided, the image setup.json containing the image processing info \
                    will be created in this directory",
    )
    parser.add_argument(
        "--thr",
        type=float,
        default=2.0,
        help="Threshold value for component significance (default: 2.0)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        # default=300,
        default=100,
        help="PNG output dpi (default: 300)",
    )

    if len(sys.argv) == 1:
        parser.print_help()
        parser.exit()

    args = parser.parse_args()

    process_image(
        NIFTI=args.nifti,
        ANAT=args.anat,
        SGN=args.sign,
        output=args.output,
        save_dir=args.dir,
        thr=args.thr,
        dpi=args.dpi,
    )
