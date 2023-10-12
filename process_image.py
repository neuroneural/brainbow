import sys

import json
import numpy as np
import nibabel as nib

from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# from nipy.labs.viz import plot_map

from nibabel.affines import apply_affine

from src.colormaps import CMAPS, DSGN
from src.utils import process_output
from src.nipy import plot_map


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

    anat = nib.load(ANAT)
    anat_affine = anat.affine
    anat_data = anat.get_fdata()

    print(f"Anat shape: {anat_data.shape}")
    print(f"Nifti shape: {nifti_data.shape}")

    # derive a few things for plotting
    vmax = nifti_data.max()
    vmin = nifti_data.min()
    mcmap = CMAPS[SGN]
    imshow_args = {"vmax": vmax, "vmin": vmin, "cmap": mcmap}

    n_features = nifti.shape[-1]
    n_cols = max([1, round(np.sqrt(n_features / 3))])
    # 3 here is the number of views; for the figure layout it accounts that
    # the column's width is greater than the row's height
    n_rows = np.ceil(n_features / n_cols).astype(int)

    print("Plotting components:")

    fig = plt.figure(figsize=(iscale * n_cols, iscale * n_rows / 3), facecolor="black")
    gs = gridspec.GridSpec(ncols=n_cols, nrows=n_rows)
    fig.subplots_adjust(
        left=0.01, right=0.99, bottom=0.01, top=0.99, wspace=0.1, hspace=0
    )

    # plot components
    for i in tqdm(range(n_features)):
        # set subplots
        ax = plt.subplot(gs[i])

        # load the ICA component and filter it according to SGN
        data = nifti_data[:, :, :, i]
        data[np.sign(data) == DSGN[SGN]] = 0
        # plot component
        if max(abs(data.flatten())) > thr + 0.2:
            max_idx = np.unravel_index(np.argmax(data), data.shape)
            cut_coords = apply_affine(nifti_affine, max_idx)

            slicer = plot_map(
                data,
                nifti_affine,
                anat=anat_data,
                anat_affine=anat_affine,
                black_bg=True,
                threshold=thr,
                cut_coords=cut_coords,
                axes=ax,
                **imshow_args,
            )

            slicer.annotate(size=8, s=f"{i+1}")

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
