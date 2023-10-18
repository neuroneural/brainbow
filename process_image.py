import sys

import json
import numpy as np

from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from nibabel.affines import apply_affine

from src.colormaps import CMAPS, DSGN
from src.data_load import load_images
from src.nipy import plot_map
from src.utils import process_output_path


def process_image(
    NIFTI,
    ANAT,
    SGN,
    output: str = None,
    save_dir: str = None,
    thr: float = 0.2,
    normalize: bool = True,
    extend: bool = False,
    dpi: int = 150,
    annotate: bool = True,
    components: list = None,
    iscale: int = 3,
):
    # create output directory (if needed) and define output extension
    output, ext = process_output_path(output, save_dir)

    print("Loading and processing the nifti files")

    nifti_data, nifti_affine, anat_data, anat_affine = load_images(
        NIFTI,
        ANAT,
        thr,
        normalize=normalize,
        components=components,
    )

    # derive a few things for plotting
    mcmap = CMAPS[SGN]  # get colormap
    n_features = nifti_data.shape[-1]
    n_cols = max([1, round(np.sqrt(n_features / 3))])
    n_rows = np.ceil(n_features / n_cols).astype(int)

    print("Plotting components:")

    # create figure and grid
    if extend:
        fig = plt.figure(figsize=(iscale * n_cols, iscale * n_rows), facecolor="black")
    else:
        fig = plt.figure(
            figsize=(iscale * n_cols, iscale * n_rows / 3), facecolor="black"
        )

    gs = gridspec.GridSpec(ncols=n_cols, nrows=n_rows)
    fig.subplots_adjust(
        left=0.01, right=0.99, bottom=0.01, top=0.99, wspace=0.1, hspace=0
    )

    # plot components
    for i in tqdm(range(n_features)):
        if extend:
            subgs = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[i])
            ax = [plt.subplot(subgs[i]) for i in range(3)]
        else:
            ax = [plt.subplot(gs[i])]  # it is a list for compatibility reasons

        # load the ICA component and filter it according to SGN
        data = nifti_data[:, :, :, i]
        data[np.sign(data) == DSGN[SGN]] = 0

        # plot component
        if np.max(np.abs(data)) > thr:
            if SGN == "neg":
                cut_idx = np.unravel_index(np.argmin(data), data.shape)
            else:
                cut_idx = np.unravel_index(np.argmax(data), data.shape)
            cut_coords = apply_affine(nifti_affine, cut_idx)

            vmax = data.max()
            vmin = -vmax if SGN == "both" else data.min()
            imshow_args = {"vmax": vmax, "vmin": vmin, "cmap": mcmap}

            slicer = plot_map(
                map=data,
                affine=nifti_affine,
                anat=anat_data,
                anat_affine=anat_affine,
                black_bg=True,
                threshold=thr,
                cut_coords=cut_coords,
                axes=ax[0],
                **imshow_args,
            )
            if annotate:
                if components is not None:
                    slicer.annotate(size=8, s=f"{components[i]}")
                else:
                    slicer.annotate(size=8, s=f"{i+1}")

            if extend:
                plot_map(
                    map=None,
                    affine=None,
                    anat=anat_data,
                    anat_affine=anat_affine,
                    black_bg=True,
                    threshold=thr,
                    cut_coords=cut_coords,
                    axes=ax[1],
                    **imshow_args,
                )
                plot_map(
                    map=data,
                    affine=nifti_affine,
                    anat=None,
                    anat_affine=None,
                    black_bg=True,
                    threshold=thr,
                    cut_coords=cut_coords,
                    axes=ax[2],
                    **imshow_args,
                )
        else:
            for axx in ax:
                axx.set_facecolor("black")
            ax[0].text(
                0.5,
                0.5,
                s="Below threshold",
                color="white",
                ha="center",
                va="center",
                size=14,
            )

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


def parse():
    import warnings
    import argparse

    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(
        prog="brainbow",
        description="A tool for brain parcellation visualization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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
                or both (both)",
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
        help="Name for the results directory (can be nested).\
            If none is provided, output will be placed in the directory where brainbow is executed\n\
                If some is provided, the image setup.json containing the image processing info \
                    will be created in this directory",
    )
    parser.add_argument(
        "--thr",
        type=float,
        default=0.2,
        help="Threshold value for component significance",
    )
    parser.add_argument(
        "--norm",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Whether the nifti data should be normalized. May produce a better looking picture, \
            but not recommended for QA.",
    )
    parser.add_argument(
        "--extend",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="If passed, in addition to overlay+underlay picture each component \
            will also have rows with only underlay/overlay",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="PNG output dpi",
    )
    parser.add_argument(
        "--annotate",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Enumerate components in the output figure",
    )
    parser.add_argument(
        "-c",
        "--components",
        nargs="*",
        type=int,
        help="Allows to provide a list of components to plot (e.g., '42 4 2'). \
                            Enumeration starts with 1.",
        default=None,
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
        normalize=args.norm,
        extend=args.extend,
        dpi=args.dpi,
        annotate=args.annotate,
        components=args.components,
    )


if __name__ == "__main__":
    parse()
