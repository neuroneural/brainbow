# brainbow
Script for visualizing ICA/ROI brain parcellation
<img src="https://raw.githubusercontent.com/neuroneural/brainbow/main/.github/images/1.png"/>

# Requirements
```
pip install brainbow
```

# Examples
```
brainbow --nifti nifti.nii --anat anat.nii
```

# Quick guide
- provide path to nifti map (`-n` flag) and anatomical underlay (`-a` flag)
- control the output with `--sign`, `--thr`, and `--norm/--no-norm` flags
- to plot only certain components use `-c` flag
- see below for more info

# Options for `brainbow`

- `--nifti`
    - path to the nifti file with 4D ICA map to convert to images (last dimension is components), or
    - path to the nifti file with 3D ROI map to convert to images (int values from 0 to `n_regions`);
    - required
- `--anat`
    - path to the anatomical image to use as an underlay
    - required

- `--output`
    - Name of the output file(s) 
    - default - brainbow-output.png/svg
    - You can specify the exact extension (png or svg). If none is provided, both extensions will be used
- `--dir`
    - directory for saving results
    - can be nested (e.g., `final/most_final`)
    - default - directory where brainbow is executed
- `--sign`
    - choices: `pos, neg, both`
    - used for filtering only positive, only negative, or all values in the components
    - ALSO controls the colormap
    - default - `both`
- `--thr`
    - threshold value for component significance
    - if component is significant, `thr` is used to mask values: `-thr < value < thr`
    - default - `0.3`
- `--norm/--no-norm`
    - whether the components from nifti should be normalized:
        - z-scored,
        - divided by the max abs value, and 
        - divided by the sign (`1 or -1`) of this max abs value.
    - may produce a better looking picture, but not recommended for QC
    - default - `True`
- `--extend/--no-extend`
    - If `True`, in addition to overlay+underlay picture each component \
            will also have a row with separate overlay/underlay
    - helpful for QC
    - default - `False`
- `--dpi`
    - dpi for png output
    - default - `150`
- `--annotate/--no-annotate`
    - If `True`, enumerate components in the output figure
    - default - `True`


# Output example

<img src="https://raw.githubusercontent.com/neuroneural/brainbow/main/.github/images/brainbow-output.png"/>