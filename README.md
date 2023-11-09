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
- provide path to nifti map (`-n/--nifti` flag) and anatomical underlay (`-a/--anat` flag)
- control the output with `--sign`, `--thr`, and `--norm/--no-norm` flags
- to plot only certain components use `-c/--component` flag
- to set cut coordinates manually, use `--cut` flag
- see below for more info

# Options for `brainbow`

- `-n/--nifti`
    - path to the nifti file with 4D ICA map to convert to images (last dimension is components), or
    - path to the nifti file with 3D ROI map to convert to images (int values from 0 to `n_regions`);
    - required
- `-a/--anat`
    - path to the anatomical image to use as an underlay
    - required

- `-o/--output`
    - name of the output file(s) 
    - default - `brainbow-output.png/svg`
    - You can specify the exact extension (png or svg). If none is provided, both extensions will be used
- `--rich`
    - if `--rich` is passed, in addition to the basic output a config file and a csv file containing cut coordinates is generated
- `-s/--sign`
    - choices: `pos, neg, both`
    - used for filtering only positive (`pos`), only negative (`neg`), or all values (`both`) in the components
    - ALSO controls the colormap
- `--thr`
    - threshold value for component significance
    - if component is significant, `thr` is used to mask values: `-thr < value < thr`
    - default - `0.2`
- `--norm/--no-norm`
    - whether the components from nifti should be normalized:
        - centered around median,
        - divided by the max abs value, and 
        - divided by the sign (`1 or -1`) of this max abs value.
    - may produce a better looking picture, but not recommended for QC
    - default - `--norm` (`True`)
- `--extend/--no-extend`
    - If `True`, in addition to overlay+underlay picture each component \
            will also have a row with separate overlay/underlay
    - helpful for QC
    - default - `--no-extend` (`False`)
- `--dpi`
    - dpi for png output
    - default - `150`
- `--annotate`
    - choices: `none, minimal, full`
    - show components indices (`minimal`), components indices and cut coordinates (`full`), or nothing (`none`) on the output figure
    - default - `minimal`
- `-c/--components`
    - allows to provide a list of components to plot (e.g., '42 4 2')
    - enumeration starts with 1
- `--cut`
    - allows to set the cut coordinates manually
    - default behavior is to use the coordinates of the max abs value
    - needs to be either :
        - a path to scv file (like the one created by '--rich' flag), or
        - a comma separated list of 3 cooridnates, which will be used for all components.
    - coordinates order should be RAS+
    - be careful when using with '--components' flag: brainbow assumes that the components order in the csv file correspond to the order provided components

# Output example

<img src="https://raw.githubusercontent.com/neuroneural/brainbow/main/.github/images/brainbow-output.png"/>