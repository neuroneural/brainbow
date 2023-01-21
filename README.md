# brainpaint
Script for visualizing ICA/ROI brain parcellation

# Requirements
```
pip install brainpaint
```

# Examples
```
brainpaint --nifti nifti.nii --anat anat.nii
```

# Options for `brainpaint`

- `--nifti`
    - path to the 4D nifti to convert to images
    - required
- `--anat`
    - path to the anatomical image to use as underlay
    - required

- `--output`
    - Name of the output file(s) 
    - default - brainpaint-output.png/svg
    - You can specify the exact extension (png or svg). If none is provided, both extensions will be used.
- `--dir`
    - name for the directory where to save results
    - can be nested (e.g., `final/most_final`)
    - default - directory where brainpaint is executed
- `--sign`
    - choices: `pos, neg, both`
    - used for filtering only positive, only negative, or both components
    - default - `both`
- `--thr`
    - threshold value for component significance
    - default - `2.0`
- `--dpi`
    - dpi for png output
    - default - `300`
