# brain-paint
Script for visualizing ICA/ROI brain parcellation

# Requirements
```
pip install brain-paint
```

# Examples
```
brain-paint --nifti nifti.nii --anat anat.nii
```

# Options for `brain-paint`

- `--nifti`
    - path to the 4D nifti to convert to images
    - required
- `--anat`
    - path to the anatomical image to use as underlay
    - required

- `--dir`
    - name for the directory where to save results
    - **brain-paint** creates a directory `brain-paint-results`, where `dir` directory will be saved
    - can be nested (e.g., `final/most_final`)
    - default - current UTC time
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
