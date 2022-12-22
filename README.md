# brain-paint
Scripts for visualizing ICA/ROI brain parcellation

# Requirements
```
conda create -n brain_paint python=3.9
conda activate brain_paint
pip install -r requirements.txt
```

# Examples
```
python process_image.py --nifti nifti.nii --anat anat.nii
```

# Options for `process_image.py`

- `--nifti`
    - path to the 4D nifti to convert to images
    - required
- `--anat`
    - path to the anatomical image to use as underlay
    - required

- `--dir`
    - name for the results directory
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
