# Requirements
- python >= 3.11
    - `conda create -n bb -y python=3.11`
    - `conda activate bb`
- `pip install brainbow`
## optional, just for demo
- `anat='/data/users2/ppopov1/bb_data_examples/anat_mni.nii.gz'`
- `ica='/data/users2/ppopov1/bb_data_examples/ica_neuromark.nii'`
- `roi='/data/users2/ppopov1/bb_data_examples/roi_schaefer.nii.gz'`

# Data examples
- `anat_mni.nii.gz` - MNI template, serves as a canvas for components
- `ica_neuromark.nii` - neuromark's 53 ICA components
- `roi_schaefer.nii.gz` - 200 ROIs from Schaefer parcellation

# `brainbow` examples
- `brainbow -a $anat -n $ica`
    - ![general output](./samples/0.png)
    - minimal example
    - produces 2 pictures (`brainbow_output.png` and `brainbow_output.svg`) in the directory from which `brainbow` is called
- `brainbow -a $anat -n $ica -o demo.png`    
    - `-o` flag allows to set the output file
- `brainbow -a $anat -n $ica --components 16 42`
    - ![1](./samples/1.png)
    - `--components` (`-c`) flag allows to select a few components for visualization
    - enumeration starts with 1
-  `brainbow -a $anat -n $ica -c 16 42 --sign neg`
    - ![2](./samples/2.png)
    - `--sign` allows to visualize only positive/negative values of components
    - options: `pos`, `neg`, `both` (default) 
-  `brainbow -a $anat -n $ica -c 16 42 --thr 0.05`
    - ![3](./samples/3.png)
    - brainbow masks values falling into `[-trh, thr]`, default threshold is `0.2`
    - by decreasing the threshold we can visualize more details of components
    - by increasing the threshold we can hide irrelevant details of components
- `brainbow -a $anat -n $ica -c 16 42 --annotate full`
    - ![4](./samples/4.png)
    - `--annotate` flag allows to control the amount of additional information on the output picture
    - `--annotate minimal` (default) shows only component indices
    - `--annotate full` shows component indices and their cut coordinates
    - `--annotate none` hides annotations
- `brainbow -a $anat -n $ica -c 16 42 --dpi 300`
    - default `dpi` for png is `150`
## Debugging opitons
- `brainbow -a $anat -n $ica -c 16 42 --extend`
    - ![5](./samples/5.png)
    - in addition to underlay+overlay, brainbow will plot underlay and overlay separately (for each component)
- `brainbow -a $anat -n $ica -c 16 42 --rich`
    - in addition to output pictures, brainbow will procude `output_config.json` config file and `output_cuts.csv` csv file with cut coordinates of each plotted component
- `brainbow -a $anat -n $ica -c 16 42 --cut 0,1,2 --annotate full`
    - ![6](./samples/6.png)
    - `--cut` allows to manually set cut coordinates
    - by default component's cut coordinates correspond to the coordinates of the max abs value
    - alternatively, you can provide path to a csv file with cut coordinates for each component
- `brainbow -a $anat -n $ica -c 16 42 --no-norm`
    - ![7](./samples/7.png)
    - by default, brainbow normalizes each component by centering it around median value, dividing by the max abs value, and flipping sign if this max abs value is negative