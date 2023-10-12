import numpy as np
import nibabel as nib


def load_images(nifti_path, anat_path, normalize: bool = False):
    nifti = nib.load(nifti_path)
    nifti_affine = nifti.affine
    nifti_data = nifti.get_fdata()

    anat = nib.load(anat_path)
    anat_affine = anat.affine
    anat_data = anat.get_fdata()

    # if nifti is ROI map, transform it to 4D
    if len(nifti_data.shape) == 3:
        nifti_data = nifti_data.astype(int)
        n_features = np.unique(nifti_data).shape[0] - 1
        new_nifti_data = np.zeros((*nifti_data.shape, n_features))
        for i in range(n_features):
            new_nifti_data[:, :, :, i] = (nifti_data == (i + 1)).astype(int)

        nifti_data = new_nifti_data

    return nifti_data, nifti_affine, anat_data, anat_affine
