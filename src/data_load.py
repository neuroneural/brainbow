import numpy as np
import nibabel as nib

from scipy.stats import zscore
from scipy.ndimage import convolve1d


def load_images(
    nifti_path, anat_path, normalize: bool = False, components: list = None
):
    nifti = nib.load(nifti_path)
    nifti_affine = nifti.affine
    nifti_data = nifti.get_fdata()

    anat = nib.load(anat_path)
    anat_affine = anat.affine
    anat_data = anat.get_fdata()

    # if nifti is ROI map (3D), transform it to 4D
    if len(nifti_data.shape) == 3:
        nifti_data = nifti_data.astype(int)

        # get # ROIs
        features = np.unique(nifti_data).astype(int)
        new_nifti_data = np.zeros((*nifti_data.shape, len(features) - 1))

        if components is not None:
            assert (
                min(components) >= 1 and max(components) <= nifti_data.shape[-1]
            ), "Provided components are out of components range in the data"
            target_components = components
        else:
            target_components = features

        kernel = np.ones((5)) / 5
        for i in features:
            if i == 0:
                continue
            # extract ROI regions
            if i in target_components:
                roi_data = (nifti_data == i).astype(float)

                # smooth the edges by convolution
                for j in range(3):
                    roi_data = convolve1d(roi_data, kernel, axis=j)
            else:
                roi_data = np.zeros(nifti_data.shape)

            new_nifti_data[:, :, :, i - 1] = roi_data

        nifti_data = new_nifti_data

    if components is not None:
        assert (
            min(components) >= 1 and max(components) <= nifti_data.shape[-1]
        ), "Provided components are out of components range in the data"
        components = np.array(components) - 1
        nifti_data = nifti_data[:, :, :, components]

    # print(nifti_data.mask)
    if normalize:
        # # mask the data
        nifti_data = np.ma.masked_where(abs(nifti_data) < 1e-8, nifti_data)

        # z-score
        shape = nifti_data.shape
        nifti_data = nifti_data.reshape(-1, shape[-1])
        for i in range(shape[-1]):
            nifti_data[:, i] = zscore(nifti_data[:, i], axis=None)
        nifti_data = nifti_data.reshape(shape)

        # normalize and flip sign
        for i in range(nifti_data.shape[-1]):
            component = nifti_data[:, :, :, i]
            max_idx = np.unravel_index(np.argmax(abs(component)), component.shape)
            component = component / component[max_idx]
            nifti_data[:, :, :, i] = component

        # unmask the data
        nifti_data = nifti_data.filled(0)

    return nifti_data, nifti_affine, anat_data, anat_affine
