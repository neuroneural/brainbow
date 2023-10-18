import numpy as np
import nibabel as nib

from scipy.stats import zscore
from scipy.ndimage import convolve1d

from sklearn import preprocessing
from nilearn.image import resample_to_img


def load_images(
    nifti_path,
    anat_path,
    thr: float = None,
    normalize: bool = False,
    components: list = None,
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

    if normalize:
        # mask the data
        resampled_mask_img = resample_to_img(anat, nifti)
        mask = resampled_mask_img.get_fdata()
        mask_correction_idx = np.where(mask > 0)
        mask[mask_correction_idx] = zscore(mask[mask_correction_idx]) + 1
        mask = mask > 0
        mask_idx = np.where(mask)

        # perform zscore normalization
        S = nifti_data[*mask_idx, :]
        S = preprocessing.StandardScaler().fit_transform(S)

        # divide by the max abs value and its sign
        S = (np.diag(1 / np.abs(S.T).max(axis=1)) @ S.T).astype("float32")

        midx = np.argmax(np.abs(S), axis=1)
        signs = np.diag(S[np.arange(S.shape[0]), midx])
        S = signs @ S

        S = S.T
        nifti_data[*mask_idx, :] = S

        mask = np.stack([mask] * nifti_data.shape[-1], axis=-1)
        nifti_data = np.ma.masked_array(nifti_data, mask=~mask)

    if thr is not None:
        nifti_data = np.ma.masked_inside(nifti_data, -thr, thr, copy=False)

    return nifti_data, nifti_affine, anat_data, anat_affine
