import numpy as np
import nibabel as nib

from scipy.ndimage import convolve1d


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

    # compute mask: used in normalization
    whole_mask = nifti_data == 0.0
    if len(whole_mask.shape) == 4:
        combined_mask = np.all(whole_mask, axis=-1)
    else:
        combined_mask = whole_mask

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
        mask_idx = np.where(~combined_mask)
        S = nifti_data[*mask_idx, :]

        S = S - np.median(S, axis=0)

        S = (np.diag(1 / np.abs(S.T).max(axis=1)) @ S.T).astype("float32")

        midx = np.argmax(np.abs(S), axis=1)
        signs = np.diag(S[np.arange(S.shape[0]), midx])
        S = signs @ S

        S = S.T
        nifti_data[*mask_idx, :] = S

        combined_mask = np.stack([combined_mask] * nifti_data.shape[-1], axis=-1)
        nifti_data = np.ma.masked_array(nifti_data, mask=combined_mask)

    if thr is not None:
        nifti_data = np.ma.masked_inside(nifti_data, -thr, thr, copy=False)

    return nifti_data, nifti_affine, anat_data, anat_affine
