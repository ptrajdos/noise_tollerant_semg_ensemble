import numpy as np
import numba as nb

@nb.jit(nopython=True)
def mask_outliers(base_competences,outlier_indicators,tol=1E-3):

    new_competences = np.copy(base_competences)

    for r in np.arange(base_competences.shape[0]):
        for c in np.arange(base_competences.shape[1]):

            if outlier_indicators[r,c] == -1:
                new_competences[r,c] = 0

        r_sum = np.sum( new_competences[r,:])
        if r_sum < tol:
            for c in np.arange(base_competences.shape[1]):
                new_competences[r,c] = 1

    return new_competences