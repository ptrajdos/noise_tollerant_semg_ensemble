
import numpy as np
import numba as nb
from dexterous_bioprosthesis_2021_raw_datasets_framework.estimators.meta.des.des_p_outlier import DespOutlier
from dexterous_bioprosthesis_2021_raw_datasets_framework.estimators.meta.des.outiler_tools import mask_outliers


class DespOutlierFullB(DespOutlier):
    def __init__(self, pool_classifiers=None, pool_outlier_classifiers=None, k=7, DFP=False, with_IH=False, safe_k=None, IH_rate=0.3, mode='selection', random_state=None, knn_classifier='knn', knne=False, DSEL_perc=0.5, n_jobs=-1):
        super().__init__(pool_classifiers, pool_outlier_classifiers, k, DFP, with_IH, safe_k, IH_rate, mode, random_state, knn_classifier, knne, DSEL_perc, n_jobs)
        """
        This class changes estimate_competence method to ignore original compatence and use outlier-score based competence.
        The competence is crisp: 0 if sample predicted as outlier, and 1 otherwise.

        """

    def estimate_competence(self, query, neighbors, distances=None, predictions=None):
        base_competences = super().estimate_competence(query, neighbors, distances, predictions)

        base_competences = np.ones_like(base_competences)

        if self.pool_outlier_classifiers_ is None:
            return base_competences

        outlier_mask = np.zeros_like(base_competences)

        n_classifiers = outlier_mask.shape[1]

        for out_classifier_id in range(n_classifiers):
            outlier_mask[:,out_classifier_id] = self.pool_outlier_classifiers_[out_classifier_id].predict(query)

        masked_competences = mask_outliers(base_competences, outlier_mask)

        return masked_competences 
    
    def select(self, competences):
        """

        Select nonzero with the highest index

        Parameters
        ----------
        competences : array of shape (n_samples, n_classifiers)
            Competence level estimated for each base classifier and test
            example.

        Returns
        -------
        selected_classifiers : array of shape (n_samples, n_classifiers)
            Boolean matrix containing True if the base classifier is selected,
            False otherwise.

        """
        if competences.ndim < 2:
            competences = competences.reshape(1, -1)


        selected_classifiers = select_highest_index(competences)

        return selected_classifiers

@nb.jit(nopython=True)
def select_highest_index(competences):

    selected_classifiers = np.zeros_like(competences)
    
    for r in range(competences.shape[0]):
        for c in  [*range(competences.shape[1])][::-1]:
            if competences[r,c] > 0:
                selected_classifiers[r,c] = 1
                break
        
        rsum = np.sum(selected_classifiers[r])

        if rsum < 1E-3:
            selected_classifiers[r] = np.ones_like(selected_classifiers[r])
            

    return selected_classifiers.astype(np.bool_)
