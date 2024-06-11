
import numpy as np
from dexterous_bioprosthesis_2021_raw_datasets_framework.estimators.meta.des.des_p_outlier import DespOutlier
from dexterous_bioprosthesis_2021_raw_datasets_framework.estimators.meta.des.outiler_tools import mask_outliers


class DespOutlierFull(DespOutlier):
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