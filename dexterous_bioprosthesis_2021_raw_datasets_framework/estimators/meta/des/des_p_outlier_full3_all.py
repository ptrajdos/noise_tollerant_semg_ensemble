
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from dexterous_bioprosthesis_2021_raw_datasets_framework.estimators.meta.des.des_p_outlier import DespOutlier
from dexterous_bioprosthesis_2021_raw_datasets_framework.estimators.meta.des.outiler_tools import mask_outliers
from pt_outlier_probability.outlier_probability_estimator import OutlierProbabilityEstimator


class DespOutlierFull3All(DespOutlier):
    def __init__(self, pool_classifiers=None, pool_outlier_classifiers=None, k=7, DFP=False, with_IH=False, safe_k=None, IH_rate=0.3, mode='selection', random_state=None, knn_classifier='knn', knne=False, DSEL_perc=0.5, n_jobs=-1):
        super().__init__(pool_classifiers, pool_outlier_classifiers, k, DFP, with_IH, safe_k, IH_rate, mode, random_state, knn_classifier, knne, DSEL_perc, n_jobs)
        """
        This class changes estimate_competence method to ignore original compatence and use outlier-score based competence.
        The competence is soft: 0 if sample predicted as outlier, and 1 otherwise.
        All competence values in interval [0;1] are allowed.

        It uses OutlierProbabilityEstimator to provide soft outlier score.
        
        """


    def _create_outlier_pool(self,X):

        if self.pool_outlier_classifiers is None:
            #Default model
            n_base_classifiers = len(self.pool_classifiers_)
            self.pool_outlier_classifiers_ = []
            for _ in range(n_base_classifiers):
                outlier_classifier = OutlierProbabilityEstimator(IsolationForest(n_estimators=10, random_state=self.random_state_),
                                                                LogisticRegression() )
                outlier_classifier.fit(X)
                self.pool_outlier_classifiers_.append(outlier_classifier)
        else:
            self.pool_outlier_classifiers_ = self.pool_outlier_classifiers

    

    def estimate_competence(self, query, neighbors, distances=None, predictions=None):
        base_competences = super().estimate_competence(query, neighbors, distances, predictions)

        base_competences = np.ones_like(base_competences).astype(np.bool_)

        return base_competences


    def select(self, competences):
        """

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

        selected_classifiers = np.ones_like(competences).astype(np.bool_)

        return selected_classifiers