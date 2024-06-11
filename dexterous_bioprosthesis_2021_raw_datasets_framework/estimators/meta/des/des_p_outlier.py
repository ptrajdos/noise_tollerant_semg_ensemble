from deslib.des.des_p import DESP
import numpy as np
from sklearn.ensemble import IsolationForest

from dexterous_bioprosthesis_2021_raw_datasets_framework.estimators.meta.des.outiler_tools import mask_outliers

class DespOutlier(DESP):
    def __init__(self, pool_classifiers=None, pool_outlier_classifiers=None , k=7, DFP=False, with_IH=False,
                  safe_k=None, IH_rate=0.3, mode='selection', random_state=None, knn_classifier='knn',
                    knne=False, DSEL_perc=0.5, n_jobs=-1):
        
        super().__init__(pool_classifiers, k, DFP, with_IH, safe_k, IH_rate, mode,
                          random_state, knn_classifier, knne, DSEL_perc, n_jobs)
        
        self.pool_outlier_classifiers = pool_outlier_classifiers

    def _create_outlier_pool(self,X):

        if self.pool_outlier_classifiers is None:
            #Default model
            n_base_classifiers = len(self.pool_classifiers_)
            self.pool_outlier_classifiers_ = []
            for _ in range(n_base_classifiers):
                outlier_classifier = IsolationForest(n_estimators=10, random_state=self.random_state_)
                outlier_classifier.fit(X)
                self.pool_outlier_classifiers_.append(outlier_classifier)
        else:
            self.pool_outlier_classifiers_ = self.pool_outlier_classifiers

    def fit(self, X, y):
        ret_val = super().fit(X, y)
        self.n_features_in_ = self.n_features_

        self._create_outlier_pool(X)
    
        return ret_val
        

    def estimate_competence(self, query, neighbors, distances=None, predictions=None):
        base_competences = super().estimate_competence(query, neighbors, distances, predictions)

        if self.pool_outlier_classifiers_ is None:
            return base_competences

        outlier_mask = np.zeros_like(base_competences)

        n_classifiers = outlier_mask.shape[1]

        for out_classifier_id in range(n_classifiers):
            outlier_mask[:,out_classifier_id] = self.pool_outlier_classifiers_[out_classifier_id].predict(query)

        masked_competences = mask_outliers(base_competences, outlier_mask)

        return masked_competences 
    
        