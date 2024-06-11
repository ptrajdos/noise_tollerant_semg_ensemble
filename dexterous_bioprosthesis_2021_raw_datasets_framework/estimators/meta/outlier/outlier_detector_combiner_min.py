from sklearn.base import BaseEstimator, ClassifierMixin

import numpy as np

from dexterous_bioprosthesis_2021_raw_datasets_framework.estimators.meta.outlier.outlier_detector_combiner import OutlierDetectorCombiner


class OutlierDetectorCombinerMin(OutlierDetectorCombiner):
    
    def __init__(self, outlier_detectors) -> None:
        super().__init__(outlier_detectors)

    def fit(self, X, y=None):
        raise NotImplementedError("Fitting operation is not implemented for this class")
    
    def fit_predict(self,X, y=None):
        raise NotImplementedError("Fitting operation is not implemented for this class")
    
    def _combine_predictions_labels(self, base_predictions):
        return np.min(base_predictions, axis=1)
    
    def _combine_predictions_soft(self, base_predictions):
        n_objects = base_predictions.shape[0]
        combined = np.zeros((n_objects,2))

        combined[:,1] =  np.min(base_predictions[:,1], axis=1)
        combined[:,[0]] = 1.0 - combined[:,[1]]

        return combined