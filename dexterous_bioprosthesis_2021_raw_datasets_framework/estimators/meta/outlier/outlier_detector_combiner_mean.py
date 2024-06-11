
from dexterous_bioprosthesis_2021_raw_datasets_framework.estimators.meta.outlier.outlier_detector_combiner import OutlierDetectorCombiner
from dexterous_bioprosthesis_2021_raw_datasets_framework.estimators.meta.outlier.outlier_detector_combiner_min import OutlierDetectorCombinerMin

import numpy as np
from scipy.stats import mode

class OutlierDetectorCombinerMean(OutlierDetectorCombiner):
    def __init__(self, outlier_detectors) -> None:
        super().__init__(outlier_detectors)

    def _combine_predictions_soft(self, base_predictions):
        n_objects = base_predictions.shape[0]
        combined = np.zeros((n_objects,2))

        combined[:,1] =  np.mean(base_predictions[:,1], axis=1)
        combined[:,[0]] = 1.0 - combined[:,[1]]

        return combined
    
    def _combine_predictions_labels(self, base_predictions):
        return mode(base_predictions,axis=1, keepdims=False)[0]
    
