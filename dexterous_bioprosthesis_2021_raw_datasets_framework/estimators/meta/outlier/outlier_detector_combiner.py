from sklearn.base import BaseEstimator, ClassifierMixin
import abc
import numpy as np


class OutlierDetectorCombiner(BaseEstimator, ClassifierMixin):
    
    def __init__(self, outlier_detectors) -> None:
        """
        Combines predictions of multiple outlier detectors. 
        If one of outliers inidicates that the instance is an outlier, then the commitee says it is an outlier.

        Arguments:
        outlier_detectors -- list/iterable of trained outlier detectors
        """
        super().__init__()

        self.outlier_detectors = outlier_detectors

    def fit(self, X, y=None):
        raise NotImplementedError("Fitting operation is not implemented for this class")
    
    def fit_predict(self,X, y=None):
        raise NotImplementedError("Fitting operation is not implemented for this class")
    
    @abc.abstractmethod
    def _combine_predictions_labels(self, base_predictions):
        """
        Combine label predictions

        Arguments:
        ----------
        base_predictions of size (n_objects, n_detectors)

        Returns:
        --------
        numpy array (n_objects,)
        """

    def predict(self, X):
        """
        Combines predictions of multiple outlier detectors

        Arguments:
        ----------
        X -- numpy array of size (n_objectx, n_attributes)

        Returns:
        numpy array of size (n_objects, n_detectors) containing predictions (-1 -- outlier, 1 non-outlier)
        """
        n_objects = X.shape[0]
        n_detectors = len(self.outlier_detectors)

        initial_predictions = np.zeros( (n_objects, n_detectors))

        for detector_id, detector in enumerate(self.outlier_detectors):

            initial_predictions[:,detector_id] = detector.predict(X)

        final_prediction = self._combine_predictions_labels(initial_predictions)

        return final_prediction
    
    def _combine_predictions_soft(self, base_predictions):
        """
        Combines soft predictions of multiple outlier detectors

        Arguments:
        ----------
        base_predictions -- numpy array of size (n_objects, 2, n_detectors)

        
        Returns:
        --------
        numpy array of size (n_objects,) containing predictions in range [0 -- outlier, 1 non-outlier]
        """

    
    def predict_proba(self, X):
        """
        Combines predictions of multiple outlier detectors

        Arguments:
        ----------
        X -- numpy array of size (n_objects, n_attributes)

        Returns:
        numpy array of size (n_objects, 2) containing soft predictions in interval [0;1]. 

        first column -- outlier score. The higher, the more outlier it is.
        second column -- non-outlier score. The higher, the less outlier it is. 
        
        First and second column sums up to one.
        """

        n_objects = X.shape[0]
        n_detectors = len(self.outlier_detectors)

        initial_predictions = np.zeros( (n_objects,2,  n_detectors))

        for detector_id, detector in enumerate(self.outlier_detectors):

            # compatible with ptOutliers package
            initial_predictions[:,:,detector_id] = detector.predict_proba(X)

        final_prediction = self._combine_predictions_soft(initial_predictions)

        return final_prediction
    
    
    def set_outlier_detectors(self, outlier_detectors):
        self.outlier_detectors =  outlier_detectors