
import numpy as np
from pt_outlier_probability.outlier_probability_estimator import OutlierProbabilityEstimator
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from dexterous_bioprosthesis_2021_raw_datasets_framework.estimators.meta.channel_combination_classifier_outliers_fast2 import ChannelCombinationClassifierOutliersFast2, channel_group_gen
from dexterous_bioprosthesis_2021_raw_datasets_framework.estimators.meta.des.des_p_outlier import DespOutlier
from dexterous_bioprosthesis_2021_raw_datasets_framework.estimators.meta.outlier.outlier_detector_combiner_min import OutlierDetectorCombinerMin
from dexterous_bioprosthesis_2021_raw_datasets_framework.preprocessing.select_attributes_transformer import SelectAttributesTransformer
from dexterous_bioprosthesis_2021_raw_datasets_framework.set_creators.set_creator import SetCreator

from dexterous_bioprosthesis_2021_raw_datasets_framework.set_creators.set_creator_feature_extractor import SetCreatorFeatureExtractor




class ChannelCombinationClassifierOutliersFast2S(ChannelCombinationClassifierOutliersFast2):

    def __init__(self,
                  model_prototype=RandomForestClassifier(),
                  outlier_detector_prototype=IsolationForest(),
                    validation_percentage=0.5,
                    es_class=DespOutlier,
                    es_arguments: dict = {},
                    random_state=0,
                    channel_features=None,
                    channel_combination_generator=channel_group_gen,
                    channel_combination_generator_options={'group_sizes':[2]},
                    use_no_fault_model=False,
                    outlier_detector_combiner_class=OutlierDetectorCombinerMin,
                    outlier_detector_combiner_params=dict(),
                    partial_train=False,
                    use_full_training_data=True) -> None:
        
        super().__init__(model_prototype, outlier_detector_prototype,
                          validation_percentage, es_class, es_arguments, random_state,
                            channel_features, channel_combination_generator, channel_combination_generator_options,
                              use_no_fault_model, outlier_detector_combiner_class, outlier_detector_combiner_params,
                                partial_train, use_full_training_data)
        
    def _prepare_base_outlier_detectors(self, X, y):
        n_channels = len(self.channel_features)

        if (not hasattr(self, "base_outlier_detectors_")) or (not self.partial_train):
            self.base_outlier_detectors_ = np.zeros( n_channels,dtype=object)
            y_o = np.ones_like(y,dtype=np.int64)
            #create outlier detectors
            for channel_id in range(n_channels):
                column_indices = self._select_channel_group_features([channel_id])
                base_detector = Pipeline([
                    ('trans', SelectAttributesTransformer( column_indices=column_indices )),
                    ('classifier', OutlierProbabilityEstimator(clone(self.outlier_detector_prototype)) )
                    ])
                base_detector.fit(X,y_o)
                self.base_outlier_detectors_[channel_id] = base_detector