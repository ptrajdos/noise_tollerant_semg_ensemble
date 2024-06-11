import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.pipeline import Pipeline
from dexterous_bioprosthesis_2021_raw_datasets_framework.estimators.meta.channel_combination_classifier_outliers2 import ChannelCombinationClassifierOutliers2
from dexterous_bioprosthesis_2021_raw_datasets_framework.estimators.meta.channel_combination_classifier_outliers_fast2 import channel_group_gen
from dexterous_bioprosthesis_2021_raw_datasets_framework.estimators.meta.des.des_p_outlier import DespOutlier
from dexterous_bioprosthesis_2021_raw_datasets_framework.estimators.meta.outlier.outlier_detector_combiner_min import OutlierDetectorCombinerMin
from dexterous_bioprosthesis_2021_raw_datasets_framework.preprocessing.select_attributes_transformer import SelectAttributesTransformer
from pt_outlier_probability.outlier_probability_estimator import OutlierProbabilityEstimator
from sklearn.base import clone

class ChannelCombinationClassifierOutliers2B(ChannelCombinationClassifierOutliers2):
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
                use_full_training_data=True,
                attribute_extractor=None,
                                              ) -> None:

        super().__init__(model_prototype, outlier_detector_prototype, validation_percentage, es_class,
                          es_arguments, random_state, channel_features, channel_combination_generator,
                            channel_combination_generator_options, use_no_fault_model, outlier_detector_combiner_class,
                              outlier_detector_combiner_params, partial_train, use_full_training_data, attribute_extractor)
        
    @property
    def channel_combination_generator_options(self):
        return self._channel_combination_generator_options
    
    @channel_combination_generator_options.setter
    def channel_combination_generator_options(self,value):
        #Does nothing!
        pass

    @channel_combination_generator_options.deleter
    def channel_combination_generator_options(self):
        del self._channel_combination_generator_options