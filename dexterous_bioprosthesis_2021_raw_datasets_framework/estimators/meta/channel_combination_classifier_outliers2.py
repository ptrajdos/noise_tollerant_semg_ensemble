from sklearn.ensemble import IsolationForest, RandomForestClassifier
from dexterous_bioprosthesis_2021_raw_datasets_framework.estimators.meta.channel_combination_classifier_outliers_fast2 import ChannelCombinationClassifierOutliersFast2, channel_group_gen
from dexterous_bioprosthesis_2021_raw_datasets_framework.estimators.meta.des.des_p_outlier import DespOutlier
from dexterous_bioprosthesis_2021_raw_datasets_framework.estimators.meta.outlier.outlier_detector_combiner_min import OutlierDetectorCombinerMin
from dexterous_bioprosthesis_2021_raw_datasets_framework.preprocessing.select_attributes_transformer import SelectAttributesTransformer
from dexterous_bioprosthesis_2021_raw_datasets_framework.set_creators.np_signal_extractors.np_signal_extractor_mav import NpSignalExtractorMav
from dexterous_bioprosthesis_2021_raw_datasets_framework.set_creators.set_creator_dwt import SetCreatorDWT

class ChannelCombinationClassifierOutliers2(ChannelCombinationClassifierOutliersFast2):
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
                    attribute_extractor = None,
                    ) -> None:
        
        super().__init__(model_prototype, outlier_detector_prototype,
                          validation_percentage, es_class, es_arguments, random_state,
                            channel_features, channel_combination_generator, channel_combination_generator_options,
                              use_no_fault_model, outlier_detector_combiner_class, outlier_detector_combiner_params,
                                partial_train, use_full_training_data)
        
        self.attribute_extractor = self.__atribute_extractor_prepare(attribute_extractor)

    def __atribute_extractor_prepare(self, attribute_extractor):
        if attribute_extractor is not None:
            return attribute_extractor
        
        extractor = SetCreatorDWT(
        num_levels=2,
        wavelet_name="db6",
        extractors=[
                NpSignalExtractorMav(),
            ])
        return extractor
        

    def _preprocess_X_y(self, X, y):
        #Raw signals at input
        Xe, ye, t = self.attribute_extractor.fit_transform(X)
        self.channel_features = self.attribute_extractor.get_channel_attribs_indices()
        return super()._preprocess_X_y(Xe, ye)
    
    def predict(self, X, y=None):
        Xe, ye, t = self.attribute_extractor.transform(X)
        return super().predict(Xe, y)
