
from deslib.des.base import BaseDES
from deslib.des.des_p import DESP
from dexterous_bioprosthesis_dataset_creator.featureextraction.feature_extractor_function import FeatureExtractorFunction, MAV2_function
from dexterous_bioprosthesis_dataset_creator.featureextraction.feature_extractor_multichannel_simple import FeatureExtractorMultichannelSimple
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, check_array, check_is_fitted
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_random_state
from sklearn.base import clone
from dexterous_bioprosthesis_2021_raw_datasets_framework.estimators.meta.des.des_p_outlier import DespOutlier
from dexterous_bioprosthesis_2021_raw_datasets_framework.estimators.meta.outlier.outlier_detector_combiner_min import OutlierDetectorCombinerMin
from dexterous_bioprosthesis_2021_raw_datasets_framework.preprocessing.select_attributes_transformer import SelectAttributesTransformer
from dexterous_bioprosthesis_2021_raw_datasets_framework.set_creators.set_creator import SetCreator

from dexterous_bioprosthesis_2021_raw_datasets_framework.set_creators.set_creator_feature_extractor import SetCreatorFeatureExtractor
import itertools

def channel_group_gen(indices, group_sizes=[2]):
    result_list = []
    for group_size in group_sizes:
        result_list +=  [ k for k in itertools.combinations(indices, group_size )]
        
    return result_list

class ChannelCombinationClassifierFast(BaseEstimator, ClassifierMixin):


    def __init__(self,model_prototype=RandomForestClassifier(), validation_percentage=0.5,
                 es_class=DESP, es_arguments:dict={}, random_state=0, 
                 channel_features=None, channel_combination_generator = channel_group_gen,
                   channel_combination_generator_options={'group_sizes':[2]},
                     stratify=False  ) -> None:
        """

        Metaclassifier that construct ensemble for multiple channel classifiers.
        It accepts transformed attributes not raw signals
        Arguments:
        model_prototype -- base classifier prototype for the ensemble
        validation_percentage -- percentage of points used construct metaclassifier
        es_class -- Class of the meta classifier 
        es_arguments:dic -- A dictionary of argumenst passed to meta classifier
        random_state -- seed for the random generator 
        channel_features -- a list that contains channel specific features
        channel_combination_generator -- function that generates channel combinations
        stratify:bool -- determines if stratification is performed
        """
        super().__init__()

        self.model_prototype = model_prototype
        self.validation_percentage = validation_percentage
        self.es_class = es_class
        self.es_arguments = es_arguments
        self.random_state = random_state
        self.channel_features = channel_features
        self.channel_combination_generator = channel_combination_generator
        self.channel_combination_generator_options = channel_combination_generator_options
        self.stratify = stratify

    def _select_channel_group_features(self, channel_group_indices):
        all_selected_features = []

        for ch_idx in channel_group_indices:
            all_selected_features+= self.channel_features[ch_idx]

        return all_selected_features
    
    def _stratify(self,y):
        if not self.stratify:
            return None
        
        uniq, counts = np.unique(y,return_counts=True)
        if all( counts>1 ):
            return y
        
        return None

    def fit(self,X,y):
        self.random_state_ = check_random_state(self.random_state)
        self.classes_ = np.unique(y)

        self.label_encoder_ = LabelEncoder()
        self.label_encoder_.fit(y)

        X_train, X_valid, y_train, y_valid = train_test_split(X,y, test_size=self.validation_percentage,
                                                               random_state=self.random_state_, stratify=self._stratify(y))
        
        y_train = self.label_encoder_.transform(y_train)
        y_valid = self.label_encoder_.transform(y_valid)
        
        n_channels = len(self.channel_features)

        self.base_classifiers_ = []

        channel_groups = self.channel_combination_generator( range(n_channels), 
                                                            **self.channel_combination_generator_options)

        
        for channel_group_id,  channel_group_indices in enumerate(channel_groups):
            # create base classifiers
            selected_channels_features = self._select_channel_group_features(channel_group_indices)
            self.base_classifiers_.append(Pipeline([
                 ('trans', SelectAttributesTransformer( column_indices= selected_channels_features)),
                 ('classifier', clone(self.model_prototype))
                 ]))
            
        #train base classifiers
        for base_classifier in self.base_classifiers_:
            base_classifier.fit(X_train, y_train)

        #train ensemble here
        self.ensemble_ = self.es_class(self.base_classifiers_, **self.es_arguments)
        self.ensemble_.fit(X_valid, y_valid)

        return self
    

    def predict(self,X,y=None):
        
        check_is_fitted(self, ("classes_", "ensemble_", "base_classifiers_", "random_state_", "label_encoder_"))

        if y is not None:
            ye = self.label_encoder_.transform(y)
            try:
                #For oracle classifier
                return self.label_encoder_.inverse_transform(self.ensemble_.predict(X,ye))
            except TypeError:
                return self.label_encoder_.inverse_transform(self.ensemble_.predict(X)) 
            except Exception as e:
                raise e
            

        return self.label_encoder_.inverse_transform(self.ensemble_.predict(X)) 
    
    def _more_tags(self):
        return {
            "_xfail_checks":{
                "check_parameters_default_constructible":
                    "transformer has 1 mandatory parameter",
            }
        }