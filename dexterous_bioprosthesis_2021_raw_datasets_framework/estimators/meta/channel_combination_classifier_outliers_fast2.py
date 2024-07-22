
from deslib.des.base import BaseDES
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

class ChannelCombinationClassifierOutliersFast2(BaseEstimator, ClassifierMixin):


    def __init__(self,model_prototype=RandomForestClassifier(), outlier_detector_prototype = IsolationForest() ,
                 validation_percentage=0.5,
                 es_class=DespOutlier, es_arguments:dict={}, random_state=0, 
                 channel_features=None, channel_combination_generator = channel_group_gen,
                   channel_combination_generator_options={'group_sizes':[2]}, use_no_fault_model=False,
                     outlier_detector_combiner_class = OutlierDetectorCombinerMin,
                   outlier_detector_combiner_params = dict(),
                   partial_train=False, 
                   use_full_training_data = True,
                       ) -> None:
        """

        Metaclassifier that construct ensemble for multiple channel classifiers.
        Some channel-group-related metaclassifiers may be rejected if (single) channel data is considered as outlier.
        It accepts transformed attributes not raw signals
        The method also includes full channel/attribute classifier. It is used when no degradated channel is detected.
        Arguments:
        model_prototype -- base classifier prototype for the ensemble
        validation_percentage -- percentage of points used construct metaclassifier
        es_class -- Class of the meta classifier 
        es_arguments:dic -- A dictionary of argumenst passed to meta classifier
        random_state -- seed for the random generator 
        channel_features -- a list that contains channel specific features
        channel_combination_generator -- function that generates channel combinations
        use_no_fault_model:bool -- determines if 'no fault model' is used 
        outlier_detector_combiner_class -- Class that combines outputs of multiple outlier detectors
        outlier_detector_combiner_params = Additional options for creating the outlier_detector_combiner_class
        partial_train: boolean -- Indicates if channel specific outlier detectors will be stored between training sessions
        use_full_training_data: boolean -- Indicates if the ensemble is training using a full training set.
        """
        super().__init__()

        self.model_prototype = model_prototype
        self.outlier_detector_prototype = outlier_detector_prototype
        self.validation_percentage = validation_percentage
        self.es_class = es_class
        self.es_arguments = es_arguments
        self.random_state = random_state
        self.channel_features = channel_features
        self.channel_combination_generator = channel_combination_generator
        self._channel_combination_generator_options = channel_combination_generator_options
        self.use_no_fault_model = use_no_fault_model
        self.outlier_detector_combiner_class = outlier_detector_combiner_class
        self.outlier_detector_combiner_params = outlier_detector_combiner_params
        self.partial_train = partial_train
        self.use_full_training_data = use_full_training_data

    @property
    def channel_combination_generator_options(self):
        return self._channel_combination_generator_options
    
    @channel_combination_generator_options.setter
    def channel_combination_generator_options(self,value):
        self._channel_combination_generator_options = value

    @channel_combination_generator_options.deleter
    def channel_combination_generator_options(self):
        del self._channel_combination_generator_options

    def _select_channel_group_features(self, channel_group_indices):
        all_selected_features = []

        for ch_idx in channel_group_indices:
            all_selected_features+= self.channel_features[ch_idx]

        return all_selected_features
    
    def _stratify(self,y):
        return y
        

    def _preprocess_X_y(self, X, y):
        """
        If some class(es) has only one object, then append another copy of this object.
        This is to perform proper stratification in datasest with 
        """    
        uniq, counts = np.unique(y,return_counts=True)
    
        one_instance_labels = uniq[counts==1]
        select_inndices = [True if i in one_instance_labels else False for i in y]

        Xc = np.append(X, X[select_inndices],axis=0)
        yc = np.append(y,y[select_inndices])

        return Xc, yc


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
                    ('classifier', clone(self.outlier_detector_prototype))
                    ])
                base_detector.fit(X,y_o)
                self.base_outlier_detectors_[channel_id] = base_detector

    def _prepare_no_fault_model(self, X, y):
        #Model to be used when no outlier channels detected
        self.no_fault_model_ = clone(self.model_prototype)
        y_full_train = self.label_encoder_.transform(y)
        self.no_fault_model_.fit(X,y_full_train)
        self.no_fault_detector_ = OutlierDetectorCombinerMin(self.base_outlier_detectors_,)

    def _create_ensemble(self, X, y):
        self.base_classifiers_ = []
        self.outlier_detectors_ = []
        n_channels = len(self.channel_features)
        channel_groups = self.channel_combination_generator( range(n_channels), 
                                                            **self._channel_combination_generator_options)
        
        if self.use_full_training_data:
            X_train, X_valid, y_train, y_valid = X,X,y,y
        else:
            X_train, X_valid, y_train, y_valid = train_test_split(X,y, test_size=self.validation_percentage,
                                                                random_state=self.random_state_, stratify=self._stratify(y))
        
        y_train = self.label_encoder_.transform(y_train)
        y_valid = self.label_encoder_.transform(y_valid)

        for channel_group_id,  channel_group_indices in enumerate(channel_groups):
            # create base classifiers
            selected_channels_features = self._select_channel_group_features(channel_group_indices)
            self.base_classifiers_.append(Pipeline([
                 ('trans', SelectAttributesTransformer( column_indices= selected_channels_features)),
                 ('classifier', clone(self.model_prototype))
                 ]))
            
            #create combined outlier detectors
            outlier_detectors = self.base_outlier_detectors_[list(channel_group_indices)]
            self.outlier_detectors_.append(
                self.outlier_detector_combiner_class(
                    outlier_detectors= outlier_detectors,
                    **self.outlier_detector_combiner_params
                )
            )
            
        #train base classifiers
        for base_classifier in self.base_classifiers_:
            base_classifier.fit(X_train, y_train)

        self.ensemble_ = self.es_class(pool_classifiers = self.base_classifiers_,
                                       pool_outlier_classifiers = self.outlier_detectors_,
                                        **self.es_arguments)
        self.ensemble_.fit(X_valid, y_valid)

    def fit(self,X,y=None):
        self.random_state_ = check_random_state(self.random_state)
        
        Xp, yp = self._preprocess_X_y(X,y)
        self.classes_ = np.unique(yp)

        self.label_encoder_ = LabelEncoder()
        self.label_encoder_.fit(yp)

        self._prepare_base_outlier_detectors(Xp,yp)
        self._prepare_no_fault_model(Xp,yp)
        
        self._create_ensemble(Xp, yp)

        return self
    

    def predict(self,X,y=None):
        
        check_is_fitted(self, ("classes_", "ensemble_", "base_classifiers_",
                               "outlier_detectors_", "random_state_", "label_encoder_",
                               "no_fault_model_", "no_fault_detector_"))
        
        if y is not None:
            ye = self.label_encoder_.transform(y)
            try:
                #For oracle classifier
                return  self.label_encoder_.inverse_transform(self.ensemble_.predict(X,ye))
            except TypeError:
                return self.label_encoder_.inverse_transform(self.ensemble_.predict(X)) 
            except Exception as e:
                raise e
        
        if  self.use_no_fault_model:
            outlier_indicator = self.no_fault_detector_.predict(X)
            outlier_object_list = [idx for idx,k in enumerate(outlier_indicator) if k<0]
            non_outlier_object_list = [idx for idx,k in enumerate(outlier_indicator) if k>=0]
            n_objects = len(X)

            out_dtype =  self.label_encoder_.classes_.dtype
            out = np.zeros( (n_objects, ), dtype=out_dtype)
            X_non_outlier = X[non_outlier_object_list]
            X_outlier = X[outlier_object_list]
            if len(outlier_object_list) > 0:
                out[outlier_object_list] = self.label_encoder_.inverse_transform(self.ensemble_.predict(X_outlier)) 
            if len(non_outlier_object_list) > 0:
                out[non_outlier_object_list] = self.label_encoder_.inverse_transform(self.no_fault_model_.predict(X_non_outlier))
            
            return out
        
        out = self.label_encoder_.inverse_transform(self.ensemble_.predict(X)) 
            
        return out
    
    def _more_tags(self):
        return {
            "_xfail_checks":{
                "check_parameters_default_constructible":
                    "transformer has 1 mandatory parameter",
            }
        }