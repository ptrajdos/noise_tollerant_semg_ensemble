from dexterous_bioprosthesis_2021_raw_datasets_framework.raw_signals.raw_signals import RawSignals
from dexterous_bioprosthesis_2021_raw_datasets_framework.set_creators.set_creator import SetCreator

from dexterous_bioprosthesis_dataset_creator.featureextraction.feature_extractor_multichannel_interface import FeatureExtractorMultichannel
import numpy as np

import pandas as pd
import re

class SetCreatorFeatureExtractor(SetCreator):
    """
    Creates a new set using feature creators
    """
    
    def __init__(self, multichannel_extractor:FeatureExtractorMultichannel) -> None:
        super().__init__()
        self.multichannel_extractor = multichannel_extractor
        self.channel_selected_attribs = None # List containing number of attributes for each channel
        

    def fit_transform(self, raw_signals:RawSignals, y=None):
        
        extracted_objs_attribs = []
        extracted_objs_classes = []
        extracted_objs_timestamps = []
        
        for raw_signal in raw_signals:
            extr_features = self.multichannel_extractor.extract_features(raw_signal.signal)
            extracted_objs_attribs.append(extr_features)
            extracted_objs_classes.append(raw_signal.object_class)
            extracted_objs_timestamps.append(raw_signal.timestamp)

        X = pd.DataFrame(extracted_objs_attribs)
        X.columns = [ nam for nam in extracted_objs_attribs[0].index]

        channel_names = []
        for c in X.columns:
            channel_name = c.split("-")[0]
            if not channel_name in channel_names:
                channel_names.append(channel_name)

        channel_names.sort()

        self.channel_selected_attribs = []

        column_names = list(X.columns) 

        for channel_name in channel_names:
            r = re.compile("{}-*".format( channel_name ))
            selected_column_names = list(filter(r.match, column_names))
            channel_indices = []
            for selected_column_name in selected_column_names:
                channel_indices.append(X.columns.get_loc(selected_column_name))

            self.channel_selected_attribs.append(channel_indices)


        X_n = X.to_numpy()

        return X_n, np.asanyarray(extracted_objs_classes), np.asanyarray(extracted_objs_timestamps)

    def fit(self, raw_signals: RawSignals, y=None):
        self.fit_transform(raw_signals)
        return self

    def transform(self, raw_signals: RawSignals):
        return self.fit_transform(raw_signals)

    def get_channel_attribs_indices(self):
        return self.channel_selected_attribs