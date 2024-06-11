import numpy as np
from dexterous_bioprosthesis_2021_raw_datasets_framework.raw_signals.raw_signals import RawSignals
from dexterous_bioprosthesis_2021_raw_datasets_framework.set_creators.set_creator import SetCreator


class SetCreatorDummy(SetCreator):
    
    def fit(self, raw_signals: RawSignals, y=None):
        return super().fit(raw_signals)
    
    def transform(self, raw_signals: RawSignals):

        X = raw_signals
        y =  np.asanyarray([rs.get_label() for rs in raw_signals])
        t = np.asanyarray([rs.get_timestamp() for rs in raw_signals])
        return X,y,t
    
    def fit_transform(self, raw_signals: RawSignals, y=None):
        self.fit(raw_signals)

        return self.transform(raw_signals) 
    
    def get_channel_attribs_indices(self):
        return None