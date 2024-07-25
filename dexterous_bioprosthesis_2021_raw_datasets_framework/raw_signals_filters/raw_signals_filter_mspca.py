from copy import deepcopy

import numpy as np
from dexterous_bioprosthesis_2021_raw_datasets_framework.raw_signals.raw_signals import RawSignals
from dexterous_bioprosthesis_2021_raw_datasets_framework.raw_signals.tools.uniform_column_detector import detect_uniform_columns_signals
from dexterous_bioprosthesis_2021_raw_datasets_framework.raw_signals_filters.raw_signals_filter import RawSignalsFilter

import mspca


class RawSignalsFilterMSPCA(RawSignalsFilter):

    def __init__(self, wavelet_func = 'db4', threshold=0.3,scale=True) -> None:
        super().__init__()
        self.wavelet_func = wavelet_func
        self.threshold = threshold
        self.scale = scale

    def fit(self, raw_signals: RawSignals) -> None:
        return super().fit(raw_signals)
    
    def _detect_uniform_columns(self, raw_signal_np, sd_eps = 1E-5):
        
        sdevs = np.std(raw_signal_np, axis=0 )
        non_uniform_bool = sdevs > sd_eps

        non_uniform_indices = [ idx for idx, b in enumerate(non_uniform_bool) if b]
        
        return non_uniform_indices

    def transform(self,raw_signals: RawSignals)->RawSignals:
    
        copied_signals = deepcopy(raw_signals)

        mspca_obj = mspca.MultiscalePCA()
                
        for raw_signal in copied_signals:
            non_uniform_idxs = self._detect_uniform_columns(raw_signal.signal)

            if len(non_uniform_idxs) == 0:
                continue

            raw_signal.signal[:,non_uniform_idxs] = mspca_obj.fit_transform(raw_signal.signal[:,non_uniform_idxs],
                                                                                wavelet_func=self.wavelet_func,
                                                                            threshold=self.threshold,
                                                                                scale=self.scale)
           
        return copied_signals
