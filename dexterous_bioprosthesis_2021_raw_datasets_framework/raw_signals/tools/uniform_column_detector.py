from dexterous_bioprosthesis_2021_raw_datasets_framework.raw_signals.raw_signal import RawSignal
from dexterous_bioprosthesis_2021_raw_datasets_framework.raw_signals.raw_signals import RawSignals

import numpy as np

def detect_uniform_columns_signals(raw_signals:RawSignals, std_eps = 1E-5):

    uniform_columners = []

    raw_signal:RawSignal
    for raw_signal_id, raw_signal in enumerate(raw_signals): 
        signal = raw_signal.to_numpy()
        sdevs = np.std(signal, axis=0)

        sdev_test = sdevs < std_eps 
        if np.any( sdev_test ):
            uniform_columners.append(
                {
                    "ID": raw_signal_id,
                    "Class":raw_signal.get_label(),
                    "Column idxs": [ idx for idx, b in enumerate(sdev_test) if b]
                }
            )

    return uniform_columners


