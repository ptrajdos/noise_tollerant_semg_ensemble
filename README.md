# A noise-tolerant dual multiclassifier system with fuzzy model applied to the sEMG-based control a bionic upper limb prosthesis

## System Requirements

Requires: __Python>=3.9.7__
All required packages may be installed via __pip__.
Tested on: __Ubuntu Linux 22.04__, __macOS  Sonoma 14.5__

## Setup
To download test data, create the virtual envirionment, and install required packages type:
```
make create_env
```
To clean virtual environment type:
```
make clean
```

The experimental results will be located in: __./experiments\_results__

## Experiments

### One class classifiers

To run the experiments on one-class classifiers type:
```
make run_one_class
```

Results will be placed in: __./experiments\_results/outlier\_detection\_experiment\_snr2__.
Directory structure:

  + Single set
    + *A[1-9]_Force_Exp_low_windowed.pickle* -- raw results (for a single set) as numpy arrays
    + *A[1-9]_Force_Exp_low_windowed.pdf* -- raw results (for a single set) as boxplots. It include results for all quality criteria, all classifiers, all SNR levels, and signal spoilers.
    + *A[1-9]_Force_Exp_low_windowed.md* -- raw results (for a single set) in tabular form (average and standard deviation). It include results for all quality criteria, all classifiers, all SNR levels.
    + *A[1-9]_Force_Exp_low_windowed_trends.pdf* -- trends in quality criteria (median, Q1, Q3) over all SNR levels. Results for all quality criteria, signal spoilers, and base one-class classifiers.
  + Ranking over all sets
    + *ALL_trends_ranks.pdf* -- average ranks plots for all SNR levels.
    + *ALL_trends_ranks.md* -- average ranks and statistical tests in tabular form.


### Ensemble classifiers

To run experiments on commitees with outlier detection type:
```
make run_commitees
```

Results will be placed in: __./experiments\_results/results\_channel\_combination\_ensemble\_fast\_full__.
Directory structure:

  + Single set:
    + *A[1-9]_Force_Exp_low_windowed.pickle* -- raw results (for a single set) as numpy arrays.
    + *A[1-9]_Force_Exp_low_windowed_snr_m1.pdf* -- boxplots (for a single set) for different SNR values, criteria, ensemble sizes, number of contaminated channels.
    + *A[1-9]_Force_Exp_low_windowed_snr_m2.pdf* -- boxplots (for a single set) for different SNR values, criteria, ensemble sizes.  Averaged over the number of contaminated channels.
    + *A[1-9]_Force_Exp_low_windowed_noise_gs.pdf* -- boxplots (for a single set) for different SNR values, and quality criteria. The impact of changing number of channels included in the committee.
    + *A[1-9]_Force_Exp_low_windowed_noise_gs.pdf* -- boxplots (for a single set) for SNR=6, and quality criteria. The impact of changing number of channels included in the committee.
  + Rankings over all sets:
    + *ALL_snr_m1_ranks.pdf" -- Average ranks plots for different SNR values, criteria, ensemble sizes. 
    + *ALL_snr_m1_ranks.md" -- Average ranks tables and statistical tests for different SNR values, criteria, ensemble sizes.  
    + *ALL_snr_m2_ranks.pdf" -- Average ranks plots for different SNR values, criteria, ensemble sizes.  Averaged over the number of contaminated channels.
    + *ALL_snr_m2_ranks.md" -- Average ranks tables and statistical tests for different SNR values, criteria, ensemble sizes. Averaged over the number of contaminated channels.
    + *ALL_noise_gs_ranks.pdf" -- Investigation of the impact of the K parameter -- classical ranks.  
    + *ALL_noise_gs_ranks.md" -- Investigation of the impact of the K parameter -- classical ranks. Tabular version with statistical tests  
    + *ALL_noise_gs_ranks_alt.pdf" -- Investigation of the impact of the K parameter -- Different values of K are ranked.  
    + *ALL_noise_gs_ranks_alt.md" -- Investigation of the impact of the K parameter -- Different values of K are ranked. Tabular version with statistical tests  
  

### Ensemble classifiers

To run experiments for comparison with reference methods type:
```
make run_reference
```

Results will be placed in: __./experiments\_results/results_channel\_combination\_ensemble\_full2__.
Directory structure:

  + Single set:
    + *A[1-9]_Force_Exp_low_windowed.pickle* -- raw results (for a single set) as numpy arrays.
    + *A[1-9]_Force_Exp_low_windowed_snr_m1.pdf* -- boxplots (for a single set) for different SNR values, criteria, ensemble sizes, number of contaminated channels.
    + *A[1-9]_Force_Exp_low_windowed_snr_m2.pdf* -- boxplots (for a single set) for different SNR values, criteria, ensemble sizes.  Averaged over the number of contaminated channels.
    + *A[1-9]_Force_Exp_low_windowed_noise_gs.pdf* -- boxplots (for a single set) for different SNR values, and quality criteria. The impact of changing number of channels included in the committee.
    + *A[1-9]_Force_Exp_low_windowed_noise_gs.pdf* -- boxplots (for a single set) for SNR=6, and quality criteria. The impact of changing number of channels included in the committee.
  + Rankings over all sets:
    + *ALL_snr_m1_ranks.pdf" -- Average ranks plots for different SNR values, criteria, ensemble sizes. 
    + *ALL_snr_m1_ranks.md" -- Average ranks tables and statistical tests for different SNR values, criteria, ensemble sizes.  
    + *ALL_snr_m2_ranks.pdf" -- Average ranks plots for different SNR values, criteria, ensemble sizes.  Averaged over the number of contaminated channels.
    + *ALL_snr_m2_ranks.md" -- Average ranks tables and statistical tests for different SNR values, criteria, ensemble sizes. Averaged over the number of contaminated channels.
    + *ALL_noise_gs_ranks.pdf" -- Investigation of the impact of the K parameter -- classical ranks.  
    + *ALL_noise_gs_ranks.md" -- Investigation of the impact of the K parameter -- classical ranks. Tabular version with statistical tests  
    + *ALL_noise_gs_ranks_alt.pdf" -- Investigation of the impact of the K parameter -- Different values of K are ranked.  
    + *ALL_noise_gs_ranks_alt.md" -- Investigation of the impact of the K parameter -- Different values of K are ranked. Tabular version with statistical tests  