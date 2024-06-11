
import os
import string
import warnings
from dexterous_bioprosthesis_2021_raw_datasets_framework.tools.stats_tools import p_val_matrix_to_vec, p_val_vec_to_matrix

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, f1_score, make_scorer, precision_score, recall_score
from sklearn.multiclass import OutputCodeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import OneClassSVM
from statsmodels.stats.multitest import multipletests
from dexterous_bioprosthesis_2021_raw_datasets_framework.estimators.meta.des.des_p_outlier_full2 import DespOutlierFull2
from dexterous_bioprosthesis_2021_raw_datasets_framework.estimators.meta.des.des_p_outlier_full3 import DespOutlierFull3
from dexterous_bioprosthesis_2021_raw_datasets_framework.estimators.meta.des.des_p_outlier_fullB import DespOutlierFullB
from dexterous_bioprosthesis_2021_raw_datasets_framework.estimators.meta.outlier.outlier_detector_combiner_mean import OutlierDetectorCombinerMean
from dexterous_bioprosthesis_2021_raw_datasets_framework.estimators.meta.outlier.outlier_detector_combiner_median import OutlierDetectorCombinerMedian
from dexterous_bioprosthesis_2021_raw_datasets_framework.estimators.meta.outlier.outlier_detector_combiner_min import OutlierDetectorCombinerMin
from dexterous_bioprosthesis_2021_raw_datasets_framework.estimators.meta.parameter_selection.gridsearchcv_oneclass2 import GridSearchCVOneClass2
from dexterous_bioprosthesis_2021_raw_datasets_framework.preprocessing.one_class.outlier_generators.outlier_generator_uniform import OutlierGeneratorUniform
from dexterous_bioprosthesis_2021_raw_datasets_framework.raw_signals.raw_signals_io import read_signals_from_dirs
from dexterous_bioprosthesis_2021_raw_datasets_framework.raw_signals_filters.raw_signals_filter_channel_idx import RawSignalsFilterChannelIdx
from dexterous_bioprosthesis_2021_raw_datasets_framework.raw_signals_filters.raw_signals_filter_mspca import RawSignalsFilterMSPCA
from dexterous_bioprosthesis_2021_raw_datasets_framework.raw_signals_spoilers.raw_signals_spoiler_multiple import RawSignalsSpoilerMultiple
from dexterous_bioprosthesis_2021_raw_datasets_framework.raw_signals_spoilers.raw_signals_spoiler_sine import RawSignalsSpoilerSine
from dexterous_bioprosthesis_2021_raw_datasets_framework.set_creators.np_signal_extractors.np_signal_extractor_mav import NpSignalExtractorMav
from dexterous_bioprosthesis_2021_raw_datasets_framework.set_creators.np_signal_extractors.np_signal_extractor_ssc import NpSignalExtractorSsc
from dexterous_bioprosthesis_2021_raw_datasets_framework.set_creators.set_creator_dwt import SetCreatorDWT



from dexterous_bioprosthesis_2021_raw_datasets_framework.raw_signals_spoilers.raw_signals_spoiler_damper import RawSignalsSpoilerDamper
from dexterous_bioprosthesis_2021_raw_datasets_framework.raw_signals_spoilers.raw_signals_spoiler_gauss import RawSignalsSpoilerGauss
from dexterous_bioprosthesis_2021_raw_datasets_framework.raw_signals_spoilers.raw_signals_spoiler_multiple import RawSignalsSpoilerMultiple
from dexterous_bioprosthesis_2021_raw_datasets_framework.raw_signals_spoilers.raw_signals_spoiler_cubicclipper import RawSignalsSpoilerCubicClipper



from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from dexterous_bioprosthesis_2021_raw_datasets_framework.set_creators.set_creator_transformer_wrapper import SetCreatorTransformerWrapper
from dexterous_bioprosthesis_2021_raw_datasets_framework.tools.warnings import warn_with_traceback

from dexterous_bioprosthesis_2021_raw_datasets_framework_experiments import settings

from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold

from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.metrics import accuracy_score

from dexterous_bioprosthesis_2021_raw_datasets_framework_experiments.tools import logger
from dexterous_bioprosthesis_2021_raw_datasets_framework.tools.progressparallel import ProgressParallel
from joblib import delayed
from copy import deepcopy


from scipy.stats import wilcoxon
import seaborn as sns

import random
from scipy.stats import rankdata

from dexterous_bioprosthesis_2021_raw_datasets_framework.estimators.meta.channel_combination_classifier_outliers2b import ChannelCombinationClassifierOutliers2B
from dexterous_bioprosthesis_2021_raw_datasets_framework.estimators.meta.channel_combination_classifier_outliers2s import ChannelCombinationClassifierOutliers2S
from dexterous_bioprosthesis_2021_raw_datasets_framework.estimators.meta.channel_combination_classifier_outliers2sb import ChannelCombinationClassifierOutliers2SB

#Plot line colors and markers
from cycler import cycler

class PlotConfigurer:

    def __init__(self) -> None:
        self.is_configured = False

    def configure_plots(self):
        if not self.is_configured:
            # print("Configuring plot")
            dcc = plt.rcParams['axes.prop_cycle']
            mcc = cycler(marker=['x','o',])
            #TODO color cycling works, lines too, but do markers do not. Why
            cc = cycler(color=['r', 'g',])
            lcc = cycler(linestyle=['-', '--', ':', '-.'])
            c = lcc * dcc
            plt.rc('axes', prop_cycle=c)
            # print('Params set', plt.rcParams['axes.prop_cycle'])
            self.is_configured = True

configurer = PlotConfigurer()



def wavelet_extractor2(wavelet_level = 2):
    extractor = SetCreatorDWT(
        num_levels=wavelet_level,
        wavelet_name="db6",
        extractors=[
                NpSignalExtractorMav(),
                NpSignalExtractorSsc(),
                # NpSignalExtractorAr(lags=2),
            ])
    return extractor


def create_extractors():

    extractors_dict = {
        "DWT": wavelet_extractor2(),
    }

    return extractors_dict




def generate_classifiers():
    classifiers = {
        "random_forest": RandomForestClassifier(n_estimators=30, random_state=0),
                }
    return classifiers

def warn_unknown_labels(y_true,y_pred):
    true_set = set(y_true)
    pred_set = set(y_pred)
    diffset = pred_set.difference(true_set)
    if len(diffset) > 0:
        warnings.warn("Diffset: {}".format(diffset))

def acc_m(y_true, y_pred, labels=None, average = None, zero_division=None):
    return accuracy_score(y_true, y_pred)

def bac_m(y_true, y_pred, labels=None, average = None, zero_division=None):
    return balanced_accuracy_score(y_true, y_pred)

def kappa_m(y_true, y_pred, labels=None, average = None, zero_division=None):
    return cohen_kappa_score(y_true, y_pred)

def f1_score_m(y_true, y_pred, labels=None, average = None, zero_division=None):
    return f1_score(y_true, y_pred, average='micro')

def precision_score_m(y_true, y_pred, labels=None, average = None, zero_division=None):
    return precision_score(y_true, y_pred, average='micro')

def recall_score_m(y_true, y_pred, labels=None, average = None, zero_division=None):
    return recall_score(y_true, y_pred, average='micro')



def generate_metrics():
    metrics = {
        "ACC":acc_m,
        "BAC":bac_m,
        "Kappa":kappa_m,
        "F1":f1_score_m, 
        "precision": precision_score_m,
        "recall": recall_score_m
    }
    return metrics




def generate_base(base_classifier, feautre_extractor, group_sizes=[2], outlier_detector_prototype=None):
    
    return Pipeline([
        ('transformer', SetCreatorTransformerWrapper(deepcopy(feautre_extractor))),
        ('classifier', deepcopy(base_classifier))
    ])

def generate_mspca(base_classifier, feautre_extractor, group_sizes=[2], outlier_detector_prototype=None):

    params = {
         'filter__scale': [False,True],
         'filter__threshold': [0.01,0.05,0.1,0.2,0.3],
         'filter__wavelet_func': ['db6','bior2.4'],
    }
    pipeline = Pipeline([
        ('filter',RawSignalsFilterMSPCA()),
        ('transformer', SetCreatorTransformerWrapper(deepcopy(feautre_extractor))),
        ('estimator',OutputCodeClassifier(estimator=deepcopy(base_classifier),random_state=0))
        ])
    
    bac_scorer = make_scorer(balanced_accuracy_score)
    gs = GridSearchCV(estimator=pipeline, param_grid=params, scoring=bac_scorer, cv=3)
    return gs


def generate_ecoc(base_classifier, feautre_extractor, group_sizes=[2], outlier_detector_prototype=None):

    params = {
         'estimator__code_size': [2,3,4,5,6]
    }

    pipeline = Pipeline([
        ('transformer', SetCreatorTransformerWrapper(deepcopy(feautre_extractor))),
        ('estimator',OutputCodeClassifier(estimator=deepcopy(base_classifier),random_state=0))
        ])
    
    bac_scorer = make_scorer(balanced_accuracy_score)
    gs = GridSearchCV(estimator=pipeline, param_grid=params, scoring=bac_scorer, cv=3)
    return gs


def generate_desp_outlier_full_soft_dm7(base_classifier, feautre_extractor, group_sizes=[7],outlier_detector_prototype=None):

    single_channel_ensemble = ChannelCombinationClassifierOutliers2SB(
        outlier_detector_prototype=deepcopy(outlier_detector_prototype),                                                               
        channel_combination_generator_options={'group_sizes':[7]}, 
        es_arguments={'k':3, 'random_state':0},
        es_class=DespOutlierFull3,
        model_prototype=deepcopy(base_classifier),
        channel_features=None,
        outlier_detector_combiner_class = OutlierDetectorCombinerMin,
        partial_train=False,
        use_no_fault_model=True, 
        attribute_extractor=feautre_extractor,
    )
    
    return single_channel_ensemble

def generate_desp_outlier_full_R(base_classifier, feautre_extractor, group_sizes=[7],outlier_detector_prototype=None):

    single_channel_ensemble = ChannelCombinationClassifierOutliers2B(
        outlier_detector_prototype=deepcopy(outlier_detector_prototype),                                                               
        channel_combination_generator_options={'group_sizes':[1,2,3,4,5,6,7]}, 
        es_arguments={'k':3, 'random_state':0},
        es_class=DespOutlierFullB,
        model_prototype=deepcopy(base_classifier),
        channel_features=None,
        outlier_detector_combiner_class = OutlierDetectorCombinerMin,
        partial_train=False,
        use_no_fault_model=True,
        attribute_extractor=feautre_extractor,
    )
    
    return single_channel_ensemble


def generate_desp_outlier_full_soft_mean(base_classifier, feautre_extractor, group_sizes=[2],outlier_detector_prototype=None):

    single_channel_ensemble = ChannelCombinationClassifierOutliers2S(
        outlier_detector_prototype=deepcopy(outlier_detector_prototype),                                                               
        channel_combination_generator_options={'group_sizes':group_sizes},
        es_arguments={'k':3, 'mode':'weighting', 'random_state':0},
        es_class=DespOutlierFull2,
        model_prototype=deepcopy(base_classifier),
        channel_features=None,
        outlier_detector_combiner_class = OutlierDetectorCombinerMean,
        partial_train=True,
        attribute_extractor=feautre_extractor,
    )
    
    return single_channel_ensemble

def generate_desp_outlier_full_soft_median(base_classifier, feautre_extractor, group_sizes=[2],outlier_detector_prototype=None):

    single_channel_ensemble = ChannelCombinationClassifierOutliers2S(
        outlier_detector_prototype=deepcopy(outlier_detector_prototype),                                                               
        channel_combination_generator_options={'group_sizes':group_sizes},
        es_arguments={'k':3, 'mode':'weighting', 'random_state':0},
        es_class=DespOutlierFull2,
        model_prototype=deepcopy(base_classifier),
        channel_features=None,
        outlier_detector_combiner_class = OutlierDetectorCombinerMedian,
        partial_train=True,
        attribute_extractor=feautre_extractor,
    )
    
    return single_channel_ensemble



def generate_methods():
    methods = {
        "B": generate_base,
        "EC": generate_ecoc,
        "MS": generate_mspca,
        "Re": generate_desp_outlier_full_R,
        "DO7":generate_desp_outlier_full_soft_dm7,
        "DOa": generate_desp_outlier_full_soft_mean,
        "DOm": generate_desp_outlier_full_soft_median,
    }  
    return methods

def generate_group_sizes():
    group_sizes={
        '8':[2,3,4],
        '9':[3,4,5],
        '10':[2,5,3],
        '11':[2,5,4],
    }   
    return group_sizes

def generate_spoiled_ch_fraction():
    spoiled_channels_fractions = {
        '0': 0/8,
        '1': 1/8,
        '2': 2/8,
        '3': 3/8,
        '4': 4/8,
        '5': 5/8,
        '6': 6/8,
        '7': 7/8,

    }
    return spoiled_channels_fractions


def generate_ocsvm():

    
    params = {
            'estimator__gamma':['auto'],
             'estimator__nu': [0.1 * (i+1) for i in range(9)]
         }

    kappa_scorer = make_scorer(balanced_accuracy_score)
    generator =  OutlierGeneratorUniform()

    pipeline = Pipeline([('scaler', StandardScaler()),  ('estimator',OneClassSVM()  )])
    gs = GridSearchCVOneClass2(pipeline,test_outlier_generator=generator,  param_grid=params, scoring=kappa_scorer,cv=3)

    return gs

def generate_outlier_detectors():
    detectors = {
        'SVM':generate_ocsvm,
    }
    return detectors

def generate_spoiler_50Hz(snr, channels_spoiled_frac):
    return RawSignalsSpoilerMultiple(
        spoilers=[RawSignalsSpoilerSine(snr=snr,channels_spoiled_frac=channels_spoiled_frac,frequency=50)],
                                      spoiler_relabalers=None)
def generate_spoiler_damper(snr, channels_spoiled_frac):
    return RawSignalsSpoilerMultiple(
        spoilers=[RawSignalsSpoilerDamper(snr=snr,channels_spoiled_frac=channels_spoiled_frac)],
                                      spoiler_relabalers=None)

def generate_spoiler_clipper(snr, channels_spoiled_frac):
    return RawSignalsSpoilerMultiple(
        spoilers=[RawSignalsSpoilerCubicClipper(snr=snr,channels_spoiled_frac=channels_spoiled_frac)],
                                      spoiler_relabalers=None)

def generate_spoiler_gauss(snr, channels_spoiled_frac):
    return RawSignalsSpoilerMultiple(
        spoilers=[RawSignalsSpoilerGauss(snr=snr,channels_spoiled_frac=channels_spoiled_frac)],
                                      spoiler_relabalers=None)

def generate_spoiler_baseline_wander(snr, channels_spoiled_frac):
    return RawSignalsSpoilerMultiple(
        spoilers=[RawSignalsSpoilerSine(snr=snr,channels_spoiled_frac=channels_spoiled_frac,frequency=1, freq_deviation=0.5)],
                                      spoiler_relabalers=None)



def generate_spoilers_gens():
    spoilers = {
        '50':generate_spoiler_50Hz,
        'Damper':generate_spoiler_damper,
        'Clipper':generate_spoiler_clipper,
        'Gauss':generate_spoiler_gauss,
        "Baseline":generate_spoiler_baseline_wander,

    }
    return spoilers

def get_snr_levels():
    return [6,5,4,3,2,1,0]



def run_experiment(input_data_dir_list, output_directory, n_splits=10, n_repeats=3,
                    random_state=0, n_jobs=1, overwrite=True,n_channels=None, append=True):

    os.makedirs(output_directory, exist_ok=True)

    metrics = generate_metrics()
    n_metrics = len(metrics)

    extractors_dict = create_extractors()
    n_extr = len(extractors_dict)

    classifiers_dict = generate_classifiers()
    n_classifiers = len(classifiers_dict)

    methods = generate_methods()
    n_methods = len(methods)

    skf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,random_state=random_state)
    n_folds = skf.get_n_splits()

    group_sizes = generate_group_sizes()
    n_group_sizes = len(group_sizes)

    detector_generators = generate_outlier_detectors()
    n_detector_generators = len(detector_generators)


    spoiler_generators =  generate_spoilers_gens()
    n_spoiler_generators = len(spoiler_generators)

    snrs = get_snr_levels()
    n_snrs = len(snrs)

    channel_spoil_fractions = generate_spoiled_ch_fraction()
    n_channel_spoil_fraction = len(channel_spoil_fractions)

    


    for in_dir in tqdm(input_data_dir_list,desc="Data sets"):

        # metrics x extractors x classifiers x n_group_sizes x detectors x spoilers x snr x ch_fraction x methods x folds
        results = np.zeros( (n_metrics, n_extr, n_classifiers,n_group_sizes,
                         n_detector_generators,
                         n_spoiler_generators,
                         n_snrs,
                         n_channel_spoil_fraction,
                          n_methods, n_folds) )
        
        set_name = os.path.basename(in_dir)

        result_file_path = os.path.join(output_directory,"{}.pickle".format(set_name))

        exists = os.path.isfile(result_file_path)

        if exists and not(overwrite):
            print("Skipping {} !".format(set_name))
            continue

        
        pre_set = read_signals_from_dirs(in_dir)
        raw_set = pre_set['accepted']


        if n_channels is not None:
            n_set_channels = raw_set[0].to_numpy().shape[1]
            n_effective_channels = min(( n_set_channels, n_channels ))
            indices = [*range(n_effective_channels)]
            filter = RawSignalsFilterChannelIdx(indices)
            raw_set = filter.fit_transform(raw_set)



        y =   np.asanyarray(raw_set.get_labels())
        num_labels = len( np.unique(y))

        def compute(fold_idx,train_idx, test_idx):
            #For tracing warnings inside multiprocessing
            warnings.showwarning = warn_with_traceback

            raw_train = raw_set[train_idx]
            y_train = raw_train.get_labels()

            raw_test = raw_set[test_idx]

            fold_res = []

            for extractor_id, extractor_name in enumerate(extractors_dict):
                extractor = extractors_dict[extractor_name]

                for base_classifier_id, base_classifier_name in enumerate(classifiers_dict):
                    base_classifier = classifiers_dict[base_classifier_name]

                    for detector_generator_id, detector_generator_name in enumerate(detector_generators):
                        outlier_detector = detector_generators[detector_generator_name]()

                        for method_id, method_name in enumerate( methods ):
                            method_creator = methods[method_name]
                            
                            
                            for group_size_id, group_size_name in enumerate(group_sizes):
                                group_size = group_sizes[group_size_name]

                                method = method_creator(base_classifier, extractor, group_size, outlier_detector_prototype=outlier_detector)    

                                method.fit(raw_train, y_train)
                            

                                for snr_id, snr in enumerate(snrs):

                                    for channel_spoil_f_id, channel_spoil_f_name in enumerate(channel_spoil_fractions):
                                        spoiled_fraction = channel_spoil_fractions[channel_spoil_f_name]

                                        for spoiler_generator_id, spoiler_generator_name  in enumerate(spoiler_generators):
                                            signal_spoiler = spoiler_generators[spoiler_generator_name](snr=snr, channels_spoiled_frac = spoiled_fraction)
                                        
                                            raw_spoiled_test = signal_spoiler.fit_transform(raw_test)
                                            raw_spoiled_test += raw_test

                                            y_test = raw_spoiled_test.get_labels()

                                            try:
                                                y_pred = method.predict(raw_spoiled_test, y_test)
                                            except TypeError:
                                                y_pred = method.predict(raw_spoiled_test)
                                            except Exception as e:
                                                raise e

                                            y_gt = y_test

                                            for metric_id, metric_name in enumerate(metrics):
                                                metric = metrics[metric_name]
                                                
                                                metric_value = metric(y_gt, y_pred)

                                                
                                                # metrics x extractors x classifiers x n_group_sizes x detectors x spoilers x snr x ch_fraction x methods x folds
                                                fold_res.append( (metric_id, extractor_id, base_classifier_id,
                                                                    group_size_id, detector_generator_id, spoiler_generator_id, snr_id,
                                                                        channel_spoil_f_id ,method_id, fold_idx, metric_value) )
            return fold_res

        results_list =  ProgressParallel(n_jobs=n_jobs, desc="K-folds", total=skf.get_n_splits(), leave=False)\
                            (delayed(compute)(fold_idx, train_idx, test_idx) for fold_idx, (train_idx, test_idx) in enumerate( skf.split(raw_set, y)) )
        
        for result_sublist in results_list:
            # metrics x extractors x classifiers x n_group_sizes x detectors x spoilers x snr x ch_fraction x methods x folds
            for metric_id, extractor_id, base_classifier_id, group_size_id, detector_generator_id, spoiler_generator_id, snr_id, channel_spoil_f_id,method_id, fold_idx, metric_value in result_sublist:
                results[metric_id, extractor_id, base_classifier_id, group_size_id,
                        detector_generator_id, spoiler_generator_id, snr_id, channel_spoil_f_id ,method_id, fold_idx] = metric_value

        
        
        
        fin_result_dict = {
                "classifier_names": [k for k in classifiers_dict],
                "metric_names": [k for k in metrics],
                "extractors": [k for k in extractors_dict],
                "group_sizes":[k for k in group_sizes],
                "outlier_detectors": [k for k in detector_generators],
                "spoilers": [k for k in spoiler_generators],
                "snrs":[k for k in snrs],
                "spoil_ch_f":[k for k in channel_spoil_fractions],
                "methods": [k for k in methods],
                "results": results
            }

        pickle.dump(obj=fin_result_dict, file=open(result_file_path,"wb"))
        
                
def analyze_results_2B(results_directory, output_directory):
    configurer.configure_plots()
    result_files = [f for f in os.listdir(results_directory) if f.endswith(".pickle") ]

    

    for result_file in result_files:
        result_file_basename = os.path.splitext(result_file)[0]
        result_file_path = os.path.join(results_directory, result_file)

        
        raw_data = pickle.load(open(result_file_path,"rb"))

        classifier_names = raw_data["classifier_names"]
        metric_names = raw_data["metric_names"]
        extractor_names = raw_data["extractors"]
        group_size_names = raw_data["group_sizes"]
        outlier_detector_names = raw_data["outlier_detectors"]
        spoiler_names = raw_data["spoilers"]
        snrs = raw_data["snrs"]
        spoiled_channels_fraction = raw_data["spoil_ch_f"]
        method_names = raw_data["methods"]
        
        # metrics x extractors x classifiers x n_group_sizes x detectors x spoilers x snr x ch_fraction x methods x folds
        results = raw_data["results"]
        n_methods = len(method_names)

        pdf_file_path = os.path.join(output_directory, "{}_snr_m1.pdf".format(result_file_basename))


        with PdfPages(pdf_file_path) as pdf:

            for metric_id, metric_name in enumerate(metric_names):
                
                for extractor_id, extractor_name in enumerate(extractor_names):
                    
                    for classifier_id, classifier_name in enumerate(classifier_names):
                        

                        for outlier_detector_id, outlier_detector_name in enumerate(outlier_detector_names):
                            

                            for spoiled_ch_f_id, spoiled_ch_f_name in enumerate(spoiled_channels_fraction):
                                
                                
                                for group_size_id, group_size_name in enumerate(group_size_names):
                                    
                                    # metrics x extractors x classifiers x n_group_sizes x detectors x spoilers x snr x ch_fraction x methods x folds
                                    # spoilers x snr x methods x folds
                                    sub_results = results[metric_id, extractor_id, classifier_id,group_size_id,outlier_detector_id, :,:,spoiled_ch_f_id] 
                                    sub_results = np.mean(sub_results, axis=0)
                                
                                    df = pd.DataFrame(columns=['snr','method','value'])

                                    for i, snr_value in enumerate(snrs):
                                        for j, method_name in enumerate(method_names):
                                            for k in range(sub_results.shape[2]):
                                                new_row = pd.DataFrame({'snr':snr_value, 'method':method_name, 'value':sub_results[i,j,k]}, index=[0])
                                                df = pd.concat([new_row,df.loc[:]]).reset_index(drop=True)


                                    sns.boxplot(x=df['snr'], y=df['value'], hue=df['method'], palette='husl')
                                    plt.title( "{}, {}, {} ,GS:{}, Od:{}, SpCh:{}".format(metric_name, extractor_name, classifier_name,
                                                                                                        group_size_name,outlier_detector_name,
                                                                                                                spoiled_ch_f_name, ))
                    
                                    pdf.savefig()
                                    plt.close()

def analyze_results_2B_ranks(results_directory, output_directory, alpha=0.05):
    configurer.configure_plots()
    result_files = [f for f in os.listdir(results_directory) if f.endswith(".pickle") ]
    n_files = len(result_files)

    #Init

    classifier_names = None
    n_classifiers = None
    metric_names = None
    n_metrics = None
    extractor_names = None
    n_extractors = None
    group_size_names = None
    n_group_sizes = None
    outlier_detector_names = None
    n_outlier_detectors = None
    spoiler_names = None
    n_spoilers = None
    snrs = None
    n_snrs = None
    spoiled_channels_fraction = None
    n_sp_channel_fraction = None
    method_names = None
    n_methods = None
    n_folds = None

    global_results = None

    for result_file_id,  result_file in enumerate(result_files):
        result_file_basename = os.path.splitext(result_file)[0]
        result_file_path = os.path.join(results_directory, result_file)

        
        raw_data = pickle.load(open(result_file_path,"rb"))
        results = raw_data["results"]

        if global_results is None:

            classifier_names = raw_data["classifier_names"]
            n_classifiers = len(classifier_names)
            metric_names = raw_data["metric_names"]
            n_metrics = len(metric_names)
            extractor_names = raw_data["extractors"]
            n_extractors = len(extractor_names)
            group_size_names = raw_data["group_sizes"]
            n_group_sizes = len(group_size_names)
            outlier_detector_names = raw_data["outlier_detectors"]
            n_outlier_detectors = len(outlier_detector_names)
            spoiler_names = raw_data["spoilers"]
            n_spoilers = len(spoiler_names)
            snrs = raw_data["snrs"]
            n_snrs = len(snrs)
            spoiled_channels_fraction = raw_data["spoil_ch_f"]
            n_sp_channel_fraction = len(spoiled_channels_fraction)
            method_names = raw_data["methods"]
            n_methods = len(method_names)
            
            # metrics x extractors x classifiers x n_group_sizes x detectors x spoilers x snr x ch_fraction x methods x folds
            
            n_folds = results.shape[-1]

            # n_files x metrics x extractors x classifiers x n_group_sizes x detectors x spoilers x snr x ch_fraction x methods x folds
            global_results = np.zeros( ( n_files, n_metrics, n_extractors, n_classifiers, n_group_sizes, n_outlier_detectors,
                                      n_spoilers, n_snrs, n_sp_channel_fraction, n_methods, n_folds ))
        
        global_results[result_file_id] = results

        pdf_file_path = os.path.join(output_directory, "{}_snr_m1_ranks.pdf".format("ALL"))
        report_file_path = os.path.join(output_directory, "{}_snr_m1_ranks.md".format("ALL"))
        report_file_handler = open(report_file_path, 'w+')


        with PdfPages(pdf_file_path) as pdf:

            for metric_id, metric_name in enumerate(metric_names):
                print("# {}".format(metric_name), file=report_file_handler)
                
                for extractor_id, extractor_name in enumerate(extractor_names):
                    print("## {}".format(extractor_name), file=report_file_handler)
                    
                    for classifier_id, classifier_name in enumerate(classifier_names):
                        print("### {}".format(classifier_name), file=report_file_handler)

                        for outlier_detector_id, outlier_detector_name in enumerate(outlier_detector_names):
                            print("#### {}".format(outlier_detector_name), file=report_file_handler)
                            
                            for spoiled_ch_f_id, spoiled_ch_f_name in enumerate(spoiled_channels_fraction):
                                print("##### SpCh {}".format(spoiled_ch_f_name), file=report_file_handler)    
                                
                                for group_size_id, group_size_name in enumerate(group_size_names):
                                    print("###### GS: {}".format(group_size_name), file=report_file_handler)    
                                    
                                    # files x metrics x extractors x classifiers x n_group_sizes x detectors x spoilers x snr x ch_fraction x methods x folds
                                    # files 0 x spoilers 1 x snr 2 x methods 3 x folds 4
                                    sub_results = global_results[:,metric_id, extractor_id, classifier_id,group_size_id,outlier_detector_id, :,:,spoiled_ch_f_id]
                                    # methods  0 x snrs 1 x ( files 2 x spoilers 3 x folds 4)
                                    sub_results_r = np.moveaxis(sub_results, [0,1,2,3,4], [2, 3, 1, 0, 4 ])
                                    sub_results = sub_results_r.reshape( (n_methods, n_snrs, -1))

                                    ranked_data = rankdata(sub_results, axis=0)
                                    # methods, snrs
                                    avg_ranks = np.mean(ranked_data,axis=-1)

                                    for method_id, method_name in enumerate(method_names):

                                        plt.plot([int(i) for i in snrs], avg_ranks[method_id,:], marker="o", label = method_name)

                                    plt.title("{}, {}, {}, {}, spFrac {}, GS {} ".format(metric_name, extractor_name, classifier_name, outlier_detector_name, spoiled_ch_f_name,group_size_name ))
                                    plt.xlabel("SNR")
                                    plt.ylabel("Criterion avg rank")
                                    plt.legend()
                                    pdf.savefig()
                                    plt.close()

                
                                    #avg_ranks (methods, snrs)
                                    mi = pd.MultiIndex.from_arrays( [ ["{}".format(metric_name) for _ in range(n_methods)], [m for m in method_names] ] )
                                    av_rnk_df = pd.DataFrame(avg_ranks.T,columns=mi ,
                                                              index=["Avg Rnk {}, snr:{}".format(a,si) for si,a in zip( snrs,string.ascii_letters)])

                                    # methods  0 x snrs 1 x ( files 2 x spoilers 3)
                                    sub_results_snr = np.mean(sub_results_r, axis=-1).reshape((n_methods, n_snrs, -1))
                                    for snr_id, (snr_name,snr_letter) in enumerate( zip(snrs, string.ascii_letters )):
                                        # methods    x ( files  x spoilers )
                                        values = sub_results_snr[:,snr_id]
                                        p_vals = np.zeros( (n_methods, n_methods) )
                                        for i in range(n_methods):
                                            for j in range(n_methods):
                                                if i == j:
                                                    continue

                                                values_squared_diff = np.sqrt (np.sum( (values[i,:] - values[j,:])**2 ) )
                                                if values_squared_diff > 1E-4:
                                                    with warnings.catch_warnings(): #Normal approximation
                                                        warnings.simplefilter("ignore")
                                                        p_vals[i,j]  = wilcoxon(values[i], values[j]).pvalue #mannwhitneyu(values[:,i], values[:,j]).pvalue
                                                else:
                                                    p_vals[i,j] = 1.0
                                        
                                        p_val_vec  = p_val_matrix_to_vec(p_vals)

                                        p_val_vec_corrected = multipletests(p_val_vec, method='hommel')
                                        corr_p_val_matrix = p_val_vec_to_matrix(p_val_vec_corrected[1],n_methods,symmetrize=True)

                                        s_test_outcome = []
                                        for i in range(n_methods):
                                            out_a = []
                                            for j in range(n_methods):
                                                if avg_ranks[i,snr_id] > avg_ranks[j, snr_id] and corr_p_val_matrix[i,j]<alpha:
                                                    out_a.append(j+1)
                                            if len(out_a) == 0:
                                                s_test_outcome.append("--")
                                            else:
                                                s_test_outcome.append("{\\scriptsize " + ",".join([ str(x) for x in out_a]) + "}") 
                                        av_rnk_df.loc["{} {}, snr:{}_T".format("Avg Rnk",snr_letter,snr_name)] = s_test_outcome
                                        av_rnk_df.sort_index(inplace=True)

                                    av_rnk_df.style.format(precision=3,na_rep="")\
                                    .format_index(escape="latex", axis=0).format_index(escape="latex", axis=1)\
                                    .to_latex(report_file_handler, multicol_align="c")
        report_file_handler.close()

def analyze_results_2C(results_directory, output_directory):
    configurer.configure_plots()
    result_files = [f for f in os.listdir(results_directory) if f.endswith(".pickle") ]

    for result_file in result_files:
        result_file_basename = os.path.splitext(result_file)[0]
        result_file_path = os.path.join(results_directory, result_file)

        
        raw_data = pickle.load(open(result_file_path,"rb"))

        classifier_names = raw_data["classifier_names"]
        metric_names = raw_data["metric_names"]
        extractor_names = raw_data["extractors"]
        group_size_names = raw_data["group_sizes"]
        outlier_detector_names = raw_data["outlier_detectors"]
        spoiler_names = raw_data["spoilers"]
        snrs = raw_data["snrs"]
        spoiled_channels_fraction = raw_data["spoil_ch_f"]
        method_names = raw_data["methods"]
        
        # metrics x extractors x classifiers x n_group_sizes x detectors x spoilers x snr x ch_fraction x methods x folds
        results = raw_data["results"]
        n_methods = len(method_names)

        pdf_file_path = os.path.join(output_directory, "{}_snr_m2.pdf".format(result_file_basename))


        with PdfPages(pdf_file_path) as pdf:

            for metric_id, metric_name in enumerate(metric_names):
                
                for extractor_id, extractor_name in enumerate(extractor_names):
                    
                    for classifier_id, classifier_name in enumerate(classifier_names):
                        

                        for outlier_detector_id, outlier_detector_name in enumerate(outlier_detector_names):
                            
                                
                            for group_size_id, group_size_name in enumerate(group_size_names):
                                
                                # metrics x extractors x classifiers x n_group_sizes x detectors x spoilers x snr x ch_fraction x methods x folds

                                sub_results = results[metric_id, extractor_id, classifier_id,group_size_id,outlier_detector_id, :,:,] # spoilers x snr x chan_frac x methods x folds
                                sub_results = np.mean(sub_results, axis=(0,2))
                            
                                df = pd.DataFrame(columns=['snr','method','value'])

                                for i, snr_value in enumerate(snrs):
                                    for j, method_name in enumerate(method_names):
                                        for k in range(sub_results.shape[2]):
                                            new_row = pd.DataFrame({'snr':snr_value, 'method':method_name, 'value':sub_results[i,j,k]}, index=[0])
                                            df = pd.concat([new_row,df.loc[:]]).reset_index(drop=True)


                                sns.boxplot(x=df['snr'], y=df['value'], hue=df['method'], palette='husl')
                                plt.title( "{}, {}, {} ,GS:{}, Od:{}".format(metric_name, extractor_name, classifier_name,
                                                                                                    group_size_name,outlier_detector_name,
                                                                                                         ))
                
                                pdf.savefig()
                                plt.close()

def analyze_results_2C_ranks(results_directory, output_directory, alpha=0.05):
    configurer.configure_plots()
    result_files = [f for f in os.listdir(results_directory) if f.endswith(".pickle") ]
    n_files = len(result_files)

    #Init

    classifier_names = None
    n_classifiers = None
    metric_names = None
    n_metrics = None
    extractor_names = None
    n_extractors = None
    group_size_names = None
    n_group_sizes = None
    outlier_detector_names = None
    n_outlier_detectors = None
    spoiler_names = None
    n_spoilers = None
    snrs = None
    n_snrs = None
    spoiled_channels_fraction = None
    n_sp_channel_fraction = None
    method_names = None
    n_methods = None
    n_folds = None

    global_results = None

    for result_file_id,  result_file in enumerate(result_files):
        result_file_basename = os.path.splitext(result_file)[0]
        result_file_path = os.path.join(results_directory, result_file)

        
        raw_data = pickle.load(open(result_file_path,"rb"))
        results = raw_data["results"]

        if global_results is None:

            classifier_names = raw_data["classifier_names"]
            n_classifiers = len(classifier_names)
            metric_names = raw_data["metric_names"]
            n_metrics = len(metric_names)
            extractor_names = raw_data["extractors"]
            n_extractors = len(extractor_names)
            group_size_names = raw_data["group_sizes"]
            n_group_sizes = len(group_size_names)
            outlier_detector_names = raw_data["outlier_detectors"]
            n_outlier_detectors = len(outlier_detector_names)
            spoiler_names = raw_data["spoilers"]
            n_spoilers = len(spoiler_names)
            snrs = raw_data["snrs"]
            n_snrs = len(snrs)
            spoiled_channels_fraction = raw_data["spoil_ch_f"]
            n_sp_channel_fraction = len(spoiled_channels_fraction)
            method_names = raw_data["methods"]
            n_methods = len(method_names)
            
            # metrics x extractors x classifiers x n_group_sizes x detectors x spoilers x snr x ch_fraction x methods x folds
            
            n_folds = results.shape[-1]

            # n_files x metrics x extractors x classifiers x n_group_sizes x detectors x spoilers x snr x ch_fraction x methods x folds
            global_results = np.zeros( ( n_files, n_metrics, n_extractors, n_classifiers, n_group_sizes, n_outlier_detectors,
                                      n_spoilers, n_snrs, n_sp_channel_fraction, n_methods, n_folds ))
        
        global_results[result_file_id] = results

        pdf_file_path = os.path.join(output_directory, "{}_snr_m2_ranks.pdf".format("ALL"))
        report_file_path = os.path.join(output_directory, "{}_snr_m2_ranks.md".format("ALL"))
        report_file_handler = open(report_file_path, 'w+')


        with PdfPages(pdf_file_path) as pdf:

            for metric_id, metric_name in enumerate(metric_names):
                print("# {}".format(metric_name), file=report_file_handler)
                
                for extractor_id, extractor_name in enumerate(extractor_names):
                    print("## {}".format(extractor_name), file=report_file_handler)
                    
                    for classifier_id, classifier_name in enumerate(classifier_names):
                        print("### {}".format(classifier_name), file=report_file_handler)
                        
                        for outlier_detector_id, outlier_detector_name in enumerate(outlier_detector_names):
                            print("#### {}".format(outlier_detector_name), file=report_file_handler)
                                
                            for group_size_id, group_size_name in enumerate(group_size_names):
                                print("##### GS {}".format(group_size_name), file=report_file_handler)
                                
                                # files x metrics x extractors x classifiers x n_group_sizes x detectors x spoilers x snr x ch_fraction x methods x folds
                                # files 0 x spoilers 1 x snr 2 x ch_fraction 3 x methods 4 x folds 5
                                sub_results = global_results[:,metric_id, extractor_id, classifier_id,group_size_id,outlier_detector_id, :,:,:]
                                # methods  0 x snrs 1 x ( files 2 x spoilers 3 x ch_frac 4 x folds 5)
                                sub_results_r = np.moveaxis(sub_results, [0,1,2,3,4,5], [2, 3, 1, 4, 0, 5])
                                sub_results = sub_results_r.reshape( (n_methods, n_snrs, -1))

                                ranked_data = rankdata(sub_results, axis=0)
                                # methods, snrs
                                avg_ranks = np.mean(ranked_data,axis=-1)

                                for method_id, method_name in enumerate(method_names):

                                    plt.plot([int(i) for i in snrs], avg_ranks[method_id,:], marker="o", label = method_name)

                                plt.title("{}, {}, {}, {}, GS {} ".format(metric_name, extractor_name, classifier_name, outlier_detector_name,group_size_name ))
                                plt.xlabel("SNR")
                                plt.ylabel("Criterion avg rank")
                                plt.legend()
                                pdf.savefig()
                                plt.close()

                                #avg_ranks (methods, snrs)
                                mi = pd.MultiIndex.from_arrays( [ ["{}".format(metric_name) for _ in range(n_methods)], [m for m in method_names] ] )
                                av_rnk_df = pd.DataFrame(avg_ranks.T,columns=mi ,
                                                            index=["Avg Rnk {}, snr:{}".format(a,si) for si,a in zip( snrs,string.ascii_letters)])
                                
                                # methods  0 x snrs 1 x ( files 2 x spoilers 3 x ch_frac 4)
                                sub_results_snr = np.mean(sub_results_r, axis=-1).reshape((n_methods, n_snrs, -1))
                                for snr_id, (snr_name,snr_letter) in enumerate( zip(snrs, string.ascii_letters )):
                                    # methods    x ( files  x spoilers )
                                    values = sub_results_snr[:,snr_id]
                                    p_vals = np.zeros( (n_methods, n_methods) )
                                    for i in range(n_methods):
                                        for j in range(n_methods):
                                            if i == j:
                                                continue

                                            values_squared_diff = np.sqrt (np.sum( (values[i,:] - values[j,:])**2 ) )
                                            if values_squared_diff > 1E-4:
                                                with warnings.catch_warnings(): #Normal approximation
                                                    warnings.simplefilter("ignore")
                                                    p_vals[i,j]  = wilcoxon(values[i], values[j]).pvalue #mannwhitneyu(values[:,i], values[:,j]).pvalue
                                            else:
                                                p_vals[i,j] = 1.0
                                    
                                    p_val_vec  = p_val_matrix_to_vec(p_vals)

                                    p_val_vec_corrected = multipletests(p_val_vec, method='hommel')
                                    corr_p_val_matrix = p_val_vec_to_matrix(p_val_vec_corrected[1],n_methods,symmetrize=True)

                                    s_test_outcome = []
                                    for i in range(n_methods):
                                        out_a = []
                                        for j in range(n_methods):
                                            if avg_ranks[i,snr_id] > avg_ranks[j, snr_id] and corr_p_val_matrix[i,j]<alpha:
                                                out_a.append(j+1)
                                        if len(out_a) == 0:
                                            s_test_outcome.append("--")
                                        else:
                                            s_test_outcome.append("{\\scriptsize " + ",".join([ str(x) for x in out_a]) + "}") 
                                    av_rnk_df.loc["{} {}, snr:{}_T".format("Avg Rnk",snr_letter,snr_name)] = s_test_outcome
                                    av_rnk_df.sort_index(inplace=True)

                                av_rnk_df.style.format(precision=3,na_rep="")\
                                .format_index(escape="latex", axis=0).format_index(escape="latex", axis=1)\
                                .to_latex(report_file_handler, multicol_align="c")

        report_file_handler.close()


def analyze_results_3(results_directory, output_directory):
    configurer.configure_plots()
    result_files = [f for f in os.listdir(results_directory) if f.endswith(".pickle") ]

    for result_file in result_files:
        result_file_basename = os.path.splitext(result_file)[0]
        result_file_path = os.path.join(results_directory, result_file)

        
        raw_data = pickle.load(open(result_file_path,"rb"))

        classifier_names = raw_data["classifier_names"]
        metric_names = raw_data["metric_names"]
        extractor_names = raw_data["extractors"]
        group_size_names = raw_data["group_sizes"]
        outlier_detector_names = raw_data["outlier_detectors"]
        spoiler_names = raw_data["spoilers"]
        snrs = raw_data["snrs"]
        spoiled_channels_fraction = raw_data["spoil_ch_f"]
        method_names = raw_data["methods"]
        
        # metrics x extractors x classifiers x n_group_sizes x detectors x spoilers x snr x ch_fraction x methods x folds
        results = raw_data["results"]
        n_methods = len(method_names)

        pdf_file_path = os.path.join(output_directory, "{}_no_noise_gs.pdf".format(result_file_basename))


        with PdfPages(pdf_file_path) as pdf:

            for metric_id, metric_name in enumerate(metric_names):
                
                for extractor_id, extractor_name in enumerate(extractor_names):
                    
                    for classifier_id, classifier_name in enumerate(classifier_names):
                        
                        for outlier_detector_id, outlier_detector_name in enumerate(outlier_detector_names):
                                                            
                            # metrics x extractors x classifiers x n_group_sizes x detectors x spoilers x snr x ch_fraction x methods x folds

                            sub_results = results[metric_id, extractor_id, classifier_id,:,outlier_detector_id, :,:,0] # group sizes x spoilers x snr x methods x folds
                            sub_results = np.mean(sub_results, axis=(1,2)) # group sizes x methods x folds
                        
                            df = pd.DataFrame(columns=['GS','method','value'])

                            gs_values = [int(v) for v in group_size_names]
                            for i, gs_value in enumerate(gs_values):
                                for j, method_name in enumerate(method_names):
                                    for k in range(sub_results.shape[2]):
                                        new_row = pd.DataFrame({'GS':gs_value, 'method':method_name, 'value':sub_results[i,j,k]}, index=[0])
                                        df = pd.concat([new_row,df.loc[:]]).reset_index(drop=True)


                            sns.boxplot(x=df['GS'], y=df['value'], hue=df['method'], palette='husl')
                            plt.title( "{}, {}, {} , Od:{}".format(metric_name, extractor_name, classifier_name,
                                                                                                outlier_detector_name,
                                                                                                            ))
            
                            pdf.savefig()
                            plt.close()


def analyze_results_3_ranks(results_directory, output_directory, alpha=0.05):
    configurer.configure_plots()
    result_files = [f for f in os.listdir(results_directory) if f.endswith(".pickle") ]

    n_files = len(result_files)

    #Init

    classifier_names = None
    n_classifiers = None
    metric_names = None
    n_metrics = None
    extractor_names = None
    n_extractors = None
    group_size_names = None
    n_group_sizes = None
    outlier_detector_names = None
    n_outlier_detectors = None
    spoiler_names = None
    n_spoilers = None
    snrs = None
    n_snrs = None
    spoiled_channels_fraction = None
    n_sp_channel_fraction = None
    method_names = None
    n_methods = None
    n_folds = None

    global_results = None


    for result_file_id,  result_file in enumerate(result_files):
        result_file_basename = os.path.splitext(result_file)[0]
        result_file_path = os.path.join(results_directory, result_file)

        raw_data = pickle.load(open(result_file_path,"rb"))
        results = raw_data["results"]

        if global_results is None:

            classifier_names = raw_data["classifier_names"]
            n_classifiers = len(classifier_names)
            metric_names = raw_data["metric_names"]
            n_metrics = len(metric_names)
            extractor_names = raw_data["extractors"]
            n_extractors = len(extractor_names)
            group_size_names = raw_data["group_sizes"]
            n_group_sizes = len(group_size_names)
            outlier_detector_names = raw_data["outlier_detectors"]
            n_outlier_detectors = len(outlier_detector_names)
            spoiler_names = raw_data["spoilers"]
            n_spoilers = len(spoiler_names)
            snrs = raw_data["snrs"]
            n_snrs = len(snrs)
            spoiled_channels_fraction = raw_data["spoil_ch_f"]
            n_sp_channel_fraction = len(spoiled_channels_fraction)
            method_names = raw_data["methods"]
            n_methods = len(method_names)
            
            # metrics x extractors x classifiers x n_group_sizes x detectors x spoilers x snr x ch_fraction x methods x folds
            
            n_folds = results.shape[-1]

            # n_files x metrics x extractors x classifiers x n_group_sizes x detectors x spoilers x snr x ch_fraction x methods x folds
            global_results = np.zeros( ( n_files, n_metrics, n_extractors, n_classifiers, n_group_sizes, n_outlier_detectors,
                                      n_spoilers, n_snrs, n_sp_channel_fraction, n_methods, n_folds ))
        
        global_results[result_file_id] = results
            

    pdf_file_path = os.path.join(output_directory, "{}_no_noise_gs_ranks.pdf".format("ALL"))
    report_file_path = os.path.join(output_directory, "{}_no_noise_gs_ranks.md".format("ALL"))
    report_file_handler = open(report_file_path, 'w+')


    with PdfPages(pdf_file_path) as pdf:

        for metric_id, metric_name in enumerate(metric_names):
            print("# {}".format(metric_name), file=report_file_handler)
            
            for extractor_id, extractor_name in enumerate(extractor_names):
                print("## {}".format(extractor_name), file=report_file_handler)
                
                for classifier_id, classifier_name in enumerate(classifier_names):
                    print("## {}".format(classifier_name), file=report_file_handler)
                    
                    for outlier_detector_id, outlier_detector_name in enumerate(outlier_detector_names):
                        print("### {}".format(outlier_detector_name), file=report_file_handler)

                        # files x metrics x extractors x classifiers x n_group_sizes x detectors x spoilers x snr x ch_fraction x methods x folds 
                        # files [0] x group sizes [1] x spoilers [2] x snr [3]  x methods [4] x folds [5] 
                        sub_results = global_results[:,metric_id, extractor_id, classifier_id,:,outlier_detector_id, :,:,0] 

                        # methods [0] x group size [1] x ( files [2] * spoilers [3] * snrs [4] * folds [5] )
                        sub_results_r = np.moveaxis(sub_results,[0,1,2,3,4,5], [2,1,3,4,0,5])
                        sub_results = sub_results_r.reshape( (n_methods, n_group_sizes, -1) )

                        ranked_data = rankdata(sub_results, axis=0)
                        # methods, group_sizes
                        avg_ranks = np.mean(ranked_data,axis=-1)

                        for method_id, method_name in enumerate(method_names):

                             plt.plot([int(i) for i in group_size_names], avg_ranks[method_id,:], marker="o", label = method_name)

                        plt.title("{}, {}, {}, {}".format(metric_name, extractor_name, classifier_name, outlier_detector_name))
                        plt.xlabel("Group size")
                        plt.ylabel("Criterion avg rank")
                        plt.legend()
                        pdf.savefig()
                        plt.close()

                        #avg_ranks (methods, snrs)
                        mi = pd.MultiIndex.from_arrays( [ ["{}".format(metric_name) for _ in range(n_methods)], [m for m in method_names] ] )
                        av_rnk_df = pd.DataFrame(avg_ranks.T,columns=mi ,
                                                    index=["Avg Rnk {}, gs:{}".format(a,si) for si,a in zip( group_size_names,string.ascii_letters)])

                        # methods [0] x group size [1] x ( files [2] * spoilers [3] * snrs [4]  )
                        sub_results_snr = np.mean(sub_results_r, axis=-1).reshape((n_methods, n_group_sizes, -1))
                        for gs_id, (gs_name,gs_letter) in enumerate( zip(group_size_names, string.ascii_letters )):
                            # methods    x ( files * spoilers * snrs  )
                            values = sub_results_snr[:,gs_id]
                            p_vals = np.zeros( (n_methods, n_methods) )
                            for i in range(n_methods):
                                for j in range(n_methods):
                                    if i == j:
                                        continue

                                    values_squared_diff = np.sqrt (np.sum( (values[i,:] - values[j,:])**2 ) )
                                    if values_squared_diff > 1E-4:
                                        with warnings.catch_warnings(): #Normal approximation
                                            warnings.simplefilter("ignore")
                                            p_vals[i,j]  = wilcoxon(values[i], values[j]).pvalue #mannwhitneyu(values[:,i], values[:,j]).pvalue
                                    else:
                                        p_vals[i,j] = 1.0
                            
                            p_val_vec  = p_val_matrix_to_vec(p_vals)

                            p_val_vec_corrected = multipletests(p_val_vec, method='hommel')
                            corr_p_val_matrix = p_val_vec_to_matrix(p_val_vec_corrected[1],n_methods,symmetrize=True)

                            s_test_outcome = []
                            for i in range(n_methods):
                                out_a = []
                                for j in range(n_methods):
                                    if avg_ranks[i,gs_id] > avg_ranks[j, gs_id] and corr_p_val_matrix[i,j]<alpha:
                                        out_a.append(j+1)
                                if len(out_a) == 0:
                                    s_test_outcome.append("--")
                                else:
                                    s_test_outcome.append("{\\scriptsize " + ",".join([ str(x) for x in out_a]) + "}") 
                            av_rnk_df.loc["{} {}, gs:{}_T".format("Avg Rnk",gs_letter,gs_name)] = s_test_outcome
                            av_rnk_df.sort_index(inplace=True)

                        av_rnk_df.style.format(precision=3,na_rep="")\
                        .format_index(escape="latex", axis=0).format_index(escape="latex", axis=1)\
                        .to_latex(report_file_handler, multicol_align="c")
    report_file_handler.close()


def analyze_results_3B(results_directory, output_directory):
    configurer.configure_plots()
    result_files = [f for f in os.listdir(results_directory) if f.endswith(".pickle") ]

    for result_file in result_files:
        result_file_basename = os.path.splitext(result_file)[0]
        result_file_path = os.path.join(results_directory, result_file)

        
        raw_data = pickle.load(open(result_file_path,"rb"))

        classifier_names = raw_data["classifier_names"]
        metric_names = raw_data["metric_names"]
        extractor_names = raw_data["extractors"]
        group_size_names = raw_data["group_sizes"]
        outlier_detector_names = raw_data["outlier_detectors"]
        spoiler_names = raw_data["spoilers"]
        snrs = raw_data["snrs"]
        spoiled_channels_fraction = raw_data["spoil_ch_f"]
        method_names = raw_data["methods"]
        
        # metrics x extractors x classifiers x n_group_sizes x detectors x spoilers x snr x ch_fraction x methods x folds
        results = raw_data["results"]
        n_methods = len(method_names)

        pdf_file_path = os.path.join(output_directory, "{}_noise_gs.pdf".format(result_file_basename))


        with PdfPages(pdf_file_path) as pdf:

            for metric_id, metric_name in enumerate(metric_names):
                
                for extractor_id, extractor_name in enumerate(extractor_names):
                    
                    for classifier_id, classifier_name in enumerate(classifier_names):
                        
                        for outlier_detector_id, outlier_detector_name in enumerate(outlier_detector_names):
                                                            
                            # metrics x extractors x classifiers x n_group_sizes x detectors x spoilers x snr x ch_fraction x methods x folds

                            sub_results = results[metric_id, extractor_id, classifier_id,:,outlier_detector_id, :,:,:] # group sizes x spoilers x snr x ch_fraction x methods x folds
                            sub_results = np.mean(sub_results, axis=(1,2,3)) # group sizes x methods x folds
                        
                            df = pd.DataFrame(columns=['GS','method','value'])

                            gs_values = [int(v) for v in group_size_names]
                            for i, gs_value in enumerate(gs_values):
                                for j, method_name in enumerate(method_names):
                                    for k in range(sub_results.shape[2]):
                                        new_row = pd.DataFrame({'GS':gs_value, 'method':method_name, 'value':sub_results[i,j,k]}, index=[0])
                                        df = pd.concat([new_row,df.loc[:]]).reset_index(drop=True)


                            sns.boxplot(x=df['GS'], y=df['value'], hue=df['method'], palette='husl')
                            plt.title( "{}, {}, {} , Od:{}".format(metric_name, extractor_name, classifier_name,
                                                                                                outlier_detector_name,
                                                                                                            ))
            
                            pdf.savefig()
                            plt.close()

def analyze_results_3B_ranks(results_directory, output_directory, alpha=0.05):
    configurer.configure_plots()
    result_files = [f for f in os.listdir(results_directory) if f.endswith(".pickle") ]

    n_files = len(result_files)

    #Init

    classifier_names = None
    n_classifiers = None
    metric_names = None
    n_metrics = None
    extractor_names = None
    n_extractors = None
    group_size_names = None
    n_group_sizes = None
    outlier_detector_names = None
    n_outlier_detectors = None
    spoiler_names = None
    n_spoilers = None
    snrs = None
    n_snrs = None
    spoiled_channels_fraction = None
    n_sp_channel_fraction = None
    method_names = None
    n_methods = None
    n_folds = None

    global_results = None


    for result_file_id,  result_file in enumerate(result_files):
        result_file_basename = os.path.splitext(result_file)[0]
        result_file_path = os.path.join(results_directory, result_file)

        raw_data = pickle.load(open(result_file_path,"rb"))
        results = raw_data["results"]

        if global_results is None:

            classifier_names = raw_data["classifier_names"]
            n_classifiers = len(classifier_names)
            metric_names = raw_data["metric_names"]
            n_metrics = len(metric_names)
            extractor_names = raw_data["extractors"]
            n_extractors = len(extractor_names)
            group_size_names = raw_data["group_sizes"]
            n_group_sizes = len(group_size_names)
            outlier_detector_names = raw_data["outlier_detectors"]
            n_outlier_detectors = len(outlier_detector_names)
            spoiler_names = raw_data["spoilers"]
            n_spoilers = len(spoiler_names)
            snrs = raw_data["snrs"]
            n_snrs = len(snrs)
            spoiled_channels_fraction = raw_data["spoil_ch_f"]
            n_sp_channel_fraction = len(spoiled_channels_fraction)
            method_names = raw_data["methods"]
            n_methods = len(method_names)
            
            # metrics x extractors x classifiers x n_group_sizes x detectors x spoilers x snr x ch_fraction x methods x folds
            
            n_folds = results.shape[-1]

            # n_files x metrics x extractors x classifiers x n_group_sizes x detectors x spoilers x snr x ch_fraction x methods x folds
            global_results = np.zeros( ( n_files, n_metrics, n_extractors, n_classifiers, n_group_sizes, n_outlier_detectors,
                                      n_spoilers, n_snrs, n_sp_channel_fraction, n_methods, n_folds ))
        
        global_results[result_file_id] = results
            

    pdf_file_path = os.path.join(output_directory, "{}_noise_gs_ranks.pdf".format("ALL"))
    report_file_path = os.path.join(output_directory, "{}_noise_gs_ranks.md".format("ALL"))
    report_file_handler = open(report_file_path, 'w+')


    with PdfPages(pdf_file_path) as pdf:

        for metric_id, metric_name in enumerate(metric_names):
            print("# {}".format(metric_name), file=report_file_handler)
            
            for extractor_id, extractor_name in enumerate(extractor_names):
                print("## {}".format(extractor_name), file=report_file_handler)
                
                for classifier_id, classifier_name in enumerate(classifier_names):
                    print("### {}".format(classifier_name), file=report_file_handler)
                    
                    for outlier_detector_id, outlier_detector_name in enumerate(outlier_detector_names):
                        print("#### {}".format(outlier_detector_name), file=report_file_handler)
   
                        # files x metrics x extractors x classifiers x n_group_sizes x detectors x spoilers x snr x ch_fraction x methods x folds 
                        # files [0] x group sizes [1] x spoilers [2] x snr [3] x ch_fraction [4] x methods [5] x folds [6] 
                        sub_results = global_results[:,metric_id, extractor_id, classifier_id,:,outlier_detector_id, :,:,:] 

                        # methods [0] x group size [1] x ( files [2] * spoilers [3] * snrs [4] * ch_fraction [5] * folds [6] )
                        sub_results_r = np.moveaxis(sub_results,[0,1,2,3,4,5,6], [2,1,3,4,5,0,6])
                        sub_results = sub_results_r.reshape( (n_methods, n_group_sizes, -1) )

                        ranked_data = rankdata(sub_results, axis=0)
                        # methods, group_sizes
                        avg_ranks = np.mean(ranked_data,axis=-1)

                        for method_id, method_name in enumerate(method_names):

                             plt.plot([int(i) for i in group_size_names], avg_ranks[method_id,:], marker="o", label = method_name)

                        plt.title("{}, {}, {}, {}".format(metric_name, extractor_name, classifier_name, outlier_detector_name))
                        plt.xlabel("Group size")
                        plt.ylabel("Criterion avg rank")
                        plt.legend()
                        pdf.savefig()
                        plt.close()

                        mi = pd.MultiIndex.from_arrays( [ ["{}".format(metric_name) for _ in range(n_methods)], [m for m in method_names] ] )
                        av_rnk_df = pd.DataFrame(avg_ranks.T,columns=mi ,
                                                    index=["Avg Rnk {}, gs:{}".format(a,si) for si,a in zip( group_size_names,string.ascii_letters)])

                        # methods [0] x group size [1] x ( files [2] * spoilers [3] * snrs [4] * ch_fraction [5] )
                        sub_results_gs = np.mean(sub_results_r, axis=-1).reshape((n_methods, n_group_sizes, -1))
                        for gs_id, (gs_name,gs_letter) in enumerate( zip(group_size_names, string.ascii_letters )):
                            # methods    x ( files * spoilers * snrs  )
                            values = sub_results_gs[:,gs_id]
                            p_vals = np.zeros( (n_methods, n_methods) )
                            for i in range(n_methods):
                                for j in range(n_methods):
                                    if i == j:
                                        continue

                                    values_squared_diff = np.sqrt (np.sum( (values[i,:] - values[j,:])**2 ) )
                                    if values_squared_diff > 1E-4:
                                        with warnings.catch_warnings(): #Normal approximation
                                            warnings.simplefilter("ignore")
                                            p_vals[i,j]  = wilcoxon(values[i], values[j]).pvalue #mannwhitneyu(values[:,i], values[:,j]).pvalue
                                    else:
                                        p_vals[i,j] = 1.0
                            
                            p_val_vec  = p_val_matrix_to_vec(p_vals)

                            p_val_vec_corrected = multipletests(p_val_vec, method='hommel')
                            corr_p_val_matrix = p_val_vec_to_matrix(p_val_vec_corrected[1],n_methods,symmetrize=True)

                            s_test_outcome = []
                            for i in range(n_methods):
                                out_a = []
                                for j in range(n_methods):
                                    if avg_ranks[i,gs_id] > avg_ranks[j, gs_id] and corr_p_val_matrix[i,j]<alpha:
                                        out_a.append(j+1)
                                if len(out_a) == 0:
                                    s_test_outcome.append("--")
                                else:
                                    s_test_outcome.append("{\\scriptsize " + ",".join([ str(x) for x in out_a]) + "}") 
                            av_rnk_df.loc["{} {}, gs:{}_T".format("Avg Rnk",gs_letter,gs_name)] = s_test_outcome
                            av_rnk_df.sort_index(inplace=True)

                        av_rnk_df.style.format(precision=3,na_rep="")\
                        .format_index(escape="latex", axis=0).format_index(escape="latex", axis=1)\
                        .to_latex(report_file_handler, multicol_align="c")
    report_file_handler.close()

def analyze_results_3C_ranks(results_directory, output_directory, alpha=0.05):
    configurer.configure_plots()
    result_files = [f for f in os.listdir(results_directory) if f.endswith(".pickle") ]

    n_files = len(result_files)

    #Init

    classifier_names = None
    n_classifiers = None
    metric_names = None
    n_metrics = None
    extractor_names = None
    n_extractors = None
    group_size_names = None
    n_group_sizes = None
    outlier_detector_names = None
    n_outlier_detectors = None
    spoiler_names = None
    n_spoilers = None
    snrs = None
    n_snrs = None
    spoiled_channels_fraction = None
    n_sp_channel_fraction = None
    method_names = None
    n_methods = None
    n_folds = None

    global_results = None


    for result_file_id,  result_file in enumerate(result_files):
        result_file_basename = os.path.splitext(result_file)[0]
        result_file_path = os.path.join(results_directory, result_file)

        raw_data = pickle.load(open(result_file_path,"rb"))
        results = raw_data["results"]

        if global_results is None:

            classifier_names = raw_data["classifier_names"]
            n_classifiers = len(classifier_names)
            metric_names = raw_data["metric_names"]
            n_metrics = len(metric_names)
            extractor_names = raw_data["extractors"]
            n_extractors = len(extractor_names)
            group_size_names = raw_data["group_sizes"]
            n_group_sizes = len(group_size_names)
            outlier_detector_names = raw_data["outlier_detectors"]
            n_outlier_detectors = len(outlier_detector_names)
            spoiler_names = raw_data["spoilers"]
            n_spoilers = len(spoiler_names)
            snrs = raw_data["snrs"]
            n_snrs = len(snrs)
            spoiled_channels_fraction = raw_data["spoil_ch_f"]
            n_sp_channel_fraction = len(spoiled_channels_fraction)
            method_names = raw_data["methods"]
            n_methods = len(method_names)
            
            # metrics x extractors x classifiers x n_group_sizes x detectors x spoilers x snr x ch_fraction x methods x folds
            
            n_folds = results.shape[-1]

            # n_files x metrics x extractors x classifiers x n_group_sizes x detectors x spoilers x snr x ch_fraction x methods x folds
            global_results = np.zeros( ( n_files, n_metrics, n_extractors, n_classifiers, n_group_sizes, n_outlier_detectors,
                                      n_spoilers, n_snrs, n_sp_channel_fraction, n_methods, n_folds ))
        
        global_results[result_file_id] = results
            

    pdf_file_path = os.path.join(output_directory, "{}_noise_gs_ranks_alt.pdf".format("ALL"))
    report_file_path = os.path.join(output_directory, "{}_noise_gs_ranks_alt.md".format("ALL"))
    report_file_handler = open(report_file_path, 'w+')


    with PdfPages(pdf_file_path) as pdf:

        for metric_id, metric_name in enumerate(metric_names):
            print("# {}".format(metric_name), file=report_file_handler)
            
            for extractor_id, extractor_name in enumerate(extractor_names):
                print("## {}".format(extractor_name), file=report_file_handler)
                
                for classifier_id, classifier_name in enumerate(classifier_names):
                    print("### {}".format(classifier_name), file=report_file_handler)
                    
                    for outlier_detector_id, outlier_detector_name in enumerate(outlier_detector_names):
                        print("#### {}".format(outlier_detector_name), file=report_file_handler)

                        # files x metrics x extractors x classifiers x n_group_sizes x detectors x spoilers x snr x ch_fraction x methods x folds
                        # files [0] x group sizes [1] x spoilers [2] x snr [3] x ch_fraction [4] x methods [5] x folds [6] 
                        sub_results = global_results[:,metric_id, extractor_id, classifier_id,:,outlier_detector_id, :,:,:] 

                        # methods [0] x group size [1] x ( files [2] * spoilers [3] * snrs [4] * ch_fraction [5] * folds [6] )
                        sub_results_r = np.moveaxis(sub_results,[0,1,2,3,4,5,6], [2,1,3,4,5,0,6])
                        sub_results = sub_results_r.reshape( (n_methods, n_group_sizes, -1) )

                        ranked_data = rankdata(sub_results, axis=1)
                        # methods, group_sizes
                        avg_ranks = np.mean(ranked_data,axis=-1)

                        for method_id, method_name in enumerate(method_names):

                             plt.plot([int(i) for i in group_size_names], avg_ranks[method_id,:], marker="o", label = method_name)

                        plt.title("{}, {}, {}, {}".format(metric_name, extractor_name, classifier_name, outlier_detector_name))
                        plt.xlabel("Group size")
                        plt.ylabel("Criterion avg rank")
                        plt.legend()
                        pdf.savefig()
                        plt.close()
                        
                        mi = pd.MultiIndex.from_arrays( [ ["{}".format(metric_name) for _ in range(n_group_sizes)], [m for m in group_size_names] ] )
                        av_rnk_df = pd.DataFrame(avg_ranks,columns=mi ,
                                                    index=["Avg Rnk {}, method:{}".format(a,si) for si,a in zip( method_names,string.ascii_letters)])
                        
                        # methods [0] x group size [1] x ( files [2] * spoilers [3] * snrs [4] * ch_fraction [5])
                        sub_results_gs = np.mean(sub_results_r, axis=-1).reshape((n_methods, n_group_sizes, -1))
                        for method_id, (method_name, method_letter) in enumerate( zip(method_names, string.ascii_letters )):
                            #group size [1] x ( files [2] * spoilers [3] * snrs [4] * ch_fraction [5])
                            values = sub_results_gs[method_id]
                            p_vals = np.zeros( (n_group_sizes, n_group_sizes) )
                            for i in range(n_group_sizes):
                                for j in range(n_group_sizes):
                                    if i == j:
                                        continue

                                    values_squared_diff = np.sqrt (np.sum( (values[i,:] - values[j,:])**2 ) )
                                    if values_squared_diff > 1E-4:
                                        with warnings.catch_warnings(): #Normal approximation
                                            warnings.simplefilter("ignore")
                                            p_vals[i,j]  = wilcoxon(values[i], values[j]).pvalue #mannwhitneyu(values[:,i], values[:,j]).pvalue
                                    else:
                                        p_vals[i,j] = 1.0
                            
                            p_val_vec  = p_val_matrix_to_vec(p_vals)

                            p_val_vec_corrected = multipletests(p_val_vec, method='hommel')
                            corr_p_val_matrix = p_val_vec_to_matrix(p_val_vec_corrected[1],n_group_sizes,symmetrize=True)

                            s_test_outcome = []
                            for i in range(n_group_sizes):
                                out_a = []
                                for j in range(n_group_sizes):
                                    # methods, group_sizes
                                    if avg_ranks[method_id,i] > avg_ranks[method_id,j] and corr_p_val_matrix[i,j]<alpha:
                                        out_a.append(j+1)
                                if len(out_a) == 0:
                                    s_test_outcome.append("--")
                                else:
                                    s_test_outcome.append("{\\scriptsize " + ",".join([ str(x) for x in out_a]) + "}") 
                            av_rnk_df.loc["{} {}, method:{}_T".format("Avg Rnk",method_letter,method_name)] = s_test_outcome
                            av_rnk_df.sort_index(inplace=True)

                        av_rnk_df.style.format(precision=3,na_rep="")\
                        .format_index(escape="latex", axis=0).format_index(escape="latex", axis=1)\
                        .to_latex(report_file_handler, multicol_align="c")
    report_file_handler.close()

def analyze_results_3D_ranks(results_directory, output_directory, alpha=0.05):
    configurer.configure_plots()
    result_files = [f for f in os.listdir(results_directory) if f.endswith(".pickle") ]

    n_files = len(result_files)

    #Init

    classifier_names = None
    n_classifiers = None
    metric_names = None
    n_metrics = None
    extractor_names = None
    n_extractors = None
    group_size_names = None
    n_group_sizes = None
    outlier_detector_names = None
    n_outlier_detectors = None
    spoiler_names = None
    n_spoilers = None
    snrs = None
    n_snrs = None
    spoiled_channels_fraction = None
    n_sp_channel_fraction = None
    method_names = None
    n_methods = None
    n_folds = None

    global_results = None


    for result_file_id,  result_file in enumerate(result_files):
        result_file_basename = os.path.splitext(result_file)[0]
        result_file_path = os.path.join(results_directory, result_file)

        raw_data = pickle.load(open(result_file_path,"rb"))
        results = raw_data["results"]

        if global_results is None:

            classifier_names = raw_data["classifier_names"]
            n_classifiers = len(classifier_names)
            metric_names = raw_data["metric_names"]
            n_metrics = len(metric_names)
            extractor_names = raw_data["extractors"]
            n_extractors = len(extractor_names)
            group_size_names = raw_data["group_sizes"]
            n_group_sizes = len(group_size_names)
            outlier_detector_names = raw_data["outlier_detectors"]
            n_outlier_detectors = len(outlier_detector_names)
            spoiler_names = raw_data["spoilers"]
            n_spoilers = len(spoiler_names)
            snrs = raw_data["snrs"]
            n_snrs = len(snrs)
            spoiled_channels_fraction = raw_data["spoil_ch_f"]
            n_sp_channel_fraction = len(spoiled_channels_fraction)
            method_names = raw_data["methods"]
            n_methods = len(method_names)
            
            # metrics x extractors x classifiers x n_group_sizes x detectors x spoilers x snr x ch_fraction x methods x folds
            
            n_folds = results.shape[-1]

            # n_files x metrics x extractors x classifiers x n_group_sizes x detectors x spoilers x snr x ch_fraction x methods x folds
            global_results = np.zeros( ( n_files, n_metrics, n_extractors, n_classifiers, n_group_sizes, n_outlier_detectors,
                                      n_spoilers, n_snrs, n_sp_channel_fraction, n_methods, n_folds ))
        
        global_results[result_file_id] = results
            

    pdf_file_path = os.path.join(output_directory, "{}_no_noise_gs_ranks_alt.pdf".format("ALL"))
    report_file_path = os.path.join(output_directory, "{}_no_noise_gs_ranks_alt.md".format("ALL"))
    report_file_handler = open(report_file_path, 'w+')

    with PdfPages(pdf_file_path) as pdf:

        for metric_id, metric_name in enumerate(metric_names):
            print("# {}".format(metric_name), file=report_file_handler)
            
            for extractor_id, extractor_name in enumerate(extractor_names):
                print("## {}".format(extractor_name), file=report_file_handler)
                
                for classifier_id, classifier_name in enumerate(classifier_names):
                    print("### {}".format(classifier_name), file=report_file_handler)
                    
                    for outlier_detector_id, outlier_detector_name in enumerate(outlier_detector_names):
                        print("#### {}".format(outlier_detector_name), file=report_file_handler)
          
                        # files x metrics x extractors x classifiers x n_group_sizes x detectors x spoilers x snr x ch_fraction x methods x folds 
                        # files [0] x group sizes [1] x spoilers [2] x snr [3]  x methods [4] x folds [5] 
                        sub_results = global_results[:,metric_id, extractor_id, classifier_id,:,outlier_detector_id, :,:,0] 

                        # methods [0] x group size [1] x ( files [2] * spoilers [3] * snrs [4] * folds [5] )
                        sub_results_r = np.moveaxis(sub_results,[0,1,2,3,4,5], [2,1,3,4,0,5])
                        sub_results = sub_results_r.reshape( (n_methods, n_group_sizes, -1) )

                        ranked_data = rankdata(sub_results, axis=1)
                        # methods, group_sizes
                        avg_ranks = np.mean(ranked_data,axis=-1)

                        for method_id, method_name in enumerate(method_names):

                             plt.plot([int(i) for i in group_size_names], avg_ranks[method_id,:], marker="o", label = method_name)

                        plt.title("{}, {}, {}, {}".format(metric_name, extractor_name, classifier_name, outlier_detector_name))
                        plt.xlabel("Group size")
                        plt.ylabel("Criterion avg rank")
                        plt.legend()
                        pdf.savefig()
                        plt.close()

                        mi = pd.MultiIndex.from_arrays( [ ["{}".format(metric_name) for _ in range(n_group_sizes)], [m for m in group_size_names] ] )
                        av_rnk_df = pd.DataFrame(avg_ranks,columns=mi ,
                                                    index=["Avg Rnk {}, method:{}".format(a,si) for si,a in zip( method_names,string.ascii_letters)])
                        
                        # methods [0] x group size [1] x ( files [2] * spoilers [3] * snrs [4])
                        sub_results_gs = np.mean(sub_results_r, axis=-1).reshape((n_methods, n_group_sizes, -1))
                        for method_id, (method_name, method_letter) in enumerate( zip(method_names, string.ascii_letters )):
                            #group size [1] x ( files [2] * spoilers [3] * snrs [4])
                            values = sub_results_gs[method_id]
                            p_vals = np.zeros( (n_group_sizes, n_group_sizes) )
                            for i in range(n_group_sizes):
                                for j in range(n_group_sizes):
                                    if i == j:
                                        continue

                                    values_squared_diff = np.sqrt (np.sum( (values[i,:] - values[j,:])**2 ) )
                                    if values_squared_diff > 1E-4:
                                        with warnings.catch_warnings(): #Normal approximation
                                            warnings.simplefilter("ignore")
                                            p_vals[i,j]  = wilcoxon(values[i], values[j]).pvalue #mannwhitneyu(values[:,i], values[:,j]).pvalue
                                    else:
                                        p_vals[i,j] = 1.0
                            
                            p_val_vec  = p_val_matrix_to_vec(p_vals)

                            p_val_vec_corrected = multipletests(p_val_vec, method='hommel')
                            corr_p_val_matrix = p_val_vec_to_matrix(p_val_vec_corrected[1],n_group_sizes,symmetrize=True)

                            s_test_outcome = []
                            for i in range(n_group_sizes):
                                out_a = []
                                for j in range(n_group_sizes):
                                    # methods, group_sizes
                                    if avg_ranks[method_id,i] > avg_ranks[method_id,j] and corr_p_val_matrix[i,j]<alpha:
                                        out_a.append(j+1)
                                if len(out_a) == 0:
                                    s_test_outcome.append("--")
                                else:
                                    s_test_outcome.append("{\\scriptsize " + ",".join([ str(x) for x in out_a]) + "}") 
                            av_rnk_df.loc["{} {}, method:{}_T".format("Avg Rnk",method_letter,method_name)] = s_test_outcome
                            av_rnk_df.sort_index(inplace=True)

                        av_rnk_df.style.format(precision=3,na_rep="")\
                        .format_index(escape="latex", axis=0).format_index(escape="latex", axis=1)\
                        .to_latex(report_file_handler, multicol_align="c")
    report_file_handler.close()


if __name__ == '__main__':
    
    np.random.seed(0)
    random.seed(0)

    
    
    # data_path2 = os.path.join(settings.DATAPATH,"ottobock")
    
    data_sets = [ os.path.join( settings.DATAPATH, "tsnre_windowed","A{}_Force_Exp_low_windowed".format(i)) for i in range(1,3) ]
    # data_sets = [data_path2]

    output_directory = os.path.join(settings.EXPERIMENTS_RESULTS_PATH,"./results_channel_combination_ensemble_full2/")

    log_dir = os.path.dirname(settings.EXPERIMENTS_LOGS_PATH)
    log_file = os.path.splitext( os.path.basename(__file__) )[0]
    logger(log_dir,log_file, enable_logging=False)
    warnings.showwarning = warn_with_traceback    
    
    run_experiment(data_sets, output_directory, n_splits=8, n_repeats=1,
                    random_state=0, n_jobs=-1, overwrite=True,n_channels=8)

    analysis_functions =[ 
                        analyze_results_2B,
                        analyze_results_2B_ranks,
                        
                        analyze_results_2C,
                        analyze_results_2C_ranks,
                        
                        analyze_results_3,
                        analyze_results_3_ranks,
                        
                        analyze_results_3B,
                        analyze_results_3B_ranks,
                        
                        analyze_results_3C_ranks,
                        
                        analyze_results_3D_ranks,
                        
                        ]
    
    ProgressParallel(backend="multiprocessing",n_jobs=-1, desc="Analysis", total=len(analysis_functions), leave=False)\
                            (delayed(fun)(output_directory, output_directory) for fun in  analysis_functions ) 