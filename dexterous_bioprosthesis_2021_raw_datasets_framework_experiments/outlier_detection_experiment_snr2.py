import os
import string
import warnings
from dexterous_bioprosthesis_2021_raw_datasets_framework.tools.stats_tools import p_val_matrix_to_vec, p_val_vec_to_matrix

import numpy as np
import pandas as pd
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.svm import OneClassSVM

from statsmodels.stats.multitest import multipletests
from dexterous_bioprosthesis_2021_raw_datasets_framework.estimators.meta.parameter_selection.gridsearchcv_oneclass2 import GridSearchCVOneClass2
from dexterous_bioprosthesis_2021_raw_datasets_framework.preprocessing.dummy_transformer import DummyTransformer
from dexterous_bioprosthesis_2021_raw_datasets_framework.preprocessing.one_class.outlier_generators.outlier_generator_uniform import OutlierGeneratorUniform
from dexterous_bioprosthesis_2021_raw_datasets_framework.raw_signals.raw_signals_io import read_signals_from_dirs
from dexterous_bioprosthesis_2021_raw_datasets_framework.raw_signals_filters.raw_signals_filter_channel_idx import RawSignalsFilterChannelIdx
from dexterous_bioprosthesis_2021_raw_datasets_framework.raw_signals_spoilers.raw_signals_spoiler_cubicclipper import RawSignalsSpoilerCubicClipper
from dexterous_bioprosthesis_2021_raw_datasets_framework.raw_signals_spoilers.raw_signals_spoiler_damper import RawSignalsSpoilerDamper
from dexterous_bioprosthesis_2021_raw_datasets_framework.raw_signals_spoilers.raw_signals_spoiler_gauss import RawSignalsSpoilerGauss
from dexterous_bioprosthesis_2021_raw_datasets_framework.raw_signals_spoilers.raw_signals_spoiler_multiple import RawSignalsSpoilerMultiple
from dexterous_bioprosthesis_2021_raw_datasets_framework.raw_signals_spoilers.raw_signals_spoiler_sine import RawSignalsSpoilerSine
from dexterous_bioprosthesis_2021_raw_datasets_framework.raw_signals_spoilers.relabelers.relabeler_constant import RelabelerConstant
from dexterous_bioprosthesis_2021_raw_datasets_framework.set_creators.np_signal_extractors.np_signal_extractor_mav import NpSignalExtractorMav
from dexterous_bioprosthesis_2021_raw_datasets_framework.set_creators.np_signal_extractors.np_signal_extractor_ssc import NpSignalExtractorSsc
from dexterous_bioprosthesis_2021_raw_datasets_framework.set_creators.set_creator_dwt import SetCreatorDWT


from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

from dexterous_bioprosthesis_2021_raw_datasets_framework_experiments import settings

from sklearn.model_selection import RepeatedStratifiedKFold

from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.metrics import accuracy_score, make_scorer

from dexterous_bioprosthesis_2021_raw_datasets_framework_experiments.tools import logger
from dexterous_bioprosthesis_2021_raw_datasets_framework.tools.progressparallel import ProgressParallel
from joblib import delayed

from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, precision_score, recall_score, f1_score
from scipy.stats import wilcoxon


from scipy.stats import rankdata
from ptranks.ranks.wra import wra


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
        "Wavelet": wavelet_extractor2()
    }

    return extractors_dict

def generate_if(pre_extractor, spoiler):

    
    params = {
            'estimator__contamination':['auto']
         }

    kappa_scorer = make_scorer(cohen_kappa_score)
    generator =  OutlierGeneratorUniform()

    pipeline = Pipeline([('scaler', DummyTransformer() ),  ('estimator',IsolationForest(n_estimators=100)  )])
    gs = GridSearchCVOneClass2(pipeline,test_outlier_generator=generator,  param_grid=params, scoring=kappa_scorer,cv=5)

    return gs


def generate_ocsvm(pre_extractor, spoiler):

    
    params = {
            'estimator__gamma':['auto'],
             'estimator__nu': [0.1 * (i+1) for i in range(9)]
         }

    kappa_scorer = make_scorer(cohen_kappa_score)
    generator =  OutlierGeneratorUniform()

    pipeline = Pipeline([('scaler', StandardScaler()),  ('estimator',OneClassSVM()  )])
    gs = GridSearchCVOneClass2(pipeline,test_outlier_generator=generator,  param_grid=params, scoring=kappa_scorer,cv=5)

    return gs


def generate_lof(pre_extractor, spoiler):

    params = {
            'estimator__n_neighbors':[1,5,9,13,17,21,25,29,33],
         }

    kappa_scorer = make_scorer(cohen_kappa_score)
    generator =  OutlierGeneratorUniform()

    pipeline = Pipeline([ ('scaler', StandardScaler()),  ('estimator',LocalOutlierFactor(novelty=True)  )])
    gs = GridSearchCVOneClass2(pipeline,test_outlier_generator=generator,  param_grid=params, scoring=kappa_scorer,cv=5)

    return gs

def generate_ee(pre_extractor, spoiler):

    params = {
            'estimator__contamination':[0.1 * (i+1) for i in range(4)],
         }

    kappa_scorer = make_scorer(cohen_kappa_score)
    generator =  OutlierGeneratorUniform()

    pipeline = Pipeline([('scaler', StandardScaler()),  ('estimator', EllipticEnvelope()  )])
    gs = GridSearchCVOneClass2(pipeline,test_outlier_generator=generator,  param_grid=params, scoring=kappa_scorer,cv=5)

    return gs


def base_classifier_generators():
    classifiers = {
        "IF": generate_if,
        "OCSVM":generate_ocsvm,
        "LOF": generate_lof,
        "EE": generate_ee,
                }
    return classifiers

def generate_spoiler_50Hz(snr):
    return RawSignalsSpoilerMultiple(
        spoilers=[RawSignalsSpoilerSine(snr=snr,channels_spoiled_frac=1,frequency=50)],
                                      spoiler_relabalers=[RelabelerConstant(-1)])
def generate_spoiler_damper(snr):
    return RawSignalsSpoilerMultiple(
        spoilers=[RawSignalsSpoilerDamper(snr=snr,channels_spoiled_frac=1)],
                                      spoiler_relabalers=[RelabelerConstant(-1)])

def generate_spoiler_clipper(snr):
    return RawSignalsSpoilerMultiple(
        spoilers=[RawSignalsSpoilerCubicClipper(snr=snr,channels_spoiled_frac=1)],
                                      spoiler_relabalers=[RelabelerConstant(-1)])

def generate_spoiler_gauss(snr):
    return RawSignalsSpoilerMultiple(
        spoilers=[RawSignalsSpoilerGauss(snr=snr,channels_spoiled_frac=1)],
                                      spoiler_relabalers=[RelabelerConstant(-1)])

def generate_spoiler_baseline_wander(snr):
    return RawSignalsSpoilerMultiple(
        spoilers=[RawSignalsSpoilerSine(snr=snr,channels_spoiled_frac=1,frequency=1, freq_deviation=0.5)],
                                      spoiler_relabalers=[RelabelerConstant(-1)])



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


def acc_m(y_true, y_pred, labels=None, average = None, zero_division=None):
    return accuracy_score(y_true, y_pred)

def bac_m(y_true, y_pred, labels=None, average = None, zero_division=None):
    return balanced_accuracy_score(y_true, y_pred)

def kappa_m(y_true, y_pred, labels=None, average = None, zero_division=None):
    return cohen_kappa_score(y_true, y_pred)

def f1_score_m(y_true, y_pred, labels=None, average = None, zero_division=None):
    return f1_score(y_true, y_pred, average='binary')

def precision_score_m(y_true, y_pred, labels=None, average = None, zero_division=None):
    return precision_score(y_true, y_pred, average='binary')

def recall_score_m(y_true, y_pred, labels=None, average = None, zero_division=None):
    return recall_score(y_true, y_pred, average='binary')

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

def run_experiment(input_data_arg_list, output_directory, n_jobs=-1, n_splits=10, n_repeats=1):

    os.makedirs(output_directory, exist_ok=True)

    extractor = create_extractors()["Wavelet"]
    
    base_classifiers_generators_dict = base_classifier_generators()
    n_classifiers = len(base_classifiers_generators_dict)

    spoilers_dict = generate_spoilers_gens()
    n_spoilers = len(spoilers_dict) 

    snr_levels = get_snr_levels()
    n_snr_levels = len(snr_levels)

    metrics_dict = generate_metrics()
    n_metrics = len(metrics_dict)

    for in_dir in tqdm(input_data_arg_list,desc="Data sets"):
        
        set_name = os.path.basename(in_dir)

        pre_set = read_signals_from_dirs(in_dir)
        raw_set = pre_set['accepted']

        n_channels = raw_set[0].signal.shape[1]

        print("Channels: ", n_channels)

        y =  np.asanyarray(raw_set.get_labels())
        y = np.ones_like(y, dtype=np.int_)
        raw_set.set_labels(y)


        skf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,random_state=0)
        n_folds = skf.get_n_splits()

        # metric x classifier x spoiler x snr_levels x channel x fold
        results = np.zeros( (n_metrics, n_classifiers,n_spoilers,n_snr_levels,n_channels, n_folds) ) 

        def compute(fold_idx,train_idx, test_idx):

            raw_train = raw_set[train_idx]
            raw_test = raw_set[test_idx]

            fold_res = []

            for channel_idx in range(n_channels):
                channel_filter =  RawSignalsFilterChannelIdx(indices_list=[channel_idx])

                raw_filtered_train = channel_filter.fit_transform(raw_train)
                raw_filtered_test = channel_filter.fit_transform(raw_test)

                X_train, y_train, _ = extractor.fit_transform(raw_filtered_train)

                for snr_level_id, snr_level in enumerate(snr_levels):


                    for spoiler_id, spoiler_name in enumerate(spoilers_dict):

                        spoiler_gen = spoilers_dict[spoiler_name]
                        spoiler = spoiler_gen(snr_level)
                        


                        raw_filtered_test_spoiled = spoiler.fit_transform(raw_filtered_test)
                        raw_filtered_test_spoiled += raw_filtered_test

                        X_test, y_test, _ = extractor.transform(raw_filtered_test_spoiled)

                        for classifier_gen_id, classifier_gen_name in tqdm( enumerate(base_classifiers_generators_dict), desc="Base classifiers", leave=False, total=n_classifiers):
                            classifier_gen = base_classifiers_generators_dict[classifier_gen_name]

                            classifier = classifier_gen(None, None)

                            classifier.fit(X_train, y_train)

                            y_predict = classifier.predict(X_test)

                            for metric_id, metric_name in enumerate(metrics_dict):
                                metric = metrics_dict[metric_name]

                                metric_val = metric(y_test, y_predict)
                                # metric x classifier x spoiler x snr_levels x channel x fold
                                fold_res.append( (metric_id, classifier_gen_id, spoiler_id,snr_level_id,channel_idx, fold_idx, metric_val ) )

            return fold_res
        
        rets =  ProgressParallel(n_jobs=n_jobs, desc="K-folds", total=skf.get_n_splits(), leave=False)\
                            (delayed(compute)(fold_idx, train_idx, test_idx) for fold_idx, (train_idx, test_idx) in enumerate( skf.split(raw_set, y)) )

        for fold_list in rets:
            for  metric_id,  classifier_id,spoiler_id,snr_level_id,channel_id, fold_idx , metric_value in fold_list:
                results[metric_id, classifier_id,spoiler_id, snr_level_id,channel_id, fold_idx] = metric_value

               
        result_file_path = os.path.join(output_directory,"{}.pickle".format(set_name))

        fin_result_dict = {
            "criteria_names": [k for k in metrics_dict],
            "classifier_names": [k for k in base_classifiers_generators_dict],
            "spoiler_names":[k for k in spoilers_dict],
            "snr_levels": snr_levels,
            "channels":[v for v in range(n_channels)],
            "results":results,
        }
        pickle.dump(obj=fin_result_dict, file=open(result_file_path,"wb"))

def analyze_results(results_directory, output_directory, stat_test=True):
    result_files = [f for f in os.listdir(results_directory) if f.endswith(".pickle") ]

    for result_file in result_files:
        result_file_basename = os.path.splitext(result_file)[0]
        result_file_path = os.path.join(results_directory, result_file)

        
        raw_data = pickle.load(open(result_file_path,"rb"))

        metric_names = raw_data["criteria_names"]
        channels = raw_data["channels"]
        classifier_names = raw_data["classifier_names"]
        spoiler_names = raw_data["spoiler_names"]
        snr_levels = raw_data["snr_levels"]
        results = raw_data["results"] # metric x classifier x spoiler x snr_levels x channel x fold

        n_channels = len(channels)
        n_classifiers = len(classifier_names)

        pdf_file_path = os.path.join(output_directory, "{}.pdf".format(result_file_basename))
        report_file_path = os.path.join(output_directory, "{}.md".format(result_file_basename))
        report_file_handler = open(report_file_path, "w")

        with PdfPages(pdf_file_path) as pdf:

            for metric_id, metric_name in enumerate(metric_names):
                print("# {}".format(metric_name),file=report_file_handler)
                for spoiler_id, spoiler_name in enumerate(spoiler_names):
                    print("## {}".format(spoiler_name), file=report_file_handler)

                    for snr_level_id, snr_level in enumerate(snr_levels):
                        print("### {}".format(spoiler_name), file=report_file_handler)

                        # classifier x (channel * fold)
                        result_subset = results[metric_id,:,spoiler_id,snr_level_id].reshape(n_classifiers,-1) 

                        plt.boxplot(result_subset.transpose())
                        plt.title( "{}, {}, SNR: {}".format(metric_name, spoiler_name,snr_level ) )
                        plt.xticks( range(1,len(classifier_names)+1), classifier_names )
                        pdf.savefig()
                        plt.close()

                        fold_means = np.mean(result_subset, axis=1)
                        fold_sdevs = np.std(result_subset, axis=1)

                        
                        for classifier_id, classifier_name in enumerate(classifier_names):
                            print("{}:\t{} +- {}".format(classifier_name, fold_means[classifier_id], fold_sdevs[classifier_id]), file=report_file_handler)

                        p_vals = np.zeros( (n_classifiers, n_classifiers) )
                        values = result_subset.transpose()

                        if stat_test:
                            for i in range(n_classifiers):
                                for j in range(n_classifiers):
                                    if i == j:
                                        continue

                                    values_squared_diff = np.sqrt (np.sum( (values[:,i] - values[:,j])**2 ) )
                                    if values_squared_diff > 1E-4:
                                        with warnings.catch_warnings(): #Normal approximation
                                            warnings.simplefilter("ignore")
                                            p_vals[i,j]  = wilcoxon(values[:,i], values[:,j]).pvalue #mannwhitneyu(values[:,i], values[:,j]).pvalue
                                    else:
                                        p_vals[i,j] = 1.0
                            
                            p_val_vec  = p_val_matrix_to_vec(p_vals)

                            p_val_vec_corrected = multipletests(p_val_vec, method='hommel')


                            corr_p_val_matrix = p_val_vec_to_matrix(p_val_vec_corrected[1],n_classifiers)

                            p_val_df = pd.DataFrame(corr_p_val_matrix, columns=classifier_names,index=classifier_names )

                            print("PVD:\n",file=report_file_handler)
                            p_val_df.to_markdown(report_file_handler)
                            print("\n", file=report_file_handler)
                            

        report_file_handler.close()

def analyze_results_trend(results_directory, output_directory):
    result_files = [f for f in os.listdir(results_directory) if f.endswith(".pickle") ]

    for result_file in result_files:
        result_file_basename = os.path.splitext(result_file)[0]
        result_file_path = os.path.join(results_directory, result_file)

        
        raw_data = pickle.load(open(result_file_path,"rb"))

        metric_names = raw_data["criteria_names"]
        channels = raw_data["channels"]
        classifier_names = raw_data["classifier_names"]
        spoiler_names = raw_data["spoiler_names"]
        snr_levels = raw_data["snr_levels"]
        results = raw_data["results"] # metric x classifier x spoiler x snr_levels x channel x fold

        n_channels = len(channels)
        n_classifiers = len(classifier_names)
        n_snr_levels = len(snr_levels)

        pdf_file_path = os.path.join(output_directory, "{}_trends.pdf".format(result_file_basename))
        

        with PdfPages(pdf_file_path) as pdf:

            for metric_id, metric_name in enumerate(metric_names):
                
                for spoiler_id, spoiler_name in enumerate(spoiler_names):
                    
                    for classifier_id, classifier_name in enumerate(classifier_names):

                        #snr_levels x (channel * fold)
                        result_subset = results[metric_id,classifier_id,spoiler_id].reshape(n_snr_levels, -1)

                        medians = np.median(result_subset,axis=1)
                        quantiles = np.quantile(result_subset,q=(0.25, 0.75),axis=1) # 2 x snr_levels

                        plt.plot(snr_levels, medians, marker="o", label = classifier_name)
                        plt.fill_between(snr_levels, quantiles[0], quantiles[1], alpha=0.3)

                    plt.title("{}, {}".format(metric_name, spoiler_name))
                    plt.xlabel("SNR level")
                    plt.ylabel("Criterion value")
                    plt.legend()
                    pdf.savefig()
                    plt.close()


def analyze_results_trend_ranks(results_directory, output_directory, alpha = 0.05):
    result_files = [f for f in os.listdir(results_directory) if f.endswith(".pickle") ]

    n_result_files = len(result_files)

    global_results = None
    metric_names = None
    channels = None
    classifier_names = None
    spoiler_names = None
    snr_levels = None

    n_metrics = None
    n_channels = None
    n_classifiers = None
    n_snr_levels = None
    n_spoilers = None


    for result_file_id,  result_file in enumerate(result_files):
        result_file_basename = os.path.splitext(result_file)[0]
        result_file_path = os.path.join(results_directory, result_file)

        
        raw_data = pickle.load(open(result_file_path,"rb"))

        if global_results is None:


            metric_names = raw_data["criteria_names"]
            channels = raw_data["channels"]
            classifier_names = raw_data["classifier_names"]
            spoiler_names = raw_data["spoiler_names"]
            snr_levels = raw_data["snr_levels"]

            n_metrics = len(metric_names)
            n_channels = len(channels)
            n_classifiers = len(classifier_names)
            n_snr_levels = len(snr_levels)
            n_spoilers = len(spoiler_names)

            #  sets x metric x classifier x spoiler x snr_levels
            global_results = np.zeros((n_result_files, n_metrics, n_classifiers, n_spoilers, n_snr_levels))

        results = raw_data["results"] # metric x classifier x spoiler x snr_levels x channel x fold

        # metric x classifier x spoiler x snr_levels x (channel * fold)
        tmp = results.reshape( (n_metrics, n_classifiers, n_spoilers, n_snr_levels,-1) )
        # metric x classifier x spoiler x snr_levels
        global_results[result_file_id] = np.mean(tmp,axis=-1)


        
    pdf_file_path = os.path.join(output_directory, "{}_trends_ranks.pdf".format("ALL"))
    report_file_path = os.path.join(output_directory, "{}_trends_ranks.md".format("ALL"))
    report_file_handler = open(report_file_path, "w+")

    

    with PdfPages(pdf_file_path) as pdf:

        for metric_id, metric_name in enumerate(metric_names):
            print("# {}".format(metric_name), file=report_file_handler)

            # sets 0 x classifiers 1 x spoilers 2  x snr_levels 3
            result_subset = global_results[:,metric_id]

            # classifiers 0 x  snr_levels 1 x (sets 2 * spoilers 3)
            result_subset = np.moveaxis(result_subset, [0,1,2,3], [2,0,3,1]).reshape( (n_classifiers, n_snr_levels,-1 ))

            ranked_data = rankdata(result_subset,axis=0)
            # classifiers x snr_levels
            avg_ranks = np.mean(ranked_data, axis=-1)
                
            for classifier_id, classifier_name in enumerate(classifier_names):

                plt.plot(snr_levels, avg_ranks[classifier_id,:], marker="o", label = classifier_name)
                

            plt.title("{}".format(metric_name))
            plt.xlabel("SNR level")
            plt.ylabel("Criterion avg rank")
            plt.legend()
            pdf.savefig()
            plt.close()

            mi = pd.MultiIndex.from_arrays( [ ["{}".format(metric_name) for _ in range(n_classifiers)], [m for m in classifier_names] ] )
            av_rnk_df = pd.DataFrame(avg_ranks.T,columns=mi ,
                                                    index=["Avg Rnk {}, SNR:{}".format(a,si) for si,a in zip( snr_levels,string.ascii_letters)])
            
            for snr_id, (gs_name,snr_letter) in enumerate( zip(snr_levels, string.ascii_letters )):
                # methods    x ( files * spoilers * snrs  )
                n_methods = n_classifiers
                values = result_subset[:,snr_id]
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
                av_rnk_df.loc["{} {}, gs:{}_T".format("Avg Rnk",snr_letter,gs_name)] = s_test_outcome
                av_rnk_df.sort_index(inplace=True)
            

            av_rnk_df.style.format(precision=3,na_rep="")\
                        .format_index(escape="latex", axis=0).format_index(escape="latex", axis=1)\
                        .to_latex(report_file_handler, multicol_align="c")

    report_file_handler.close()
       
import random
if __name__ == '__main__':

    
    # data_path2 = os.path.join(settings.DATAPATH,"ottobock")

    random.seed(0)
    np.random.seed(0)

    data_sets = [ os.path.join( settings.DATAPATH, "tsnre_windowed","A{}_Force_Exp_low_windowed".format(i)) for i in range(1,10) ]
    # data_sets = [ data_path2]


    output_directory = os.path.join(settings.EXPERIMENTS_RESULTS_PATH,"./outlier_detection_experiment_snr2/") 

    log_dir = os.path.dirname(__file__)
    log_file = os.path.splitext( os.path.basename(__file__) )[0]
    logger(log_dir,log_file, enable_logging=False)    
    
    # run_experiment(data_sets, output_directory, n_jobs=-1, n_splits=8, n_repeats=1)
    analyze_results(output_directory, output_directory,stat_test=False)
    analyze_results_trend(output_directory, output_directory)
    analyze_results_trend_ranks(output_directory, output_directory)
