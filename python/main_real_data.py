#!/home/handl/tools/miniconda3/envs/myenv/bin/python
#$ -S /home/handl/tools/miniconda3/envs/myenv/bin/python
'''
Created on 28.01.2019

@author: Lisa Handl
'''

import os
import sys
sys.path.append("")

import pandas
import numpy as np
import sklearn.linear_model as lm
import feature_models
import GPy
import itertools

import time
import datetime

import data as data_mod
import models
import util


def printTestErrors(pred_raw, test_y_raw, heading=None, indent=0):
    prefix = " "*indent
    errors = np.abs(test_y_raw - pred_raw)
    mean_err = np.mean(errors)
    median_err = np.median(errors)
    corr = np.corrcoef(pred_raw, test_y_raw)[0,1]
    std = np.std(errors)
    q75, q25 = np.percentile(errors, [75 ,25])
    iqr = q75 - q25
    if (heading is not None):
        print(prefix + heading)
        print(prefix + len(heading)*'-')
    print(prefix + "Mean abs. error:", mean_err)
    print(prefix + "Median abs. error:", median_err)
    print(prefix + "Correlation:", corr)
    print()
    return [mean_err, median_err, corr, std, iqr]

    
def main():
     
    # print time stamp
    print("Program started:", "{:%d.%m.%Y %H:%M:%S}".format(datetime.datetime.now()))
    print(flush=True)
     
    # set paths and parameters
    home_path = "/home/handl/"
      
    feature_model_path = os.path.join(home_path, "data/aging/modelFeatures/blinear_all")
    conf_path = os.path.join(feature_model_path, "confidences")
    output_path = os.path.join(home_path, "data/aging/modelWenda")
        
    n_jobs = 10
    # power parameters for weight function
    kwnet_values = [1,2,3,4, 6, 8, 10, 14, 18, 25, 35]
    repetitions = 10
    
    print("feature_model_path:", feature_model_path)
    print("conf_path:", conf_path)
    print("output_path:", output_path)
    print("n_jobs:", n_jobs)
    print("weighting parameters:", kwnet_values)
    print("repetitions:", repetitions)
    print()
    
    print("Reading data...", end=" ", flush=True)
    start = time.time()
    
    data = data_mod.DataDNAmethPreprocessed()
    
    end = time.time()
    print("took", util.sec2str(end-start))
    print()
    print("data:", data)
    print(flush=True)
    
    # collect and aggregate tissue information
    test_tissues = data.test.pheno_table["tissue_complete"]
    
    test_tissues_aggregated = test_tissues.copy()
    test_tissues_aggregated[test_tissues_aggregated == "whole blood"] = "blood"
    test_tissues_aggregated[test_tissues_aggregated == "menstrual blood"] = "blood"
    test_tissues_aggregated[test_tissues_aggregated == "Brain MedialFrontalCortex"] = "Brain Frontal"

    crbm_samples = np.array(test_tissues == "Brain CRBM")
    
    normalizer_x = util.StandardNormalizer
    normalizer_y = util.HorvathNormalizer
    
    feature_model_type = feature_models.FeatureGPR
    feature_model_params = {"kernel": GPy.kern.Linear(input_dim=data.training.getNofCpGs()-1)}

    model = models.Wenda(data.training.meth_matrix, data.training.age, data.test.meth_matrix, 
                                 normalizer_x, normalizer_y, feature_model_path, feature_model_type, feature_model_params, conf_path, 
                                 n_jobs=n_jobs)
    print(model)
    print(flush=True)
     
    # fit feature models
    print("Fitting feature models...", flush=True)
    model.fitFeatureModels()
    print("Collecting confidences...", flush=True)
    model.collectConfidences()
   
    
       
    
    # Predict with CV on training data ---
    print("\nPredicting with cross-validation on training data...")
       
    for i in range(repetitions):
        print("- repetition ", i, "\n", flush=True)
           
        # setup output path
        adaptive_cv_path = os.path.join(output_path, "wnet_cv/repetition_{0:02d}".format(i))
        os.makedirs(adaptive_cv_path, exist_ok=True)
        print("  path:", adaptive_cv_path, "\n", flush=True)
           
        # iterate over weighting functions, predict and save errors
        for k_wnet in kwnet_values:
            print("  - k_wnet =", k_wnet, end="\n    ", flush=True)
            weighting_function = lambda x: np.power(1-x, k_wnet)
            predictions = model.predictWithTrainingDataCV(weight_func=weighting_function, grouping=test_tissues_aggregated, alpha=0.8, n_splits=10,
                                                          predict_path=os.path.join(adaptive_cv_path, "glmwnet_pow{0:d}".format(k_wnet)))
            np.savetxt(os.path.join(adaptive_cv_path, "predictions_k{0:d}.txt".format(k_wnet)), predictions)
            
            errors_all = printTestErrors(predictions, data.test.age, "Weighted elastic net (full data):", indent=4)
            errors_crbm = printTestErrors(predictions[crbm_samples], data.test.age[crbm_samples], "Weighted elastic net (CRBM):", indent=4)
                      
            table = pandas.DataFrame(np.vstack([errors_all, errors_crbm]), 
                                     index=["all", "crbm"], 
                                     columns=["mean", "median", "corr", "std", "iqr"])
            table.to_csv(os.path.join(adaptive_cv_path, "errors_k{0:d}.csv".format(k_wnet)), sep=";", quotechar='"')
            
        np.savetxt(os.path.join(adaptive_cv_path, "powers.txt"), kwnet_values)
     
 
     
    # Predict with similarity-lambda regression ---
    print("\nPredicting with prior knowledge / similarity-lambda regression...")
      
    # setup output path
    adaptive_lambdareg_path = os.path.join(output_path, "wnet_lambda_regression")
    os.makedirs(adaptive_lambdareg_path, exist_ok=True)
    print("path:", adaptive_lambdareg_path, "\n", flush=True)
          
    # read similarities translated from GTEx paper
    tissue_similarity = data_mod.TissueSimilarity()
      
    # Compute similarity of each tissue with training data
    test_tissue_frequencies = test_tissues_aggregated.value_counts()
    test_tissues_unique = test_tissue_frequencies.index.values
    similarity = pandas.Series(index=test_tissues_unique, dtype=np.float64)
    for t in test_tissues_unique:
        similarity[t] = tissue_similarity.compute_similarity(data.training.pheno_table["tissue_detailed"],
                                                             test_tissues[test_tissues_aggregated == t])
    similarity.to_csv(os.path.join(adaptive_lambdareg_path, "tissue_similarity.csv"), sep=";")
      
    # iterate over weighting functions 
    for k_wnet in kwnet_values:
        print("- k_wnet =", k_wnet, end="\n  ", flush=True)
          
        # compute all weighted elastic net coefficient paths (without cross-validation)
        weighting_function = lambda x: np.power(1-x, k_wnet)
        wnet_models = model.trainFinalWeightedModels(weight_func=weighting_function, grouping=test_tissues_aggregated, alpha=0.8, 
                                                     predict_path=os.path.join(adaptive_lambdareg_path, "wnet_paths_pow{0:d}".format(k_wnet)), n_splits=0)
        prediction_paths = model.computePredictionPaths(wnet_models, test_tissues_aggregated)
  
        # set up tables for hold-out errors
        error_table = pandas.DataFrame(columns=["combination", "tissue", "mean", "median", "corr", "std", "iqr"], dtype=np.float64)
        error_table["combination"] = error_table["combination"].astype(np.int64)
        error_table.set_index(["combination", "tissue"], inplace=True)
     
        # iterate over all possible training-validation splits and ...
        # (we use 3 of the tissues with >10 samples for training in each run)
        tissue_combinations = list(itertools.combinations(test_tissues_unique[test_tissue_frequencies > 20], 3))
        for i in range(len(tissue_combinations)):
            print("  - combination", i, end="\n")
              
            fit_tissues = tissue_combinations[i]
            holdout_tissues = [t for t in test_tissues_unique if t not in fit_tissues]
            predict_path = os.path.join(adaptive_lambdareg_path, "glmwnet_pow{0:d}_comb{1:02d}".format(k_wnet, i))
            os.makedirs(predict_path, exist_ok=True)
             
            # ... compute error paths and optimal lambdas for fit_tissues
            known_optimal_lambdas = dict()
            for t in fit_tissues:
                error_path = np.full(prediction_paths[t].shape[1], np.nan)
                for j in range(len(error_path)):
                    error_path[j] = np.mean(np.abs(prediction_paths[t][:,j] - data.test.age[test_tissues_aggregated == t]))
                index_min = np.argmin(error_path)
                known_optimal_lambdas[t] = wnet_models[t].lambda_path_[index_min]
              
            # ... fit a line to similarity-lambda relationship
            optimal_lambda_model = lm.LinearRegression()
            optimal_lambda_model.fit(similarity[list(fit_tissues)].values.reshape(-1, 1), 
                                     np.log([known_optimal_lambdas[t] for t in fit_tissues]))
            np.savetxt(os.path.join(predict_path, "similarity_lambda_fit.txt"), [optimal_lambda_model.intercept_, optimal_lambda_model.coef_])
              
  
            # ... predict optimal lambdas on hold-out tissues
            predicted_lambdas = np.exp(optimal_lambda_model.predict(similarity[holdout_tissues].values.reshape(-1, 1)))
            predicted_lambdas = pandas.Series(data=predicted_lambdas, index=holdout_tissues, dtype=np.float64)
            print("predicted lambdas:")
            print(predicted_lambdas, "\n")
            predicted_lambdas.to_csv(os.path.join(predict_path, "predicted_lambdas.csv"), sep=";")
             
            # check if lambda is within range and retrain if it is not
            # (only small lambdas matter, for large ones the model remains the same with all coefficients=0)
            for t in holdout_tissues:
                if predicted_lambdas[t] < wnet_models[t].lambda_path_[-1]:
                    lambda_path = np.append(wnet_models[t].lambda_path_, predicted_lambdas[t])
                    weights = weighting_function(model._getConfidences(t, test_tissues_aggregated))
                    wnet_models[t] = model._trainOneWeightedModel(t, weights=weights, alpha=0.8, lambda_path=lambda_path, n_splits=0, 
                                                                  predict_path=os.path.join(adaptive_lambdareg_path, "wnet_paths_pow{0:d}".format(k_wnet)))
             
            # ... predict output for hold-out tissues using the predicted optimal lambdas
            predictions = model.predictWithPriorKnowledge(wnet_models, test_tissues_aggregated, predicted_lambdas.to_dict())
  
            # ... compute and save errors per tissue
            for t in holdout_tissues:
                errors_t = printTestErrors(predictions[t], data.test.age[test_tissues_aggregated == t], 
                                           "Weighted elastic net with prior knowledge ({}):".format(t), indent=4)
                error_table.loc[(i, t),:] = errors_t
                np.savetxt(os.path.join(predict_path, "predictions {}.txt".format(t)), predictions[t])
  
        # write results
        error_table.to_csv(os.path.join(adaptive_lambdareg_path, "errors_pow{0:d}.csv".format(k_wnet)), sep=";", quotechar='"')
  
    np.savetxt(os.path.join(adaptive_lambdareg_path, "powers.txt"), kwnet_values)
 
 
 
    # Predict with non-adaptive reference model ---
    print("\nPredicting with non-adaptive reference model...")
                   
    # set up output path
    ref_path = os.path.join(output_path, "reference_10fold_cv")
    os.makedirs(ref_path, exist_ok=True)
    print("path:", ref_path, "\n", flush=True)
         
    # repeat n times, so I can report means + standard deviations
    for i in range(repetitions):
        print("- repetition ", i, flush=True)
        predict_path = os.path.join(ref_path, "reference_rep{0:02d}".format(i))
        os.makedirs(predict_path, exist_ok=True)
        # fit model
        modelSimple = models.ElasticNetLm(alpha=0.8, n_splits=10, norm_x=normalizer_x, norm_y=normalizer_y)
        modelSimple.fit(data.training.meth_matrix, data.training.age)
        # calculate predictions
        predSimpleNet = modelSimple.predictNet(data.test.meth_matrix)
        predSimple = modelSimple.predict(data.test.meth_matrix)
        np.savetxt(os.path.join(predict_path, "predictions.txt"), np.vstack([predSimple, predSimpleNet]).T)
                      
        # check test error (for all samples and for CBRM samples)    
        errors_net_all = printTestErrors(predSimpleNet, data.test.age, "Simple model EN (full data):")
        errors_net_crbm = printTestErrors(predSimpleNet[crbm_samples], data.test.age[crbm_samples], 
                        "Simple model EN (CRBM):")
        errors_lm_all = printTestErrors(predSimple, data.test.age, "Simple model lm (full data):", indent=2)
        errors_lm_crbm = printTestErrors(predSimple[crbm_samples], data.test.age[crbm_samples], 
                        "Simple model lm (CRBM):", indent=2)
        table = pandas.DataFrame(np.vstack([errors_lm_all, errors_lm_crbm, errors_net_all, errors_net_crbm]), 
                                 index=["lm_all", "lm_crbm", "net_all", "net_crbm"], 
                                 columns=["mean", "median", "corr", "std", "iqr"])
        table.to_csv(os.path.join(predict_path, "errors.csv"), sep=";", quotechar='"')
        modelSimple.savetxt(os.path.join(predict_path, "model"))
 
 
 
 
    # Predict with weighted-EN baseline ---
    print("\nPredict with weighted-EN baseline (simpler confidences based on marginals) ...\n", flush=True)
    ref_wnet_path = os.path.join(output_path, "reference_wnet_ecdf")
    os.makedirs(ref_wnet_path, exist_ok=True)
     
    ref_wnet_conf_path = os.path.join(home_path, "data/aging/modelFeatures/marginal_ecdf/confidences")
    baseline_wnet = models.BaselineWeightedEN(data.training.meth_matrix, data.training.age, data.test.meth_matrix,
                                               normalizer_x, normalizer_y, ref_wnet_conf_path, n_jobs=n_jobs)
    print("Collecting confidences...", flush=True)
    baseline_wnet.collectConfidences()
     
    for i in range(repetitions):
        print("- repetition ", i, flush=True)
        for k_wnet in kwnet_values:
            print("  - k_wnet =", k_wnet, end="\n    ", flush=True)
            ref_wnet_predict_path = os.path.join(ref_wnet_path, "repetition_{0:02d}/reference_wnet_k{1}".format(i, k_wnet))
            weighting_function = lambda x: np.power(1-x, k_wnet)
            ref_wnet_predictions = baseline_wnet.predictWithTrainingDataCV(weight_func=weighting_function, grouping=test_tissues_aggregated, 
                                                                           alpha=0.8, n_splits=10, predict_path=ref_wnet_predict_path)
            np.savetxt(os.path.join(ref_wnet_predict_path, "predictions.txt"), ref_wnet_predictions)
           
            errors_all = printTestErrors(ref_wnet_predictions, data.test.age, "Weighted EN baseline (full data):", indent=4)
            errors_crbm = printTestErrors(ref_wnet_predictions[crbm_samples], data.test.age[crbm_samples], "Weighted EN baseline (CRBM):", indent=4)
                        
            table = pandas.DataFrame(np.vstack([errors_all, errors_crbm]), 
                                     index=["all", "crbm"], 
                                     columns=["mean", "median", "corr", "std", "iqr"])
            table.to_csv(os.path.join(ref_wnet_predict_path, "errors.csv".format(k_wnet)), sep=";", quotechar='"')


    # print time stamp
    print("Program finished:", "{:%d.%m.%Y %H:%M:%S}".format(datetime.datetime.now()))


if __name__ == "__main__":
    print("working directory:", os.getcwd(), "\n")
    main()


