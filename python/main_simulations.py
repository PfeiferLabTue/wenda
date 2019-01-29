#!/home/handl/tools/miniconda3/envs/myenv/bin/python
#$ -S /home/handl/tools/miniconda3/envs/myenv/bin/python
'''
Created on 28.01.2019

@author: Lisa Handl
'''

import os
import sys
sys.path.append("")

from main_real_data import printTestErrors
import feature_models
import models
import simulation
import util

import GPy
import datetime
import numpy as np
import pandas
import sklearn.linear_model as lm


class PretrainedModels:
    '''
    Helper class to save and access pretrained weighted models.
    '''
    def __init__(self):
        self.all_wnet_models = dict()
        self.all_prediction_paths = dict()
    
    def set(self, altered_percentage, simulation, wnet_models, prediction_paths):
        self.all_wnet_models = self._set_in_dict(altered_percentage, simulation, wnet_models, self.all_wnet_models)
        self.all_prediction_paths = self._set_in_dict(altered_percentage, simulation, prediction_paths, self.all_prediction_paths)
    
    def _set_in_dict(self, key1, key2, new_value, target_dict):
        if key1 not in target_dict:
            target_dict[key1] = dict()
        target_dict[key1][key2] = new_value
        return target_dict

    def get(self, altered_percentage, simulation, include_paths=True):
        models = self.all_wnet_models[altered_percentage][simulation]
        if include_paths:
            paths = self.all_prediction_paths[altered_percentage][simulation]
            return models, paths
        else:
            return models
    

def pretrain_weighted_EN_paths(simulation_dirs, transformation_class, altered_percentages, outcoef_scaling_factor, n_target, 
                               source_parameters, model_parameters, adaptive_confidence_dir, weighting_function, grouping, predict_path):
    '''
    ...
    
    Expects simulation data and confidences to be simulated and computed before, respectively, and will throw errors if not.
    '''
    pretrained_models = PretrainedModels()
    # iterate over simulations ...
    for simulation_dir in simulation_dirs:
        # get source data
        sim = simulation.SimulationScenario(simulation_dir, **source_parameters)
        sim.read_source_data(out=False)
        
        # ... and target domains ...
        for altered_percentage in altered_percentages:
            # get target data
            sim.set_target_domain(transformation_class, altered_percentage, outcoef_scaling_factor, n_target)
            sim.read_target_data(out=False)
            # set up adaptive model
            adaptive_model = models.Wenda(x_train=sim.source_data.input.as_matrix(), y_train=sim.source_data.output, 
                                                  x_test=sim.target_data.input.as_matrix(), feature_model_dir=sim.paths.feature_model_dir,
                                                  confidences_dir=adaptive_confidence_dir(sim), **model_parameters)
            # compute models / predictions for entire lambda path
            wnet_models = adaptive_model.trainFinalWeightedModels(weight_func=weighting_function, grouping=grouping, alpha=0.8,
                                                                  predict_path=predict_path(sim), n_splits=0, out=False)
            prediction_paths = adaptive_model.computePredictionPaths(wnet_models, grouping) # dict with groups as keys
            # set results to pretrained_models
            pretrained_models.set(altered_percentage, simulation_dir, wnet_models, prediction_paths)
            # print dot to monitor progress
            print(".", end="", flush=True)
    print("\n")
    return pretrained_models

    
def compute_known_optimal_lambdas(pretrained_models, fit_percentages, simulation_dirs, source_parameters,
                                  transformation_class, outcoef_scaling_factor, n_target, grouping):
    known_optimal_lambdas = list()
    for altered_percentage in fit_percentages:
        for simulation_dir in simulation_dirs:
            # read target data
            sim = simulation.SimulationScenario(simulation_dir, **source_parameters, 
                                                transformation_class=transformation_class, altered_percentage=altered_percentage, 
                                                outcoef_scaling_factor=outcoef_scaling_factor, n_target=n_target)
            sim.read_target_data(out=False)
            # retrieve pretrained models / prediction paths
            wnet_models, prediction_paths = pretrained_models.get(altered_percentage, simulation_dir)
            for g in prediction_paths.keys():
                error_path = np.full(prediction_paths[g].shape[1], np.nan)
                for k in range(len(error_path)):
                    error_path[k] = np.mean(np.abs(prediction_paths[g][:,k] - sim.target_data.output[grouping == g]))
                index_min = np.argmin(error_path)
                lambda_opt = wnet_models[g].lambda_path_[index_min]
                one_row_frame = pandas.DataFrame({"Simulation": [simulation_dir], "AlteredPercentage": [altered_percentage], "GroupID": [g], 
                                                  "LambdaOpt": [lambda_opt]})
                known_optimal_lambdas.append(one_row_frame)
            # print dot to monitor progress
            print(".", end="", flush=True)
    print("\n")         
    known_optimal_lambdas = pandas.concat(known_optimal_lambdas, ignore_index=True)
    known_optimal_lambdas = known_optimal_lambdas[["Simulation", "AlteredPercentage", "GroupID", "LambdaOpt"]]
    return known_optimal_lambdas


def ensure_lambda_within_range(sim, adaptive_model, pretrained_models, predicted_lambda,
                               weighting_function, grouping, predict_path):
    '''
    Check if predicted lambda is within range of pretrained models and retrain if it is not.
    (only small lambdas matter, for large ones the model remains the same with all coefficients=0)
    '''
    # retrieve pretrained models
    wnet_models, prediction_paths = pretrained_models.get(sim.altered_percentage, sim.paths.simulation_dir)
    for g in wnet_models:
        if predicted_lambda < wnet_models[g].lambda_path_[-1]:
            # append predicted lambda to lambda_path and retrain
            lambda_path = np.append(wnet_models[g].lambda_path_, predicted_lambda)
            weights = weighting_function(adaptive_model._getConfidences(g, grouping))
            wnet_models[g] = adaptive_model._trainOneWeightedModel(g, weights=weights, alpha=0.8, lambda_path=lambda_path, 
                                                                   n_splits=0, predict_path=predict_path(sim))
    print()
    # update and return pretrained models
    pretrained_models.set(sim.altered_percentage, sim.paths.simulation_dir, wnet_models, prediction_paths)
    return pretrained_models


def predict_with_prior_knowledge(sim, adaptive_model, pretrained_models, predicted_lambda, grouping):
    # retrieve pretrained models and set up dict for predicted lambdas
    wnet_models = pretrained_models.get(sim.altered_percentage, sim.paths.simulation_dir, include_paths=False)
    predicted_lambdas = {key: predicted_lambda for key in wnet_models.keys()}
    # get predictions for each group
    predictions_dict = adaptive_model.predictWithPriorKnowledge(wnet_models, grouping, predicted_lambdas)
    # merge dict into one array
    predictions = np.full_like(sim.target_data.output, np.nan)
    for g in predictions_dict.keys():
        predictions[grouping == g] = predictions_dict[g]
    return predictions


def main():

    # print time stamp
    print('Program started:', '{:%d.%m.%Y %H:%M:%S}'.format(datetime.datetime.now()), flush=True)
    print()
    
    # base parameters
    home_dir = "/home/handl" 
    #     base_dir = os.path.join(home_dir, "data/simulated/BN-test")
    base_dir = os.path.join(home_dir, "data/simulated/BN-final-20graphs-3000-1000")
    n_jobs = 5
    
    # simulation indexes / directories
    simulation_indexes = list(range(10))
    simulation_dirs = [os.path.normpath(os.path.join(base_dir, "Sim-{0:02d}".format(i+1))) for i in simulation_indexes]
    
    # source domain parameters
    input_noise_var = 0.1
    output_noise_var = 0.1
    n_relevant = 20
    n_source = 3000
    
    # target domain parameters
    transformation_class = simulation.CompDependencyInversion
    altered_percentages = [0, 10, 20, 30]
    outcoef_scaling_factor = 0
    n_target = 1000
    n_per_group = 100
    
    # adaptive model parameters
    kwnet_values = [1,2,3,4, 6, 8, 10, 14, 18, 25, 35]
    normalizer_x = util.StandardNormalizer
    normalizer_y = util.StandardNormalizer
    feature_model_type = feature_models.FeatureGPR
    feature_model_params = {'kernel': GPy.kern.Linear(input_dim=999)}
    
    # package parameters into dicts for easier access
    source_parameters = {"input_noise_var": input_noise_var,
                         "output_noise_var": output_noise_var,
                         "n_relevant": n_relevant,
                         "n_source": n_source}
    
    model_parameters = {"norm_x": normalizer_x,
                        "norm_y": normalizer_y,
                        "feature_model_type": feature_model_type,
                        "feature_model_params": feature_model_params,
                        "n_jobs": n_jobs}

    # print parameters for log
    print("base_dir:", base_dir)
    print("simulation_indexes:", simulation_indexes)
    print("n_jobs:", n_jobs)
    print()
    print("source_parameters:", source_parameters)
    print("transformation class:", transformation_class)
    print("altered_percentages:", altered_percentages)
    print("outcoef_scaling_factor:", outcoef_scaling_factor)
    print("n_target:", n_target)
    print("n_per_group:", n_per_group)
    print()
    print("model_parameters:", model_parameters)
    print("kwnet_values:", kwnet_values)
    print()
    
    
    # check domain/group size and translate to grouping vector
    if n_target % n_per_group != 0:
        raise ValueError("n_per_group must be a factor of n_target!")
    grouping = np.repeat(range(n_target // n_per_group), n_per_group)
    
    
    for simulation_dir in simulation_dirs:
        print("SIMULATION: {0:s}".format(os.path.basename(simulation_dir)))
        print("------------------\n")
         
        sim = simulation.SimulationScenario(simulation_dir, **source_parameters)
        sim.read_or_create_source_data()
 
        # fit non-adaptive reference model
        simple_model = models.ElasticNetLm(alpha=0.8, n_splits=10, norm_x=normalizer_x, norm_y=normalizer_y)
        simple_model.fit(sim.source_data.input.as_matrix(), sim.source_data.output)
   
           
        for altered_percentage in altered_percentages:
             
            print("\n- Altered percentage:", altered_percentage, end="\n\n  ")
             
            # set / update target domain and load or create corresponding data
            sim.set_target_domain(transformation_class, altered_percentage, outcoef_scaling_factor, n_target)
            sim.read_or_create_target_data()
         
            # Predict with adaptive model and CV on training data (wenda-cv) ---
            print("\n  Predicting with adaptive model (with CV on training data)...\n")
            # setup output path
            adaptive_cv_path = os.path.join(sim.paths.target_result_dir, "wnet_cv")
            os.makedirs(adaptive_cv_path, exist_ok=True)
            print("  path:", adaptive_cv_path, "\n", flush=True)
            # fit model
            adaptive_confidence_dir = os.path.join(sim.paths.target_confidence_dir, "confidences_blinear")
            adaptive_model = models.Wenda(x_train=sim.source_data.input.as_matrix(), y_train=sim.source_data.output, 
                                                  x_test=sim.target_data.input.as_matrix(), feature_model_dir=sim.paths.feature_model_dir, 
                                                  confidences_dir=adaptive_confidence_dir, **model_parameters)
#                                                   normalizer_x, normalizer_y, 
#                                                   feature_model_type, feature_model_params, n_jobs=n_jobs)
            print(" ", adaptive_model, "\n", flush=True)
            print("  Fitting feature models...", end="\n  ", flush=True)
            adaptive_model.fitFeatureModels()
            sys.stderr.flush()
            print("  Computing confidences...", end="\n  ", flush=True)
            adaptive_model.collectConfidences()
            sys.stderr.flush()
            # iterate over weighting functions, predict and save errors
            for k_wnet in kwnet_values:
                print("\n  - k_wnet =", k_wnet, end="\n    ", flush=True)
                weighting_function = lambda x: np.power(1-x, k_wnet)
                adaptive_cv_predict_path = os.path.join(adaptive_cv_path, "glmwnet_pow{0:d}_dsize{1:d}".format(k_wnet, n_per_group))
                predictions = adaptive_model.predictWithTrainingDataCV(weight_func=weighting_function, grouping=grouping, alpha=0.8, 
                                                                       n_splits=10, predict_path=adaptive_cv_predict_path)
                np.savetxt(os.path.join(adaptive_cv_predict_path, "predictions.txt"), predictions)
                # check test error
                print()
                errors = printTestErrors(predictions, sim.target_data.output, "Weighted elastic net:", indent=4)
                table = pandas.DataFrame([errors], columns=["mean", "median", "corr", "std", "iqr"])
                table.to_csv(os.path.join(adaptive_cv_predict_path, "errors.csv"), sep=";", quotechar='"')
                 
            np.savetxt(os.path.join(adaptive_cv_path, "kwnet_values.txt"), kwnet_values)
   
   
            # Predict with non-adaptive reference model (en-ls) ---
            print("\n  Predicting with non-adaptive reference model...\n", flush=True)
            ref_path = os.path.join(sim.paths.target_result_dir, "reference_10fold_cv")
            os.makedirs(ref_path, exist_ok=True)
            print("  path:", ref_path, "\n", flush=True)
            # calculate predictions (model has been fitted before)
            pred_simple_net = simple_model.predictNet(sim.target_data.input.as_matrix())
            pred_simple = simple_model.predict(sim.target_data.input.as_matrix())
            np.savetxt(os.path.join(ref_path, "predictions.txt"), np.vstack([pred_simple, pred_simple_net]).T)
            # check test error
            errors_simple_net = printTestErrors(pred_simple_net, sim.target_data.output, "Simple model EN:", indent=2)
            errors_simple_lm = printTestErrors(pred_simple, sim.target_data.output, "Simple model lm:", indent=2)
            table = pandas.DataFrame(np.vstack([errors_simple_lm, errors_simple_net]), 
                                     index=["lm", "net"], 
                                     columns=["mean", "median", "corr", "std", "iqr"])
            table.to_csv(os.path.join(ref_path, "errors.csv"), sep=";", quotechar='"')
            simple_model.savetxt(os.path.join(ref_path, "model"))     
             
             
            # Predict with weighted-EN baseline (wenda-mar) ---
            print("\n  Predicting with weighted-EN baseline ...\n")
            ref_wnet_path = os.path.join(sim.paths.target_result_dir, "reference_wnet_ecdf")
            os.makedirs(ref_wnet_path, exist_ok=True)
            print("  path:", ref_wnet_path, "\n", flush=True)
            # fit model
            simple_confidence_dir = os.path.join(sim.paths.target_confidence_dir, "confidences_simple_ecdf")
            baseline_wnet = models.BaselineWeightedEN(sim.source_data.input.as_matrix(), sim.source_data.output, 
                                                       sim.target_data.input.as_matrix(), normalizer_x, normalizer_y, 
                                                       simple_confidence_dir, n_jobs=n_jobs)
            print("  Collecting simple confidences...", end="\n  ", flush=True)
            baseline_wnet.collectConfidences()
            sys.stderr.flush()
            # iterate over weighting functions, predict and save errors
            for k_wnet in kwnet_values:
                print("\n  - k_wnet =", k_wnet, end="\n    ", flush=True)
                ref_wnet_predict_path = os.path.join(ref_wnet_path, "reference_wnet_k{}_dsize{}".format(k_wnet, n_per_group))
                weighting_function = lambda x: np.power(1-x, k_wnet)
                ref_wnet_predictions = baseline_wnet.predictWithTrainingDataCV(weight_func=weighting_function, grouping=grouping, 
                                                                               alpha=0.8, n_splits=10, predict_path=ref_wnet_predict_path)
                np.savetxt(os.path.join(ref_wnet_predict_path, "predictions.txt"), ref_wnet_predictions)
                # check test error
                print()
                ref_wnet_errors = printTestErrors(ref_wnet_predictions, sim.target_data.output, "Weighted EN baseline:", indent=4)
                table = pandas.DataFrame(np.vstack([ref_wnet_errors]), columns=["mean", "median", "corr", "std", "iqr"])
                table.to_csv(os.path.join(ref_wnet_predict_path, "errors.csv".format(k_wnet)), sep=";", quotechar='"')
           


    # Predict with prior knowledge (wenda-pn) ---
    print("\nPredicting with similarity-lambda regression...\n", flush=True)
    
    # define similarity
    similarity = lambda p: 1-p/100
    
    # define paths (as functions of simulation)
    adaptive_lambdareg_path = lambda sim: os.path.join(sim.paths.target_result_dir, "wnet_lambda_regression_0")
    adaptive_confidence_dir = lambda sim: os.path.join(sim.paths.target_confidence_dir, "confidences_blinear")
   
    for k_wnet in kwnet_values:
        print("- k_wnet =", k_wnet, "\n", flush=True)
        
        weighting_function = lambda x: np.power(1-x, k_wnet)
        predict_path = lambda sim: os.path.join(adaptive_lambdareg_path(sim), "wnet_k{0:d}_dsize{1:d}".format(k_wnet, n_per_group))
        
        # compute all weighted elastic net coefficient paths (without cross-validation)
        print("  Train weighted elastic-net coefficient paths " + 
              "({} simulations, percentages={})...".format(len(simulation_dirs), altered_percentages), end="\n  ", flush=True)
        pretrained_models = pretrain_weighted_EN_paths(simulation_dirs, transformation_class, altered_percentages, outcoef_scaling_factor, 
                                                       n_target, source_parameters, model_parameters, adaptive_confidence_dir,
                                                       weighting_function, grouping, predict_path)
        
        # iterate over all possible training-validation splits and ...
        # (i.e., always hold one percentage out for testing, fit lambda regression on the rest)
        for j in range(len(altered_percentages)):
            holdout_percentage = altered_percentages[j]
            fit_percentages = [p for p in altered_percentages if p != holdout_percentage]
            print("  - holdout_percentage =", holdout_percentage, "\n")
 
            # ... compute error paths and optimal lambdas for fit percentages
            print("    Compute known optimal lambdas " + 
                  "({} simulations, percentages={})...".format(len(simulation_dirs), fit_percentages), end="\n    ", flush=True)
            known_optimal_lambdas = compute_known_optimal_lambdas(pretrained_models, fit_percentages, simulation_dirs, source_parameters,
                                                                  transformation_class, outcoef_scaling_factor, n_target, grouping)
              
            # ... fit a line to similarity-lambda relationship and predict for holdout percentage
            print("    Fit similarity-lambda relationship and predict for holdout percentage...\n")
            optimal_lambda_model = lm.LinearRegression()
            training_similarites = similarity(known_optimal_lambdas.loc[:,"AlteredPercentage"].values)
            optimal_lambda_model.fit(training_similarites.reshape(-1, 1), 
                                     np.log(known_optimal_lambdas.loc[:,"LambdaOpt"]))
            predicted_lambda = np.exp(optimal_lambda_model.predict(similarity(holdout_percentage)))
            print("    optimal_lambda_model.coef_:", optimal_lambda_model.coef_)
            print("    predicted lambda:", predicted_lambda)
 
            
            # ... predict output for hold-out tissues using the predicted optimal lambdas
            print("\n    Predicting for holdout percentage...")
            for simulation_dir in simulation_dirs:
                print("\n    - Simulation:", os.path.basename(simulation_dir), "\n", flush=True)
                # set up simulation scenario and data
                sim = simulation.SimulationScenario(simulation_dir, **source_parameters, 
                                                    transformation_class=transformation_class, altered_percentage=holdout_percentage, 
                                                    outcoef_scaling_factor=outcoef_scaling_factor, n_target=n_target)
                sim.read_source_data(out=False)
                sim.read_target_data(out=False)
                # construct model
                adaptive_model = models.Wenda(x_train=sim.source_data.input.as_matrix(), y_train=sim.source_data.output, 
                                                      x_test=sim.target_data.input.as_matrix(), feature_model_dir=sim.paths.feature_model_dir,
                                                      confidences_dir=adaptive_confidence_dir(sim), **model_parameters)
                
                # check if lambda is within range of pretrained models and retrain if it is not
                print("      Ensure lambda range includes predicted lambda...", flush=True)
                pretrained_models = ensure_lambda_within_range(sim, adaptive_model, pretrained_models, predicted_lambda,
                                                               weighting_function, grouping, predict_path)
                # compute and save predictions
                predictions = predict_with_prior_knowledge(sim, adaptive_model, pretrained_models, predicted_lambda, grouping)
                np.savetxt(os.path.join(predict_path(sim), "predictions.txt"), predictions)
                
                # compute and save errors
                errors = printTestErrors(predictions, sim.target_data.output, 
                                         "Weighted elastic net with prior knowledge:", indent=6)
                table = pandas.DataFrame(np.vstack([errors]), columns=["mean", "median", "corr", "std", "iqr"])
                table.to_csv(os.path.join(predict_path(sim), "errors.csv"), sep=";", quotechar='"')
                
                # save lambda-similarity fit and predicted lambdas (same for each simulation)
                np.savetxt(os.path.join(predict_path(sim), "similarity_lambda_fit.txt"), [optimal_lambda_model.intercept_, optimal_lambda_model.coef_])
                np.savetxt(os.path.join(predict_path(sim), "predicted_lambda.csv"), [predicted_lambda])

    
    # print time stamp
    print('Program finished:', '{:%d.%m.%Y %H:%M:%S}'.format(datetime.datetime.now()))
    print()
    sys.stdout.flush()
    
    
  

if __name__ == '__main__':
    print("working directory:", os.getcwd(), "\n")
    main()
    
    
