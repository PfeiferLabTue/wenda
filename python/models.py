'''
This module contains the main functionality of the Wenda method and the models we
compare it to.

Created on 28.01.2019

@author: Lisa Handl
'''

import warnings
import os
import sys
import importlib
import numpy as np
import multiprocessing
import random
import numbers

import glmnet
import sklearn.linear_model as lm
import util
import statsmodels.distributions.empirical_distribution


class ElasticNetLm:
    '''
    Class representing a model which combines elastic net and least squares regression.
    
    The idea is to first fit an elastic net model and then fit a linear model using
    least squares based on the features selected by the elastic net.
    
    @see S. Horvath, "DNA methylation age of human tissue and cell types", 
         Genome Biology, vol. 14, no. 10, 2013.
    '''      
    def __init__(self, alpha, n_splits, norm_x=None, norm_y=None, lambda_path=None, 
                 scoring="mean_squared_error", shuffle=True):
        '''
        Initializes the model.
        
        Parameters:
        -----------
        @param alpha:        mixing parameter for the elastic net
        @param lambda_path:  (optional) path for regularization parameter lambda
        '''
        # initialize elastic net model
        self.modelNet = glmnet.ElasticNet(alpha=alpha, lambda_path=lambda_path, standardize=False, scoring=scoring, n_splits=n_splits)
        self.modelLm = self.modelLm = lm.LinearRegression()
        self._norm_x = norm_x
        if self._norm_x is None:
            self._norm_x = util.NoNormalizer()
        self._norm_y = norm_y
        if self._norm_y is None:
            self._norm_y = util.NoNormalizer()
        self._norm_x_trained = None
        self._norm_y_trained = None
        self._shuffle = shuffle
      
    def fit(self, x, y, lamb="lambda_max"):
        self._norm_x_trained = self._norm_x(x)
        self._norm_y_trained = self._norm_y(y)
        self._x_train = self._norm_x_trained.normalize(x)
        self._y_train = self._norm_y_trained.normalize(y)
        # shuffle indexes before cross-validation if necessary
        train_indexes = np.arange(x.shape[0])
        if self._shuffle:
            random.shuffle(train_indexes)
        self.modelNet.fit(self._x_train[train_indexes], self._y_train[train_indexes])
        # choose the right lambda
        if lamb == "lambda_max":
            lamb = self.modelNet.lambda_max_
        elif lamb == "lambda_1se":
            lamb = self.modelNet.lambda_best_
        elif not isinstance(lamb, numbers.Number):
            raise ValueError("lamb must be 'lambda_max', 'lambda_1se' or a number!")
        self._lambda_opt = lamb
        # find index of optimal lambda and save nonzero elements
        self._index_opt = np.argmin(np.abs(self.modelNet.lambda_path_ - self._lambda_opt))
        self.nonzero = (self.modelNet.coef_path_[:,self._index_opt] != 0)
        self.modelLm = self.modelLm.fit(self._x_train[:, self.nonzero], self._y_train)
        
        
    def predict(self, new_x, lamb=None):
        if lamb is None:
            modelLm = self.modelLm
        else:
            index = np.argmin(np.abs(self.modelNet.lambda_path_ - lamb))
            nonzero = (self.modelNet.coef_path_[:,index] != 0)
            modelLm = lm.LinearRegression()
            modelLm = modelLm.fit(self._x_train[:, nonzero], self._y_train)
        # predict with linear regression model
        new_x_norm = self._norm_x_trained.normalize(new_x)
        pred = modelLm.predict(new_x_norm[:, self.nonzero])
        return self._norm_y_trained.unnormalize(pred)
    
      
    def predictNet(self, new_x, lamb=None):
        if lamb is None:
            lamb = self._lambda_opt
        new_x_norm = self._norm_x_trained.normalize(new_x)
        pred = np.atleast_1d(self.modelNet.predict(new_x_norm, lamb=lamb))
        return self._norm_y_trained.unnormalize(pred)
            
    
    def savetxt(self, output_dir):
#         if not self._fitted:
#             raise RuntimeError("Model was not fitted, can only save fitted models!")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        np.savetxt(os.path.join(output_dir, "alpha.txt"), [self.modelNet.alpha])
        np.savetxt(os.path.join(output_dir, "lambda_opt.txt"), [self._lambda_opt])
        np.savetxt(os.path.join(output_dir, "lambda_min.txt"), [self.modelNet.lambda_max_])
        np.savetxt(os.path.join(output_dir, "lambda_1se.txt"), [self.modelNet.lambda_best_])
        np.savetxt(os.path.join(output_dir, "lambda_path.txt"), self.modelNet.lambda_path_)
        np.savetxt(os.path.join(output_dir, "coef_path.txt"), self.modelNet.coef_path_)
        np.savetxt(os.path.join(output_dir, "intercept_path.txt"), self.modelNet.intercept_path_)
        np.savetxt(os.path.join(output_dir, "coef_lm.txt"), self.modelLm.coef_)
        np.savetxt(os.path.join(output_dir, "intercept_lm.txt"), [self.modelLm.intercept_])
        


class WeightedElasticNet(glmnet.ElasticNet):
    ''' 
    Wrapper around the glmnet ElasticNet class, with some custom defaults.
    '''
    
    def __init__(self, weights, alpha, n_splits, lambda_path=None, standardize=True, scoring="mean_squared_error", shuffle=True):
        super().__init__(alpha=alpha, lambda_path=lambda_path, standardize=standardize, scoring=scoring, n_splits=n_splits)
        self._penalty_weights = weights
        self._shuffle = shuffle
    
    def fit(self, x, y):
        # shuffle indexes to randomize CV
        train_indexes = np.arange(x.shape[0])
        if self._shuffle:
            random.shuffle(train_indexes)
        # determine lambda_path
        if self.lambda_path is None:
            lambda_max = self._compute_lambda_max(x, y, self._penalty_weights)
            lambda_min = np.min([self.min_lambda_ratio*lambda_max, 1e-4])
            self.lambda_path = np.geomspace(lambda_max, lambda_min, num=self.n_lambda)
        super().fit(x[train_indexes], y[train_indexes], relative_penalties=self._penalty_weights)
    
    def predict(self, new_x, lamb=None):
        if lamb is None:
            if self.lambda_max_ is None:
                raise RuntimeError("Regularization parameter is undetermined! Please use cross-validation or supply lamb.")
            else:
                lamb = self.lambda_max_
        return super().predict(new_x, lamb=lamb)
    
    def savetxt(self, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        np.savetxt(os.path.join(output_dir, "alpha.txt"), [self.alpha])
        np.savetxt(os.path.join(output_dir, "lambda_path.txt"), self.lambda_path_)
        np.savetxt(os.path.join(output_dir, "weights.txt"), self._penalty_weights)
        np.savetxt(os.path.join(output_dir, "coef_net_path.txt"), self.coef_path_)
        np.savetxt(os.path.join(output_dir, "intercept_net_path.txt"), self.intercept_path_)
        try:
            np.savetxt(os.path.join(output_dir, "lambda_max.txt"), [self.lambda_max_])
        except AttributeError:
            pass
        try:
            np.savetxt(os.path.join(output_dir, "lambda_1se.txt"), [self.lambda_best_])
        except AttributeError:
            pass   
    
    def _compute_lambda_max(self, x, y, relative_penalties):
        '''
        Computes the maximum lambda in an automatic lambda path like glmnet does.
        
        This is the first value of lambda for which all coefficients are zero.
        '''
        normalized_penalties = relative_penalties*len(relative_penalties)/np.sum(relative_penalties)
        scalar_products = np.abs(np.dot(y, x)) / np.array(normalized_penalties)
        lambda_max = np.max(scalar_products) / (x.shape[0]*self.alpha)
        return lambda_max
    

class Wenda:
    '''
    Class for the Wenda method.
    
    The idea is to use feature models to estimate how confident we are that
    a feature in a target domain fits well to the training data. Then, a tailored
    predictor can be fitted, weighting features with according to the confidences.
    '''
    
    _feature_model_format = 'model_{0:05d}'
    
    def __init__(self, x_train, y_train, x_test, norm_x, norm_y, feature_model_dir, feature_model_type, feature_model_params, confidences_dir, n_jobs=1,
                 maxtasksperchild=20, accept_partial_fit=False):
        # store normalizers
        self._norm_x = norm_x(x_train)
        self._norm_y = norm_y(y_train)
        
        # store normalized data
        self._x_train = self._norm_x.normalize(x_train)
        self._y_train = self._norm_y.normalize(y_train)
        self._x_test = self._norm_x.normalize(x_test)
        self._checkData()
        
        self._feature_model_type = feature_model_type
        self._feature_model_params = feature_model_params
        
        # check if feature models are fitted, create path if necessary
        self._feature_model_dir = feature_model_dir
        self._feature_models_fitted = self._checkFeatureModels(accept_partial_fit)

        # check if confidences exist, create path if necessary
        self._confidences_dir = confidences_dir
        if not os.path.exists(self._confidences_dir):
            os.makedirs(self._confidences_dir)
            self._confidences = None
        else:
            conf_path = os.path.join(self._confidences_dir, 'all_confidences.csv')
            if not os.path.exists(conf_path):
                self._confidences = None
            else:
                self._confidences = np.loadtxt(conf_path, delimiter=';')
                self._confidences = np.asfortranarray(self._confidences)
                if self._confidences.shape[0] != x_test.shape[0]:
                    raise RuntimeError("Incompatible confidences and test set: " +
                                       "confidence vectors: {0:d}, test samples: {1:d}"
                                       .format(self._confidences.shape[0], x_test.shape[0]))
                if self._confidences.shape[1] != x_train.shape[1]:
                    raise RuntimeError("Imcompatible confidences and training set: " +
                                       "confidence features: {0:d}, training features: {1:d}"
                                       .format(self._confidences.shape[1]), x_train.shape[1])
        # initialize status and variables
        self.n_jobs = n_jobs
        self.maxtasksperchild = maxtasksperchild

    def _checkData(self):
        assert self._x_train.shape[1] == self._x_test.shape[1], \
               "Incompatible datasets! Training and test data must " + \
               "have the same number of features!"
        assert self._x_train.shape[0] == self._y_train.shape[0], \
               "Incompatible datasets! Training data x and y must " + \
               "have the same number of samples!"
               
    def _checkFeatureModels(self, accept_partial_fit=False):
        if not os.path.exists(self._feature_model_dir):
            os.makedirs(self._feature_model_dir)
            return "none"
        else:
            existing_files = os.listdir(self._feature_model_dir)
            model_files = [self._feature_model_format.format(i) for i in range(self._x_train.shape[1]+1)]
            files_found = np.sum(np.in1d(existing_files, model_files))
            if files_found == 0:
                return "none"
            elif files_found == self._x_train.shape[1]:
                return "complete"
            elif files_found > self._x_train.shape[1]:
                raise RuntimeError(("Incompatible fitted feature models: " + 
                                    "{0:d} feature model files found, {1:d} were expected! " +
                                    "Use different feature_model_dir!")
                                   .format(files_found, self._x_train.shape[1]))
            elif accept_partial_fit:
                return "partial"
            else:
                raise RuntimeError(("Incompatible or incomplete fitted feature models: " + 
                                    "Found {0:d} feature model files, expected {1:d}!" + 
                                    "Consider using a different feature_mode_dir or setting " + 
                                    "accept_partial_fit=True!")
                                   .format(files_found, self._x_train.shape[1]))

    def fitFeatureModels(self):
        # skip if feature models have been fitted before
        if self._feature_models_fitted == "complete":
            warnings.warn("Unnecessary call to fitFeatureModels did nothing: " +
                          "Seems like feature models have been fitted before.", 
                          RuntimeWarning)
            sys.stderr.flush()
            return
        if self._feature_models_fitted == "partial":
            warnings.warn("Completing feature models partially trained in a previous run." +
                          "Existing models are not retrained.", 
                          RuntimeWarning)
            sys.stderr.flush()
        # otherwise, fit and save them
        if self.n_jobs == 1:
            for i in range(self._x_train.shape[1]):
                self._fitOneFeatureModel(i)
        else:
            old_values = self._limit_numpy_threads()
            pool = multiprocessing.Pool(processes=self.n_jobs, maxtasksperchild=self.maxtasksperchild)
            pool.map(self._fitOneFeatureModel, range(self._x_train.shape[1]))
            pool.close()
            self._reset_numpy_threads(old_values)
            print()
    
    def _fitOneFeatureModel(self, i):
        model_path = os.path.join(self._feature_model_dir, self._feature_model_format.format(i))
        if os.path.exists(model_path):
            return
        else:
            is_i = np.in1d(np.arange(self._x_train.shape[1]), i)
            data_x = self._x_train[:,~is_i]
            data_y = self._x_train[:,is_i]
            model = self._feature_model_type(**self._feature_model_params)
            model.fit(data_x, data_y)
            model.savetxt(os.path.join(self._feature_model_dir, self._feature_model_format.format(i)))
            print(".", end="", flush=True)

    def collectConfidences(self):
        # skip if confidences have been loaded before
        if self._confidences is not None:
            warnings.warn("Unnecessary call to collectConfidences did nothing: " +
                          "Confidences have been computed/loaded before.", 
                          RuntimeWarning)
            sys.stderr.flush()
            return
        if self.n_jobs == 1:
            self._confidences = np.empty(shape=self._x_test.shape)
            self._confidences.fill(np.nan)
            for i in range(self._x_train.shape[1]):
                self._confidences[:,i] = self._computeConfidences(i).squeeze()
        else:
            old_values = self._limit_numpy_threads()
            pool = multiprocessing.Pool(processes=self.n_jobs, maxtasksperchild=self.maxtasksperchild)
            confidences = pool.starmap(self._computeConfidences, [[i] for i in range(self._x_train.shape[1])])
            self._confidences = np.column_stack(confidences)
            pool.close()
            self._reset_numpy_threads(old_values)
        print()
        np.savetxt(os.path.join(self._confidences_dir, 'all_confidences.csv'), self._confidences, delimiter=';')

    def _computeConfidences(self, feature):
        # construct datasets
        is_feature = np.in1d(np.arange(self._x_train.shape[1]), feature)
        data_train_x = self._x_train[:,~is_feature]
        data_train_y = self._x_train[:,is_feature]
        data_test_x = self._x_test[:,~is_feature]
        data_test_y = self._x_test[:,is_feature]
        # read model
        model_path = os.path.join(self._feature_model_dir, self._feature_model_format.format(feature))
        model = self._feature_model_type.read(model_path, data_train_x, data_train_y, **self._feature_model_params)
        # compute confidences
        confidences = model.getConfidence(data_test_x, data_test_y)
        print(".", end="", flush=True)
        del model
        return confidences

    def _getConfidences(self, group, grouping):
        confidences = self._confidences[np.array(group==grouping),]
        confidences = np.mean(confidences, axis=0)
        return confidences


    def predictWithTrainingDataCV(self, weight_func, grouping, alpha, n_splits, predict_path, lambda_path=None):
        if n_splits < 3:
            raise ValueError("n_splits must be at least 3 for cross-validation!")
        models = self.trainFinalWeightedModels(weight_func, grouping, alpha, predict_path, n_splits, lambda_path)
        groups = np.unique(grouping)
        predictions = np.full(self._x_test.shape[0], np.nan)
        for g in groups:
            predictions[grouping==g] = models[g].predict(self._x_test[grouping==g,:])
        predictions = self._norm_y.unnormalize(predictions)
        return predictions
        
        
    def trainFinalWeightedModels(self, weight_func, grouping, alpha, predict_path, n_splits, lambda_path=None, out=True):
        if self._confidences is None:
            raise RuntimeError("Confidences must be computed/loaded before predicting!")
        pool = multiprocessing.Pool(processes=self.n_jobs, maxtasksperchild=self.maxtasksperchild)
        groups = np.unique(grouping)
        models = pool.starmap(self._trainOneWeightedModel,
                              [[g, weight_func(self._getConfidences(g, grouping)), alpha, lambda_path, n_splits, predict_path, out] for g in groups])
        pool.close()
        if out:
            print()
        return {groups[i]: models[i] for i in range(len(groups))}
    
    def _trainOneWeightedModel(self, group, weights, alpha, lambda_path, n_splits, predict_path, out=True):
        model = WeightedElasticNet(weights, alpha, n_splits=n_splits, lambda_path=lambda_path, standardize=False)
        model.fit(self._x_train, self._y_train)
        model.savetxt(os.path.join(predict_path, "model_{}".format(group)))
        if out:
            print(".", end="", flush=True)
        return model
    
    def computePredictionPaths(self, models, grouping):
        groups = np.unique(grouping)
        predictions = dict()
        for g in groups:
            predictions[g] = models[g].predict(self._x_test[grouping==g,:], lamb=models[g].lambda_path_)
            predictions[g] = self._norm_y.unnormalize(predictions[g])
        return predictions
    
    def predictWithPriorKnowledge(self, models, grouping, lambda_dict):
        '''
        Can be called for only a subset of tissues by supplying lambda only for some of them.
        '''
        groups = list(lambda_dict.keys())
        predictions = dict()
        for g in groups:
            predictions[g] = models[g].predict(self._x_test[grouping==g,:], lamb=lambda_dict[g])
            predictions[g] = self._norm_y.unnormalize(predictions[g])
        return predictions
    
    
    def __repr__(self):
        representation = """Wenda(
            _x_train=array(shape={0}),
            _y_train=array(shape={1}),
            _x_test=array(shape={2}),
            _norm_x={3!r},
            _norm_y={4!r}
            _feature_model_dir={5!r},
            _feature_models_fitted={6!r},
            _feature_model_type={7!r},
            _feature_model_params={8!r}
            _confidences_dir={9!r},
            """.format(self._x_train.shape,
                        self._y_train.shape,
                        self._x_test.shape,
                        self._norm_x,
                        self._norm_y,
                        self._feature_model_dir,
                        self._feature_models_fitted,
                        self._feature_model_type,
                        self._feature_model_params,
                        self._confidences_dir)
        if self._confidences is None:
            conf_part = "_confidences={0},".format(self._confidences)
        else:
            conf_part = "_confidences=array(shape={0}),".format(self._confidences.shape)
        representation = representation + conf_part + """
            n_jobs={0!r}
            )""".format(self.n_jobs)
        return representation
  
    def _limit_numpy_threads(self):
        env_variables = ["MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "OMP_NUM_THREADS"]
        old_values = [None]*len(env_variables)
        for i in range(len(env_variables)):
            if env_variables[i] in os.environ:
                old_values[i] = os.environ[env_variables[i]]
            os.environ[env_variables[i]] = "1"
        importlib.reload(np)
        return old_values
    
    def _reset_numpy_threads(self, old_values=[None]*3):
        env_variables = ["MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "OMP_NUM_THREADS"]
        for i in range(len(env_variables)):
            if old_values[i] is None:
                del os.environ[env_variables[i]]
            else:
                os.environ[env_variables[i]] = old_values[i]
        importlib.reload(np)


class BaselineWeightedEN:
    
    '''
    Code for the baseline with simpler feature weights (wenda-mar).
    '''
    
    def __init__(self, x_train, y_train, x_test, norm_x, norm_y, confidences_dir, n_jobs=1, maxtasksperchild=20):
        
        # store normalizers
        self._norm_x = norm_x(x_train)
        self._norm_y = norm_y(y_train)
        
        # store normalized data
        self._x_train = self._norm_x.normalize(x_train)
        self._y_train = self._norm_y.normalize(y_train)
        self._x_test = self._norm_x.normalize(x_test)

        # check if confidences exist, create path if necessary
        self._confidences_dir = confidences_dir
        if not os.path.exists(self._confidences_dir):
            os.makedirs(self._confidences_dir)
            self._confidences = None
        else:
            conf_path = os.path.join(self._confidences_dir, 'all_confidences.csv')
            if not os.path.exists(conf_path):
                self._confidences = None
            else:
                self._confidences = np.loadtxt(conf_path, delimiter=';')
                self._confidences = np.asfortranarray(self._confidences)
                if self._confidences.shape[0] != x_test.shape[0]:
                    raise RuntimeError("Incompatible confidences and test set: " +
                                       "confidence vectors: {0:d}, test samples: {1:d}"
                                       .format(self._confidences.shape[0], x_test.shape[0]))
                if self._confidences.shape[1] != x_train.shape[1]:
                    raise RuntimeError("Imcompatible confidences and training set: " +
                                       "confidence features: {0:d}, training features: {1:d}"
                                       .format(self._confidences.shape[1]), x_train.shape[1])
                
        # initialize status and variables
        self.n_jobs = n_jobs
        self.maxtasksperchild = maxtasksperchild

    def collectConfidences(self):
        # skip if confidences have been loaded before
        if self._confidences is not None:
            warnings.warn("Unnecessary call to collectConfidences did nothing: " +
                          "Confidences have been computed/loaded before.", 
                          RuntimeWarning)
            sys.stderr.flush()
            return
        pool = multiprocessing.Pool(processes=self.n_jobs, maxtasksperchild=self.maxtasksperchild)
        confidences = pool.starmap(self._computeConfidences, [[i] for i in range(self._x_train.shape[1])])
        self._confidences = np.column_stack(confidences)
        pool.close()
        print()
        np.savetxt(os.path.join(self._confidences_dir, 'all_confidences.csv'), self._confidences, delimiter=';')
    
    # estimate simple confidences, using the ECDF of the feature in the training data
    def _computeConfidences(self, feature):
        empirical_cdf = statsmodels.distributions.empirical_distribution.ECDF(self._x_train[:,feature])
        tmp = empirical_cdf(self._x_test[:,feature])
        confidences = 2*np.minimum(tmp, 1-tmp)
        print(".", end="", flush=True)
        return confidences

    def _getConfidences(self, group, grouping):
        confidences = self._confidences[np.array(group==grouping),]
        confidences = np.mean(confidences, axis=0)
        return confidences
    
    def predictWithTrainingDataCV(self, weight_func, grouping, alpha, n_splits, predict_path, lambda_path=None):
        if n_splits < 3:
            raise ValueError("n_splits must be at least 3 for cross-validation!")
        models = self.trainFinalWeightedModels(weight_func, grouping, alpha, predict_path, n_splits, lambda_path)
        groups = np.unique(grouping)
        predictions = np.full(self._x_test.shape[0], np.nan)
        for g in groups:
            predictions[grouping==g] = models[g].predict(self._x_test[grouping==g,:])
        predictions = self._norm_y.unnormalize(predictions)
        return predictions
        
    def trainFinalWeightedModels(self, weight_func, grouping, alpha, predict_path, n_splits, lambda_path=None):
        if self._confidences is None:
            raise RuntimeError("Confidences must be computed/loaded before predicting!")
        pool = multiprocessing.Pool(processes=self.n_jobs, maxtasksperchild=self.maxtasksperchild)
        groups = np.unique(grouping)
        models = pool.starmap(self._trainOneWeightedModel,
                              [[g, weight_func(self._getConfidences(g, grouping)), alpha, lambda_path, n_splits, predict_path] for g in groups])
        pool.close()
        print()
        return {groups[i]: models[i] for i in range(len(groups))}
    
    def _trainOneWeightedModel(self, group, weights, alpha, lambda_path, n_splits, predict_path):
        model = WeightedElasticNet(weights, alpha, lambda_path=lambda_path, standardize=False, n_splits=n_splits)
        model.fit(self._x_train, self._y_train)
        model.savetxt(os.path.join(predict_path, "model_{}".format(group)))
        print(".", end="", flush=True)
        return model
 




