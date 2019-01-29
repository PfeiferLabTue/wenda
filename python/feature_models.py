'''
This module contains classes contains the feature models for Wenda.

Created on 28.01.2019

@author: Lisa Handl
'''

import numpy as np
import GPy
import os
from scipy.stats import norm

import util

class FeatureModel:
    '''
    Base class for feature models.
    '''
    def __init__(self):
        '''
        Initializes the model.
        '''
        raise NotImplementedError('The abstract base class "' + self.__class__.__name__ 
                                  + '" cannot be initialized!')

    def fit(self, x, y):
        '''
        Fits the model to data.
        
        Parameters:
        -----------
        @param x: inputs (rows=samples)
        @param y: output 
        '''
        raise NotImplementedError('The class "' + self.__class__.__name__ +
                                  '" does not implement the method fit!')
        
    def predict(self, new_x):
        '''
        Predicts for new input data.
        
        Parameters:
        -----------
        @param x_new: new inputs (rows=samples)
        '''
        raise NotImplementedError('The class "' + self.__class__.__name__ +
                                  '" does not implement the method predict!')
    
    def savetxt(self, output_dir):
        '''
        Saves the model to txt files.
        
        Parameters:
        -----------
        @param output_dir: directory to which txt files are saved.
        '''
        raise NotImplementedError('The class "' + self.__class__.__name__ +
                                  '" does not implement the method savetxt!')


class FeatureGPR(FeatureModel):
    '''
    Feature model using Gaussian process regression.
    '''
    
    def __init__(self, kernel):
        '''
        Initializes a FeatureGPR model with the given kernel.
        
        @param kernel: the kernel to use in the Gaussian process model
        '''
        self._kernel = kernel.copy()
        self._model = None
        self._fitted = False

    def fit(self, x, y):
        '''
        Fits the FeatureGPR model to data.
        
        Parameters:
        -----------
        @param x: inputs (rows=samples)
        @param y: output 
        '''
        if self._fitted:
            raise RuntimeError("This model has been fitted to data before!")
        self._model = GPy.models.GPRegression(x, y, kernel=self._kernel)
        self._model.optimize()
        self._fitted = True
    
    def predict(self, x_new):
        '''
        Draws predictions from the FeatureGPR model.
        
        Parameters:
        -----------
        @param x_new: new inputs (rows=samples)
        '''
        if not self._fitted:
            raise RuntimeError("Cannot predict from model before calling fit()!")
        return self._model.predict(x_new)[0]
        
    def savetxt(self, output_dir):
        '''
        Saves the fitted model to a txt file.
        
        More precisely, the param_array of the GPRegression object is saved.
        '''
        if not self._fitted:
            raise RuntimeError("Model was not fitted, can only save fitted models!")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        np.savetxt(os.path.join(output_dir, 'param_array.txt'), self._model.param_array)

    @classmethod
    def read(cls, input_dir, x, y, kernel):
        '''
        Reads the fitted model from a txt file.
        '''
        feature_model = cls(kernel)
        feature_model._model = util.readGPRModel(os.path.join(input_dir, 'param_array.txt'), x, y, kernel) 
        feature_model._fitted = True
        return feature_model
    
    def getConfidence(self, x_test, y_test):
        '''
        Computes confidences for each sample in a test dataset.
        '''
        mu, sigma_sq = self._model.predict(x_test)
        res_normed = (y_test - mu) / np.sqrt(sigma_sq)
        confidences = (1 - abs(norm.cdf(res_normed) - norm.cdf(-res_normed)))
        return confidences
        

    
    