'''
Created on 28.01.2019

@author: Lisa Handl
'''

import os
import platform

import pandas
import numpy as np


_base_path_windows = "//fs-bi/share/mm/home/handl/data/aging"
#     _base_path_linux = "/TL/stat_learn/work/handl/data/aging"
_base_path_linux = "/home/handl/data/aging"

# determine correct base path
if (platform.system() == "Windows"):
    _base_path = _base_path_windows
elif (platform.system() == "Linux"):
    _base_path = _base_path_linux
else:
    raise RuntimeError("No base path known for platform '" + platform.system() + "'")


class MethylationDataset:
    '''
    Simple class for DNA methylation datasets.
    
    Contains methylation and phenotype information.
    
    Attributes:
    -----------
    @var meth_table:    pandas data frame containing methylation levels
                        (rows=CpGs, columns=samples)
    @var pheno_table:   pandas data frame containing phenotype information
                        (rows=samples, columns=phenotype variables)
    @var meth_matrix:   methylation matrix as needed for models
                        (rows=samples, columns=CpGs)
    @var age:           age column as needed for models (not transformed)
    '''
    
    def __init__(self, methylation, phenotype, _meth_path=None, _pheno_path=None):
        '''
        Constructs a DNA methylation dataset based on methylation 
        and phenotype information.
        
        Parameters:
        ----------
        @param methylation:    methylation levels as a pandas data frame
                               (as exported from RnBeads using meth())
        @param phenotype:      phenotype information as a pandas data frame
                               (as exported from RnBeads using pheno())
        '''
        # save methylation and phenotype information
        self.meth_table = methylation
        self.pheno_table = phenotype
        assert self.meth_table.shape[1] == self.pheno_table.shape[0], \
               """Incompatible dimensions of methylation levels and phenotype info!:
               methylation samples: {0:d}, phenotype samples: {1:d}""" \
               .format(self.meth_table.shape[1], self.pheno_table.shape[0])
        
        # save paths (if supplied)
        self._meth_path = _meth_path
        self._pheno_path = _pheno_path
        if (self._meth_path is not None): 
            self._meth_path = os.path.normpath(self._meth_path)
        if (self._pheno_path is not None):
            self._pheno_path = os.path.normpath(self._pheno_path)
        
        # save methylation matrix and age as needed for model input
        self.meth_matrix = np.asfortranarray(self.meth_table.as_matrix().T)
        self.age = np.asfortranarray(self.pheno_table.loc[:, 'age'].as_matrix())
    
    @classmethod
    def read(cls, meth_path, pheno_path, sep=' ', quotechar='"'):
        '''
        Reads a DNA methylation dataset from two csv files.
        
        Parameters:
        ----------
        @param meth_path:     path of the file containing methylation levels
                              (as exported from RnBeads using meth())
        @param pheno_path:    path of the file containing phenotype information
                              (as exported from RnBeads using pheno())
        @param sep:           (optional) separator of csv files, default: ' '
        @param quotechar:     (optional) quotechar of csv files, default: '"'
        
        @return: a MethylationDataset object containing the data read. 
        '''
        methylation = pandas.read_csv(meth_path, sep=sep, quotechar=quotechar)
        phenotype = pandas.read_csv(pheno_path, sep=sep, quotechar=quotechar)
        return cls(methylation, phenotype, meth_path, pheno_path)
    
    def getSampleSize(self):
        '''
        Returns the sample size of the DNA methylation dataset.
        
        @return: the sample size
        '''
        return self.meth_table.shape[1]
    
    def getNofCpGs(self):
        '''
        Returns the number of CpG sites in the DNA methylation dataset.
        
        @return: the number of CpG sites
        '''
        return self.meth_table.shape[0]
    
    def __str__(self):
        return '''%s(%d samples, %d CpGs, 
                   meth_path=%r,
                   pheno_path=%r)''' \
                   % (self.__class__.__name__, self.getSampleSize(), self.getNofCpGs(),
                      self._meth_path, self._pheno_path)


class DataDNAmethPreprocessed:
    '''
    Class for the DNA methylation data used to evaluate wenda.
    
    Contains all paths to the data preprocessed by Michael Scherer
    (without sex chromosomes, imputed, reduced to ~13000 features).
    '''
    _path_training_meth = "training/3-TrainingReducedMeth.csv"
    _path_training_pheno = "training/3-TrainingReducedPheno.csv"
    _path_test_meth = "test/3-TestReducedMeth.csv"
    _path_test_pheno = "test/3-TestReducedPheno.csv"
    
    def __init__(self):
        '''
        Reads data from Michael Scherer and stores it in a DataDNAmethPreprocessed object.
        '''
        # read training data
        train_meth_path = os.path.join(_base_path, self._path_training_meth)
        train_pheno_path = os.path.join(_base_path, self._path_training_pheno)
        self.training = MethylationDataset.read(train_meth_path, train_pheno_path)

        # read test data  
        test_meth_path = os.path.join(_base_path, self._path_test_meth)
        test_pheno_path = os.path.join(_base_path, self._path_test_pheno)
        self.test = MethylationDataset.read(test_meth_path, test_pheno_path)
    
    def __str__(self):
        return '''%s(
        training=%r,
        test=%r
        )''' % (self.__class__.__name__, self.training, self.test)


class TissueSimilarity():
    '''
    Reads the tissue similarity translated from the similarities in the following GTEx Nature paper.
    
    @see: Aguet, F. et al. Genetic effects on gene expression across human tissues. NATURE 550, 204–213 (2017).
    '''
    _path_similarity = os.path.join(_base_path, "tissueSimilarityFromNaturePaper/cis_translated.csv")
    
    def __init__(self):
        self._similarity_table = pandas.read_csv(self._path_similarity, sep=";", index_col=0)
    
    def compute_similarity(self, tissue_series1, tissue_series2):
        tissue_freq1 = tissue_series1.value_counts(dropna=False) / len(tissue_series1)
        tissue_freq2 = tissue_series2.value_counts(dropna=False) / len(tissue_series2)
        sim_cutout = self._similarity_table.loc[tissue_freq1.index.values, tissue_freq2.index.values]
        sim_total = np.dot(np.dot(tissue_freq1.values, sim_cutout), tissue_freq2.values)
        return sim_total
