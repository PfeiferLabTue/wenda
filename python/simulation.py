'''
Created on 28.01.2019

@author: Lisa Handl
'''

import os
import re
import glob

import numpy as np
import pandas

from graph import Graph, GaussianDAG
from builtins import isinstance
import itertools


class CompGaussianDAGInputModel:
    '''
    Class to simulate dependent inputs based on multiple GaussianDAGs.
    '''
    _graph_name_format = "Graph{0:d}"
    _gaussian_dag_name_format = "gaussian_dag_{0:02d}"
    
    def __init__(self, gaussian_dags):
        '''
        Initializes the input model.
        '''
        self._gaussian_dags = gaussian_dags
        self._varnames = self._construct_varnames(gaussian_dags)
        
    
    @classmethod
    def create_random(cls, graph_dir, input_noise_var):
        '''
        Create a random input model from the specified graphs.
        
        Generates random input weights and normalizes them so that each node has variance 1.
        '''
        graphs = cls._read_graphs_from_BIF(graph_dir)
        gaussian_dags = cls._simulate_gaussian_dags(graphs, input_noise_var)
        return cls(gaussian_dags)
    
    
    @classmethod
    def read(cls, gaussian_dag_dir, gaussian_dag_name_format=None):
        '''
        Read a CompGaussianDAGInputModel.
        '''
        if (not os.path.exists(gaussian_dag_dir)) or (not os.path.isdir(gaussian_dag_dir)):
            raise FileNotFoundError("Couldn't find gaussian_dag_dir: " + gaussian_dag_dir)
        if gaussian_dag_name_format is None:
            gaussian_dag_name_format = cls._gaussian_dag_name_format
        pattern = os.path.join(gaussian_dag_dir, re.sub("{.*?}", "*", gaussian_dag_name_format))
        dag_paths = glob.glob(pattern)
        # try to read gaussian_dags
        gaussian_dags = list()
        for i in range(len(dag_paths)):
            path = os.path.join(gaussian_dag_dir, gaussian_dag_name_format.format(i))
            gaussian_dags.append(GaussianDAG.read(path))
        # return if successful
        return cls(gaussian_dags)
    
    
    def simulate(self, n=1):
        '''
        Draw samples from the distribution specified by the CompGaussianDAGInputModel.
        '''
        # simulate from probabilistic DAGs
        result = pandas.DataFrame()
        for i in range(len(self._gaussian_dags)):
            # simulate from one DAG and convert to data frame
            sample = self._gaussian_dags[i].simulate(n)
            sample_frame = pandas.DataFrame.from_dict(sample)
            # append graph index to column names
            sample_frame.columns = ['graph{}-{}'.format(i, cname) for cname in sample_frame.columns]
            # merge with simulations from previous graphs
            result = pandas.concat([result, sample_frame], axis=1)
        # consistency check regarding the order of variables
        if np.sum(result.columns == self._varnames) != len(self._varnames):
            raise RuntimeError("Inconsistent order of variables!")
        return result
    
    
    def write(self, out_dir, gaussian_dag_name_format=None):
        '''
        Write a CompGaussianDAGInputModel.
        '''
        if gaussian_dag_name_format is None:
            gaussian_dag_name_format = self._gaussian_dag_name_format
        for i in range(len(self._gaussian_dags)):
            self._gaussian_dags[i].to_csv(os.path.join(out_dir, gaussian_dag_name_format.format(i)))
    
    
    def copy(self):
        '''
        Creates an independent copy of the CompGaussianDAGInputModel.
        '''
        gaussian_dag_copies = [dag.copy() for dag in self._gaussian_dags]
        return CompGaussianDAGInputModel(gaussian_dag_copies)
    
    
    def get_varnames(self, dag_indexes=None):
        if dag_indexes is None:
            return self._varnames
        dag_indexes = np.atleast_1d(dag_indexes)
        if len(dag_indexes) == 1:
            prefixes = "graph{}".format(dag_indexes[0])
        else:
            prefixes = tuple(["graph{}".format(i) for i in dag_indexes])
        altered = [name.startswith(prefixes) for name in self._varnames]
        return list(itertools.compress(self._varnames, altered))
           
    def get_indexes(self, varnames):
        '''
        Translates variables names to indexes.
        '''
        # not very fast, could be improved with better data structure
        return [self._varnames.index(name) for name in varnames]
        
    
    @classmethod
    def _read_graphs_from_BIF(cls, graph_dir, graph_name_format=None):
        '''
        Reads (non-weighted) graphs underlying the model from BIF format (as provided by BNGenerator).
        '''
        # search and count graph files
        if graph_name_format is None:
            graph_name_format = cls._graph_name_format
        pattern = os.path.join(graph_dir, re.sub("{.*?}", "*", graph_name_format) + ".xml")
        graph_files = glob.glob(pattern)
        # read them 
        n_components = len(graph_files)
        graphs = list()
        if n_components == 0:
            raise RuntimeError("No graph files found in directory: '{0:s}'!".format(graph_dir))
        for i in range(n_components):
            bif_name = os.path.join(graph_dir, graph_name_format.format(i+1) + ".xml")
            graph = Graph.readFromBIF(bif_name)
            graphs.append(graph)
        return graphs

    
    @classmethod
    def _write_graphs_to_SIF(cls, graphs, out_dir):
        '''
        Writes graphs to SIF format (e.g. for visualization).
        '''
        for i in range(len(graphs)):
            sif_name = os.path.join(out_dir, cls._graph_name_format.format(i+1) + ".sif")
            graphs[i].writeToSIF(sif_name)
    
    
    @classmethod
    def _construct_varnames(self, graphs):
        '''
        Constructs variable names from graph indices and node names.
        
        The alphabetic order of these names also specifies the order of 
        variables in a simulated dataset.
        '''
        varnames = list()
        for i in range(len(graphs)):
            graph = graphs[i]
            vertex_names = ["graph{}-{}".format(i, v) for v in graph.get_vertices()]
            vertex_names.sort()
            varnames.extend(vertex_names)
        return varnames
        
    
    @classmethod
    def _simulate_gaussian_dags(cls, graphs, input_noise_var):
        '''
        Randomly draws edge weights for GaussianDAGs.
        
        Coefficients are drawn from a multivariate standard normal distribution 
        and normalized to ensure that the variance of each node is 1.
        '''
        gaussian_dags = list()
        for i in range(len(graphs)):
            # set root distributions
            roots = graphs[i].get_roots()
            root_dist = dict()
            for r in roots:
                root_dist[r] = [0,1]
            # create GaussianDAG with NaN edge weights
            edge_weights = dict()
            for e in graphs[i].get_edges():
                edge_weights[e] = np.nan
            gaussian_dag = GaussianDAG(graphs[i]._graph_dict, edge_weights, root_dist, sigma=np.sqrt(input_noise_var))
            # choose random weights for conditional distributions (following the topological ordering)
            ordering = gaussian_dag.get_topological_ordering()
            contributions = dict()
            for vertex in ordering:
                gaussian_dag, contributions = cls._draw_weights_for(gaussian_dag, vertex, contributions)
            gaussian_dags.append(gaussian_dag)
        return gaussian_dags
    
          
    @classmethod
    def _draw_weights_for(cls, gaussian_dag, vertex, contributions):
        '''
        Function to draw edge weights for a specific vertex.
        
        Caomputes edge weights for ancestors recursively, returns if edge weights have been returned before.
        '''
        # root node? update contributions and leave
        if vertex in gaussian_dag.get_roots():
            contributions = gaussian_dag.compute_contributions(vertex, contributions)
            return gaussian_dag, contributions
        # weights are not yet drawn -> do so
        parents = gaussian_dag.get_parents(vertex)
        # simulate and store (unnormalized) edge weights
        weights = np.random.normal(size=len(parents))
        for i in range(len(parents)):
            gaussian_dag._edge_weights[(parents[i], vertex)] = weights[i]
        # compute variance of vertex with given edge weights
        var = gaussian_dag.compute_variance(vertex, contributions)
        # normalize and update weights
        weights = weights * np.sqrt((1-gaussian_dag._sigma**2)/(var-gaussian_dag._sigma**2))
        for i in range(len(parents)):
            gaussian_dag._edge_weights[(parents[i], vertex)] = weights[i]
        # compute contributions of current vertex (with final weights)
        contributions = gaussian_dag.compute_contributions(vertex, contributions)
        return gaussian_dag, contributions

    
    def get_combined_gaussian_dag(self):
        '''
        Combines all GaussianDAGs into one.
        
        Uses the same name pattern for vertices as in self._varnames.
        '''
        combined_dag = GaussianDAG()
        for i in range(len(self._gaussian_dags)):
            # add vertices
            for v in self._gaussian_dags[i].get_vertices():
                combined_dag.add_vertex("graph{}-".format(i) + v)
            # add edges
            for (v1, v2) in self._gaussian_dags[i].get_edges():
                weight = self._gaussian_dags[i]._edge_weights[(v1, v2)]
                combined_dag.add_edge("graph{}-".format(i) + v1, "graph{}-".format(i) + v2, weight)
            # set root distributions
            for v in self._gaussian_dags[i].get_roots():
                mu, sigma = self._gaussian_dags[i]._root_distribution[v]
                combined_dag.set_root_distribution("graph{}-".format(i) + v, mu, sigma)
            # set / check input noise
            if combined_dag._sigma is None:
                combined_dag._sigma = self._gaussian_dags[i]._sigma
            else:
                if not combined_dag._sigma == self._gaussian_dags[i]._sigma:
                    raise ValueError("Inconsistent input noise levels!")
        return combined_dag
    

class SparseLinearOutputModel:
    '''
    Class to simulate a sparse linear output model.
    '''
    def __init__(self, coefficients, noise_variance):
        '''
        Initializes the output mode.l
        '''
        self._coefficients = coefficients
        self._noise_sigma = np.sqrt(noise_variance)
    
    @classmethod
    def create_random(cls, input_model, noise_variance, n_rel=None, prob_rel=None):
        '''
        Randomly draws coefficients. Coefficients are nonzero with probability prob_rel.
        Non-zero coefficients are drawn from a multivariate standard normal distribution
        and normalized to ensure an output variance of 1.
        '''
        # check inputs
        if (n_rel is None) + (prob_rel is None) in [0,2]:
            raise ValueError("Exactly one of n_rel and prob_zero has to be specified!")
        if prob_rel is not None and (prob_rel < 0 or prob_rel > 1):
            raise ValueError("Parameter out of range: prob_zero must be a probability in [0,1]!")
        if not isinstance(input_model, CompGaussianDAGInputModel):
            raise ValueError("input_model must be in class CompGaussianDAGInputModel!")
        # pick indexes of relevant variables
        if n_rel is None:
            rel_indexes = cls._pick_relevant_by_prob(prob_rel, input_model)
        else:
            rel_indexes = cls._pick_n_relevant(n_rel, input_model)
        # draw random coefficients for relevant variables
        out_coef = np.zeros(shape=len(input_model._varnames))
        out_coef[rel_indexes] = np.random.normal(size=len(rel_indexes))
        # get combined graph and add output node
        combined_dag = input_model.get_combined_gaussian_dag()
        combined_dag.add_vertex("output")
        for index in rel_indexes:
            combined_dag.add_edge(input_model._varnames[index], "output", out_coef[index])
        var = combined_dag.compute_variance("output")
        # normalize and save coefficients
        out_coef = out_coef * np.sqrt((1-noise_variance)/(var-combined_dag._sigma**2))
        return cls(out_coef, noise_variance)
    
    
    @classmethod
    def _pick_n_relevant(cls, n, input_model):
        # fix number of relevant variables per graph
        quotient = n // len(input_model._gaussian_dags)
        remainder = n % len(input_model._gaussian_dags)
        n_per_graph = np.repeat(quotient, len(input_model._gaussian_dags))
        incr_indexes = np.random.choice(range(len(n_per_graph)), remainder, replace=False)
        n_per_graph[incr_indexes] = n_per_graph[incr_indexes] + 1
        # draw specified number of variables per graph
        rel_indexes = list()
        for i in range(len(n_per_graph)):
            variables = np.random.choice(input_model._gaussian_dags[i].get_vertices(), n_per_graph[i], replace=False)
            indexes = [input_model._varnames.index("graph{}-{}".format(i, var)) for var in variables]
            rel_indexes.extend(indexes)
        return np.array(rel_indexes)
    
    def _pick_relevant_by_prob(self, prob_rel, input_model):
        relevant = (np.random.uniform(size=len(input_model._varnames)) >= prob_rel)
        return np.arange(len(input_model._varnames))[relevant]
    
    def simulate(self, inputs):
        lin_comb = np.dot(inputs, self._coefficients)
        return lin_comb + self._noise_sigma*np.random.normal(size=len(lin_comb))
    
    @classmethod
    def read(cls, path, coef_name="coef.txt", sigma_name="sigma.txt"):
        '''
        Reads a SparseLinearOutputModel from the specified location.
        '''
        if (not os.path.exists(path)) or (not os.path.isdir(path)):
            raise FileNotFoundError("Couldn't find directory: " + path)
        # read coefficients
        coefficients = np.loadtxt(os.path.join(path, coef_name))
        noise_sigma = np.loadtxt(os.path.join(path, sigma_name))
        return cls(coefficients, noise_sigma**2)
    
    
    def copy(self):
        '''
        Creates an independent copy of the SparseLinearOutputModel.
        '''
        return SparseLinearOutputModel(self._coefficients.copy(), self._noise_sigma**2)
    
    
    def write(self, out_dir, coef_name="coef.txt", sigma_name="sigma.txt"):
        os.makedirs(out_dir)
        np.savetxt(os.path.join(out_dir, coef_name), self._coefficients)
        np.savetxt(os.path.join(out_dir, sigma_name), [self._noise_sigma])


class DataSimulationModel:
    '''
    Class for the model level of a simulated dataset, containing a 
    CompGaussianDAGInputModel and a SparseLinearOutputModel.
    
    More comfortable for reading / writing / simulating.
    '''
    
    def __init__(self, input_model, output_model):
        self.input_model = input_model
        self.output_model = output_model
        self._check_consistency()
    
    def simulate(self, n):
        '''
        Simulates and returns input for n samples with this model.
        
        Parameters:
        -----------
        @param n: number of samples to simulate
        '''
        input_data = self.input_model.simulate(n)
        output_data = self.output_model.simulate(input_data)
        return SimulationDataset(input_data, output_data)
    
    @classmethod
    def create_random(cls, graph_dir, input_noise_var, output_noise_var, n_rel=None, prob_rel=None):
        '''
        Creates a random DataSimulationModel by passing the parameters to 
        CompGaussianDAGInputModel and SparseLinearOutputModel.
        '''
        input_model = CompGaussianDAGInputModel.create_random(graph_dir, input_noise_var)
        output_model = SparseLinearOutputModel.create_random(input_model, output_noise_var, n_rel, prob_rel)
        return cls(input_model, output_model)
    
    @classmethod
    def read(cls, input_model_dir, output_model_dir, gaussian_dag_name_format=None, out_coef_name="coef.txt", out_sigma_name="sigma.txt"):
        '''
        Reads a DataSimulationModel from the specified input/output directories.
        '''
        input_model = CompGaussianDAGInputModel.read(input_model_dir, gaussian_dag_name_format=gaussian_dag_name_format)
        output_model = SparseLinearOutputModel.read(output_model_dir, coef_name=out_coef_name, sigma_name=out_sigma_name)
        return cls(input_model, output_model)

    
    def copy(self):
        '''
        Creates an independent copy of the DataSimulationModel.
        '''
        input_model_copy = self.input_model.copy()
        output_model_copy = self.output_model.copy()
        return DataSimulationModel(input_model_copy, output_model_copy)
    
    
    def write(self, input_model_dir, output_model_dir, gaussian_dag_name_format=None, out_coef_name="coef.txt", out_sigma_name="sigma.txt"):
        '''
        Writes a DataSimulationModel to the specified input/output directories.
        '''
        self.input_model.write(input_model_dir, gaussian_dag_name_format=gaussian_dag_name_format)
        self.output_model.write(output_model_dir, coef_name=out_coef_name, sigma_name=out_sigma_name)


    def _check_consistency(self):
        # check if number of inputs is consistent
        if len(self.input_model._varnames) != len(self.output_model._coefficients):
            raise RuntimeError("Inconsistent number of variables " + 
                               "({} in input model, {} in output model)!".format(len(self.input_model._varnames), 
                                                                                 len(self.output_model._coefficients)))


class SimulationDataset:
    
    def __init__(self, input_data, output_data):
        self.input = input_data
        self.output = output_data
        self._check_consistency()
    
    def write(self, input_path, output_path=None):
        self._write_input(input_path)
        self._write_output(output_path)
        
    def _write_input(self, input_path):
        self._check_path(input_path)
        self.input.to_csv(input_path, sep=";")
        
    def _write_output(self, output_path):
        self._check_path(output_path)
        np.savetxt(output_path, self.output)
        
    @staticmethod
    def _check_path(path):
        # check if file exists
        if os.path.exists(path):
            raise FileExistsError("Cannot overwrite file:", path)
        # create directory if necessary
        dirname = os.path.dirname(path)
        if not os.path.exists(dirname):
            os.makedirs(dirname, exist_ok=True)
            
    
    @classmethod
    def read(cls, input_path, output_path):
        input_data = cls._read_input(input_path)
        output_data = cls._read_output(output_path)
        return cls(input_data, output_data)
    
    @staticmethod
    def _read_input(input_path):
        input_data = pandas.read_csv(input_path, sep=";", index_col=0)
        return input_data
    
    @staticmethod
    def _read_output(output_path):
        output_data = np.loadtxt(output_path)
        return output_data
    
    def _check_consistency(self):
        # check if dimensions are consistent
        if self.input.shape[0] != len(self.output):
            raise RuntimeError("Inconsistent number of samples " + 
                               "({} in input data, {} in output data)!".format(self.input.shape[0], len(self.output)))
            

class CompTransformation:

    '''
    Abstract base class to change dependencies in some components of a ComponentGaussianDAG.

    Describes a transformation of a DataSimulationModel to simulate the change 
    from source to target domain by changing dependencies in some of the
    GaussianDAGs and scaling the output coefficients of the corresponding variables.
    
    Needs ComponentGaussianDAG and SparseLinearOutputModel as source input and output 
    models, respectively.
    '''
    def __init__(self, altered_components, outcoef_scaling_factor):
        '''
        Initializes a CompTransformation object.
        '''
        if self.__class__ is CompTransformation:
            raise TypeError('The abstract base class "' + self.__class__.__name__ 
                                  + '" cannot be initialized!')
        self.altered_components = altered_components
        self.outcoef_scaling_factor = outcoef_scaling_factor

    @classmethod
    def read(cls, path, components_name="altered_components.txt", scaling_factor_name="outcoef_scaling_factor.txt"):
        '''
        Reads a CompTransformation object from the specified directory.
        '''
        altered_components = np.atleast_1d(np.loadtxt(os.path.join(path, components_name), dtype=np.int64))
        outcoef_scaling_factor = np.loadtxt(os.path.join(path, scaling_factor_name))
        return cls(altered_components, outcoef_scaling_factor)
    
    @classmethod
    def create_random(cls, source_simulation_model, altered_percentage, outcoef_scaling_factor):
        '''
        Creates a CompTransformation object by randomly picking which components to change.
        '''
        n_components = len(source_simulation_model.input_model._gaussian_dags)
        n_altered = n_components*altered_percentage/100
        if not n_altered == int(n_altered):
            raise RuntimeError(("Cannot change {} out of {} components. Please choose altered_percentage "
                               + "as a multiple of {}!").format(n_altered, n_components, 100/n_components))
        altered_components = np.random.choice(range(n_components), size=int(n_altered), replace=False)
        return cls(altered_components, outcoef_scaling_factor)
    
    def write(self, out_dir, components_name="altered_components.txt", scaling_factor_name="outcoef_scaling_factor.txt"):
        '''
        Writes a CompTransformation object to the specified directory.
        '''
        os.makedirs(out_dir)
        np.savetxt(os.path.join(out_dir, components_name), self.altered_components, fmt="%d")
        np.savetxt(os.path.join(out_dir, scaling_factor_name), [self.outcoef_scaling_factor])
        
    def apply(self, source_simulation_model):
        '''
        Applies the DomainTransformation to a DataSimulationModel.
        
        Parameters:
        -----------
        @param data_simulation_model: The data simulation model
        
        @return: A modified DataSimulationModel resulting from the transformation.
        '''
        input_model = self._apply_to_input_model(source_simulation_model)
        output_model = self._apply_to_output_model(source_simulation_model)
        return DataSimulationModel(input_model, output_model)
        
    def _apply_to_input_model(self, source_simulation_model):
        '''
        Declaration of method to apply transformation to input model.
        
        Should be implemented by subclass.
        
        Parameters:
        -----------
        @param data_simulation_model: The data simulation model
        
        @return: transformed input model
        '''
        raise NotImplementedError('The class "' + self.__class__.__name__ +
                                  '" does not implement the method _apply_to_input_model!')
    
    def _apply_to_output_model(self, source_simulation_model):
        '''
        Applies transformation to output model.
        
        Scales coefficients of altered variables by self.outcoef_scaling_factor.

        Parameters:
        -----------
        @param data_simulation_model: The data simulation model
        
        @return: transformed output model
        '''
        # determine indexes of altered variables
        prefixes = tuple(["graph{}".format(i) for i in self.altered_components])
        altered = [name.startswith(prefixes) for name in source_simulation_model.input_model._varnames]
        # apply transformation to output model
        output_model = source_simulation_model.output_model.copy()
        output_model._coefficients[altered] = self.outcoef_scaling_factor*output_model._coefficients[altered]
        return output_model


class CompDependencyInversion(CompTransformation):
    '''
    Class to "invert" dependencies in some components of a ComponentGaussianDAG.
    
    Multiplies all edge weights in some of the GaussianDAGs with -1 (destroying 
    previous dependencies) and scales the output coefficients of the corresponding 
    variables.
    
    @see: CompTransformation
    '''

    def _apply_to_input_model(self, source_simulation_model):
        '''
        Applies the DomainTransformation to input model.
        
        @see: CompTransformation
        '''
        # apply transformation to input model
        # (multiply all edge weights in altered components by -1)
        input_model = source_simulation_model.input_model.copy()
        for i in self.altered_components:
            old_weights = input_model._gaussian_dags[i]._edge_weights
            new_weights = {edge: -old_weights[edge] for edge in old_weights}
            input_model._gaussian_dags[i]._edge_weights = new_weights
        return input_model
        
        
class CompDependencyRemoval(CompTransformation):
    '''
    Class to remove dependencies in some components of a ComponentGaussianDAG.
    
    Removes all edges in some of the GaussianDAGs (destroying previous 
    dependencies) and scales the output coefficients of the corresponding 
    variables.
    
    @see: CompTransformation
    '''
    
    def _apply_to_input_model(self, source_simulation_model):
        '''
        Applies the DomainTransformation to input model.
        
        @see: CompTransformation
        '''
        # apply transformation to input model
        # (remove all edges in altered components)
        input_model = source_simulation_model.input_model.copy()
        for i in self.altered_components:
            edgeless_dag = GaussianDAG()
            for v in input_model._gaussian_dags[i].get_vertices():
                edgeless_dag.add_vertex(v)
                edgeless_dag.set_root_distribution(v, 0, 1)
            input_model._gaussian_dags[i] = edgeless_dag
        return input_model

        

class SimulationScenario:
    '''
    Class to represent an entire simulation scenario, corresponding paths, etc.
    '''
    class SimulationPaths:
        '''
        Helper class to construct and store all relevant paths in a central place.
        '''
        def __init__(self, simulation_dir, input_noise_var, output_noise_var, n_relevant):
            # base directories
            self.simulation_dir = simulation_dir
            self.input_dir = os.path.join(simulation_dir, "input_noise_{0:.1f}".format(input_noise_var))
            self.output_dir = os.path.join(self.input_dir, "output_noise_{0:.1f}_rel_{1:d}".format(output_noise_var, n_relevant))
            self.feature_model_dir = os.path.join(self.input_dir, "feature_models_blinear")
            # directories for source model
            self.source_input_model_dir = os.path.join(self.input_dir, "input_model")
            self.source_output_model_dir = os.path.join(self.output_dir, "output_model")
            self.source_model_graph_dir = os.path.join(self.simulation_dir, "graphs")
            # paths for source data
            self.source_input_data_path = os.path.join(self.input_dir, "simulated_inputs/input_train.csv")
            self.source_output_data_path = os.path.join(self.output_dir, "simulated_output/output_train.csv")
            self.set_empty_target_domain()

        def set_target_domain_with_mismatch(self, transformation_class, altered_percentage, outcoef_scaling_factor):
            # basic target domain directories
            self.target_dir = os.path.join(self.output_dir, "{0:s}_p_{1:d}_a_{2:.2f}".format(transformation_class.__name__, 
                                                                                          altered_percentage, outcoef_scaling_factor))
            self.domain_transformation_dir = os.path.join(self.target_dir, "domain_transformation")
            # paths for target data
            self.target_input_data_path = os.path.join(self.target_dir, "simulated_data/input.csv")
            self.target_output_data_path = os.path.join(self.target_dir, "simulated_data/output.csv")
            # directories for target confidences and results
            self.target_confidence_dir = self.target_dir
            self.target_result_dir = os.path.join(self.target_dir, "model_results")
        
        def set_target_domain_without_mismatch(self):
            # basic target domain directories
            self.domain_transformation_dir = None
            # paths for target data
            self.target_input_data_path = os.path.join(self.input_dir, "simulated_inputs/input_test.csv")
            self.target_output_data_path = os.path.join(self.output_dir, "simulated_output/output_test.csv")
            # directories for target confidences and results
            self.target_confidence_dir = self.input_dir
            self.target_result_dir = os.path.join(self.output_dir, "model_results")
            
        def set_empty_target_domain(self):
            # initialize target domain directories
            self.domain_transformation_dir = None
            # paths for target data
            self.target_input_data_path = None
            self.target_output_data_path = None
            # directories for target confidences and results
            self.target_confidence_dir = None
            self.target_result_dir = None
            
        def __str__(self):
            result = """SimulationPaths(
                simulation_dir={0!r},
                input_dir={1!r},
                output_dir={2!r},
                feature_model_dir={3!r},
                source_input_model_dir={4!r},
                source_output_model_dir={5!r},
                source_model_graph_dir={6!r},
                source_input_data_path={7!r},
                source_output_data_path={8!r},
                domain_transformation_dir={9!r},
                target_input_data_path={10!r},
                target_output_data_path={11!r},
                target_confidence_dir={12!r},
                target_result_dir={13!r}
            )""".format(self.simulation_dir,
                        self.input_dir,
                        self.output_dir,
                        self.feature_model_dir, 
                        self.source_input_model_dir,
                        self.source_output_model_dir,
                        self.source_model_graph_dir,
                        self.source_input_data_path,
                        self.source_output_data_path,
                        self.domain_transformation_dir,
                        self.target_input_data_path,
                        self.target_output_data_path,
                        self.target_confidence_dir,
                        self.target_result_dir)
            return result
        
        
    def __init__(self, simulation_dir, input_noise_var, output_noise_var, n_relevant, n_source,
                 transformation_class = None, altered_percentage=None, outcoef_scaling_factor=None, n_target=None):
        # set up paths
        self.paths = self.SimulationPaths(simulation_dir, input_noise_var, output_noise_var, n_relevant)
        # save source simulation parameters
        self.input_noise_var = input_noise_var
        self.output_noise_var = output_noise_var
        self.n_relevant = n_relevant
        self.n_source = n_source
        # initialize source model and data variables
        self.source_model = None
        self.source_data = None
        # initialize target domain parameters / variables
        self.set_target_domain(transformation_class, altered_percentage, outcoef_scaling_factor, n_target)

    
    def set_target_domain(self, transformation_class, altered_percentage, outcoef_scaling_factor, n_target):
        '''
        Sets or updates target domain paths / parameters of the simulation scenario.
        '''
        # set / update paths
        if (transformation_class is None) and (altered_percentage is None) and (outcoef_scaling_factor is None) and (n_target is None):
            self.paths.set_empty_target_domain()
        elif (transformation_class is None) or (altered_percentage is None) or (outcoef_scaling_factor is None) or (n_target is None):
            raise ValueError("Target parameters have to be all None or all not None!")
        elif altered_percentage == 0:
            self.paths.set_target_domain_without_mismatch()
        else:
            self.paths.set_target_domain_with_mismatch(transformation_class, altered_percentage, outcoef_scaling_factor)
        # set / update target simulation parameters
        self.transformation_class = transformation_class
        self.altered_percentage = altered_percentage
        self.outcoef_scaling_factor = outcoef_scaling_factor
        self.n_target = n_target
        # reset target model / data
        self.domain_transformation = None
        self.target_model = None
        self.target_data = None
        
        
    def read_source_data(self, out=True):
        self.source_data = SimulationDataset.read(self.paths.source_input_data_path, self.paths.source_output_data_path)
        self._check_sample_size(self.source_data, self.n_source)
        if out:
            print("Read dataset with {} samples.".format(self.source_data.input.shape[0]))
    
    def read_target_data(self, out=True):
        self.target_data = SimulationDataset.read(self.paths.target_input_data_path, self.paths.target_output_data_path)
        self._check_sample_size(self.target_data, self.n_target)
        if out:
            print("Read dataset with {} samples.".format(self.target_data.input.shape[0]))
        
        
    def read_or_create_source_data(self, out=True):
        # case 1: full dataset exists  
        try:
            self.read_source_data(out=out)
        # case 2: dataset (or part of it) needs to be simulated -> read model
        except FileNotFoundError:
            self.read_or_create_source_model(out=out)
            self.source_data = self.complete_or_create_dataset(self.paths.source_input_data_path, self.paths.source_output_data_path, 
                                                               self.source_model, self.n_source, out=out)
        
    
    def read_or_create_target_data(self, out=True):
        # case 1: full dataset exists  
        try:
            self.read_target_data(out=out)
        # case 2: dataset (or part of it) needs to be simulated -> read model
        except FileNotFoundError:
            # read target model
            self.read_or_create_target_model(out=out)
            self.target_data = self.complete_or_create_dataset(self.paths.target_input_data_path, self.paths.target_output_data_path,
                                                               self.target_model, self.n_target, out=out)  
        
        
    def read_or_create_source_model(self, out=True):
        if self.source_model is None:
            self.source_model = self.read_or_create_model(self.paths.source_input_model_dir, self.paths.source_output_model_dir, 
                                                          self.paths.source_model_graph_dir, 
                                                          self.input_noise_var, self.output_noise_var, n_rel=self.n_relevant, out=out)
    
    def read_or_create_target_model(self, out=True):
        self.read_or_create_source_model(out=out)
        # special case: no mismatch
        if self.altered_percentage == 0:
            if out:
                print("Setting target model without mismatch.", flush=True)
            self.domain_transformation = None
            self.target_model = self.source_model
        # normal case: mismatch
        else:
            # read or create transformation and target domain model
            if out:
                print("Reading or generating domain transformation...", flush=True)
            self.domain_transformation = self.read_or_create_transformation(self.paths.domain_transformation_dir, self.source_model, 
                                                                            self.transformation_class, self.altered_percentage, 
                                                                            self.outcoef_scaling_factor, out=out)
            self.target_model = self.domain_transformation.apply(self.source_model)
    
    
    @classmethod
    def read_or_create_model(cls, input_model_dir, output_model_dir, graph_dir, input_noise_var, 
                             output_noise_var, n_rel=None, prob_rel=None, gaussian_dag_name_format=None, 
                             out_coef_name="coef.txt", out_sigma_name="sigma.txt", out=True):
            
        # case 1: full model exists    
        try:
            model = DataSimulationModel.read(input_model_dir, output_model_dir, 
                                                        gaussian_dag_name_format, out_coef_name, out_sigma_name)
            cls._check_input_noise(model.input_model, input_noise_var)
            cls._check_output_noise(model.output_model, output_noise_var)
            if out:
                print("Read DataSimulationModel with {} Bayesian networks and {} variables.".format(
                    len(model.input_model._gaussian_dags), len(model.input_model._varnames)), flush=True)
            return model
        except FileNotFoundError:
            pass
        # case 2: input model exists, output model doesn't
        try:
            input_model = CompGaussianDAGInputModel.read(input_model_dir, gaussian_dag_name_format=gaussian_dag_name_format)
            cls._check_input_noise(input_model, input_noise_var)
            output_model = SparseLinearOutputModel.create_random(input_model, output_noise_var, n_rel, prob_rel)
            output_model.write(output_model_dir, out_coef_name, out_sigma_name)
            model = DataSimulationModel(input_model, output_model)
            if out:
                print("Read input model with {} Bayesian networks and {} variables, created output model.".format(
                    len(model.input_model._gaussian_dags), len(model.input_model._varnames)), flush=True)
            return model
        except FileNotFoundError:
            pass
        # case 3: neither input nor output model exist
        model = DataSimulationModel.create_random(graph_dir, input_noise_var, output_noise_var, n_rel, prob_rel)
        model.write(input_model_dir, output_model_dir, gaussian_dag_name_format, out_coef_name, out_sigma_name)
        if out:
            print("Randomly generated DataSimulationModel with {} Bayesian networks and {} variables.".format(
                    len(model.input_model._gaussian_dags), len(model.input_model._varnames)), flush=True)
        return model
    
    
    @classmethod
    def read_or_create_transformation(cls, path, source_model, transformation_class, altered_percentage, 
                                      outcoef_scaling_factor, components_name="altered_components.txt", 
                                      scaling_factor_name="outcoef_scaling_factor.txt", out=True):
        try:
            transformation = transformation_class.read(path, components_name, scaling_factor_name)
            # check parameters
            if not np.isclose(transformation.outcoef_scaling_factor, outcoef_scaling_factor):
                raise RuntimeError("Found transformation with wrong scaling factor ({} instead of {})!".format(
                    transformation.outcoef_scaling_factor, outcoef_scaling_factor))
            if not np.isclose(100*len(transformation.altered_components)/len(source_model.input_model._gaussian_dags), altered_percentage):
                raise RuntimeError("Found transformation with wrong altered percentage ({}% instead of {}%)!".format(
                    100*len(transformation.altered_components)/len(source_model.input_model._gaussian_dags), altered_percentage))
            if out:
                print("Read transformation ({} altered components, scaling factor {}).".format(
                    len(transformation.altered_components), transformation.outcoef_scaling_factor))
            return transformation
        except FileNotFoundError:
            pass
        transformation = transformation_class.create_random(source_model, altered_percentage, outcoef_scaling_factor)
        transformation.write(path, components_name, scaling_factor_name)
        if out:
            print("Created transformation ({} altered components, scaling factor {}).".format(
                len(transformation.altered_components), transformation.outcoef_scaling_factor))
        return transformation
    
    
    @classmethod
    def complete_or_create_dataset(cls, input_path, output_path, model, n, out=True):
        '''
        Can be called when reading a complete dataset failed.
        
        Will simulate output for existing input or simulate and write the full dataset.
        '''
        # case 1: input data exists, output doesn't
        try:
            input_data = SimulationDataset._read_input(input_path)
            output_data = model.output_model.simulate(input_data)
            data = SimulationDataset(input_data, output_data)
            # check size
            if data.input.shape[0] != n:
                raise RuntimeError("Found dataset with wrong size ({} instead of {})!".format(data.input.shape[0], n))
            data._write_output(output_path)
            if out:
                print("Read input dataset with {} samples, simulated output.".format(data.input.shape[0]))
            return data
        except FileNotFoundError:
            # case 2: dataset doesn't exist -> simulate
            data = model.simulate(n)
            data.write(input_path, output_path)
            if out:
                print("Simulated dataset with {} samples.".format(data.input.shape[0]))
            return data
        
    
    @classmethod
    def _check_sample_size(cls, dataset, n):
        if dataset.input.shape[0] != n:
            raise RuntimeError("Found dataset with wrong size ({} instead of {})!".format(dataset.input.shape[0], n))
    
    @classmethod
    def _check_input_noise(cls, input_model, input_noise_var):
        for graph in input_model._gaussian_dags:
            if not np.isclose(graph._sigma**2, input_noise_var):
                raise RuntimeError("Found input model with wrong noise variance " + 
                                   "({} instead of {})!".format(graph._sigma**2, input_noise_var))
     
    @classmethod           
    def _check_output_noise(cls, output_model, output_noise_var):
        if not np.isclose(output_model._noise_sigma**2, output_noise_var):
            raise RuntimeError("Found output model with wrong noise variance " + 
                               "({} instead of {})!".format(output_model._noise_sigma**2, output_noise_var))



