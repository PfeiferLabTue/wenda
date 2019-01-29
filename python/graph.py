'''
Created on 28.01.2019

@author: Lisa Handl
'''

import warnings
from lxml import etree
import numpy as np
import os

class Graph:
    
    def __init__(self, graph_dict=None):
        if graph_dict is None:
            self._graph_dict = dict()
            self._reverse_dict = dict()
        else:
            self._graph_dict = graph_dict
            # make dict for reverse edges
            self._reverse_dict = dict()
            # add all vertices
            for vertex in self._graph_dict.keys():
                self._reverse_dict[vertex] = []
            # add reverse edges
            for vertex1 in self._graph_dict.keys():
                for vertex2 in self._graph_dict[vertex1]:
                    edges = self._reverse_dict[vertex2]
                    edges.append(vertex1)
                    self._reverse_dict[vertex2] = edges
        
    def add_vertex(self, vertex):
        if vertex in self._graph_dict:
            warnings.warn("Vertex {} was already contained in the graph! (Did nothing.)".format(vertex), RuntimeWarning)
        else:
            self._graph_dict[vertex] = []
            self._reverse_dict[vertex] = []
    
    def add_edge(self, vertex1, vertex2):
        if (vertex1 not in self._graph_dict) or (vertex2 not in self._graph_dict):
            raise RuntimeError("One of the vertices {} and {} is not in the graph!")
        fw_edges = self._graph_dict[vertex1]
        fw_edges.append(vertex2)
        self._graph_dict[vertex1] = fw_edges
        bw_edges = self._reverse_dict[vertex2]
        bw_edges.append(vertex1)
        self._reverse_dict[vertex2] = bw_edges
    
    def get_vertices(self):
        return list(self._graph_dict.keys())
    
    def get_edges(self):
        edge_list = []
        for vertex1 in self._graph_dict:
            for vertex2 in self._graph_dict[vertex1]:
                edge_list.append((vertex1, vertex2))
        return edge_list
    
    def get_children(self, vertex):
        if vertex not in self._graph_dict:
            raise RuntimeError("Graph does not contain a vertex {}".format(vertex))
        return self._graph_dict[vertex]
    
    def get_parents(self, vertex):
        if vertex not in self._graph_dict:
            raise RuntimeError("Graph does not contain a vertex {}".format(vertex))
        return self._reverse_dict[vertex]
    
    def get_roots(self):
        '''
        Find and return vertices without parents.
        '''
        roots = list()
        for vertex in self._reverse_dict:
            if len(self._reverse_dict[vertex]) == 0:
                roots.append(vertex)
        return roots
    
    
    @classmethod
    def readFromBIF(cls, path):
        tree = etree.parse(path)
        root = tree.getroot()
        graph = cls()
        # read and check basic structure
        if root.tag != "BIF":
            raise IOError("Illegal file format! First tag must be BIF.")
        networks = root.getchildren()
        if len(networks) != 1:
            raise IOError("Illegal number of networks! Must be 1.")
        if networks[0].tag != "NETWORK":
            raise IOError("Expected tag NETWORK, found {}!".format(networks[0].tag))
        # add variables as nodes
        variables = networks[0].findall("VARIABLE")
        for variable in variables:
            if variable.get("TYPE") != "nature":
                raise IOError("Illegal node type '{}'! Currently, only 'nature' nodes are supported.".format(variable.get("TYPE")))
            name = variable.find("NAME").text
            graph.add_vertex(name)
        # add definitions as edges
        definitions = networks[0].findall("DEFINITION")
        for definition in definitions:
            vertex2 = definition.find("FOR")
            vertices1 = definition.findall("GIVEN")
            for vertex1 in vertices1:
                graph.add_edge(vertex1.text, vertex2.text)
        return graph
    
    def writeToSIF(self, path):
        with open(path, 'w') as file:
            for vertex1 in self._graph_dict.keys():
                file.write("{0:s}".format(vertex1))
                edges = self._graph_dict[vertex1]
                if len(edges) > 0:
                    file.write(" edge")
                    for vertex2 in edges:
                        file.write(" {0:s}".format(vertex2))
                file.write("\n")


class GaussianDAG(Graph):
    
    def __init__(self, graph_dict=None, edge_weights=None, root_distribution=None, sigma=None):
        super(GaussianDAG, self).__init__(graph_dict)
        if edge_weights is None:
            self._edge_weights = dict()
        else:
            self._edge_weights = edge_weights
        if root_distribution is None:
            self._root_distribution = dict()
        else:
            self._root_distribution = root_distribution
        self._sigma = sigma
        self._node_ordering = None
    
    def add_edge(self, vertex1, vertex2, weight):
        if (vertex1 not in self._graph_dict) or (vertex2 not in self._graph_dict):
            raise RuntimeError("One of the vertices {} and {} is not in the graph!")
        fw_edges = self._graph_dict[vertex1]
        fw_edges.append(vertex2)
        self._graph_dict[vertex1] = fw_edges
        bw_edges = self._reverse_dict[vertex2]
        bw_edges.append(vertex1)
        self._reverse_dict[vertex2] = bw_edges
        edge = (vertex1, vertex2)
        if edge in self._edge_weights:
            raise RuntimeError("Tried to add duplicate edge from {} to {}!".format(vertex1, vertex2))
        self._edge_weights[edge] = weight
        
        
    def set_root_distribution(self, vertex, mu_a, sigma_a):
        self._root_distribution[vertex] = [mu_a, sigma_a]
        
        
    def simulate(self, n):
        values = dict()
        for vertex in self._graph_dict:
            values = self._draw_values_for(vertex, values, n)
        return values
        
    
    def compute_variance(self, vertex, contributions=None):
        '''
        Computes the variance of a vertex.
        '''
        # compute contributions if no intermediate result is given
        if contributions is None:
            contributions = self.compute_contributions(vertex)
        # compute variance of vertex based on parents weights and covariances
        var = 0
        parents = self.get_parents(vertex)
        for i in range(len(parents)):
            var = var + self._edge_weights[(parents[i], vertex)]**2
            for j in range(i):
                cov = self.compute_covariance(parents[i], parents[j], contributions)
                var = var + 2*cov*self._edge_weights[(parents[i], vertex)]*self._edge_weights[(parents[j], vertex)]
        var = var + self._sigma**2
        return var 
    
    def compute_covariance(self, vertex1, vertex2, contributions=None):
        '''
        Computes the covariance of two vertices.
        '''
        # compute contributions if no intermediate result is given
        if contributions is None:
            contributions = self.compute_contributions(vertex1)
            contributions = self.compute_contributions(vertex2, contributions)
        # compute covarance of vertices based on contributions and noise variance
        cov = 0
        roots = self.get_roots()
        for v in self.get_vertices():
            if (v in contributions[vertex1]) and (v in contributions[vertex2]):
                if v in roots:
                    cov = cov + contributions[vertex1][v]*contributions[vertex2][v]
                else:
                    cov = cov + contributions[vertex1][v]*contributions[vertex2][v]*(self._sigma**2)
        return cov
    
    
    def compute_contributions(self, vertex, contributions=None):
        '''
        Computes contributions of all of its ancestors to vertex.
        
        Contributions are saved in a dict in the following form:
        The entry contributions[v1][v2] tells how much of v1 is in v2. 
        
        Parameters:
        -----------
        @param vertex:            number of samples to simulate
        @param contributions:     (optional) intermediate result, 
                                  to which the new contributions are added
        
        @return: The dict containing contributions of all vertices to vertex
        '''
        if contributions is None:
            contributions = dict()
        parents = self.get_parents(vertex)
        # root node? Contributes 1 to itself
        if len(parents) == 0:
            contributions[vertex] = {vertex: 1}
            return contributions
        # ensure parent contributions have been computed
        for p in parents:
            if p not in contributions:
                contributions = self.compute_contributions(p, contributions)
        # compute contributions to vertex
        contrib_vertex = dict()
        for vertex2 in self.get_vertices():
            for p in parents:
                if vertex2 in contributions[p]:
                    if vertex2 not in contrib_vertex:
                        contrib_vertex[vertex2] = contributions[p][vertex2]*self._edge_weights[(p, vertex)]
                    else:
                        contrib_vertex[vertex2] = contrib_vertex[vertex2] + contributions[p][vertex2]*self._edge_weights[(p, vertex)]
        contrib_vertex[vertex] = 1
        contributions[vertex] = contrib_vertex
        return contributions
    
    
    def _draw_values_for(self, vertex, values, n):
        parents = self._reverse_dict[vertex]
        # case 1: vertex has been simulated before
        if vertex in values:
            return values
        # case 2: vertex is a root
        if len(parents) == 0:
            if vertex not in self._root_distribution:
                raise RuntimeError("Undefined distribution for vertex {}".format(vertex))
            mu_a, sigma_a = self._root_distribution[vertex]
            values[vertex] = np.random.normal(mu_a, sigma_a, n)
            return values
        # case 3: vertex has parents
        mu = np.zeros(n)
        for p in parents:
            if p not in values:
                values = self._draw_values_for(p, values, n)
            mu = mu + self._edge_weights[(p, vertex)]*values[p]
        if self._sigma == 0:
            values[vertex] = mu
        else:
            values[vertex] = mu + np.random.normal(0, self._sigma, n)
        return values
    
    def get_topological_ordering(self):
        # compute if not done before (using depth-first search), otherwise return directly
        if self._node_ordering is None:            
            sorted_list = list()
            marks = dict() # 0 is "in progress", 1 is "marked"
            for v in self.get_vertices():
                sorted_list, marks = self._sorting_visit(v, sorted_list, marks)
#             # translate into indexes
#             for i in range(len(sorted_list)):
#                 marks[sorted_list[i]] = i
            self._node_ordering = sorted_list
        return self._node_ordering
    
    def _sorting_visit(self, vertex, sorted_list, marks):
        # case 1: not visited before
        if vertex not in marks:
            marks[vertex] = 0
            # mark children first
            for c in self.get_children(vertex):
                sorted_list, marks = self._sorting_visit(c, sorted_list, marks)
            marks[vertex] = 1
            sorted_list.insert(0, vertex)
            return sorted_list, marks
        # case 2: visited and finished before
        elif marks[vertex] == 1:
            return sorted_list, marks
        # case 3: visited but not finished before
        elif marks[vertex] == 0:
            raise RuntimeError("Found cycle in GaussianDAG!!!")
        # case 4: something unexpected
        else:
            raise RuntimeError("Unknown mark {}, should be 0 or 1!".format(marks[vertex]))
                
    

    def to_csv(self, directory, edge_name="edges.csv", root_name="roots.csv", noise_name="noise.csv"):
        os.makedirs(directory)
        with open(os.path.join(directory, edge_name), 'w') as f:
            f.write("source,target,strength,positive\n")
            for node1 in self._graph_dict:
                for node2 in self._graph_dict[node1]:
                    weight = self._edge_weights[(node1, node2)]
                    f.write("{0:s},{1:s},{2:f},{3!s}\n".format(node1, node2, np.abs(weight), weight > 0))
        with open(os.path.join(directory, root_name), 'w') as f:
            f.write("root,mu,sigma\n")
            for a in self._root_distribution:
                mu, sigma = self._root_distribution[a]
                f.write("{0:s},{1:f},{2:f}\n".format(a, mu, sigma))
        with open(os.path.join(directory, noise_name), 'w') as f:
            f.write("noise_sigma\n")
            f.write("{0:f}\n".format(self._sigma))
    
    @classmethod
    def read(cls, directory, edge_name="edges.csv", root_name="roots.csv", noise_name="noise.csv"):
        graph = Graph()
        edge_weights = dict()
        with open(os.path.join(directory, edge_name), 'r') as f:
            line = f.readline()
            if not line == "source,target,strength,positive\n":
                raise IOError("Unexpected file format!")
            for line in f:
                splitted = line.split(",")
                node1 = splitted[0]
                node2 = splitted[1]
                weight = float(splitted[2])
                if node1 not in graph.get_vertices():
                    graph.add_vertex(node1)
                if node2 not in graph.get_vertices():
                    graph.add_vertex(node2)
                if not splitted[3].startswith("True"):
                    weight = -weight
                graph.add_edge(node1, node2)
                edge_weights[(node1, node2)] = weight
        root_distribution = dict()
        with open(os.path.join(directory, root_name), 'r') as f:
            line = f.readline()
            if line != "root,mu,sigma\n":
                raise IOError("Unexpected file format!")
            for line in f:
                splitted = line.split(",")
                node = splitted[0]
                mu = float(splitted[1])
                sigma = float(splitted[2])
                root_distribution[node] = [mu, sigma]
        with open(os.path.join(directory, noise_name), 'r') as f:
            line = f.readline()
            if line != "noise_sigma\n":
                raise IOError("Unexpected file format!")
            noise_sigma = float(f.readline())
        return cls(graph._graph_dict, edge_weights, root_distribution, noise_sigma)
    
    def copy(self):
        return GaussianDAG(self._graph_dict.copy(), self._edge_weights.copy(), self._root_distribution.copy(), self._sigma)
        
