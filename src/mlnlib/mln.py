"""
Author: Eszter Bokanyi
E-mail: e.bokanyi@uva.nl

This file is created in the context of the POPNET project:
https://popnet.io

This file contains the MultiLayerNetwork class, which is the main class of
the package. It contains tools to work with a large multilayer network
using different edge types and layers efficiently.

There are three main ingredients:
    * nodes: pd.DataFrame with node attributes
    * A: scipy.sparse.csr_matrix with edge types encoded
    * layers: pd.DataFrame with layer information

The network is either loaded from disk, using the library mode
    e.g.
    >>> mln = MultiLayerNetwork(load_from_library=True, library_path="my_library")
    the referred folder should contain the following files:
        * edges.npz (scipy.sparse.csr_matrix of size NxN with binary linktypes)
        * nodes.csv.gz or nodes_{pd.__version__}.pkl (pandas dataframe with node attributes, at least 'label' column)
        * layers.csv (pandas dataframe with layer information, at least 'layer' column)
or from raw CSV files using the RawCSVtoMLN class from preparation.py
    e.g.
    >>> mln = MultiLayerNetwork(load_from_config=True, config_path="config.json")
    see the documentation of RawCSVtoMLN for more information
    see config.json for an example configuration file

It can also be initialized using in-memory objects.
    e.g.
    >>> mln = MultiLayerNetwork(nodes, edges, layers)

After loading the three key elements, class attributes and methods work the same.

Nodes have a label that can be of arbirary type, and an id that is an integer. The methods
`to_id` and `to_label` convert between the two representations. The nodes dataframe is
stored in `self.nodes`, and it always has at least a column called `label` that contains
the node labels, and an `id` column that contains the node ids. Apart from label and id,
there can be several more columns to store node properties.

The adjacency matrix self.A is stored in a `scipy.sparse.csr_matrix` class, that only
saves nonzero elements, and on which scipy csgraph algorithms run. The node index 'id'
is the same as the row and column index of the adjacency matrix.

The adjacency matrix contains integers that encode layers if viewed as binary numbers.

The layer information is stored in self.layers, which is a pandas DataFrame. It has the
following columns:
    * layer: the layer id, which is an integer between 0 and L-1, where L is the number
      of layers stored in self.L
    * label: the human-readable name of the layer
    * binary: the binary representation of the layer, which is power of 2 between  0 and
      2**L-1, where L is the number of layers
    * group: the group to which the layer belongs. This is useful for aggregating layers.

For further information, see the documentation of the MultiLayerNetwork class.

It is possible to export the network to disk either using the three files as described
above, or to a graphml file. The latter is useful for visualization purposes.

It is also possible to convert the network to igraph or networkx objects.

"""

# importing libraries
import pandas as pd
import polars as pl
import numpy as np
import igraph as ig
import networkx as nx

import gzip

from mlnlib.preparation import RawCSVtoMLN

from scipy.sparse import csr_matrix, save_npz, load_npz, eye, block_diag, kron
import json

from scipy.special import comb
 
from copy import deepcopy
import os

from datetime import datetime
 
# define constants
# how many nodes should one export in a networkx object
nx_node_limit = 10000
# how deep the BFS should go when getting the egonetworks
ego_depth_limit = 3
    
class MultiLayerNetwork:
    def __init__(
        self, 
        nodes = "",
        edges = "",
        layers = "",
        load_from_library = False,
        library_path = "",
        load_from_config = False,
        config_path = "",
        verbose = False,
        adjacency_element = "binary",
        use_polars = False,
        **kwargs
    ):
        """
        This class contains methods and attributes to work with a large
        multilayer network using different edge types and layers efficiently.

        There are three main ingredients:
            * nodes: pd.DataFrame with node attributes
            * edges: scipy.sparse.csr_matrix with edge types
            * layers: pd.DataFrame with layer information

 
        The network is either loaded
            1. from disk, using the library mode
                    e.g.
                    >>> mln = MultiLayerNetwork(load_from_library=True, library_path="my_library")
                    the referred folder should contain the following files:
                        * edges.npz (scipy.sparse.csr_matrix of size NxN with binary linktypes)
                        * nodes.csv.gz or nodes_{pd.__version__}.pkl (pandas dataframe with node attributes, at least 'label' column)
                        * layers.csv (pandas dataframe with layer information, at least 'layer' column)
            2. from in-memory objects
                    e.g.
                    >>> mln = MultiLayerNetwork(nodes, edges, layers)
            3. from raw CSV files using the RawCSVtoMLN class from preparation.py
                    e.g.
                    >>> mln = MultiLayerNetwork(load_from_config=True, config_path="config.json")
                    see the documentation of RawCSVtoMLN for more information
                    see config.json for an example configuration file

        After loading the three key elements, class attributes and methods work
        the same.

        Nodes have a label that can be of arbirary type, and an id that is
        an integer between 0 and N-1, where N is the number of nodes. The methods
        `to_id` and `to_label` convert between the two representations. The
        nodes dataframe is stored in `self.nodes`, and it always has at least
        a column called `label` that contains the node labels, and an `id`
        column that contains the node ids.

        Apart from label and id, there can be several more columns to store
        node properties. If necessary, a codebook CSV can be added to translate
        column values into a human-readable form.
 
        The adjacency matrix self.A is stored in a `scipy.sparse.csr_matrix`
        class, that only saves nonzero elements, and on which scipy csgraph
        algorithms run. The node index 'id' is the same as the row and column
        index of the adjacency matrix.

        The adjacency matrix contains integers that encode layers if viewed
        as binary numbers. Each possible layer is assigned an integer of the
        form 2**i. For example, if both type i and type j edge is present
        between two people, then the corresponding value in self.A would be
        2**i+2**j. It means that we can test for a certain layer using
        bitwise AND operation very cheaply. E.g. a certain element of self.A is
        7, then 7=1+2+4 which means that layers 0,1, and 2 are present
        between the two people, and 7&2 = 2 in Python (it behaves like a mask
        111 & 010 = 010).

        scipy.csr matrices are cheap to slice rowwise, but beware, some
        operation that seem straightforward in numpy might be costly (e.g.
        getting random coordinates after each other or colwise slicing)! If
        something is running too long, consult the scipy reference manual.

        Whenever a new instance is created from a parent instance, the node
        mapping is updated to refer to the new number of nodes and the new
        adjacency matrix.
 
        The layer information is stored in self.layers, which is a pandas 
        DataFrame. It has the following columns:
            * layer: the layer id, which is an integer between 0 and L-1, where
              L is the number of layers stored in self.L
            * label: the human-readable name of the layer
            * binary: the binary representation of the layer, which is power of 2
              between  0 and 2**L-1, where L is the number of layers
            * group: the group to which the layer belongs. This is useful for
              aggregating layers.
        
        There are two methods to convert between the different representations
        of layers:
            * `convert_layer_representation` converts between binary, label and
              layer representations
            * `convert_layer_binary_to_list` converts a binary representation
              to a list of layers that are present in the binary representation.
              It is possible to convert both to layer and label representations.

        Parameters:
            -------------
            nodes : pandas dataframe
                dataframe with node attributes, at least 'label' column
            edges : scipy.sparse.csr_matrix
                sparse matrix with edge types
            layers : pandas dataframe
                dataframe with layer information, at least 'layer' column
            load_from_library : bool, default False
                if True, load from library
            library_path : string, default ""
                path to library folder
            load_from_config : bool, default False
                if True, load from raw CSV files
            config_path : string, default ""
                path to config file
            verbose : bool, default False
                if True, print verbose output
            adjacency_element : string, default "binary"
                if "binary", then the adjacency matrix is binary, if "weight",
                then the adjacency matrix is weighted and has no layer information
            -------------

        Attributes:
            -------------
            nodes : pandas dataframe
                dataframe with node attributes, at least 'label' column
            A : scipy.sparse.csr_matrix
                sparse matrix with edge types
            layers : pandas dataframe
                dataframe with layer information, at least 'layer' column
            N : int
                number of nodes (does not necessarily correspond to number of active
                nodes in longitudinal networks)
            L : int
                number of layers
            adjacency_element : string
                if "binary", then the adjacency matrix is binary, if "weight",
                then the adjacency matrix is weighted and has no layer information
            igraph : igraph object
                igraph object of the network if created
            layer_adjacency_matrix : dict
                dictionary containing layer adjacency matrices if calculated
            group_adjacency_matrix : dict
                dictionary containing group adjacency matrices if calculated
            -------------

        Methods:
            -------------
            get_filtered_network: selecting subgraph
            get_layer_adjacency_matrix: getting layer adjacency matrix
            clear_layer_adjacency_matrices: clearing layer adjacency matrix storage
            clear_group_adjacency_matrices: clearing group adjacency matrix storage
            clear_all_adjacency_matrices: clearing layer and group adjacency matrix storage
            get_edgelist: getting edgelist in a pandas dataframe
            to_igraph: getting igraph object
            to_networkx: getting networkx object
            convert_layer_representation: converting between binary, label and layer representations
            convert_layer_binary_to_list: converting binary representation to list of layers
            to_id: converting node label to id
            to_label: converting node id to label
            -------------
        """
        
        """
        
        """
        
        # verbose printing for debugging purposes
        # function by kindall at StackOverflow:
        # https://stackoverflow.com/questions/5980042/how-to-implement-the-verbose-or-v-option-into-a-script
        if verbose:
            def verboseprint(*args):
                # Print each argument separately so caller doesn't need to
                # stuff everything to be printed into a single string
                for arg in args:
                    print(arg)
                print
            self.verboseprint = verboseprint
        else:
            self.verboseprint = lambda *a: None      # do-nothing function
        # check if a valid input exists
        if not load_from_library and not load_from_config and "csr_matrix" not in str(type(edges)):
            raise ValueError("You should either load a saved MLN object, load from raw files, or pass an edge adjacency matrix.")
    
        self.use_polars = use_polars

        # TODO document these
        self.null_values = kwargs.get("null_values",{})
        self.load_dtypes = kwargs.get("load_dtypes",{})
        self.final_dtypes = kwargs.get("final_dtypes",{})

        if "usecols" in kwargs:
            self.usecols = kwargs["usecols"]
        
        # loading from library
        if load_from_library:
            self.nodes, self.A, self.layers = self.load(library_path) 
        # loading from config
        elif load_from_config:
            print("""
            You are loading from raw CSV files.
            This is a slow process, and it is recommended to use the library mode.
            Use the export method to save time in the future.
                  
            E.g.:
            >>> mln.export("my_library")
            """)

            config = json.load(open(config_path))
            preparer = RawCSVtoMLN(**config)
            preparer.init_all()
            self.nodes, self.A, self.layers = preparer.nodes, preparer.A, preparer.layers
        # loading from in-memory objects
        else:
            self.nodes, self.A, self.layers = nodes, edges, layers

        self._map_id_to_label = dict(zip(self.nodes["id"],self.nodes["label"]))
        self._map_label_to_id = dict(zip(self.nodes["label"],self.nodes["id"]))
        self.N = self.A.shape[0]
        self.L = self.layers.shape[0]

        self.init_layer_dict()

        # helper variables for later use
        self.igraph = None
        self.layer_adjacency_matrix = {}
        self.group_adjacency_matrix = {}

        # set kwargs as attributes
        for k,v in kwargs.items():
            setattr(self, k, v)


        if "codebook" in kwargs:
            self.codebook = kwargs["codebook"]
        else:
            self.codebook = None
        self.init_codebook()

        self._verbose = verbose
        self._max_bin_linktype = self.layers["binary"].max()
        # TODO work with this attribute to switch between weight and binary
        self.adjacency_element = adjacency_element
 
        # passing some variables to children instances that don't change
        # shortens subsequent calls
        self._to_pass = {
            "layers" : self.layers,
            "codebook" : self.codebook,
            "use_polars" : self.use_polars,
            "_verbose" : self._verbose,
            "_layer_conversion_dict" : self._layer_conversion_dict,
            "_max_bin_linktype" : self._max_bin_linktype
        }
 
    def load(self, path):
        """
        If load_from_library is True, this function loads nodes, edges, and layers.
        
        Parameters:
        -------------
            path : string
                path to library folder

        Returns:
        --------
            nodes : string or None
                resulting path to a nodelist with attributes
            edges : string or None
                resulting path to an edgelist / adjacency matrix
            layers : string or None
                resulting path to a list of layers
        """

        # check if save_path exists
        if not os.path.exists(path):
            raise ValueError(f"Path {path} does not exist.")
        # check if save_path is a directory
        if not os.path.isdir(path):
            raise ValueError(f"Path {path} is not a directory.")
        # check if save_path has a /
        if path!="" and not path.endswith("/"):
            path += "/"

        edges_path = os.path.join(path, "edges.npz")
        nodes_path = os.path.join(path, f"nodes_{pd.__version__}.pkl")
        if not os.path.exists(nodes_path):
            nodes_path = os.path.join(path, "nodes.csv.gz")
        if not os.path.exists(edges_path):
            raise ValueError(f"Edges file {edges_path} does not exist.")
        if not os.path.exists(nodes_path):
            raise ValueError(f"Nodes file {nodes_path} does not exist.")
        layers_path = os.path.join(path, "layers.csv")
        if not os.path.exists(layers_path):
            raise ValueError(f"Layers file {layers_path} does not exist.")
        
        # if nodes is a pkl file, load picke, if csv.gz, load csv
        if nodes_path.endswith(".pkl"):
            nodes = pd.read_pickle(nodes_path)
        else:
            if hasattr(self,'usecols'):
                if self.use_polars:
                    kwargs = {"columns": self.usecols}
                else:
                    kwargs = {"usecols": self.usecols}
            else:
                kwargs = {}

            if self.use_polars:
                nodes = pl.read_csv(
                    nodes_path,
                    has_header=True,
                    null_values = self.null_values,
                    dtypes = self.load_dtypes,
                    low_memory=True, **kwargs
                )
                for col in self.load_dtypes:
                    if col in self.final_dtypes and self.load_dtypes[col]!=self.final_dtypes[col] and col in nodes.columns:
                        self.nodes = nodes.with_columns(nodes[col].cast(self.final_dtypes[col]))
            else:
                nodes = pd.read_csv(
                    nodes_path, 
                    index_col=None, 
                    header=0,
                    dtype = self.load_dtypes, **kwargs
            )
        # load edges from npz file
        edges = load_npz(edges_path)

        # load layers from csv file
        layers = pd.read_csv(layers_path, index_col=None, header=0)

        return nodes, edges, layers
                
    def init_layer_dict(self):
        """
        Initialize dictionary for efficient conversion from layer, label, and 
        binary layer representations to each other.
        Resulting dictionary is stored in self._layer_conversion_dict.
        
        Returns:
            -------------
            None
        """
        # machine readable
        binary = self.layers["binary"].tolist()
        # short
        layer = self.layers["layer"].tolist()
        # human-readable
        label = self.layers["label"].tolist()
        
        bin_to_layer = dict(zip(binary, layer))
        bin_to_label = dict(zip(binary, label))
        label_to_layer = dict(zip(label, layer))
        label_to_bin = dict(zip(label, binary))
        layer_to_bin = dict(zip(layer, binary))
        layer_to_label = dict(zip(layer, label))
        
        self._layer_conversion_dict = {
            'binary_to_layer' : bin_to_layer,
            'binary_to_label' : bin_to_label,
            'layer_to_label' : layer_to_label,
            'layer_to_binary' : layer_to_bin,
            'label_to_binary' : label_to_bin,
            'label_to_layer' : label_to_layer
        }
    
    
    def init_codebook(self):
        """
        # Initialize self.codebook with data from codebook. File should be .csv.
        # If no (valid) file is given, self.codebook = None

        Columns of the codebook should be:
            - column
            - key
            - value
        
        Parameters:
            -------------
            codebook : string
                string containing the (path to the) codebook CSV
            -------------
        """
        # check if codebook is given
        if self.codebook is not None:
            # check if self.codebook is a string and a valid path
            if isinstance(self.codebook, str) and os.path.exists(self.codebook):
                try:
                    # load codebook
                    self.codebook = pd.read_csv(self.codebook, index_col=None, header=0)
                except:
                    raise ValueError("Not a valid codebook file.")        
    
    def get_filtered_network(
            self, 
            nodes_selected=None, 
            layers_selected=[], 
            groups_selected=[], 
            node_type="label",
            layer_type="label",
            keep_node_alignment = False
            ):
        """
        Returns MultiLayerNetwork based on node and edge filtering.
 
        Parameters:
            -------------
            nodes_selected : list, default None
                list of node labels or ids to be selected
                which node representation is selected is controlled by node_type
            layers_selected : list, default []
                list of layer labels or ids to be selected
                which layer representation is selected is controlled by layer_type
            groups_selected : list, default []
                list of group labels to be selected
                this adds all layers in the groups to layers_selected
        Returns:
            -------------
            MultiLayeredNetwork
                filtered network with less data but the same structure as
                the parent object
        """

        if layer_type not in ["label", "layer"]:
            raise ValueError(f"Invalid layer_type '{layer_type}'. Please choose from 'label' or 'layer'.")
        
        if len(groups_selected)>0:
            if "group" not in self.layers.columns:
                raise ValueError("No group information found in self.layers. Please add a column called 'group' to self.layers.")
            # get layers corresponding to group
            for g in groups_selected:
                if g not in self.layers["group"].unique().tolist():
                    raise ValueError(f"Invalid group '{g}'. Please choose from {self.layers['group'].unique().tolist()}.")
                layers_selected += self.layers[self.layers["group"] == g][layer_type].tolist()

        # if there is any node selection, then decrease matrix size and grab the
        # relevant rows from the node attributes table
        if nodes_selected is not None: # and len(selected_nodes) > 0:
            # remove duplicates from list
            nodes_selected = np.unique(nodes_selected)
            # mapping label to node ids, creating node mapping
            if node_type == "label":
                nodes_selected = np.array([self.to_id(i) for i in nodes_selected])
            elif node_type == "id":
                nodes_selected = np.array(nodes_selected)
            else:
                raise ValueError(f"Invalid node_type '{node_type}'. Please choose from 'label' or 'id'.")
            
            # creating True/False mask for faster selection
            idx = np.array(np.zeros(self.N,dtype=bool))
            idx[nodes_selected] = True

            if keep_node_alignment:
                i,j = self.A.nonzero()
                test = np.isin(i,nodes_selected) & np.isin(j,nodes_selected)
                selected_i = i[test]
                del i
                selected_j = j[test]
                del j
                selected_data = self.A.data[test]
                del test
                A_selected = csr_matrix((selected_data,(selected_i,selected_j)),shape=(self.N,self.N),dtype=self.A.dtype)
                # keep full node dataframe
                if self.use_polars:
                    nodes_selected = self.nodes.clone() 
                else:
                    nodes_selected = self.nodes.copy()
                # filter down active nodes
                nodes_selected["active"] &= idx
            else:
                # slicing the adjacency matrix
                A_selected = self.A[idx,:][:,idx]
                # slicing the attribute table
                if self.use_polars:
                    nodes_selected = self.nodes.filter(idx)
                    nodes_selected["id"] = np.arange(sum(idx))
                else:
                    nodes_selected = self.nodes.iloc[nodes_selected].reset_index(drop=True)
                    nodes_selected["id"] = nodes_selected.index
        else:
            A_selected = deepcopy(self.A)
            nodes_selected = self.nodes
 
        if len(layers_selected)>0:
            for l in layers_selected:
                if l not in self.layers[layer_type].tolist():
                    raise ValueError(f"Invalid layer '{l}' for layer_type '{layer_type}'. Please choose from {self.layers[layer_type].tolist()}.")
            # adding up the binary codes for the full layers from the argumentif len(layers)>0:
            # based on type of layer value:
            binary_repr = sum([self._layer_conversion_dict[layer_type + "_to_binary"][layer] for layer in layers_selected])
            # select corresponding edges
            A_selected.data = A_selected.data & binary_repr
            # compress sparse matrix
            A_selected.eliminate_zeros()

        f = MultiLayerNetwork(
            nodes = nodes_selected,
            edges = A_selected,
            **self._to_pass
        )
 
        return f
    
    def get_layer_adjacency_matrix(self, layer, layer_type = 'layer', store = False, dtype='int64'):
        """
        Creates a binary adjacency matrix for one of the layers or one of the groups.

        Parameters:
            -------------
            layer : string
                layer or group to be selected
            layer_type : string, default 'layer'
                type of layer representation. Possible values are 'layer', 'label', and 'group'
            store : bool, default False
                if True, store the adjacency matrix in self.layer_adjacency_matrix or self.group_adjacency_matrix
            dtype : string, default 'int64'
                datatype of the adjacency matrix, smaller type means smaller memory footprint
            -------------
        """
        if layer_type not in ["label", "layer", "binary", "group"]:
            raise ValueError(f"Invalid layer_type '{layer_type}'. Please choose from 'label' or 'layer' or 'binary'.")
        
        if layer not in self.layers[layer_type].tolist():
            raise ValueError(f"Invalid layer '{layer}' for layer_type '{layer_type}'. Please choose from {self.layers[layer_type].tolist()}.")
        
        if layer_type == "group":
            if "group" not in self.layers.columns:
                raise ValueError("No group information found in self.layers. Please add a column called 'group' to self.layers.")
            if layer not in self.layers["group"].tolist():
                raise ValueError(f"Invalid group '{layer}'. Please choose from {self.layers['group'].unique().tolist()}.")
            # get layers corresponding to group
            layers = self.layers[self.layers["group"] == layer]["layer"].unique().tolist()
            # get corresponding binary representation
            binary_repr = sum([self._layer_conversion_dict["layer_to_binary"][layer] for layer in layers])
        else:
            # get binary layer value if it's not already given in binary
            if layer_type != "binary":
                binary_repr = self._layer_conversion_dict[layer_type + "_to_binary"][layer]
            else:
                binary_repr = layer

        # get layer value if it's not already given in layer
        if layer_type != "layer" and layer_type != "group":
            l = self._layer_conversion_dict[layer_type + "_to_layer"][layer]
        else:
            l = layer

        # check if layer is already stored in layer/group adjacency matrices
        if layer_type!="group" and l in self.layer_adjacency_matrix:
            return self.layer_adjacency_matrix[l]
        if layer_type=="group" and layer in self.group_adjacency_matrix:
            return self.group_adjacency_matrix[layer]

        # if it's not already stored, create it
        A_layer = deepcopy(self.A)
        A_layer.data = A_layer.data & binary_repr
        A_layer.eliminate_zeros()
        A_layer = A_layer.sign()

        if dtype != 'int64':
            A_layer = A_layer.astype(dtype)

        # store if necessary
        if store:
            if layer_type == "group":
                self.group_adjacency_matrix[layer] = A_layer
            else:
                self.layer_adjacency_matrix[l] = A_layer    

        return A_layer
    
    def clear_layer_adjacency_matrices(self):
        """
        Clears all layer adjacency matrices.

        This method resets the layer adjacency matrix attribute to an empty dictionary, 
        effectively removing all previously stored layer adjacency matrices.
        """
        self.layer_adjacency_matrix = {}

    def clear_group_adjacency_matrices(self):
        """
        Clears all group adjacency matrices.

        This method resets the group adjacency matrix attribute to an empty dictionary, 
        effectively removing all previously stored group adjacency matrices.
        """
        self.group_adjacency_matrix_adjacency_matrix = {}

    def clear_all_adjacency_matrices(self):
        """
        Clears all adjacency matrices, group and layer.

        This method resets the group and layer adjacency matrix attribute to an empty dictionary,
        effectively removing all previously stored adjacency matrices.
        """
        self.clear_layer_adjacency_matrices()
        self.clear_group_adjacency_matrices()
    
    def report_time(self, message = "", init=False):
        """
        This is a helper function for optimization time measurements.

        Parameters:
        -----------
            message : str, default ""
                Some hints on the measurement may be given here. If empty, 
                only the elapsed time will be printed.
            init : bool, default False
                If True, start timer, if False, measure elapsed time since
                last call.
        """
        if init:
            self.tic = datetime.now().timestamp()
            print("Initialized timer.")
        else:
            self.toc = datetime.now().timestamp()
            elapsed = self.toc - self.tic
            if message == "":
                print(f"Time elapsed: {elapsed/1000:.3f} seconds.")
            else:
                print(message + "\n" + f"Time elapsed: {elapsed:.5f} seconds.")
            self.tic = datetime.now().timestamp()
 
    def get_edgelist(self, edge_attribute = "binary"):
        """
        This function returns a  pandas dataframe containing the edge list
        representing sparse matrix stored in self.A. 

        Parameters:
            -----------
            edge_attribute : string, default "binary"
                controls what column is returned in the dataframe. Possible
                values are "binary", "layer", "label", and "weight". If None,
                only the source and target columns are returned.
 
        Returns
           -------
            edgelist : pandas dataframe containig the edge list representing self.A
                columns are source, target, and the one specified in edge_attribute
        """
        # self.report_time(init=True)

        # getting edges and binary linktypes
        edges = np.array(self.A.nonzero()).T
        weights = np.array([self.A.data]).T
        # self.report_time(message = "Getting data from sparse matrix.")
        if edge_attribute == "binary" or edge_attribute=="weight":
            # combine edges and weights
            edges_with_weights = np.concatenate((edges, weights), axis=1)
            if self.use_polars:
                edgelist = pl.DataFrame(edges_with_weights, schema = ["source", "target", edge_attribute])
            else:
                edgelist = pd.DataFrame(edges_with_weights, columns = ["source", "target", edge_attribute])

            if self.use_polars:
                # mapping back edges to the original labels
                edgelist = edgelist.with_column(
                    pl.col("source").map_dict(self.to_label).alias("source"),
                    pl.col("target").map_dict(self.to_label).alias("target")
                )
            else:
                # mapping back edges to the original labels
                edgelist["source"] = edgelist["source"].map(self.to_label)
                edgelist["target"] = edgelist["target"].map(self.to_label)

        elif edge_attribute == "layer" or edge_attribute == "label":
            edges_with_weights = np.concatenate((edges, weights), axis=1)
            if self.use_polars:
                edgelist = pl.DataFrame(edges_with_weights, schema = ["source", "target", "binary"])
            else:
                edgelist = pd.DataFrame(edges_with_weights, columns = ["source", "target", "binary"])
            # self.report_time(message = "Creating edgelist dataframe.")
            if self.use_polars:
                edgelist = edgelist.with_column(
                    pl.col("source").map_dict(self.to_label).alias("source"),
                    pl.col("target").map_dict(self.to_label).alias("target")
                )
            else:
                # mapping back edges to the original labels
                edgelist["source"] = edgelist["source"].map(self.to_label)
                edgelist["target"] = edgelist["target"].map(self.to_label)
            # self.report_time(message = "Remapping node labels.")
    
            # add colnames and (human) readable link types
            # if an edge has multiple linktypes, it is listed multiple times with the linktype code          
            # convert all unique binary linktypes to their labels
            link_dict = {}
            for link in edgelist["binary"].unique():
                link_dict[link] = self.convert_layer_binary_to_list(link, output_type=edge_attribute)
            # self.report_time(message = "Unfolded binary link identifiers.")
            # print(edgelist.head())

            # get pairs (binary_linktype, label) of each link
            if self.use_polars:
                # get pairs (binary_linktype, label) of each link
                edgelist = edgelist.with_column(
                    pl.col("binary").map_dict(link_dict).alias(edge_attribute)
                )
            else:
                edgelist[edge_attribute] = edgelist["binary"].map(link_dict)
            # self.report_time(message = "Mapped binary link identifiers.")
            # print(edgelist.head())

            if self.use_polars:
                 edgelist.drop("binary")
            else:
                edgelist.drop("binary", axis=1, inplace=True)
            edgelist = edgelist.explode(edge_attribute)
            # self.report_time(message = "Exploded binary link identifiers.")
            # print(edgelist.head())
            # self.report_time(message = "Adding that to dataframe?")
            # print(edgelist.head())
        elif edge_attribute is None:
            if self.use_polars:
                edgelist = pl.DataFrame(edges, schema = ["source", "target"])
            else:
                edgelist = pd.DataFrame(edges, columns = ["source", "target"])
            if self.use_polars:
                edgelist = edgelist.with_column(
                    pl.col("source").map_dict(self.to_label).alias("source"),
                    pl.col("target").map_dict(self.to_label).alias("target")
                )
            else:
                # mapping back edges to the original labels
                edgelist["source"] = edgelist["source"].map(self.to_label)
                edgelist["target"] = edgelist["target"].map(self.to_label)
        else:
            raise ValueError(f"Invalid edge_attribute '{edge_attribute}'. Please choose from 'binary', 'layer', 'label' or 'weight' or None.")
        
        return edgelist
    
    def to_igraph(self, directed=True, edge_attributes=True, node_attributes=False, replace_igraph=False, edge_attribute_type="binary"):
        """
        This function returns an igraph object of the sparse matrix stored in
        self.A. Edge attributes (layer types) and node attributes (from
        self.nodes) can be added to this object.
 
        Parameters:
            -----------
            directed : boolean, default True
                mode of returned igraph object: True for directed, False for
                undirected graph
            edge_attributes : boolean, default True
                True if weights or layer types should be stored in the igraph object
            node_attributes : boolean, default False
                True if node attributes should be added to igraph object
            replace_igraph : boolean, default False
                True if self.igraph should be replaced by the new igraph object
            edge_attribute_type : string, default "binary"
                Type of edge attribute. Options: "binary", "layer", "label" or "weight"
        Returns
           -------
            g : igraph object describing graph from input_data
        """
 
        # set mode directed / undirected
        if directed: mode = 'directed'
        else: mode = 'undirected'
 
        # create igraph object
        if edge_attributes:
            g = ig.Graph.Weighted_Adjacency(self.A, mode=mode, attr="layer")
        else:
            g = ig.Graph.Adjacency(self.A.sign(), mode=mode)
        
        if edge_attributes:
            if edge_attribute_type == "binary":     
                # obtain and add (human readable) link types
                layer_dict = {}
                for layer in set(g.es["layer"]):
                    layer_dict[layer] = self.convert_layer_binary_to_list(layer, output_type="layer")

                # rewrite link types to human readable
                g.es["layer"] = [layer_dict[x] for x in g.es["layer"]]
            elif edge_attribute_type == "weight":
                g.es["weight"] = g.es["layer"]
                del g.es["layer"]
            else:
                raise ValueError(f"Invalid edge_attribute_type '{edge_attribute_type}'. Please choose from 'binary' or 'weight'.")
                
        # add node attributes to graph from self.node_attributes
        if node_attributes:
            for col_name in self.nodes.columns:
                # first turning the series into a list improves performance
                g.vs[col_name] = list(self.nodes[col_name])
        else:
            # Add only label
            g.vs["label"] = list(self.nodes["label"])
        
        # store igraph object as mln attribute
        if self.igraph is None or replace_igraph:
            self.igraph = g
        
        return g
 
    def to_networkx(self, 
                    directed = True, 
                    edge_attributes = True, 
                    node_attributes = False, 
                    edge_attribute_type = "binary",
                    layer_type = "layer",
                    ignore_limit = False):
        """
        This function returns a networkx object of the sparse matrix stored in
        self.A. Edge attributes (layer types) and node attributes (from
        self.nodes) can be added to this object.
 
        Parameters:
            -----------
            directed : boolean, default True
                mode of returned igraph object: True for directed, False for
                undirected graph
            edge_attributes : boolean, default True
                True if edge attributes (layer types) from should be
                added to igraph object.
            node_attributes : boolean, default False
                True if node attributes should be added to networkx object
                obtained from self.node_attributes The "label" column is always
                added
            ignore_limit : boolean, default False
                False if nx object can have at most nx_node_limit nodes Set to
                True to ignore this limit
 
        Returns:
            -------
            g : networkx object based on self.A
        """
 
        if not ignore_limit and self.A.shape[0] > nx_node_limit:
            print(f"Networkx limit ({nx_node_limit}) exceeded. We recommend using igraph for larger graphs.")
            print("To ignore this limit, please use \'ignore_limit=True\'")
            return
        
        # set graph_type to directed or undirected
        if directed: graph_type = nx.DiGraph
        else: graph_type = nx.Graph
 
        # create networkx graph
        g = nx.from_scipy_sparse_array(self.A, create_using=graph_type)
 
        # add remaining attributes
        if node_attributes:
            for col_name in self.nodes:
                cur_col = list(self.nodes[col_name])
                attribute_list = dict(zip(list(range(len(cur_col))), cur_col))
                nx.set_node_attributes(g, attribute_list, col_name)
        else:
            # always add "label" column
            nx.set_node_attributes(g, self._map_id_to_label, "label")
        
        # obtain and add (human readable) link types use dict for optimization
        link_dict = {}
        
        if edge_attributes:
            if edge_attribute_type == "weight":
                    for s, t, d in g.edges(data=True):
                        d["weight"] = g[s][t]["weight"]
            elif edge_attribute_type == "binary":
                if edge_attributes:
                    for s, t, d in g.edges(data=True):
                        weight = g[s][t]["weight"]
        
                        if weight in link_dict:
                            g[s][t]["layer"] = link_dict[weight]
                        else:
                            link_types = self.convert_layer_binary_to_list(weight,output_type=layer_type)
                            link_dict[weight] = link_types
                            g[s][t]["layer"] = link_types
                        # remove weights
                        d.pop("weight", None)
 
        return g
    
    def convert_layer_representation(self, layer, input_type="layer", output_type="binary"):
        """
        This function converts a single layer type or list of layer types to
        another representation.
 
        Parameters:
            -----------
            layer : int, string or list, no default
                input layer type(s) to be converted
            input_type : string
                Type of input. Options: "label", "layer" and "binary"
            output_type : string
                Type of output. Options: "label", "layer" and "binary"

        Returns:
            -----------
            layers : string or list
                converted layer type(s)
         """
        
        dict_name = input_type + '_to_' + output_type
        try:
            d = self._layer_conversion_dict[dict_name]
        except KeyError:
            print('Error: dictionary value not found. Please choose from "layer", "label", "binary"')
            return None
        
        try:
            if type(layer) == list:
                return [d[link] for link in layer]
            else:
                return d[layer]
        except:
            print(f'Error: invalid linktype found: {input_type} to {output_type}')
            return None
    
    def convert_layer_binary_to_list(self,num,output_type="layer"):
        """
        Based on the integer binary layer type, returns a list with the layers.
 
        e.g.: 
        >>> self.convert_layer_binary_to_list(3) returns ["aunt/uncle","co-parent"]
 
        Parameters:
            --------------
            num: int
                integer number to be converted to binary and returned as linktypes
            output_type: string, default "layer"
                type of output. Options: "label", "layer" and "binary"
        Returns:
            ---------------
            list(type(output type of self.convert_layer_representation))
                list of layers corresponding to binary integer value
        """

        # sanity check, max value is 1111....1 with as many 1s as there are layers self.L converted to decimal
        if np.log2(float(num)) >= self.layers.index[-1]+1:
            raise ValueError(f"Layer binary value {num} is not a valid linktype in the network.")

        return [self.convert_layer_representation(2**i, input_type='binary', output_type=output_type)\
                 for i in range(self.layers.index[-1]+1) if int(num)&(2**i)>0]
    
    def save_to_graphml(self, 
                        file_name, 
                        directed = True, 
                        edge_attributes = True, 
                        node_attributes = False, 
                        overwrite = False, 
                        edge_attribute_type="binary"):
        """
        Save self.igraph to GraphML file called file_name. This can be read into 
        Gephi or other external software.

        If self.igraph is None, it first creates an igraph object.

        Parameters:
            -------------
            file_name : string
                file to write graph to. Extension should be .graphml
            directed : boolean, default True
                mode of returned igraph object: True for directed, False for
                undirected graph
            edge_attributes : boolean, default True
                True if weights or layer types should be stored in the igraph object
            node_attributes : boolean, default False
                True if node attributes should be added to igraph object
            overwrite : boolean, default False
                if True, overwrites existing files
            edge_attribute_type : string, default "binary"
                Type of edge attribute. Options: "binary", "layer", "label" or "weight"
            -------------
        """
        _, extension = os.path.splitext(file_name)

        if extension == ".graphml":
            if self.igraph is None or overwrite:
                self.igraph = self.to_igraph(
                    node_attributes=node_attributes,
                    edge_attributes=edge_attributes,
                    directed=directed,
                    edge_attribute_type=edge_attribute_type
                )
            else:
                raise Warning("Warning: igraph object already exists, not creating a new one - using it as you've created it earlier." +\
                              "If you want to overwrite it, call the function with overwrite=True.")
            self.igraph.write_graphml(file_name)
        else:
            raise ValueError(f"Error: {extension} is not a valid file extension for graphml saving.")
    
    def export_edges(self, file_name):
        """
        Write self.A to file called file_name. The file extension is read to
        determine the type of output. Options are:
        - ".npz" or no extension: Binary (default)
        - ".csv" ".csv.gz": Edgelist format
        Note: if self.igraph does not yet exist, an igraph object will be
        generated
 
        Parameters:
            -----------
            file_name : str
                file to write graph to. Extension should contain the type of output
                [".npz", ".csv", ".csv.gz"]

        """
        try:
            f = open(file_name, "w")
        except:
            raise ValueError(f"Error: {file_name} could not be opened.")
        
        _, extension = os.path.splitext(file_name)
 
        if extension in ['.csv','.gz']:
            # get edgelist of self.A
            edgelist = self.get_edgelist(edge_attribute = "layer")
        
            # write to csv file
            if file_name.endswith(".csv"):
                edgelist.to_csv(file_name, index=False)
            else:
                edgelist.to_csv(file_name, index=False, compression = 'gzip')
        elif extension == ".npz":
            save_npz(file_name, self.A)
        else:
            raise ValueError(f"Error: {extension} is not a valid file extension for edge saving.")
             
    def export_nodes(self, file_name):
        """
        Write self.nodes to file to .csv or .csv.gz file

        Parameters:
            -----------
            file_name : str
                file to write graph to. Extension should contain the type of output
                [".csv", ".csv.gz"]
        """
        try:
            f = open(file_name, "w")
            f.close()
        except:
            raise ValueError(f"Error: {file_name} could not be opened.")
        
        _, extension = os.path.splitext(file_name)
        if extension == ".csv":
            if self.use_polars:
                self.nodes.write_csv(file_name, include_header = True)
            else:
                self.nodes.to_csv(file_name)
        elif extension == ".gz":
            if self.use_polars:
                with gzip.open(file_name,"wb") as f:
                    self.nodes.write_csv(f, include_header = True)
            else:
                self.nodes.to_csv(file_name, compression="gzip")
        else:
            raise ValueError(f"Error: {extension} is not a valid file extension for node saving.")
            
    def save(self, path = "", overwrite = False, **kwargs):
        """
        This function saves the MultiLayerNetwork instance to a given path.
        It can be read from this path later with the library mode.

        Parameters:
            -------------
            path : string, default ""
                path to save the network to. If empty, saves to current directory
            overwrite : boolean, default False
                if True, overwrites existing files
            node_file : string, default ""
                name of file to save node attributes to
            edge_file : string, default ""
                name of file to save edge list to
            layer_file : string, default ""
                name of file to save layer information to
            -------------
        """
        # check if path is empty, give warning if yes
        if path == "":
            print("Warning: no save path given, files will be saved in current directory.")
            path += "./"
        elif path[-1] != "/":
            path += "/"
        
        # check if save_path exists, if not, create it
        if not os.path.isdir(path):
            os.mkdir(path)
        else:
            if not overwrite:
                # warning about overwriting
                print(f'The folder "{path}" already exists, call function with `overwrite = True` keyword argument if you\'re sure!')
                return
        if "node_file" not in kwargs:
            node_file = path + "nodes.csv.gz"
        else:
            node_file = kwargs["node_file"]
        if "edge_file" not in kwargs:
            edge_file = path + "edges.npz"
        else:
            edge_file = kwargs["edge_file"]
        if "layer_file" not in kwargs:
            layer_file = path + "layers.csv"
        else:
            layer_file = kwargs["layer_file"]
        if "codebook_file" not in kwargs:
            codebook_file = path + "codebook.csv"
        
        self.export_edges(edge_file)
        self.export_nodes(node_file)
        self.export_layers(layer_file)
        self.export_codebook(codebook_file)

    def export_layers(self, file_name):
        """
        This function exports the layer information to a csv file.

        Parameters:
            -------------
            file_name : string
                file to write layer dataframe to. Extension should be .csv
            -------------
        """
        self.layers.to_csv(file_name, index=False, header=True)

    def export_codebook(self, file_name):
        """
        This function exports the codebook to a csv file.

        Parameters:
            -------------
            file_name : string
                file to write codebook to. Extension should be .csv
            -------------
        """
        if self.codebook is not None:
            self.codebook.to_csv(file_name, index=False, header=True)
        
    def get_egonetwork(self, ego_label, depth=1, return_list=False, ignore_limit=False):
        """
        Starting out from a given node, this function returns the network at a
        given depth around that selected node.
 
        Parameters:
        -----------
 
            ego_label : int
                label of selected node
            depth : int, default 1
                depth in which to return network
            return_list : boolean, default False
                If False, returns an mln object representing the egonetwork
                Otherwise, a list of nodes in the egonetwork is returned (faster)
            ignore_limit : boolean, default False
                False if depth can be at most ego_depth_limit
                Set to True to ignore this limit
 
        Returns
        -------
            MultiLayerNetwork instance from egonetwork or list of nodes in egonetwork
        """
        
        if not ignore_limit and depth > ego_depth_limit:
            raise ValueError(f"Depth {depth} is too large for ego network." +\
                             f"Please use a maximum depth of {ego_depth_limit}" +\
                             f"To ignore, use ignore_limit=True")
        
        ego = self.to_id(ego_label)
 
        # separating the first round allows to skip the expensive for loop,
        # leading to a substantial speedup
        selected = [ego] + self.A[ego].indices.tolist()

        # iteratively search at further steps
        for k in range(1, depth):
            next_neighbors = self.A[selected].indices
            # there might be circles when looking for new neighbors, so we take unique integer ids
            selected = np.unique(np.concatenate((selected, next_neighbors)))
 
        selected = list(map(lambda x: self._map_id_to_label[x], selected))
 
        # return mln or list of nodes in egonetwork
        if return_list:
            return selected
        else:
            return self.get_filtered_network(nodes_selected = selected)
    
    def create_affiliation_matrix(self, key, affil_edgelist):
        """
        This method takes a node label -> affiliation bipartite edgelist, and
        creates scipy sparse representation for it, storing the mapping of
        affiliation ids in a dict.
 
        Parameters:
        -----------
            key : str
                the name of the bipartite structure for storage
            affil_edgelist : list[list]
                list of 2-tuples containing bipartite edgelist
                first node in the edges should correspond to a label in the mln instance
                second node can be anything, e.g. work IDs
 
        Returns
        -------
            None
 
        Creates a dictionary entry with key `key` into  mln.affiliation_matrix
        attribute with the following structure:
 
        mln.affiliation_matrix[key] = {
            'A': bipartite sparse adjacency matrix, mln.N times M
            'M' : if M is the number of unique elements in the second class
            'column_map_label_to_nid' : mapping IDs in second class to matrix column integer indices
            'column_map_nid_to_label' : mapping column integer indices back to second class IDs
        }
        """
        if not hasattr(self,"affiliation_matrix"):
            self.affiliation_matrix = {}
 
        self.affiliation_matrix[key] = {}
 
        unique_affiliations = sorted(list(set([v for k,v in affil_edgelist])))
 
        M = len(unique_affiliations)
 
        self.affiliation_matrix[key]['column_id_to_label'] = {i:elem for i,elem in enumerate(unique_affiliations)}
        self.affiliation_matrix[key]['column_label_to_id'] = {elem:i for i,elem in enumerate(unique_affiliations)}
 
        self.affiliation_matrix[key]['M'] = M
 
        # col number
        i = []
        j = []
        for k,v in affil_edgelist:
                i.append(self.to_id(k))
                j.append(self.affiliation_matrix[key]['column_label_to_id'][v])
 
        # sparse adjacency matrix for affiliations (work, school, region etc.)
        A = csr_matrix((np.ones(len(i)),(i,j)), shape=(self.N,M), dtype = 'int')
 
        self.affiliation_matrix[key]['A'] = A
 
    def get_binary_adjacency(self, dtype='int64'):
        """
        Downcast all values in self.A to a binary value. Only 1s occur in the
        resulting matrix.

        Parameters:
            -------------
            dtype : string, default "int64"
                datatype to downcast to
            -------------
        
        Returns:
            -------------
                A : sparsegraph A which is a downcasted version of self.A

        """
        if dtype != 'int64':
            return self.A.sign().astype(dtype)
        else:
            return self.A.sign()
    
    def to_id(self, labels):
        """
        This function converts a label, or list of labels to their corresponding
        ids
 
        Parameters:
            -----------
            labels : int or list, no default
                A label or list of labels to convert to NIDs
        Returns:
            -----------
            ids : int or list
                ids corresponding to node ids of given labels
         """
 
        if type(labels) != list:
            return self._map_label_to_id[labels]
        else:
            return [self._map_label_to_id[elem] for elem in labels]
 
        
    def to_label(self, ids):
        """
        This function converts an id, or list of ids to their corresponding
        labels
 
        Parameters:
            -----------
            nids : int or list, no default
                An id or list of ids to convert to labels
        Returns:
            -----------
            labels : int or list
                labels corresponding to user ids of given id(s)
 
        """
 
        if type(ids) != list:
            return self._map_id_to_label[ids]
        else:
            return [self._map_id_to_label[elem] for elem in ids]

    def get_degrees(self, selected_nodes=[]):
        """
        Calculate degree for selected nodes.

        If you also want to select layers, we suggest using the get_filtered_network method first.
        """
        if len(selected_nodes)==0:
            selected_nodes = list(self.nodes["label"])
        
        selected_nodes = [self.to_id(n) for n in selected_nodes]

        return dict(zip([self.to_label(n) for n in selected_nodes], self.A[selected_nodes, :].sign().sum(axis=0).tolist()[0]))

    def get_clustering_coefficient(self, selected_nodes=[],batchsize=100000):
        """
        Calculate clustering coefficient for selected nodes with selected layers.

        Parameters:
        -----------
            selected_nodes : list, default None
                which nodes to compute cc for, if None, use all
            batchsize : int, default 100000
                chunks in which to split up matrix multiplication bc of memory
                issues

        Returns:
        --------
            dict of label -> cc values
            
        """
        if len(selected_nodes)==0:
            selected_nodes = list(self.nodes["label"])
        
        selected_nodes = [self.to_id(n) for n in selected_nodes]

        if batchsize > len(selected_nodes):
            batchsize = len(selected_nodes)

        A = self.get_binary_adjacency()

        # find all triangles
        triangles = np.empty(0, dtype=np.int64)
        # B = A @ A @ A contains the number of paths of length 3 between B[i,j]
        # so B.diagonal() contains the number of triangles:
        #     paths of length 3 between a node and itself
        # // 2 as every path is found in both directions
        # B will not fit in memory, so we do this in steps
        for i in range(0, len(selected_nodes), batchsize):
            A_ = A[selected_nodes[i:i+batchsize],:]
            res = (A_ @ A @ A_.T).diagonal() // 2
            triangles = np.concatenate((triangles, res))

        # print("Triangles",dict(zip(self.nodes["label"],triangles)))

        A = A[selected_nodes,:]

        # compute the denominator
        # sum of neighbor degrees, over 2
        l = comb(A.sum(axis=1), 2)
        # sum of: neighbor degrees over two
        r = csr_matrix((comb(A.data, 2), A.indices, A.indptr),(len(selected_nodes),self.N)).sum(axis=1)
        P = np.array(l - r).T[0]

        # we ensure that division by zero errors are correctly handled and nan/inf
        # values are avoided

        # triangles / P
        clustering_coefficient = np.divide(triangles, P, out=np.zeros(len(selected_nodes)), where=P!=0)

        return dict(zip([self.to_label(n) for n in selected_nodes],clustering_coefficient))

    def get_supra_adjacency_matrix(self,dtype='int64'):
        """
        Decompressing the binary format into a supra-adjacency matrix.

        Parameters: 
        ------------
            dtype : string, default "int64"
                datatype to downcast to

        Returns
        -------
            0/1 supra-adjacency matrix of shape (self.L*self.N,self.L*self.N)
        """
        # layer couplings
        self.sA = kron(np.ones((self.L,self.L)),eye(self.N)) - eye(self.L * self.N)
        # diagonal matrices
        d = []
        for l in self.layers["layer"]:
            if l not in self.layer_adjacency_matrix:
                d.append(self.get_layer_adjacency_matrix(layer=l))
            else:
                d.append(self.layer_adjacency_matrix[l])
        self.sA += block_diag(d)

        if dtype!='int64':
            self.sA = self.sA.astype(dtype)

        return self.sA
    
    def get_grouped_mln(self):
        """
        Returns an MLN object with the same nodelist, 
        but modified edge adjacency matrix and layer
        dataframe according to the 'group' column in
        self.layers. Combines edges whose layertype belongs to
        the group to into one single layer.
        """
        self.gA = csr_matrix((self.N,self.N),dtype=np.uint8)
        groups = self.layers["group"].unique()

        self.g_layers = pd.DataFrame(groups,columns=["label"])
        self.g_layers["layer"] = self.g_layers.index+1
        self.g_layers["binary"] = self.g_layers.index.map(lambda i: int(2**i))

        print(self.g_layers)

        for g in groups:
            b = self.g_layers.set_index("label")["binary"].loc[g]
            print(f"Creating adjacency matrix for group layer {g}...")
            if g not in self.group_adjacency_matrix:
                glA = self.get_layer_adjacency_matrix(g,layer_type="group",dtype=np.uint8)
            else:
                glA = csr_matrix(self.group_adjacency_matrix[g],shape=(self.N,self.N),dtype=np.uint8)
            print("Done.")
            print(f"Adding {glA.nnz} edges to layer {g}...")
            self.gA+=glA*b
            print("Done.")


        return MultiLayerNetwork(
            nodes = self.nodes,
            edges = self.gA,
            layers = self.g_layers
        )
    
    def get_excess_closure(self, selected_nodes=[], node_type="label", selected_layers=[], layer_type="layer", batchsize=100000):
        """
        Calculate the measure called excess closure for selected nodes and layers.

        See Bokany et al. 2023 for more details.
        """

        # ===================== input handling =========================

        # handling layer input
        self.verboseprint("Handling layer input...")
        if layer_type not in ["label", "layer", "binary", "group"]:
            raise ValueError(f"Invalid layer_type '{layer_type}'. Please choose from 'label' or 'layer' or 'binary'.")

        for layer in selected_layers:
            if layer not in self.layers[layer_type].tolist():
                raise ValueError(f"Invalid layer '{layer}' for layer_type '{layer_type}'. Please choose from {self.layers[layer_type].unique().tolist()}.")

        if len(selected_layers) == 0:
            selected_layers = self.layers[layer_type].unique().tolist()

        # get corresonding binary representation
        binary_repr = sum([self._layer_conversion_dict[f"{layer_type}_to_binary"][layer] for layer in selected_layers])

        # handling node input
        if len(selected_nodes) == 0:
            selected_nodes = self.nodes["id"].tolist()
        else:
            if node_type == "label":
                selected_nodes = [self.to_id(n) for n in selected_nodes]
            elif node_type == "id":
                pass
            else:
                raise ValueError(f"Invalid node_type '{node_type}'. Please choose from 'label' or 'id'.")
        self.verboseprint("Done.")

        # ====================== actual calculation ========================
        # obtain a copy of the sparse adjacency matrix such that each element
        # A[i,j] contains the number of links between i and j
        self.verboseprint("Getting multiplexity counts of full matrix...")
        A = deepcopy(self.A)
        A.data = A.data & binary_repr
        # this is necessary, otherwise the denominators will not be correct
        # (we're using indptr and indices to calculate neighbor degrees)
        A.eliminate_zeros()
        

        # create lookup table for mapping binary values to number of layers / multiplexity
        lookup = {}
        for num in np.unique(A.data):
            lookup[num] = bin(num).count("1")
        # apply lookup table
        A.data = np.array([lookup[x] for x in A.data])
        self.verboseprint("Done.")

        self.verboseprint("Total number of triangles...")
        # find all triangles
        triangles = np.empty(0, dtype=np.int64)
        # B = A @ A @ A contains the number of paths of length 3 between B[i,j]
        # so B.diagonal() contains the number of triangles:
        #     paths of length 3 between a node and itself
        # // 2 as every path is found in both directions
        # B will not fit in memory, so we do this in steps
        for i in range(0, len(selected_nodes), batchsize):
            self.verboseprint(f"\tBatch {i}/{len(selected_nodes)//batchsize+1}")
            start = i
            end = i + batchsize
            # if end is larger than matrix size, let it be the matrix size
            if end > len(selected_nodes):
                end = len(selected_nodes)
            A_ = A[selected_nodes[start:end], :]
            res = (A_ @ A @ A_.T).diagonal() // 2
            triangles = np.concatenate((triangles, res))
        self.verboseprint("Done.")

        self.verboseprint("Pure triangle count...")
        # get number of triangles which only use 1 layer
        pure_triangles = np.zeros(len(selected_nodes))
        for layer in selected_layers:
            self.verboseprint(f"\tLayer {layer}...")
            t = np.empty(0, dtype=np.int64)
            lA = self.get_layer_adjacency_matrix(layer=layer, layer_type=layer_type)
            self.verboseprint("\tStart calc...")
            for i in range(0, len(selected_nodes), batchsize):
                start = i
                end = i + batchsize
                # if end is larger than matrix size, let it be the matrix size
                if end > len(selected_nodes):
                    end = len(selected_nodes)
                A_ = lA[selected_nodes[start:end], :]
                res = (A_ @ lA @ A_.T).diagonal() // 2
                t = np.concatenate((t, np.array(res)))
            self.verboseprint("\tEnd calc...")
            pure_triangles += t
        self.verboseprint("Done.")

        self.verboseprint("Calculating denominator...")
        # decrease matrix size to only selected nodes
        A = A[selected_nodes, :]

        # compute the denominator
        # sum of neighbor degrees, over 2
        l = comb(A.sum(axis=1), 2)
        # sum of: neighbor degrees over 2
        r = csr_matrix((comb(A.data, 2), A.indices, A.indptr), (len(selected_nodes), self.N)).sum(axis=1)
        P = np.array(l - r).T[0]
        # we ensure that division by zero erros are correctly handled and nan/inf
        # values are avoided
        self.verboseprint("Done.")

        # pure triangles / P
        Cpure = np.divide(pure_triangles, P, out=np.zeros(len(selected_nodes)), where=P!= 0)
        # triangles / P
        Cunique = np.divide(triangles, P, out=np.zeros(len(selected_nodes)), where=P!= 0)

        # (Cunique - Cpure / 1 - Cpure)
        excess_closure = np.divide(Cunique-Cpure, 1-Cpure, out=np.zeros(len(selected_nodes)), where=(1-Cpure)!= 0)
        clustering_coefficient = Cunique

        return {
            "clustering_coefficient": dict(zip([self.to_label(n) for n in selected_nodes], clustering_coefficient)),
            "excess_closure": dict(zip([self.to_label(n) for n in selected_nodes], excess_closure))
        }

    def add_nodes(self,nodes_df):
        if "label" not in nodes_df:
            raise ValueError("The dataframe should contains at least a column called label.")
        M = nodes_df.shape[0]
        nodes_df["id"] = range(self.N,self.N+M)
        self.N = self.N + M
        self._map_id_to_label.update(dict(zip(nodes_df["id"],nodes_df["label"])))
        self._map_label_to_id.update(dict(zip(nodes_df["label"],nodes_df["id"])))
        self.nodes = pd.concat([self.nodes,nodes_df],axis=0)
        self.nodes.index = self.nodes["id"]
        self.A = block_diag([self.A,csr_matrix((M,M))],format="csr")

    def remove_nodes(self,nodes):
        if any(self.nodes.index!=self.nodes.id):
            raise ValueError("Node dataframe not aligned correcly for operation.")
        mask = ~self.nodes["label"].isin(nodes)
        self.nodes = self.nodes[mask].copy()
        self.nodes.reset_index(drop=True,inplace=True)
        self.nodes["id"] = self.nodes.index
        self._map_id_to_label = dict(zip(self.nodes["id"],self.nodes["label"]))
        self._map_label_to_id = dict(zip(self.nodes["label"],self.nodes["id"]))
        self.N = mask.sum()
        self.A = self.A[mask,:][:,mask]

    def get_aggregated_network(self, aggregation_column=None, keep_layers = False, convert_A = "none"):
            """
            Return an aggregated network over a certain column in self.nodes.

            It looks at unique values in the column, and aggregates the edges for
            each unique value. The resulting network has as many nodes as there
            are unique values in the column. The edges are aggregated by counting all
            edges from all layers that go between the groups.

            The resulting network is a MultiLayerNetwork object, but instead of the binary
            adjacency matrix, it has a weighted adjacency matrix, where the weights are
            the edge counts between the groups.
            
            Parameters:
                -------------
                aggregation_column : string, default None
                    the column in self.nodes based on which the edges should be aggregated
                -------------
            """

            if not aggregation_column in self.nodes.columns:
                raise ValueError(f"Column {aggregation_column} not found in self.nodes. Possible candidates are: ", self.nodes.columns)
            
            
            if self.use_polars:
                mask = self.nodes["active"] & self.nodes[aggregation_column].is_not_null()
                affil_edgelist = list(zip(self.nodes["label"].filter(mask), self.nodes[aggregation_column].filter(mask)))
            else:
                mask = self.nodes["active"] & (~pd.isnull(self.nodes[aggregation_column]))
                affil_edgelist = list(zip(self.nodes["label"][mask], self.nodes[aggregation_column][mask]))

            self.create_affiliation_matrix(aggregation_column,affil_edgelist)

            if convert_A == "boolean":
                G = self.affiliation_matrix[aggregation_column]["A"].T * self.A.sign() * self.affiliation_matrix[aggregation_column]["A"]
            elif convert_A == "multiplexity":
                A = deepcopy(self.A)
                # create lookup table for mapping binary values to number of layers / multiplexity
                lookup = {}
                for num in np.unique(A.data):
                    lookup[num] = bin(num).count("1")
                # apply lookup table
                A.data = np.array([lookup[x] for x in A.data])
                G = self.affiliation_matrix[aggregation_column]["A"].T * A * self.affiliation_matrix[aggregation_column]["A"]
            elif convert_A == "none":
                G = self.affiliation_matrix[aggregation_column]["A"].T * self.A * self.affiliation_matrix[aggregation_column]["A"]
            else:
                ValueError("Invalid value for convert_A, please choose from none, booelan, or multiplexity.")
            uniques = [self.affiliation_matrix[aggregation_column]["column_id_to_label"][i] for i in range(self.affiliation_matrix[aggregation_column]["M"])]
            if 'weight' not in self.nodes.columns:
                # count occurrences in each aggregation group
                grp_counts = self.affiliation_matrix[aggregation_column]["A"].sum(axis=0).tolist()[0]
            else:
                # sum up previous weights
                grp_counts = self.nodes[mask].groupby(aggregation_column)['weight'].sum().loc[uniques]

            if self.use_polars:
                nodes_selected = pl.DataFrame({
                        'label' : uniques, 
                        'weight' : grp_counts, 
                        "id" : [i for i in range(self.affiliation_matrix[aggregation_column]["M"])]
                    }
                )
            else:
                nodes_selected = pd.DataFrame(data={
                        'label' : uniques, 
                        'weight' : grp_counts, 
                        "id" : [i for i in range(self.affiliation_matrix[aggregation_column]["M"])]
                    }
                )

            # add data related to particular aggregation level
            if self.use_polars:
                aggregation_nodes = self.nodes.select([c for c in self.nodes.columns if aggregation_column.split('_')[0] in c]).groupby([aggregation_column]).first()
                nodes_selected = nodes_selected.join(aggregation_nodes, left_on=['label'], right_on=[aggregation_column], how='left')
            else:
                aggregation_nodes = self.nodes.groupby([aggregation_column]).head(1)[[c for c in self.nodes.columns if aggregation_column.split('_')[0] in c]]
                nodes_selected = nodes_selected.merge(aggregation_nodes, left_on=['label'], right_on=[aggregation_column], how='left')
            
            f = MultiLayerNetwork(
                edges = G.tocsr(),
                nodes = nodes_selected,
                **self._to_pass
            )

            # this is only providing raw counts for all selected layers
            # TODO: it would be more elegant to keep layers!
            selection_layers = pd.DataFrame(data={'layer' : 1, 'label' : 'count', 'label_long' : 'count_'+convert_A, 'layer' : 'count', 'binary' : 1}, index=[0])
            selection_layers_dict = {
                'binary_to_layer': {1 : 1},
                'binary_to_layer' : {1 : 'count'},
                'layer_to_layer' : {1 : 'count'},
                'layer_to_binary' : {1 : 1},
                'layer_to_binary' : {'count' : 1},
                'layer_to_layer' : {'count' : 1}
            }

            f.layers = selection_layers
            f._layer_conversion_dict = selection_layers_dict
            # The nodes have changed, therefore new mappings for to_nid and to_label are passed.
            f._map_label_to_id = dict(zip(f.nodes["label"],f.nodes["id"]))
            f._map_id_to_label = dict(zip(f.nodes["id"],f.nodes["label"]))
            f.adjacency_element = "weight"

            return f









                    