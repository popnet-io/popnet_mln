# change working directory to repo root
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)) + '/..')
 
# importing libraries
import pandas as pd
from pandas.io.parsers import read_csv
import matplotlib
from matplotlib import cm
import numpy as np
import igraph as ig
import networkx as nx
from src.preparation import RawCSVtoMLN

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
        **kwargs
    ):
        """
        This class contains methods and attributes to work with a large
        multilayer network using different edge types and layers efficiently.
 
        The network is either loaded from an already saved sparse matrix and a
        node attribute CSV using the following attributes:
            * adjacency_file:  scipy.sparse.csr matrix saved as an `npz` file
            * node_attribute_file: CSV to be loaded as a pd.DataFrame, rows in
              order of previous matrix rows should contain a column called
              "label" that is the primary key for node identification
        
        Or in-memory objects can also be given to the constructor:
            * adjacency_matrix: scipy.sparse.csr matrix
            * node_attribute_dataframe: pd.DataFrame, rows in order of previous
              matrix rows
 
        Pre-stored files with standard names can be called with the shorthand:
            * from_library: str
        Then the npz and csv.gz files will be read from the
        library_path/from_library folder similarly to the first method.
 
        After loading these two key elements, class attributes and methods work
        the same.
 
        The adjacency matrix self.A is stored in a `scipy.sparse.csr_matrix`
        class, that only saves nonzero elements, and on which scipy csgraph
        algorithms run. People are indexed from 0 to N-1, where N is the total
        number of nodes in this network.
 
        Two dictionaries mapping user id (label) to integer node ids (NID) and
        back are created based on the node attribute file in the
        `self.map_label_to_nid` and `self.map_nid_to_label` attributes. These
        mappings only refer to one instance of the class - if you create a
        subgraph (see later) the coding is going to change, and the mapping
        should be found in the new instance that represents the subgraph.
        
        The adjacency matrix contains integers that encode linktypes if viewed
        as binary numbers. Each possible linktype is assigned an integer of the
        form 2**i. For example, if both type i and type j edge is present
        between two people, then the corresponding value in self.A would be
        2**i+2**j. It means that we can test for a certain edgetype using
        bitwise AND operation very cheaply. E.g. a certain element of self.A is
        7, then 7=1+2+4 which means that edgetypes 0,1, and 2 are present
        between the two people, and 7&2 = 2 in Python (it behaves like a mask
        111 & 010 = 010).
        
        scipy.csr matrices are cheap to slice rowwise, but beware, some
        operation that seem straightforward in numpy might be costly (e.g.
        getting random coordinates after each other or colwise slicing)! If
        something is running too long, consult the scipy reference manual.
 
        Node attributes are stored in `self.node_attributes` which is a
        `pandas.DataFrame`. It is possible to store human-readable or longer
        column names in `self.attribute_colnames`, and code tables for used
        variables in  `self.attribute_code_table`, keys of that variable are the
        colnames in the values of `self.attribute_colnames`.
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
        
        # loading from library
        if load_from_library:
            self.nodes, self.A, self.layers = self.load(library_path) 
        # loading from in-memory objects
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
        else:
            self.nodes, self.A, self.layers = nodes, edges, layers

        self.map_id_to_label = dict(zip(self.nodes["id"],self.nodes["label"]))
        self.map_label_to_id = dict(zip(self.nodes["label"],self.nodes["id"]))
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

        self.verbose = verbose
        self.max_bin_linktype = self.layers["binary"].max()
 
        # passing some variables to children instances that don't change
        # shortens subsequent calls
        self._to_pass = {
            "layers" : self.layers,
            "codebook" : self.codebook,
            "verbose" : self.verbose,
            "layers_dict" : self.layer_conversion_dict,
            "max_bin_linktype" : self.max_bin_linktype
        }
 
    def load(self, path):
        """
        If library mode is used, this function checks if the path is valid and the correct
        files are stored in it. When this is the case, it returns the full path to the 
        adjacency_file and node_attribute_file.
        
        Parameters:
        -------------
            save_path : string
                
            load_saved :    bool
                contains which folder from library to access: save_path/load_saved/

        Returns:
        --------
            edges : string or None
                resulting path to an edgelist / adjacency matrix
            nodes : string or None
                resulting path to a nodelist with attributes
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
            nodes = pd.read_csv(nodes_path, index_col=None, header=0)
        # load edges from npz file
        edges = load_npz(edges_path)

        # load layers from csv file
        layers = pd.read_csv(layers_path, index_col=None, header=0)

        return nodes, edges, layers
                
    def init_layer_dict(self):
        """
        Initialize dictionary for efficient conversion from link, label, code and 
        binary label to each other. Resulting dictionary is stored in self.layers_dict
        
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
        
        self.layer_conversion_dict = {
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
            layer_type="label"
            ):
        """
        Returns MultiLayerNetwork based on edge and node filtering.
        
        Possibilities:
 
        1. Edge filtering: defining a list of layers through the kwarg `layers`.
           E.g. `mln.get_edgelist(layers = ["parent", "household"])`
 
        2. Node filtering: given a list of labels, select subgraph spanned by
           those nodes, consisting only of selected edgetypes.
 
        Parameters:
            -------------
            layers : list of int or str, default []
                list of layers to include in returned object
                see list of possible layers in self.layers

            selected_nodes : list of int, default None
                if None, return all nodes in the parent object
                if list, select nodes and their spanned subgraph with given
                labels into returned object
 
        Returns:
            -------------
            MultiLayeredNetwork
                filtered network with less data but the very same structure as
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
            # slicing the adjacency matrix
            A_selected = self.A[idx,:][:,idx]
            # slicing the attribute table
            nodes_selected = self.nodes.iloc[nodes_selected].reset_index(drop=True)
        else:
            A_selected = deepcopy(self.A)
            nodes_selected = self.nodes
 
        if len(layers_selected)>0:
            for l in layers_selected:
                if l not in self.layers[layer_type].tolist():
                    raise ValueError(f"Invalid layer '{l}' for layer_type '{layer_type}'. Please choose from {self.layers[layer_type].tolist()}.")
            # adding up the binary codes for the full layers from the argumentif len(layers)>0:
            # based on type of layer value:
            binary_repr = sum([self.layer_conversion_dict[layer_type + "_to_binary"][layer] for layer in layers_selected])
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
            binary_repr = sum([self.layer_conversion_dict["layer_to_binary"][layer] for layer in layers])
        else:
            # get binary layer value if it's not already given in binary
            if layer_type != "binary":
                binary_repr = self.layer_conversion_dict[layer_type + "_to_binary"][layer]
            else:
                binary_repr = layer

        # get layer value if it's not already given in layer
        if layer_type != "layer" and layer_type != "group":
            l = self.layer_conversion_dict[layer_type + "_to_layer"][layer]
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


    def get_aggregated_network(self, aggregation_column=None, keep_layers = False):
        """
        Return an aggregated network over a certain column in self.nodes.
        
        Parameters:
            -------------
            aggregation_column : string, default None
                the column in self.nodes based on which the edges should be aggregated
            -------------
        """
        A_selected = deepcopy(self.A)

        if not aggregation_column in self.nodes.columns:
            raise ValueError(f"Column {aggregation_column} not found in self.nodes. Possible candidates are: ", self.nodes.columns)
        
        grps, uniques = pd.factorize(self.nodes[aggregation_column])

        # count how much each group occurs
        def count_grps(lst):
            count_dict = {}
            for num in lst:
                if num in count_dict:
                    count_dict[num] += 1
                else:
                    count_dict[num] = 1
            return count_dict
        
        grp_counts = count_grps(grps)

        i, j = list(range(len(grps))), grps
        # sparse adjacency matrix for affiliations (work, school, region etc.)
        ft = csr_matrix((np.ones(len(i)),(i,j)), shape=(self.N,len(grp_counts.keys())), dtype = 'int').T

       
        grouped_A = ft.dot(A_selected.sign())
        grouped_A_2 = grouped_A.dot(ft.T)
        grouped_A_2 = csr_matrix(grouped_A_2)

        nodes_selected = pd.DataFrame(data={
                'label' : uniques, 
                'weight' : [grp_counts[i] for i in range(0,len(uniques))]}, 
                index=[i for i in range(0,len(uniques))]
        )
        nodes_selected.reset_index(inplace=True)
        nodes_selected.rename(columns={'index' : 'id'}, inplace=True)

        # add data related to particular aggregation level
        aggregation_nodes = self.nodes.groupby([aggregation_column]).head(1)[[c for c in self.nodes.columns if aggregation_column.split('_')[0] in c]]
        nodes_selected = nodes_selected.merge(aggregation_nodes, left_on=['label'], right_on=[aggregation_column], how='left')
        
        f = MultiLayerNetwork(
            edges = grouped_A_2,
            nodes = nodes_selected,
            **self._to_pass
        )

        # this is only providing raw counts for all selected layers
        # TODO: it would be more elegant to keep layers!
        selection_layers = pd.DataFrame(data={'layer' : 1, 'edge_label' : 'count', 'edge_label_long' : 'count', 'layer' : 'count', 'binary' : 1}, index=[0])
        selection_layers_dict = {
            'binary_to_layer': {1 : 1},
            'binary_to_layer' : {1 : 'count'},
            'layer_to_layer' : {1 : 'count'},
            'layer_to_binary' : {1 : 1},
            'layer_to_binary' : {'count' : 1},
            'layer_to_layer' : {'count' : 1}
        }

        f.layers = selection_layers
        f.layer_conversion_dict = selection_layers_dict
        # The nodes have changed, therefore new mappings for to_nid and to_label are passed.
        f.map_label_to_id = dict(zip(f.nodes["label"],f.nodes["id"]))
        f.map_id_to_label = dict(zip(f.nodes["id"],f.nodes["label"]))

        return f
    
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

        If without_linktypes is True, then the edges correspond to the nonzero
        values of self.A. If it is False, binary weights in self.A are decoded.
        
        If one does not need the linktypes, then the resulting columns are:
        "source", "target"

        Otherwise, the columns are: "source", "target", "binary",
        "linktype"
 
        Returns
           -------
            edgelist : pandas dataframe containig the edge list representing self.A
        """
        # self.report_time(init=True)

        # getting edges and binary linktypes
        edges = np.array(self.A.nonzero()).T
        weights = np.array([self.A.data]).T
        # self.report_time(message = "Getting data from sparse matrix.")
        if edge_attribute == "binary" or edge_attribute=="weight":
            # combine edges and weights
            edges_with_weights = np.concatenate((edges, weights), axis=1)
            edgelist = pd.DataFrame(edges_with_weights, columns = ["source", "target", edge_attribute])

            # mapping back edges to the original labels
            edgelist["source"] = edgelist["source"].map(self.to_label)
            edgelist["target"] = edgelist["target"].map(self.to_label)

        elif edge_attribute == "layer" or edge_attribute == "label":
            edges_with_weights = np.concatenate((edges, weights), axis=1)
            edgelist = pd.DataFrame(edges_with_weights, columns = ["source", "target", "binary"])
            # self.report_time(message = "Creating edgelist dataframe.")
    
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
            edgelist[edge_attribute] = edgelist["binary"].map(link_dict)
            # self.report_time(message = "Mapped binary link identifiers.")
            # print(edgelist.head())

            edgelist.drop("binary", axis=1, inplace=True)
            edgelist = edgelist.explode(edge_attribute)
            # self.report_time(message = "Exploded binary link identifiers.")
            # print(edgelist.head())
            # self.report_time(message = "Adding that to dataframe?")
            # print(edgelist.head())
        else:
            raise ValueError(f"Invalid edge_attribute '{edge_attribute}'. Please choose from 'binary', 'layer', 'label' or 'weight'.")
        
        return edgelist
    
    def to_igraph(self, directed=True, edge_attributes=True, node_attributes=False, replace_igraph=False):
        """
        This function returns an igraph object of the sparsematrix stored in
        self.A. Edge attributes (link types) and node attributes (from
        self.node_attributes) can be added to this object. If self.igraph has
        not yet been initialized, this is set to the 
 
        Parameters:
            -----------
            directed : boolean, default True
                mode of returned igraph object: True for directed, False for
                undirected graph
            edge_attributes : boolean, default True
                True if edge attributes from self.node_attributes should be
                added to igraph object.
            node_attributes : boolean, default False
                True if node attributes should be added to igraph object
                obtained from self.node_attributes Note: the "label" column is
                always added.
            replace_igraph : boolean, default False
                When the mln object already has an igraph object, it will by
                default not be replaced by the newly generated object. To
                replace the object, set replace_igraph to True
 
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
            # obtain and add (human readable) link types
            layer_dict = {}
            for layer in set(g.es["layer"]):
                layer_dict[layer] = self.convert_layer_binary_to_list(layer, output_type="layer")

            # rewrite link types to human readable
            g.es["layer"] = [layer_dict[x] for x in g.es["layer"]]
            
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
 
    def to_networkx(self, directed = True, edge_attributes = True, node_attributes = False, ignore_limit = False):
        """
        This function returns a networkx object of the sparsematrix stored in
        self.A. Edge attributes (link types) and node attributes (from
        self.node_attributes) can be added to this object.
 
        Parameters:
            -----------
            directed : boolean, default True
                mode of returned igraph object: True for directed, False for
                undirected graph
            edge_attributes : boolean, default True
                True if edge attributes from self.node_attributes should be
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
 
        # create igraph graph
        g = nx.from_scipy_sparse_matrix(self.A, create_using=graph_type)
 
        # add remaining attributes
        if node_attributes:
            for col_name in self.nodes:
                cur_col = list(self.nodes[col_name])
                attribute_list = dict(zip(list(range(len(cur_col))), cur_col))
                nx.set_node_attributes(g, attribute_list, col_name)
        else:
            # always add "label" column
            nx.set_node_attributes(g, self.map_id_to_label, "label")
        
        # obtain and add (human readable) link types use dict for optimization
        link_dict = {}
        
        for s, t, d in g.edges(data=True):
            if edge_attributes:
                weight = g[s][t]["weight"]
 
                if weight in link_dict:
                    g[s][t]["link_types"] = link_dict[weight]
                else:
                    link_types = self.convert_layer_binary_to_list(weight)
                    link_dict[weight] = link_types
                    g[s][t]["link_types"] = link_types
 
            # remove weights
            d.pop("weight", None)
 
        return g
    
    def convert_layer_representation(self, layer, input_type="layer", output_type="binary"):
        """
        This function converts a single linktype or list of linktypes to
        their corresponding linklabels. Input can be binary, code or label.
        All possible values for each are in self.layers
 
        Parameters:
            -----------
            layer : int, string or list, no default
                A single binary code or name of a linktype
            input_type : string
                Type of input. Options: "label", "layer" and "binary"
            output_type : string
                Type of input. Options: "label", "layer" and "binary"

        Returns:
            -----------
            links : string or list
                linktype (binary, layer or name) corresponding to the given types
                If one of the linktypes is not found, returns None
         """
        
        dict_name = input_type + '_to_' + output_type
        try:
            d = self.layer_conversion_dict[dict_name]
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
        Based on the integer linktype, returns a list with the layers.
 
        E.g.: self.decompose_binary_linktype(3) returns ["aunt/uncle","co-parent"]
 
        Parameters:
            --------------
            num: int
                integer number to be converted to binary and returned as linktypes
        Returns:
            ---------------
            list(type(output type of self.convert_layer_representation))
                list of layers corresponding to binary integer value
        """

        if np.log2(float(num)) > self.layers.index[-1]+1:
            raise ValueError(f"Layer binary value {num} is not a valid linktype in the network.")

        return [self.convert_layer_representation(2**i, input_type='binary', output_type=output_type)\
                 for i in range(self.layers.index[-1]) if int(num)&(2**i)>0]
    
    def save_to_graphml(self, file_name, directed = True, edge_attributes = True, node_attributes = False, overwrite = False):
        _, extension = os.path.splitext(file_name)

        if extension == ".graphml":
            if self.igraph is None or overwrite:
                self.igraph = self.to_igraph(
                    node_attributes=node_attributes,
                    edge_attributes=edge_attributes,
                    directed=directed
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
        - ".graphml" : From igraph object to graphml format
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
        Write self.node_attributes to file to .csv or .csv.gz file
        graphml
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
            self.nodes.to_csv(file_name)
        elif extension == ".gz":
            self.nodes.to_csv(file_name, compression="gzip")
        else:
            raise ValueError(f"Error: {extension} is not a valid file extension for node saving.")
            
    def save(self, path = "", overwrite = False, **kwargs):
        """
        This function saves the MultiLayeredNetwork instance to a given path.

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
        
        self.export_edges(edge_file)
        self.export_nodes(node_file)
        self.export_layers(layer_file)

    def export_layers(self, file_name):
        self.layers.to_csv(file_name, index=False, header=True)
        
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
                Otherwise, a list of nodes in the egonetwork is returned
            ignore_limit : boolean, default False
                False if depth can be at most ego_depth_limit
                Set to True to ignore this limit
 
        Returns
        -------
            MultiLayeredNetwork instance from egonetwork
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
            selected = pd.unique(np.concatenate((selected, next_neighbors)))
 
        selected = list(map(lambda x: self.map_id_to_label[x], selected))
 
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
            mln : MultiLayeredNetwork
                the mln instance we would like to add the bipartite structure to
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
 
        unique_affiliations = set([v for k,v in affil_edgelist])
 
        M = len(unique_affiliations)
 
        self.affiliation_matrix[key]['column_to_label'] = {i:elem for i,elem in enumerate(unique_affiliations)}
        self.affiliation_matrix[key]['column_to_id'] = {elem:i for i,elem in enumerate(set([v for k,v in affil_edgelist]))}
 
        self.affiliation_matrix[key]['M'] = M
 
        # col number
        i = []
        j = []
        for k,v in affil_edgelist:
                i.append(self.to_id(k))
                j.append(self.affiliation_matrix[key]['column_to_id'][v])
 
        # sparse adjacency matrix for affiliations (work, school, region etc.)
        A = csr_matrix((np.ones(len(i)),(i,j)), shape=(self.N,M), dtype = 'int')
 
        self.affiliation_matrix[key]['A'] = A
 
    def to_binary_adjacency(self):
        """
        Downcast all values in self.A to a binary value. Only 1s occur in the
        resulting matrix.
        
        Returns:
            -------------
                A : sparsegraph A which is a downcasted version of self.A
        """
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
            nids : int or list
                NIDs corresponding to node ids of given labels
         """
 
        if type(labels) != list:
            return self.map_label_to_id[labels]
        else:
            return [self.map_label_to_id[elem] for elem in labels]
 
        
    def to_label(self, ids):
        """
        This function converts an NID, or list of NIDs to their corresponding
        labels
 
        Parameters:
            -----------
            nids : int or list, no default
                An NID or list of NIDs to convert to labels
        Returns:
            -----------
            labels : int or list
                labels corresponding to user ids of given NIDs
 
        """
 
        if type(ids) != list:
            return self.map_id_to_label[ids]
        else:
            return [self.map_id_to_label[elem] for elem in ids]

    def get_degrees(self, selected_nodes=[]):
        """
        Calculate degree for selected nodes.

        If you also want to select layers, we suggest using the get_filtered_network method first.
        """
        if len(selected_nodes)==0:
            selected_nodes = self.nodes["label"].tolist()
        
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
            dict with first key "clustering_coefficient", then dict of label -> cc values
            
        """
        if len(selected_nodes)==0:
            selected_nodes = self.nodes["label"].tolist()
        
        selected_nodes = [self.to_id(n) for n in selected_nodes]

        if batchsize > len(selected_nodes):
            batchsize = len(selected_nodes)

        A = self.to_binary_adjacency()

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

    def get_supra_adj_matrix(self):
        """
        Decompressing the binary format into a supra-adjacency matrix.

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
        return self.sA