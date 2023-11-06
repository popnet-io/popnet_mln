# importing libraries
import pandas as pd
from pandas.io.parsers import read_csv
import matplotlib
from matplotlib import cm
import numpy as np
import igraph as ig
import networkx as nx

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
    
class MultiLayeredNetwork:
    
    def __init__(
        self, 
        adjacency_file = "",
        node_attribute_file = "",
        layer_file = "",
        adjacency_matrix =  None,
        node_attribute_dataframe = None,
        node_attribute_metadata = None,
        from_library = "",
        library_path = "",
        colors_file = "",
        attribute_colnames = "",
        attribute_code_table = "",
        verbose=False
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
 
        Example loadings:
 
        ```
        # FROM FILES
        # import custom class for the network from mln
        import MultiLayeredNetwork

        # read the whole network
        popnet = MultiLayeredNetwork(
            adjacency_file = 'path_to/adjacency.npz',
            node_attribute_file = 'path_to/attributes.csv.gz'
        )

        # FROM MEMORY
        # import custom class for the network
        from mln import MultiLayeredNetwork

        # read the whole network
        popnet = MultiLayeredNetwork(
            adjacency_matrix = A, # NxN scipy.sparse.csr_matrix
            node_attribute_file = df # pd.DataFrame containing N rows in the order of the matrix
        )

        # FROM LIBRARY
        # import custom class for the network
        from mln import MultiLayeredNetwork

        # read the whole network
        popnet = MultiLayeredNetwork(
            from_library="full"
        )
 
        ```
 
        Parameters
            ----------
                adjacency_file : str, default ""
                    path of `npz` file containing `scipy.sparse.csr_matrix`
                    saved in binary format
                node_attribute_file : str, default ""
                    path of (gzipped) CSV file containing node atributes comma
                    separated, containing 'label' column
                adjacency_matrix : scipy.sparse.csr_matrix, default None
                    scipy.sparse.csr_matrix of size NxN encoding the adjacency
                    matrix
                node_attribute_dataframe : pandas.DataFrame, default None
                    pandas.DataFrame containing exactly N rows corresponding to
                    the matrix rowsm containing 'label' column other columns
                    having attributes of graph data
                layer_file : str, default None
                    path of csv file containing layers, should contain column called "layer"
                from_library : str, default ""
                    short name to get network read from `library_path`
                    subdirectory
                library_path : str, default ""
                    path where library entries are stored
                colors_file : str, default ""
                    JSON defining layer colors for visualization
                attribute_colnames : str, default ""
                    JSON containing translations for column names
                attribute_code_table : str, default ""
                    JSON containing translation of possible attribute values for
                    all columns
                verbose : bool, default False
                    if True, prints information while loading  / manipulating data
 
        Attributes
            ---------
                A : scipy.sparse.csr_matrix[int64]
                    adjacency matrix NxN, each element encodes the linktypes in
                    a binary fashion
                node_attributes : pd.DataFrame
                    table containing node attributes label column containing label,
                    "label" column containing integer / string label
                map_label_to_nid : dict
                    label (int or str) -> 0...N-1 integer NID (int) mapping
                map_nid_to_label : dict
                    0...N-1 integer NID (int) -> label (int or str) mapping
                N : int
                    number of nodes
                layers : pd.DataFrame
                    table containing info on layers/linktypes in the networks
                linktypes_dict : dict
                    dictionary containing dictionaries for efficient conversion from 
                    binary -> code, binary -> label
                    code -> binary, code -> label
                    label -> binary, label -> code
                colors : dict
                    layer (str) -> hex color (str) for plotting
                attribute_colnames : dict
                    node attribute column names (str) -> human-readable node
                    attribute column names (str) 
                attribute_code_table : dict
                    human-readable node attribute column names (str) -> dict of code
                    keys -> code values corresponding to different columns
                    useful for interpreting results e.g.
                    "gender" : {
                        "1": "Man",
                        "2": "Woman",
                        "9": "Unknown" 
                    }
                mode : str, default "directed"
                    whether the loaded data is directed or undirected, by default, it is loaded directed
                    undirected means it is ensured that self.A is symmetric
          
        Returns
            ---------
                MultiLayeredNetwork instance
        """
        
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
        if not from_library and adjacency_file == "" and adjacency_matrix is None:
            print("Error: should either select a library, give an adjacency_matrix file " + \
            "or give an adjacency matrix!")
            return None  
        
        if from_library:
            adjacency_file, node_attribute_file = self.get_library_files(library_path, from_library)
            if adjacency_file == None:
                print("Error: files could not be opened. Please give a valid library path!")
                return None
            
        self.verboseprint("Reading node attribute files...")
        self.init_node_attributes(node_attribute_file, node_attribute_dataframe)
        self.verboseprint("Done.")

        self.verboseprint("Initializing metadata...")
        self.init_meta_data()
        self.verboseprint("Done.")

        self.verboseprint("Creating linktypes...")
        self.init_linktypes(layer_file)
        self.mode = "directed"

        self.verboseprint("Done.")

        self.verboseprint("Creating attribute metadata...")
        self.init_node_attribute_metadata(node_attribute_metadata)
        self.verboseprint("Done.")
        
        self.verboseprint("Loading adjacency matrix...")
        if self.init_sparse_matrix(adjacency_file, adjacency_matrix) == -1:
            print("Error: could not initialize adjacency matrix. Please give a valid sparsegraph " \
                    + "or file in one of the following formats: [.npz, .csv, .csv.gz]")
            return None
        self.verboseprint("Done.")
        
        self.verboseprint("Initializing colors and attributes...")
        # self.init_colors(colors_file)
        self.init_attribute_colnames(attribute_colnames)
        self.verboseprint("Creating attribute code table and metadata...")
        self.init_attribute_code_table(attribute_code_table)
        self.verboseprint("Done.")
        self.igraph = None
        self.verboseprint("Done.")

        self.library_path = library_path
 
        # pathnames to give to children instances at inheritance, shortens following inits
        self._to_pass = {
            "library_path" : library_path,
            "layer_file" : layer_file,
            "colors_file" : colors_file,
            "attribute_colnames" : attribute_colnames,
            "attribute_code_table" : attribute_code_table,
            "node_attribute_metadata" : node_attribute_metadata,
            "verbose" : verbose
        }
 
    def get_library_files(self, library_path, from_library):
        """
        If library mode is used, this function checks if the path is valid and the correct
        files are stored in it. When this is the case, it returns the full path to the 
        adjacency_file and node_attribute_file.
        
        Parameters:
            -------------
            library_path : string
                string containing the path to the library
            from_library : string
                contains which folder from library to access: library_path/from_library/
 
        Returns:
            adjacency_file : string
                resulting path to adjacency_file
            node_attribute_file : string
                resulting path to node_attribute_file
            None, None : None
                if the library path or directory in library is not valid
            -------------
        """
        if from_library:
            if not os.path.isdir(library_path):
                print(f'Library path "{library_path}" does not seem to exist, check again.')
            elif not os.path.isdir(library_path + "/" + from_library):
                print(f'Library path "{library_path}" does not seem to exist, check again.')
            else :
                adjacency_file = library_path + "/" + from_library + "/adjacency.npz"
                # path to the pickle of the node attributes pickle with the current
                # pandas version and desired from_library
                if f"attributes_{pd.__version__}.pkl" in os.listdir(library_path + "/"  + from_library):
                    node_attribute_file = library_path + "/"  + from_library + f"/attributes_{pd.__version__}.pkl"
                else:
                    node_attribute_file = library_path + "/"  + from_library + f"/attributes.csv.gz"
                
                return adjacency_file, node_attribute_file
        return None, None
 
 
    def init_node_attributes(self, node_attribute_file="", node_attribute_dataframe=None):
        """
        Initialize self.node_attributes. Either set to node_attribute_dataframe or read
        from .csv, .csv.gz, or pickle file. If no node_attribute file or dataframe is given,
        self.node_attributes is initialized to None
        
        Parameters:
            -------------
            node_attribute_file : string, default ""
                string containing the (path to) node attribute file. Type: .csv, .csv.gz or pickle
                Type is derived from the file extension
            node_attribute_dataframe : pandas.DataFrame, default None
                pandas dataframe containing node attributes
            -------------
        """
        try:
            if node_attribute_dataframe is not None:
                self.node_attributes = node_attribute_dataframe
            elif node_attribute_file.endswith(".pkl"):
                # try loading a pickle of the node attributes file
                # in case there is no pickle compatible with the current pandas version
                # and the desired library, load the .csv.gz of that library and create a
                # fresh pickle
                try:
                    self.node_attributes = pd.read_pickle(node_attribute_file)
                except FileNotFoundError:
                    print("Warning: Current pandas version is incompatible with saved pickles. Using .csv.gz file and creating pickle")
                    l = node_attribute_file.split("/")
                    library_path = "/".join(l[:-2])
                    pickle_dir = l[-2]
                    l = l[-1].split("_")
                    pandas_version = l[-1][:-4]
                    from_library = "_".join(l[1:-1])
                    
                    # node attribute files in library should always be named "attributes.csv.gz"
                    self.verboseprint(f"Loading attributes.csv.gz")
                    self.node_attributes = pd.read_csv(f"{library_path}/{from_library}/attributes.csv.gz", compression="gzip")
                    self.verboseprint("Done. Pickling with current pandas version")
                    self.node_attributes.to_pickle(f"{library_path}/{from_library}/attributes_{pd.__version__}.pkl")
            elif node_attribute_file.endswith(".csv.gz"):
                self.node_attributes = pd.read_csv(node_attribute_file, compression="gzip")
            else:
                self.node_attributes = pd.read_csv(node_attribute_file)
        except:
            if node_attribute_file != "":
                print(f"Error: node_attribute_file {node_attribute_file} could not be opened")
            self.node_attributes = None

        if "label" not in self.node_attributes.columns:
            self.node_attributes = None
            print(f'Error: column "label" could not be found in the node attribute file {node_attribute_file}!')
    
    def init_node_attribute_metadata(self, node_attribute_metadata=""):
        if node_attribute_metadata is not None and node_attribute_metadata.endswith(".csv"):
            self.node_attribute_metadata = pd.read_csv(node_attribute_metadata)

    def init_meta_data(self):
        """
        Initialize meta_data: self.map_label_to_nid, self.map_nid_to_label,
        self.N (numer of nodes), and storage dictionary for layer adjacency
        matrices based on NID and label columns of node_attributes stored in
        self.node_attributes
        """
        if self.node_attributes is None:
            self.map_nid_to_label = None
            self.map_label_to_nid = None
            self.N = None
            return
        
        # getting global NID mappings for overlapping layers.
        # integer NID -> label
        self.map_nid_to_label = dict(zip(self.node_attributes.index.values,self.node_attributes['label'].values))
 
        # label -> NID
        self.map_label_to_nid = {v:k for k,v in self.map_nid_to_label.items()}
        self.node_attributes['NID'] = self.node_attributes.index
 
        # number of nodes
        self.N = max(self.map_nid_to_label.keys())+1

        # storage for layer adjacency matrices
        self.layer_adj = {}
 
    def get_sparsematrix_from_csv(self, csv_file):
        """
        Generate sparsematrix from .csv file
 
        Parameters:
            -------------
            csv_file : string
                file containing edgelist. Should have the following columns:
                source, target, binary_linktype
 
        Returns:
            A_new : csr_matrix
                resulting sparsematrix
            -------------
        """
        # get edgelists and columns corresponding to source, target and linktype
        # assumed layout: source, target, linktypes
        # self.verboseprint("Getting sparse matrix from csv...")
        if csv_file.endswith("gz"):
            edgelist = pd.read_csv(csv_file, header = 0, compression = "gzip", sep=',').drop_duplicates()
        else:
            edgelist = pd.read_csv(csv_file, header = 0, sep=',').drop_duplicates()
        try:
            source_list = edgelist["source"]
            target_list = edgelist["target"]
            if edgelist.shape[1]>=3:
                linktypes = list(edgelist["linktype"])
            else:
                linktypes = []
        except:
            print("Error: input edgelist should have at least two columns!")
            return
        
        # get number of nodes
        all_nodes = pd.concat([source_list,target_list]).unique()

        # Initialize linktype
        if self.layers is None:
            # self.verboseprint("Making self.layers...")
            # Get largest binary linktype value
            nr_links = len(np.unique(linktypes))
            self.max_bin_linktype = 2**nr_links
            
            # Generate based on binary linktypes in A
            columns = ["layer", "name", "binary"]
            labels = pd.unique(linktypes)
            binary_linktypes = [2**i for i in range(nr_links)]
            data = np.array([labels] + [labels] + [binary_linktypes]).T
            
            # Initialize linktypes
            self.layers = pd.DataFrame(data, columns=columns)
            # self.verboseprint(self.layers)
            self.init_linktypes_dict()

        if len(linktypes) == 0:
            binary_linktypes = [1 for i in range(len(source_list))]
        else:
            binary_linktypes = self.convert_linktype(linktypes, input_type="layer", output_type="binary")
        
        # initialise metadata if not initialised (no attribute data)
        # ids in edgelist are labels, mapped to nodeids
        if self.node_attributes is None:
            self.map_label_to_nid = {all_nodes[i] : i for i in range(len(all_nodes))}
            self.map_nid_to_label = {i : all_nodes[i] for i in range(len(all_nodes))}
            self.N = max(self.map_nid_to_label.keys())+1
        
        # convert label used in .csv edgelist to ID
        source_list = [self.map_label_to_nid[i] for i in source_list]
        target_list = [self.map_label_to_nid[i] for i in target_list]
        
        # create sparsematrix, use ids used in .csv file and add edgetype
        A_new = csr_matrix(((binary_linktypes),(source_list,target_list)), dtype = np.int64, shape=(self.N, self.N))
        return A_new
    
    def init_sparse_matrix(self, adjacency_file="", adjacency_matrix=None):
        """
        Initialize self.A sparsematrix. Either give a sparse matrix or a file
        from which the sparse matrix can be initialized. Valid file types are
        .npz binary (containing a scipy.csr_matrix), or .csv, .csv.gz (for
        edgelists). If no attributes are given, this function initializes label
        with artificial IDs
        
        Parameters:
            -------------
            adjacency_file : string
                string containing the (path to) adjacency file. Type: .npz binary, .csv or .csv.gz
                Type is derived from the file extension
            adjacency_matrix : scipy.csr_matrix
                sparse matrix containing the edges
        
        Returns:
            -------------
                -1 if no sparsematrix is initialized: minimal requirement to initialize 
                MultiLayeredNetwork object
        """
        try :
            if adjacency_matrix is not None:
                assert(isinstance(adjacency_matrix, csr_matrix))
                self.A = adjacency_matrix
        except :
            print("Error: not a valid adjacency matrix!")
            return -1
        
        try:
            if adjacency_matrix is None and adjacency_file.endswith(".npz"):
                self.A = load_npz(adjacency_file)
            elif adjacency_matrix is None and adjacency_file.endswith(".csv.gz") or adjacency_file.endswith(".csv") :
                # self.verboseprint("Trying to get from csv.gz or csv.")
                self.A = self.get_sparsematrix_from_csv(adjacency_file)
        except:
            print(f"Error: Adjacency_file {adjacency_file} could not be loaded succesfully")
            return -1
       
        # Initialize node attribute / label data
        if self.node_attributes is None:
            # If map_label_to_nid is not yet initialized, initialize here
            if self.map_label_to_nid is None:
                self.N = self.A.shape[0]
                # if only the matrix is given (no .csv), then node_id is assumed to equal label
                self.map_label_to_nid = {i : i for i in range(self.N)}
                self.map_nid_to_label = {i : i for i in range(self.N)}
            
            # Add label data to node_attributes
            self.node_attributes = pd.DataFrame({"label": [i for i in self.map_nid_to_label.keys()] })
 
    def init_linktypes(self, linktypes_file):
        """
        Initialize self.layers and self.layers_dict. linktypes_file should
        be .json. If no (valid) file is given, self.layers is initialized
        based on the largest binary link in self.A

        # TODO give an example for the JSON(?)
        
        Parameters:
            -------------
            linktypes_file : string
                string containing the (path to) linktypes_file. Type: .csv
        Returns:
            -------------
            None
        """ 
        try:
            self.verboseprint(f"Reading linktypes file {linktypes_file}...")
            self.layers = pd.read_csv(linktypes_file,index_col=None,header=0, sep=',')
            # Check columns
            
            if not("name" in self.layers.columns or "layer" in self.layers.columns):
                print("Error: linktypes should at least have column \"name\" or \"layer\"")

            # Add new columns if not included
            if "name" in self.layers and not "layer" in self.layers:
                self.layers["layer"] = self.layers["name"]
            elif "layer" in self.layers and not "name" in self.layers:
                self.layers["name"] = self.layers["layer"]
            if not "binary" in self.layers:
                self.layers["binary"] = [2**i for i in range(len(self.layers))]
            max_bin_link = self.layers["binary"].max()
            self.max_bin_linktype = int(np.floor(np.log2(max_bin_link)))
            self.init_linktypes_dict()
        except:
            if linktypes_file != "":
                print(f"Warning: linktypes_file {linktypes_file} is not valid.")
                print("Obtaining linktypes from adjacency matrix.")
            self.verboseprint("self.layers still None")
            self.layers = None
    
    def init_linktypes_dict(self):
        """
        Initialize dictionary for efficient conversion from link, label, code and 
        binary label to each other. Resulting dictionary is stored in self.layers_dict
        
        Returns:
            -------------
            None
        """
        binary = self.layers["binary"].tolist()
        codes = self.layers["layer"].tolist()
        labels = self.layers["name"].tolist()
        
        bin_to_code = dict(zip(binary, codes))
        bin_to_label = dict(zip(binary, labels))
        code_to_label = dict(zip(codes, labels))
        code_to_bin = dict(zip(codes, binary))
        label_to_bin = dict(zip(labels, binary))
        label_to_code = dict(zip(labels, codes))
        
        self.layers_dict = {
            'binary_to_layer' : bin_to_code,
            'binary_to_name' : bin_to_label,
            'layer_to_name' : code_to_label,
            'layer_to_binary' : code_to_bin,
            'name_to_binary' : label_to_bin,
            'name_to_layer' : label_to_code
        }
    
    def init_colors(self, colors_file):
        """
        # Initialize self.colors with data from colors_file. File should be .json.
        # If no (valid) file is given, self.colors = {all : black_color}
        
        Parameters:
            -------------
            colors_file : string
                string containing the (path to) colors_file. Type: .json
            -------------
        """
        # creating colors for the five main layers
        try:
            f = open(colors_file)
            self.colors = json.load(f)
        except FileNotFoundError:
            if colors_file != "":
                print(f"Error: colors_file {colors_file} could not be opened")
            
            # No layers, initialize default color
            if "layer" not in self.layers.columns:
                self.colors = {"all" : "#000000"}
                return
            
            # Generate color for each layer, if more than 10 colors use Spectarl scheme
            layers = np.unique(self.layers["layer"])
            if len(layers) <= 10:
                colors = matplotlib.cm.tab10(np.linspace(0, 1, len(layers)))
            else:
                colors = matplotlib.cm.Spectral(np.linspace(0, 1, len(layers)))
            tel = 0
            self.colors = {}
            for l in layers:
                self.colors[l] = matplotlib.colors.to_hex(colors[tel])
                tel += 1
 
    def init_attribute_colnames(self, attribute_colnames):
        """
        # Initialize self.attribute_colnames with data from colors_file. File should be .json.
        # If no (valid) file is given, self.attribute_colnames = None
        
        Parameters:
            -------------
            attribute_colnames : string
                string containing the (path to) attribute_colnames. Type: .json
            -------------
        """
        # adding attribute dicts and translate the column names
        try:
            f = open(attribute_colnames)
            self.attribute_colnames = json.load(f)

            # names stay the same if attribute_colnames do not exist in node_attributes.
            self.node_attributes.rename(columns=self.attribute_colnames, inplace=True)
        except FileNotFoundError:
            if attribute_colnames != "":
                print(f"Error: attribute_colnames file {attribute_colnames} could not be opened")
            self.attribute_colnames = None
    
    def init_attribute_code_table(self, attribute_code_table):
        """
        # Initialize self.attribute_code_table with data from attribute_code_table. File should be .json.
        # If no (valid) file is given, self.attribute_code_table = None
        
        Parameters:
            -------------
            attribute_colnames : string
                string containing the (path to) attribute_colnames. Type: .json
            -------------
        """
        try:
            f = open(attribute_code_table)
            self.attribute_code_table = json.load(f)

            node_attribute_metadata_list = []
            for attribute in self.attribute_code_table.keys():
                for k,v in self.attribute_code_table[attribute].items():
                    node_attribute_metadata_list.append([attribute, k, v])

            self.node_attribute_metadata = pd.DataFrame(node_attribute_metadata_list, columns=['attribute', 'key', 'value'])
        except Exception as e:
            if attribute_code_table != "":
                print(f"attribute_code_table file {attribute_code_table} could not be opened")
            self.attribute_code_table = None
    
    def get_filtered_network(self, layers = [], selected_nodes = None, use_label=True):
        """
        Returns MultiLayeredNetwork based on edge and node filtering.
        
        Possibilities:
 
        1. Edge filtering: defining a list of layers through the kwarg `layers`.
           E.g. `mln.get_edgelist(full_layers = ["family", "household"])`
 
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

        # if there is any node selection, then decrease matrix size and grab the
        # relevant rows from the node attributes table
        if selected_nodes is not None: # and len(selected_nodes) > 0:
            # remove duplicates from list
            selected_nodes = np.unique(selected_nodes)
            print("use_label",use_label)
            # mapping label to NIDs, creating node mapping
            if use_label == True:
                selected_nodes = np.array([self.to_nid(i) for i in selected_nodes])
            
            # creating True/False mask for faster selection
            idx = np.array(np.zeros(self.N,dtype=bool))
            idx[selected_nodes] = True
            # slicing the adjacency matrix
            selection_A = self.A[idx,:][:,idx]
            # slicing the attribute table
            selection_node_attributes = self.node_attributes.iloc[selected_nodes].reset_index(drop=True)

 
        else:
            selection_A = deepcopy(self.A)
            selection_node_attributes = self.node_attributes
 
        if len(layers)>0:
            # adding up the binary codes for the full layers from the argumentif len(layers)>0:
            # adding up the binary codes for the full layers from the argument
            # based on type of layer value:
            layer_index = "layer"
            if isinstance(layers[0], str):
                if use_relations:
                    layer_index = "edge_label"
                else:
                    layer_index = "name"
            
            print("Layer index",layer_index)
            binary_repr = sum([self.layers.set_index("layer").loc[l]["binary"] for l in layers])
            print("Binary repr",binary_repr)
            # select corresponding edges
            selection_A.data = selection_A.data & binary_repr
            # compress sparse matrix
            selection_A.eliminate_zeros()

        f = MultiLayeredNetwork(
            adjacency_matrix = selection_A,
            node_attribute_dataframe = selection_node_attributes,
            **self._to_pass
        )

        # TODO make this more elegant! Why is this not happening by itself?
        f.layers = self.layers
        f.layers_dict = self.layers_dict
        f.max_bin_linktype = self.max_bin_linktype
 
        return f

    def get_aggregated_network(self, aggregation_attribute=None):
        """
        Return an aggregated network over a certain attribute.
        
        Parameters:
            -------------
            aggregation_attribute : string, default None
                the column in the node_attributes over which the edges should be aggregated
            -------------
        """
        ## TODO: Check inbouwen voor aggregatie met minder dan vijf nodes.
        selection_A = deepcopy(self.A)

        if not aggregation_attribute in self.node_attributes.columns:
            return None
        
        grps, uniques = pd.factorize(self.node_attributes[aggregation_attribute])

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

        # sign om alle integers om te zetten naar 0/1.
        # hier een som over is telling van aantal relaties
        grouped_A = ft.dot(selection_A.sign())
        grouped_A_2 = grouped_A.dot(ft.T)

        grouped_A_2 = csr_matrix(grouped_A_2)

        selection_node_attributes = pd.DataFrame(data={'label' : uniques, 'weight' : [grp_counts[i] for i in range(0,len(uniques))]}, index=[i for i in range(0,len(uniques))])

        # add data related to particular aggregation level
        aggregation_node_attribute = self.node_attributes.groupby([aggregation_attribute]).head(1)[[c for c in self.node_attributes.columns if aggregation_attribute.split('_')[0] in c]]
        selection_node_attributes = selection_node_attributes.merge(aggregation_node_attribute, left_on=['label'], right_on=[aggregation_attribute], how='left')
        
        f = POPNET_data(
            adjacency_matrix = grouped_A_2,
            node_attribute_dataframe = selection_node_attributes,
            **self._to_pass
        )

        # TODO make this more elegant! Why is this not happening by itself?
        # afwijken van normale layers, aangezien de matrix nu is samengesteld door aantal relaties tussen groepen.
        selection_layers = pd.DataFrame(data={'layer' : 1, 'edge_label' : 'count', 'edge_label_long' : 'count', 'name' : 'count', 'binary' : 1}, index=[0])
        selection_layers_dict = {
            'binary_to_layer': {1 : 1},
            'binary_to_name' : {1 : 'count'},
            'layer_to_name' : {1 : 'count'},
            'layer_to_binary' : {1 : 1},
            'name_to_binary' : {'count' : 1},
            'name_to_layer' : {'count' : 1}
        }
        selection_max_bin_linktype = 1

        f.layers = selection_layers
        f.layers_dict = selection_layers_dict
        f.max_bin_linktype = selection_max_bin_linktype
        # The nodes have changed, therefore new mappings for to_nid and to_label are passed.
        f.map_label_to_nid = selection_node_attributes[["label", "NID"]].set_index('label').to_dict()['NID']
        f.map_nid_to_label = selection_node_attributes[["label", "NID"]].set_index('NID').to_dict()['label']

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
 
    def get_edgelist(self, without_linktypes = True, with_weights=False):
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
        if without_linktypes:
            if with_weights:
                # combine edges and weights
                edges_with_weights = np.concatenate((edges, weights), axis=1)
                edgelist = pd.DataFrame(edges_with_weights, columns = ["source", "target", "weight"])
            else:
                edgelist = pd.DataFrame(edges, columns = ["source", "target"])
            # self.report_time(message = "Creating edgelist dataframe.")
        else:
            edgelist = pd.DataFrame(np.concatenate((edges, weights), axis=1))
            # self.report_time(message = "Creating edgelist dataframe.")
    
            # mapping back edges to the original labels
            edgelist[0] = edgelist[0].map(self.to_label)
            edgelist[1] = edgelist[1].map(self.to_label)
            # self.report_time(message = "Remapping node labels.")
    
            # add colnames and (human) readable link types
            # if an edge has multiple linktypes, it is listed multiple times with the linktype code
            edgelist.columns = ["source", "target", "binary"]
            
            # convert all unique binary linktypes to their labels
            link_dict = {}
            for link in edgelist["binary"].unique():
                link_dict[link] = self.decompose_binary_linktype_2(link)
            # self.report_time(message = "Unfolded binary link identifiers.")
            # print(edgelist.head())


            # get pairs (binary_linktype, label) of each link
            edgelist["layer_label"] = edgelist["binary"].map(link_dict)
            # self.report_time(message = "Mapped binary link identifiers.")
            # print(edgelist.head())

            # explode values and divide pairs over binary_linktype and linktype column
            edgelist = edgelist.explode("layer_label")
            # self.report_time(message = "Exploded binary link identifiers.")
            # print(edgelist.head())
            edgelist["layer"], edgelist["label"] = edgelist["layer_label"].str
            edgelist.drop(["binary","layer_label"],axis=1,inplace=True)
            # self.report_time(message = "Adding that to dataframe?")
            # print(edgelist.head())
        
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
        g = ig.Graph.Weighted_Adjacency(self.A, mode=mode, attr="link_types")
        
        if edge_attributes:     
            # obtain and add (human readable) link types
            link_dict = {}
            for link_type in set(g.es["link_types"]):
                link_dict[link_type] = self.decompose_binary_linktype(link_type)

            g.es["link_types"] = [link_dict[x] for x in g.es["link_types"]]
            
        # add node attributes to graph from self.node_attributes
        if node_attributes:
            for col_name in self.node_attributes.columns:
                # first turning the series into a list improves performance
                g.vs[col_name] = list(self.node_attributes[col_name])
        else:
            # Add only label
            g.vs["label"] = list(self.node_attributes["label"])
        
        # store igraph object as mln attribute
        if self.igraph is None or replace_igraph :
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
            for col_name in self.node_attributes:
                cur_col = list(self.node_attributes[col_name])
                attribute_list = dict(zip(list(range(len(cur_col))), cur_col))
                nx.set_node_attributes(g, attribute_list, col_name)
        else:
            # always add "label" column
            map_nid_to_label_dict = dict(zip(np.arange(len(self.map_nid_to_label)), self.map_nid_to_label))
            nx.set_node_attributes(g, map_nid_to_label_dict, "label")
        
        # obtain and add (human readable) link types use dict for optimization
        link_dict = {}
        
        for s, t, d in g.edges(data=True):
            if edge_attributes:
                weight = g[s][t]["weight"]
 
                if weight in link_dict:
                    g[s][t]["link_types"] = link_dict[weight]
                else:
                    link_types = self.decompose_binary_linktype(weight)
                    link_dict[weight] = link_types
                    g[s][t]["link_types"] = link_types
 
            # remove weights
            d.pop("weight", None)
 
        return g
    
    def convert_linktype(self, linktypes, input_type="layer", output_type="binary"):
        """
        This function converts a single linktype or list of linktypes to
        their corresponding linklabels. Input can be binary, code or label.
        All possible values for each are in self.layers
 
        Parameters:
            -----------
            linktypes : int, string or list, no default
                A single binary code or name of a linktype
            input_type : string
                Type of input. Options: "name", "layer" and "binary"
            output_type : string
                Type of input. Options: "name", "layer" and "binary"

        Returns:
            -----------
            links : string or list
                linktype (binary, layer or name) corresponding to the given types
                If one of the linktypes is not found, returns None
         """
        
        dict_name = input_type + '_to_' + output_type
        try:
            d = self.layers_dict[dict_name]
        except KeyError:
            print('Error: dictionary value not found. Please choose from "name", "layer", "binary"')
            return None
        
        try:
            if type(linktypes) == list:
                return [d[link] for link in linktypes]
            else:
                return d[linktypes]
        except:
            print(f'Error: invalid linktype found: {input_type} to {output_type}')
            return None
    
    def decompose_binary_linktype(self,num):
        """
        Based on the integer linktype, returns a list with the linktype numbers.
 
        E.g.: self.decompose_binary_linktype(3) returns ["aunt/uncle","co-parent"]
 
        Parameters:
            --------------
            num: int
                integer number to be converted to binary and returned as linktypes
        Returns:
            ---------------
            list(str)
                list of string linktypes corresponding to integer value
        """
        return [self.convert_linktype(2**i, input_type='binary', output_type='name') for i in range(int(np.log2(self.max_bin_linktype))) if int(num)&(2**i)>0]
    
    def decompose_binary_linktype_2(self,num):
        """
        Based on the integer linktype, returns a list of tuples (linktype number, linktype).
 
        E.g.: self.decompose_binary_linktype(3) returns [(1, "aunt/uncle"), (2,"co-parent")]
 
        Parameters:
            --------------
            num: int
                integer number to be converted to binary and returned as linktype typles
        Returns:
            ---------------
            list(str)
                list of string linktypes corresponding to integer value
        """
        # TODO check it thoroughly
        return [(self.convert_linktype(2**i, input_type='binary', output_type='layer'), self.convert_linktype(2**i, input_tdecomposeype='binary', output_type='name')) for i in range(int(np.log2(self.max_bin_linktype))) if int(num)&(2**i)>0]
    
    def export_graph(self, file_name):
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
                [".npz", ".csv", ".csv.gz", ".graphml"]

        """
        try:
            f = open(file_name, "w")
        except:
            print(f"Error: {file_name} could not be opened")
            print("No graph exported")
            return
        
        _, extension = os.path.splitext(file_name)
 
        if extension in ['.csv','.gz']:
            # get edgelist of self.A
            edgelist = self.get_edgelist()
        
            # write to csv file
            if file_name.endswith(".csv"):
                edgelist.to_csv(file_name, index=False)
            else:
                edgelist.to_csv(file_name, index=False, compression = 'gzip')
        elif extension == ".graphml":
            if self.igraph is None :
                self.igraph = self.to_igraph()
            self.igraph.write_graphml(file_name)
        else:
            save_npz(file_name, self.A)
             
    def export_node_attributes(self, file_name,):
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
            print(f"Error: {file_name} could not be opened.")
            print("No node attributes exported.")
            return
        
        _, extension = os.path.splitext(file_name)
        if extension == ".csv":
            self.node_attributes.to_csv(file_name)
        else:
            self.node_attributes.to_csv(file_name, compression="gzip")
            
    def export(self, graph_file_name = "", node_attribute_file_name = "", library_name = "", overwrite = False):
        """
        Write self.A and self.node_attributes to files that can be read by this
        class to initialize the multilayered network.
 
        Parameters:
            -----------
            graph_file_name : str, default ""
                .npz file to write graph to
            node_attribute_file_name : str, default ""
                .csv.gz file to write the node attributes to
            library_name : str, default ""
                library name in which to save network files for later use, e.g.
                "amsterdam_parents" is going to create adjacency.npz and
                attributes.csv.gz within library_name folder under
                self.library_path can be later used in __init__ e.g.
                MultiLayeredNetwork(from_library = "amsterdam_parents") reads
                back to exported files
            overwrite : bool, default False
                if library name already exists, overwrite existing files if True
        """
 
        # if the user wants to save in library
        if library_name != "":
            # if the given library entry (folder) already exists
            if os.path.isdir(self.library_path + "/" + library_name):
                if not overwrite:
                    # warning about overwriting
                    print(f'Library "{self.library_path}/{library_name}" already exists, call function with `overwrite = True` keyword argument if you\'re sure!')
                    return
                else:
                    # set target filenames
                    graph_file_name = self.library_path + "/" + library_name + "/adjacency.npz"
                    node_attribute_file_name = self.library_path + "/" + library_name + "/attributes.csv.gz"
            # if the folder does not exists
            else:
                # create folder
                os.mkdir(self.library_path + "/" + library_name)
                # set target filenames
                graph_file_name = self.library_path + "/" + library_name + "/adjacency.npz"
                node_attribute_file_name = self.library_path + "/" + library_name + "/attributes.csv.gz"
        
        # check if given filenames are valid
        if graph_file_name == "" or \
            node_attribute_file_name == "" or \
            not graph_file_name.endswith(".npz") or \
            not node_attribute_file_name.endswith(".csv.gz"):
 
            print(f"You have not given correct destinations! Check again:")
            print(f"\tEdge file (should be .npz): {graph_file_name}")
            print(f"\tNode attribute file (should be .csv.gz): {node_attribute_file_name}")
            return
        
        self.export_graph(graph_file_name)
        self.export_node_attributes(node_attribute_file_name)
        
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
            print(f"Please use a maximum depth of {ego_depth_limit}")
            print("To ignore this limit, please use \'ignore_limit=True\'")
            return
        
        ego = self.to_nid(ego_label)
 
        # separating the first round allows to skip the expensive for loop,
        # leading to a substantial speedup
        selected = [ego] + self.A[ego].indices.tolist()

        # iteratively search at further steps
        for k in range(1, depth):
            next_neighbors = self.A[selected].indices
            # there might be circles when looking for new neighbors, so we take unique integer ids
            selected_nodes = pd.unique(np.concatenate((selected, next_neighbors)))
 
        selected = list(map(lambda x: self.map_nid_to_label[x], selected))
 
        # return mln or list of nodes in egonetwork
        if return_list:
            return selected
        else:
            return self.get_filtered_network(selected_nodes = selected)
    
    def create_affiliation_matrix(self, key, affil_edgelist):
        """
        This method takes a person -> affiliation bipartite edgelist, and
        creates scipy sparse representation for it, storing the mapping of
        affiliation IDs in a dict.
 
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
 
        self.affiliation_matrix[key]['column_map_nid_to_label'] = {i:elem for i,elem in enumerate(unique_affiliations)}
        self.affiliation_matrix[key]['column_map_label_to_nid'] = {elem:i for i,elem in enumerate(set([v for k,v in affil_edgelist]))}
 
        self.affiliation_matrix[key]['M'] = M
 
        # col number
        i = []
        j = []
        for k,v in affil_edgelist:
                i.append(self.map_label_to_nid[k])
                j.append(self.affiliation_matrix[key]['column_map_label_to_nid'][v])
 
        # sparse adjacency matrix for affiliations (work, school, region etc.)
        A = csr_matrix((np.ones(len(i)),(i,j)), shape=(self.N,M), dtype = 'int')
 
        self.affiliation_matrix[key]['A'] = A
 
    def downcast_to_binary(self):
        """
        Downcast all values in self.A to a binary value. Only 1s occur in the
        resulting matrix.
        
        Returns:
            -------------
                A : sparsegraph A which is a downcasted version of self.A
        """
        return self.A.astype(bool).astype(np.int16)
    
    def to_nid(self, labels):
        """
        This function converts a label, or list of labels to their corresponding
        NIDs
 
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
            return self.map_label_to_nid[labels]
        else:
            return [self.map_label_to_nid[elem] for elem in labels]
 
        
    def to_label(self, nids):
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
 
        if type(nids) != list:
            return self.map_nid_to_label[nids]
        else:
            return [self.map_nid_to_label[elem] for elem in nids]

    def get_degrees(self, selected_nodes=None, selected_layers=None):
        """
        Calculate degree for selected nodes with selected layers.
        """
        if selected_nodes is None:
            selected_nodes = self.node_attributes["label"].tolist()
        
        selected_nodes = [self.to_nid(n) for n in selected_nodes]

        if selected_layers is None:
            return dict(zip([self.to_label(n) for n in selected_nodes], self.downcast_to_binary()[selected_nodes, :].sum(axis=0).tolist()[0]))
        else:
            res = {}

            for layer in selected_layers:
                if layer not in self.layer_adj:
                    _ = self.get_single_layer_adj_matrix(layer=layer,store=True)
                res[layer] = dict(zip([self.to_label(n) for n in selected_nodes],self.layer_adj[selected_nodes,:].T.sum()))

            return res

    def get_clustering_coefficient(self, selected_nodes=None, selected_layers=None, batchsize=100000):
        """
        Calculate clustering coefficient for selected nodes with selected layers.

        Parameters:
        -----------
            selected_nodes : list, default None
                which nodes to compute cc for, if None, use all
            selected_layers : list of str
                which layers to include in the computation
            batchsize : int, default 100000
                chunks in which to split up matrix multiplication bc of memory
                issues

        Returns:
        --------
            dict with first key "clustering_coefficient", then dict of label -> cc values
            
        """
        if selected_layers is None:
            selected_layers = list(self.layers["layer"])
        if selected_nodes is None:
            selected_nodes = self.node_attributes["label"].tolist()
        
        selected_nodes = [self.to_nid(n) for n in selected_nodes]

        if batchsize > len(selected_nodes):
            batchsize = len(selected_nodes)

        # obtain a copy of the sparse adjacency matrix such that each element
        # A[i,j] contains the number of links between i and j
        A = csr_matrix(self.A.shape, dtype=np.int64)

        for layer in selected_layers:
            if layer not in self.layer_adj:
                _ = self.get_single_layer_adj_matrix(layer=layer,store=True)
            A += self.layer_adj[layer]

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

        return {
            "clustering_coefficient": dict(zip([self.to_label(n) for n in selected_nodes],clustering_coefficient)),
        }

    def get_single_layer_adj_matrix(self,layer=None,store=False,storage_name=""):
        """
        Getting 0/1 adjacency matrix of a single layer or linktype.

        Parameters:
        -----------
            layer: str, int or list[int]
                if str, whole layer
                if int, one linktype correspoding to linktype code
                if list[int] several linktypes
            store: bool, default False
                store adjacency matrix in self.layer_adj
            storage_name: str, default None
                if layer is int or list, giving a name to the stored adjacency patrix e.g. parents

        Returns
        -------
            scipy.sparse
                0/1 adjacency matrix of shape (self.N,self.N)
        """
        if type(layer)==str:
            lA = self.get_filtered_network(full_layers=[layer]).downcast_to_binary()
            if store:
                self.layer_adj[layer] = lA
        elif type(layer)==int:
            lA = self.get_filtered_network(layer_codes=[layer]).downcast_to_binary()
            if store and storage_name!="" and storage_name not in self.layers["layer"]:
                self.layer_adj[storage_name] = lA
            else:
                print("Please give a correct storage name that is nonempty and not one of the layer names!")
        elif type(layer)==list:
            lA = self.get_filtered_network(layer_codes=layer).downcast_to_binary()
            if store and storage_name!="" and storage_name not in self.layers["layer"]:
                self.layer_adj[storage_name] = lA
            else:
                print("Please give a correct storage name that is nonempty and not one of the layer names!")
        return lA

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
            if l not in self.layer_adj:
                _ = self.get_single_layer_adj_matrix(layer=l,store=True)
            d.append(self.layer_adj[l])
        self.sA += block_diag(d)
        return self.sA