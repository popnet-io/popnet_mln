# importing libraries
import pandas as pd
from pandas.io.parsers import read_csv
import numpy as np
import igraph as ig
import networkx as nx
 
from scipy.sparse import csr_matrix, save_npz, load_npz
import json
 
from copy import deepcopy
import os

from datetime import datetime
 
# define constants
nx_node_limit = 10000
ego_depth_limit = 3
 
class MultiLayeredNetwork:
    
    def __init__(
        self, 
        adjacency_file = "",
        node_attribute_file = "",
        adjacency_matrix =  None,
        node_attribute_dataframe = None,
        from_library = "",
        library_path = "/data/projecten/popnet/library",
        linktypes_file = '/data/projecten/popnet/meta/linktypes_en_full.csv',
        layer_binary_repr_file = '/data/projecten/popnet/meta/layer_binary_codes.json',
        colors_file = '/data/projecten/popnet/meta/layer_colors.json',
        attribute_colnames = '/data/projecten/popnet/meta/attribute_colnames.json',
        attribute_code_table = '/data/projecten/popnet/meta/attribute_code_table.json',
        verbose=True
    ):
        """
        This class contains methods and attributes to work with the population-level
        social network (or its parts) of the Netherlands using different edge types and layers efficiently.
 
        The network is either loaded from an already saved sparse matrix and a node attribute CSV using the
        following attributes:
            * adjacency_file:  scipy.sparse.csr matrix saved as an `npz` file
            * node_attribute_file: CSV to be loaded as a pd.DataFrame, rows in order of previous matrix rows
                TODO: should contain a column called RINPERSOON that is the primary key for node identification?
        
        Or in-memory objects can also be given to the constructor:
            * adjacency_matrix: scipy.sparse.csr matrix
            * node_attribute_dataframe: pd.DataFrame, rows in order of previous matrix rows
 
        Pre-stored files with standard names can be called with the shorthand:
            * from_library: str
        Then the npz and csv.gz files will be read from the library_path/from_library folder similarly to the first method.
 
        After loading these two key elements, class attributes and methods work the same.
 
        The adjacency matrix self.A is stored in a `scipy.sparse.csr_matrix` class, that only saves nonzero
        elements, and on which scipy csgraph algorithms run. People are indexed from 0 to N-1, where N is the
        total number of nodes in this network.
 
        Two dictionaries mapping hashed CBS ids to integer ids and back are created based on the node attribute file in the self.nodemap
        and self.nodemap_back attributes. These mappings only refer to one instance of the class - if you create a subgraph (see later)
        the coding is going to change, and the mapping should be found in the new instance that represents the subgraph.
        
        The adjacency matrix contains integers that encode linktypes if viewed as binary numbers.
        Each possible linktype is assigned an integer of the form 2**i. For example, if both type i and type j
        edge is present between two people, then the corresponding value in self.A would be 2**i+2**j. It means that we can test for a certain
        edgetype using bitwise AND operation very cheaply. E.g. a certain element of self.A is 7, then 7=1+2+4 which means
        that edgetypes 0,1, and 2 are present between the two people, and 7&2 = 2 in Python (it behaves like a mask 111 & 010 = 010).
        
        scipy.csr matrices are cheap to slice rowwise, but beware, some operation that seem straightforward in numpy might be costly 
        (e.g. getting random coordinates after each other or colwise slicing)! If something is running too long, consult the scipy reference manual.
 
        Node attributes are stored in self.node_attributes pandas.DataFrame, column names are translated into English based on self.attribute_colnames, 
        CBS code tables of used variables are accessible from self.attribute_code_table, keys of that variable are the English colnames.
 
        Example loadings:
 
        ```
        # FROM FILES
        # insert class path to sys path to access popnet
        import sys
        sys.path.insert(0,'/data/projecten/popnet/src/')
 
        # import custom class for the network
        from mln import MultiLayeredNetwork
 
        # read the whole 2018 POPNET network
        popnet = MultiLayeredNetwork(
            adjacency_file = '/data/projecten/popnet/library/full/adjacency.npz',
            node_attribute_file = '/data/projecten/popnet/library/full/attributes.csv.gz'
        )
 
        # FROM MEMORY
        import sys
        sys.path.insert(0,'/data/projecten/popnet/src/')
 
        # import custom class for the network
        from mln import MultiLayeredNetwork
 
        # read the whole 2018 POPNET network
        popnet = MultiLayeredNetwork(
            adjacency_matrix = A, # NxN scipy.sparse.csr_matrix
            node_attribute_file = df # pd.DataFrame containing N rows in the order of the matrix
        )
 
        # FROM LIBRARY
        import sys
        sys.path.insert(0,'/data/projecten/popnet/src/')
 
        # import custom class for the network
        from mln import MultiLayeredNetwork
 
        # read the whole 2018 POPNET network
        popnet = MultiLayeredNetwork(
            from_library="amsterdam_parents"
        )
 
        ```
 
        Parameters
            ----------
                adjacency_file : str, default ""
                    path of `npz` file containing `scipy.sparse.csr_matrix` saved in binary format
                node_attribute_file : str, default ""
                    path of (gzipped) CSV file containing node atributes comma separated, containing 'RINPERSOON' column
                adjacency_matrix : scipy.sparse.csr_matrix, default None
                    scipy.sparse.csr_matrix of size NxN encoding the adjacency matrix
                node_attribute_dataframe : pandas.DataFrame
                    pandas.DataFrame containing exactly N rows corresponding to the matrix rowsm containing 'RINPERSOON' column
                    other columns having attributes of graph data
                from_library : str, default ""
                    short name to get network read from `library_path` subdirectory
                library_path : str, default "/data/projecten/popnet/library"
                    path where library entries are stored
                linktypes_file : str, default '/data/projecten/popnet/meta/linktypes_en_full.csv'
                    csv file mapping linktype binary codes and integer codes to layers and string labels
                layer_binary_repr_file : str, default '/data/projecten/popnet/meta/layer_binary_codes.json'
                    JSON mapping layer names to binary equivalents
                colors_file : str, default '/data/projecten/popnet/meta/layer_colors.json'
                    JSON defining layer colors for visualization
                attribute_colnames : str, default '/data/projecten/popnet/meta/attribute_colnames.json'
                    JSON containing the Dutch -> English CBS attribute column name translations
                attribute_code_table = '/data/projecten/popnet/meta/attribute_code_table.json'
                    JSON containing English-translated possible attribute values for all columns
 
        Attributes
            ---------
                A : scipy.sparse.csr_matrix[int64]
                    adjacency matrix NxN, each element encodes the linktypes in a binary fashion
                node_attributes : pd.DataFrame
                    table containing node attributes RINPERSOON column containing CBS id,
                    ID column containing integer ID
                nodemap : dict
                    RINPERSOON CBS ID (int) -> 0...N-1 integer ID (int) mapping
                nodemap_back : dict
                    0...N-1 integer ID (int) -> RINPERSOON CBS ID (int) mapping
                N : int
                    number of nodes
                linktypes : pd.DataFrame
                    table containing info on linktypes in the networks
                linktype_dict : dict
                    linktype (int) -> long label (str) e.g. for plotting
                layer_binary_repr : dict
                    layer (str) -> binary encoding (int) e.g. for slicing
                colors : dict
                    layer (str) -> hex color (str) for plotting
                attribute_colnames : dict
                    Dutch node attribute column names (str) -> English node attribute column names (str) 
                attribute_code_table : dict
                    English node attribute column names (str) -> dict of code keys -> code values corresponding to different columns
                    useful for interpreting results
                    e.g. "gender" : {
                            "1": "Man",
                            "2": "Woman",
                            "9": "Unknown"
                            }
                
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
            print("Error. Should either select a library, give an adjacency_matrix file " + \
            "or give an adjacency matrix!")
            return None  
        
        if from_library:
            adjacency_file, node_attribute_file = self.get_library_files(library_path, from_library)
            if adjacency_file == None:
                print("Error files could not be opened. Please give a valid library path!")
                return None
        self.verboseprint("Reading node attribute files...")
        self.init_node_attributes(node_attribute_file, node_attribute_dataframe)
        self.verboseprint("Done.")
        self.init_meta_data()
        
        self.verboseprint("Loading adjacency matrix...")
        if self.init_sparse_matrix(adjacency_file, adjacency_matrix) == -1:
            print("Error: could not initialize adjacency matrix. Please give a valid sparsegraph " \
                    + "or file in one of the following formats: [.npz, .csv, .csv.gz]")
            return None
        self.verboseprint("Done.")
        
        self.init_linktypes(linktypes_file)
        self.init_layer_binary_repr(layer_binary_repr_file)
        self.init_colors(colors_file)
        self.init_attribute_colnames(attribute_colnames)
        self.init_attribute_code_table(attribute_code_table)
        self.igraph = None
 
        # pathnames to give to children instances at inheritance, shortens following inits
        self._to_pass = {
            "library_path" : library_path,
            "linktypes_file" : linktypes_file,
            "layer_binary_repr_file" : layer_binary_repr_file,
            "colors_file" : colors_file,
            "attribute_colnames" : attribute_colnames,
            "attribute_code_table" : attribute_code_table,
            "verbose" : verbose
        }
 
    def get_library_files(self, library_path, from_library, pickle_dir="pickles"):
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
            pickle_dir : string, default "pickles"
                contains which folder of the library contains pickle files
 
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
                node_attribute_file = library_path + "/" + pickle_dir + f"/attributes_{from_library}_{pd.__version__}.pkl"
                
                return adjacency_file, node_attribute_file
        return None, None
 
 
    def init_node_attributes(self, node_attribute_file="", node_attribute_dataframe=None):
        """
        Initialize self.node_attributes. Either set to node_attribute_dataframe or read
        from .csv or .csv.gz file. If no node_attribute file or dataframe is given,
        self.node_attributes is initialized to None
        
        Parameters:
            -------------
            node_attribute_file : string
                string containing the (path to) node attribute file. Type: .csv or .csv.gz
                Type is derived from the file extension
            node_attribute_dataframe : pandas dataframe
                Pandas dataframe containing node attributes
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
                    print("Current pandas version is incompatible with saved pickles. Use .csv.gz file")
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
                    self.node_attributes.to_pickle(f"{library_path}/{pickle_dir}/attributes_{from_library}_{pd.__version__}.pkl")
            elif node_attribute_file.endswith(".csv.gz"):
                self.node_attributes = pd.read_csv(node_attribute_file, compression="gzip")
            else:
                self.node_attributes = pd.read_csv(node_attribute_file)
        except:
            if node_attribute_file != "":
                print(f"node_attribute_file {node_attribute_file} could not be opened")
            self.node_attributes = None
        
        # TODO: what to do with RINPERSOON?
        # if no node attributes, create artificial RINPERSOON attribute elsewhere
 
    def init_meta_data(self):
        """
        Initialize meta_data: self.nodemap, self.nodemap_back and self.N (numer of nodes)
        based on ID and RINPERSOON columns of node_attributes stored in self.node_attributes
        """
        if self.node_attributes is None:
            self.nodemap_back = None
            self.nodemap = None
            self.N = None
            return
        
        # TODO: this only works if the node_attributes dataframe contains RINPERSOON
        # getting global node_id mappings for overlapping layers.
        # integer ID -> RINPERSOON
        self.nodemap_back = self.node_attributes['RINPERSOON'].to_dict()
 
        # RINPERSOON -> integer ID
        self.nodemap = {v:k for k,v in self.nodemap_back.items()}
        self.node_attributes['ID'] = self.node_attributes.index
 
        # number of nodes
        self.N = max(self.nodemap_back.keys())+1
 
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
        # TODO I don't see how this function binds correctly to the existing structure
        # How does it couple to node attributes?
        # TODO: check if corrected in this version
        
        # get edgelists and columns corresponding to source, target and linktype
        # assumed layout: source, target, linktypes
        if csv_file.endswith("gzip"):
            edgelist = pd.read_csv(csv_file, skiprows=[0], header = None, compression = "gzip").drop_duplicates()
        else:
            edgelist = pd.read_csv(csv_file, skiprows=[0], header = None).drop_duplicates()
        try:
            source_list = edgelist[0]
            target_list = edgelist[1]  
            if edgelist.shape[1]>=3:
                binary_linktype = edgelist[2]
            else:
                binary_linktype = []
        except:
            print("Input edgelist should have at least two columns!")
            return
 
        # get number of nodes
        all_nodes = source_list.append(target_list).unique()

        if len(binary_linktype) == 0:
            binary_linktype = [1 for i in range(len(source_list))]
        
        # initialise metadata if not initialised (no attribute data)
        if self.node_attributes is None:
            self.nodemap = {all_nodes[i] : i for i in range(len(all_nodes))}
            self.nodemap_back = {i : all_nodes[i] for i in range(len(all_nodes))}
            self.N = max(self.nodemap_back.keys())+1

        # Convert RINPERSOON used in .csv edgelist to ID
        # TODO check: is this correct?
        source_list = [self.nodemap[i] for i in source_list]
        target_list = [self.nodemap[i] for i in target_list]
        
        # create sparsematrix, use ids used in .csv file and add edgetype
        A_new = csr_matrix(((binary_linktype),(source_list,target_list)), dtype = np.int64)
        return A_new
    
    def init_sparse_matrix(self, adjacency_file="", adjacency_matrix=None):
        """
        Initialize self.A sparsematrix. Either give a sparse matrix or a file from which the sparse matrix
        can be initialized. Valid file types are .npz binary (containing a scipy.csr_matrix), or 
        .csv, .csv.gz (for edgelists). If no attributes are given,
        this function initializes RINPERSOON with artificial IDs
        
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

            elif adjacency_matrix is None:# adjacency_file.endswith(".csv.gz") or adjacency_file.endswith(".csv") :
                # TODO: initialize node attributes before initialization from csv
                self.A = self.get_sparsematrix_from_csv(adjacency_file)
        except:
            print(f"Error: Adjacency_file {adjacency_file} could not be loaded succesfully")
            return -1
       
        # Initialize node attribute / RINPERSOON data
        # TODO: use nodemap or nodemap_back
        if self.node_attributes is None:
            # If nodemap is not yet initialized, initialize here
            if self.nodemap is None:
                self.N = self.A.shape[0]
                # if only the matrix is given (no .csv), then node_id is assumed to equal label
                self.nodemap = {i : i for i in range(self.N)}
                self.nodemap_back = {i : i for i in range(self.N)}
            
            # Add label data to node_attributes
            self.node_attributes = pd.DataFrame({"RINPERSOON": [i for i in self.nodemap_back.keys()] })
 
    def init_linktypes(self, linktypes_file):
        """
        Initialize self.linktypes and self.linktype_dict. linktypes_file should be .json.
        If no (valid) file is given, self.linktypes = None
        
        Parameters:
            -------------
            linktypes_file : string
                string containing the (path to) linktypes_file. Type: .json
        Returns:
            -------------
            None
        """
 
        #TODO: when new mln is created, should inherit old linktypes or pass files again?
        try: 
            self.linktypes = pd.read_csv(linktypes_file)
            self.linktype_dict = self.linktypes["label_long"].to_dict()
            self.linktype_binary_dict = dict(zip(self.linktypes["binary_linktype"], self.linktypes["label"]))
        except:
            if linktypes_file != "":
                print(f"Linktypes_file {linktypes_file} could not be opened!")
            self.linktypes = None
            self.linktype_dict = None
            self.linktype_binary_dict = None
    
    def init_layer_binary_repr(self, layer_binary_repr_file):
        """
        Initialize self.layer_binary_repr layer_binary_repr_file should be .json.
        If no (valid) file is given, self.layer_binary_repr = None
        
        Parameters:
            -------------
            layer_binary_repr_file : string
                string containing the (path to) layer_binary_repr_file. Type: .json
        
        Returns:
            -------------
            None
        """
        # layer -> integer representations: summing up all binary numbers corresponding to a certain layer
        try:
            f = open(layer_binary_repr_file)
            self.layer_binary_repr = json.load(f)
        except:
            if layer_binary_repr_file != "":
                print(f"layer_binary_repr_file {layer_binary_repr_file} could not be opened")
            self.layer_binary_repr = None
    
    def init_colors(self, colors_file):
        """
        Initialize self.colors with data from colors_file. File should be .json.
        If no (valid) file is given, self.colors = None
        
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
                print(f"colors_file {colors_file} could not be opened")
            self.colors = None
 
    def init_attribute_colnames(self, attribute_colnames):
        """
        Initialize self.attribute_colnames with data from colors_file. File should be .json.
        If no (valid) file is given, self.attribute_colnames = None
        
        Parameters:
            -------------
            attribute_colnames : string
                string containing the (path to) attribute_colnames. Type: .json
            -------------
        """
        # adding attribute dicts
        try:
            f = open(attribute_colnames)
            self.attribute_colnames = json.load(f)
        except FileNotFoundError:
            if attribute_colnames != "":
                print(f"attribute_colnames file {attribute_colnames} could not be opened")
            self.attribute_colnames = None
    
    def init_attribute_code_table(self, attribute_code_table):
        """
        Initialize self.attribute_code_table with data from attribute_code_table. File should be .json.
        If no (valid) file is given, self.attribute_code_table = None
        
        Parameters:
            -------------
            attribute_colnames : string
                string containing the (path to) attribute_colnames. Type: .json
            -------------
        """
        try:
            f = open(attribute_code_table)
            self.attribute_code_table = json.load(f)
        except:
            if attribute_code_table != "":
                print(f"attribute_code_table file {attribute_code_table} could not be opened")
            self.attribute_code_table = None
    
    def get_filtered_network(self, layer_codes = [] , full_layers = [], selected_nodes = None):
        """
        Returns MultiLayeredNetwork based on edge and node filtering.
        
        Possibilities:
 
        1. Edge filtering 1: defining a list of full layers through the kwarg `full_layers`. List of full layers:
                - family
                - household
                - neighbors
                - school
                - work
            E.g. to get the full family and household layers:
            `mln.get_edgelist(full_layers = ["family", "household"])`
 
        2. Edge filtering 2: defining layer codes listed in the `mln.linktypes` DataFrame though the kwarg `layer_codes`.
            E.g. to select the parent and grandchild links:
            `mln.get_edgelist(layer_codes = [104,108])`
 
        3. Edge filtering 1+2: mixing the two definitions. 
            E.g. to get all family connections, and elementary school classmates:
            `mln.get_edgelist(full_layers = ["family"], layer_codes = [501])`
 
        3. Node filtering: given a list of RINPERSOON IDs, select subgraph spanned by those nodes, consisting only of selected edgetypes.
 
        Parameters:
            -------------
            layer_codes : list of int, default []
                list of layer codes to include in returned object
                see list of possible layer codes in self.linktypes
            full_layers : list of str, default []
                list of full layers to include in returned object
                5 possible types: "family","work","neighbor","school","household"
            selected_nodes : list of int, default None
                if None, return all nodes in the parent object
                if list, select nodes and their spanned subgraph with given RINPERSOON ids into returned object
 
        Returns:
            -------------
            MultiLayeredNetwork
                filtered network with less data but the very same structure as the parent object
        """
 
        # if there is any node selection, then decrease matrix size and grab the
        # relevant rows from the node attributes table
        if selected_nodes is not None and len(selected_nodes) > 0:
            # remove duplicates from list
            selected_nodes = pd.unique(selected_nodes)
            # mapping CBS IDs to integer IDs, creating node mapping
            selected_nodes_int = np.array([self.nodemap[i] for i in selected_nodes])
            # slicing the adjacency matrix
            # for small selections, it is faster to do all slicing on the csr_matrix
            if len(selected_nodes_int) < 250:
                try:
                    # the resulting matrix is transposed, but using .T returns a
                    # csc_matrix so it must be converted back
                    selection_A = self.A[selected_nodes_int, selected_nodes_int.reshape(-1,1)].T.tocsr()
                    # for unknown reasons this code can crash, so as a backup we
                    # simply use the other approach
                except:
                    selection_A = self.A[selected_nodes_int].tocsc()[:,selected_nodes_int].tocsr()
            # for large selections, it is faster to first slice on the rows,
            # then convert to a csc_matrix and do column slicing there
            else:
                selection_A = self.A[selected_nodes_int].tocsc()[:,selected_nodes_int].tocsr()

            selection_node_attributes = self.node_attributes.iloc[selected_nodes_int].reset_index(drop=True)
            selection_node_attributes['ID'] = selection_node_attributes.index
 
        else:
            selection_A = deepcopy(self.A)
            selection_node_attributes = self.node_attributes
 
        binary_repr = 0
        if len(full_layers)>0:
            # adding up the binary codes for the full layers from the argument
            binary_repr += sum([self.layer_binary_repr[l] for l in np.unique(full_layers)])
        if len(layer_codes)>0:
            for c in layer_codes:
                b = self.linktypes.set_index("code").loc[c]["binary_linktype"]
                # if the code was not already present in the full layers before
                if (binary_repr & b) == 0:
                    binary_repr += b
 
        # if there was any edge selection, then filter edges
        if len(layer_codes) > 0  or len(full_layers) > 0:
            selection_A.data = selection_A.data & binary_repr
            # compress sparse matrix
            selection_A.eliminate_zeros()
 
        return MultiLayeredNetwork(
            adjacency_matrix = selection_A,
            node_attribute_dataframe = selection_node_attributes,
            **self._to_pass
        )

    def report_time(self, message = "", init=False):
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
 
    def get_edgelist(self, without_linktypes = True):
        """
        This function returns a  pandas dataframe containing the edge list representing
        sparse matrix stored in self.A. 

        If without_linktypes is True, then the edges correspond to the nonzero values of 
        self.A. If it is False, binary weights in self.A are decoded.
        
        If one does not need the linktypes, then the resulting columns are:
        "source", "target"

        Otherwise, the columns are:
        "source", "target", "binary_linktype", "linktype"
 
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
            edgelist = pd.DataFrame(edges, columns = ["source", "target"])
            # self.report_time(message = "Creating edgelist dataframe.")
        else:
            edgelist = pd.DataFrame(np.concatenate((edges, weights), axis=1))
            # self.report_time(message = "Creating edgelist dataframe.")
    
            # mapping back edges to the original CBS IDs
            edgelist[0] = edgelist[0].map(self.nodemap_back)
            edgelist[1] = edgelist[1].map(self.nodemap_back)
            # self.report_time(message = "Remapping node ids.")
    
            # add colnames and (human) readable link types
            # if an edge has multiple linktypes, it is listed multiple times with the linktype code
            edgelist.columns = ["source", "target", "binary_linktype"]
            
            # convert all unique binary linktypes to their labels
            link_dict = {}
            for link in edgelist['binary_linktype'].unique():
                link_dict[link] = self.decompose_binary_linktype_2(link)
            # self.report_time(message = "Unfolded binary link identifiers.")
            # print(edgelist.head())


            # get pairs (binary_linktype, label) of each link
            edgelist["binary_linktype"] = edgelist["binary_linktype"].map(link_dict)
            # self.report_time(message = "Mapped binary link identifiers.")
            # print(edgelist.head())

            # explode values and divide pairs over binary_linktype and linktype column
            edgelist = edgelist.explode("binary_linktype")
            # self.report_time(message = "Exploded binary link identifiers.")
            # print(edgelist.head())
            edgelist["binary_linktype"], edgelist["linktype"] = edgelist["binary_linktype"].str
            # self.report_time(message = "Adding that to dataframe?")
            # print(edgelist.head())
        
        return edgelist
    
    def to_igraph(self, directed=True, edge_attributes=True, node_attributes=False, replace_igraph=False):
        """
        This function returns an igraph object of the sparsematrix stored in self.A.
        Edge attributes (link types) and node attributes (from self.node_attributes) 
        can be added to this object.
        If self.igraph has not yet been initialized, this is set to the 
 
        Parameters:
            -----------
            directed : boolean, default True
                mode of returned igraph object: True for directed, False for undirected graph
            edge_attributes : boolean, default True
                True if edge attributes from self.node_attributes should be added to igraph object.
            node_attributes : boolean, default False
                True if node attributes should be added to igraph object
                obtained from self.node_attributes
                Note: RINPERSOON is always added
            replace_igraph : boolean, default False
                When the mln object already has an igraph object, it will by default not be replaced
                by the newly generated object. To replace the object, set replace_igraph to True
 
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
            # Add only RINPERSOON
            g.vs["RINPERSOON"] = list(self.node_attributes["RINPERSOON"])
        
        # store igraph object as mln attribute
        if self.igraph is None or replace_igraph :
            self.igraph = g
        
        return g
 
    def to_networkx(self, directed = True, edge_attributes = True, node_attributes = False, ignore_limit = False):
        """
        This function returns a networkx object of the sparsematrix stored in self.A.
        Edge attributes (link types) and node attributes (from self.node_attributes) 
        can be added to this object.
 
        Parameters:
            -----------
            directed : boolean, default True
                mode of returned igraph object: True for directed, False for undirected graph
            edge_attributes : boolean, default True
                True if edge attributes from self.node_attributes should be added to igraph object.
            node_attributes : boolean, default False
                True if node attributes should be added to networkx object
                obtained from self.node_attributes
                RINPERSOON is always added
            ignore_limit : boolean, default False
                False if nx object can have at most nx_node_limit nodes
                Set to True to ignore this limit
 
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
            # always add RINPERSOON
            nodemap_back_dict = dict(zip(list(range(len(self.nodemap_back))), self.nodemap_back))
            nx.set_node_attributes(g, nodemap_back_dict, "RINPERSOON")
        
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
        return [self.linktype_binary_dict.get(2**i) for i in range(22) if num&(2**i)>0]
    
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
        return [(self.linktype_binary_dict.get(2**i), 2**i) for i in range(22) if num&(2**i)>0]
    
    def export_graph(self, file_name):
        """
        Write self.A to file called file_name. The file extension is read to determine 
        the type of output. Options are:
        - ".npz" or no extension: Binary (default)
        - ".csv" ".csv.gz": Edgelist format
        - ".graphml" : From igraph object to graphml format
          Note: if self.igraph does not yet exist, an igraph object will be generated
 
        Parameters:
            -----------
            file_name : str
                file to write graph to. Extension should contain the type of output
                [".npz", ".csv", ".csv.gz", ".graphml"]

        """
        try:
            f = open(file_name, "w")
        except:
            print(f"Error {file_name} could not be opened")
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
        elif extension == "graphml":
            if self.igraph is None :
                self.igraph = self.to_igraph()
            self.igraph.write_graphml(file_name)
        else:
            save_npz(file_name, self.A)
             
    def export_node_attributes(self, file_name,):
        """
        Write self.node_attributes to file to .csv or .csv.gz file
        
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
            print(f"Error {file_name} could not be opened.")
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
                library name in which to save network files for later use, e.g. "amsterdam_parents"
                is going to create adjacency.npz and attributes.csv.gz within library_name folder under self.library_path
                can be later used in __init__ e.g. MultiLayeredNetwork(from_library = "amsterdam_parents") reads back to exported files
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
        
    def get_egonetwork(self, ego_rinpersoon, depth=1, return_list=False, ignore_limit=False):
        """
        Starting out from a given node, this function returns the networks at a given
        depth around that selected node.
 
        Parameters:
        -----------
 
            ego_rinpersoon : int
                RINPERSOON ID of selected node
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
        
        ego = self.nodemap[ego_rinpersoon]
 
        # separating the first round allows to skip the expensive for loop,
        # leading to a substantial speedup
        selected = [ego] + self.A[ego].indices.tolist()

        # iteratively search at further steps
        for k in range(1, depth):
            next_neighbors = self.A[selected].indices
            # there might be circles when looking for new neighbors, so we take unique integer ids
            selected_nodes = pd.unique(np.concatenate((selected, next_neighbors)))
 
        selected = list(map(lambda x: self.nodemap_back[x], selected))
 
        # return mln or list of nodes in egonetwork
        if return_list:
            return selected
        else:
            return self.get_filtered_network(selected_nodes = selected)
    
    def create_affiliation_matrix(self, key, affil_edgelist):
        """
        This method takes a person -> affiliation bipartite edgelist, and creates
        scipy sparse representation for it, storing the mapping of IDs in a dict.
 
        Parameters:
        -----------
            mln : MultiLayeredNetwork
                the mln instance we would like to add the bipartite structure to
            key : str
                the name of the bipartite structure for storage
            affil_edgelist : list[list]
                list of 2-tuples containing bipartite edgelist
                first node in the edges should correspond to a RINPERSOON ID in the mln instance
                second node can be anything, e.g. work IDs
 
        Returns
        -------
            None
 
        Creates a dictionary entry with key `key` into  mln.affiliation_matrix attribute with the following structure:
 
        mln.affiliation_matrix[key] = {
            'A': bipartite sparse adjacency matrix, mln.N times M
            'M' : if M is the number of unique elements in the second class
            'column_nodemap' : mapping IDs in second class to matrix column integer indices
            'column_nodemap_back' : mapping column integer indices back to second class IDs
        }
        """
        if not hasattr(self,"affiliation_matrix"):
            self.affiliation_matrix = {}
 
        self.affiliation_matrix[key] = {}
 
        unique_affiliations = set([v for k,v in affil_edgelist])
 
        M = len(unique_affiliations)
 
        self.affiliation_matrix[key]['column_nodemap_back'] = {i:elem for i,elem in enumerate(unique_affiliations)}
        self.affiliation_matrix[key]['column_nodemap'] = {elem:i for i,elem in enumerate(set([v for k,v in affil_edgelist]))}
 
        self.affiliation_matrix[key]['M'] = M
 
        # col number
        i = []
        j = []
        for k,v in affil_edgelist:
                i.append(self.nodemap[k])
                j.append(self.affiliation_matrix[key]['column_nodemap'][v])
 
        # sparse adjacency matrix for affiliations (work, school, region etc.)
        A = csr_matrix((np.ones(len(i)),(i,j)), shape=(self.N,M), dtype = 'int')
 
        self.affiliation_matrix[key]['A'] = A
 
    def downcast_to_binary(self):
        """
        Downcast all values in self.A to a binary value. Only 1s 
        occur in the resulting matrix
        
        Returns:
            -------------
                A : sparsegraph A which is a downcasted version of self.A
        """
        return self.A.astype(bool).astype(np.int16)
    
    def to_intid(self, rinpersoon_ids):
        """
        This function makes it easier to convert between CBS RINPERSOON ids
        and the numbering of the matrix rows and columns. 
 
        Parameters:
            -----------
            rinpersoon_ids : int or list, no default
                CBS ids to convert to matrix ids, can be one single integer or a list of integers
        Returns:
            -----------
            int_ids : int or list
                integer ids corresponding to input
 
        """
 
        if type(rinpersoon_ids) == int:
            return self.nodemap[rinpersoon_ids]
        else:
            return [self.nodemap[elem] for elem in rinpersoon_ids]
 
        
    def to_cbsid(self, intids):
            """
            This function makes it easier to convert between CBS RINPERSOON ids
            and the numbering of the matrix rows and columns. 
 
            Parameters:
                -----------
                intids : int or list, no default
                    integer ids to convert to CBS ids, can be one single integer or a list of integers
            Returns:
                -----------
                rinpersoon_ids : int or list
                    CBS ids corresponding to input
 
            """
 
            if type(intids) == int:
                return self.nodemap_back[intids]
            else:
                return [self.nodemap_back[elem] for elem in intids]
