"""
Author: Eszter Bokanyi, 2023
E-mail: e.bokanyi@uva.nl

This file is created in the context of the POPNET project:
https://popnet.io

This script is able to prepare a multilayer network suitable for reading
with mln.py from a very general set of CSV files.

It accepts an arbitrary number of nodelist and edgelist CSV input files, reads through them,
and merges the node properties, and converts the edgelists to a scipy.sparse adjacency matrix.
The conversion happens through a layer.csv file, that contains the different layers and their
binary representation. If no nodelist or layer file is given, these are inferred from the edgelist.

Every input CSV is assumed to have a header and no index col.

The config parameter colmap is a JSON file or a dict that maps the column names of the input files
to the column names of the output files. If the map is empty ("none"), the column is dropped.

The main node file is used to determine the node set, and the other node files are merged into it.

The symmetrize parameters control whether the edgelists are symmetrized. If symmetrize_all is True,
all layers are symmetrized, thus, the network is undirected. 
Otherwise only the layers listed in symmetrize are symmetrized.

The config file is a JSON file with the following structure:
    
    {
        "node_conf": {
            "input_folder_prefix": "",
            "files": [],
            "colmap": "",
            "sep": ";",
            "main_file": 0,
            "output": ""
        },
        "edge_conf": {
            "input_folder_prefix": "",
            "files": [],
            "colmap": "",
            "sep": ";",
            "output": ""
        },
        "layer_conf": {
            "raw_file": "",
            "file": "",
            "output": "",
            "symmetrize": [],
            "symmetrize_all": False,
            "raw_sep":",",
            "sep": ",",
            "colors": ""
        },
        "output_folder": ""
    }

The resulting files are saved to the output folder, if given, otherwise to the current directory:
    * nodes.csv.gz: node dataframe
    * edges.npz: adjacency matrix
    * layers.csv: layer dataframe
"""

from typing import Dict, List, Any, Optional
import pandas as pd
from scipy.sparse import csr_matrix, save_npz
import numpy as np

import json
import gc
from time import sleep, time
import sys

# check whether we are in repo root, if not, change working directory
import os
if os.getcwd().endswith("src"):
    os.chdir("..")

print(f"We are currently working in the {os.getcwd()} directory.")

class RawCSVtoMLN:
    """
    Convert raw CSV node/edge inputs into an MLN library on disk.

    This utility builds the three core artifacts consumed by MultiLayerNetwork:
    - nodes.csv.gz (node attributes with at least 'label' and 'id')
    - edges.npz (NxN csr_matrix with binary-encoded layers)
    - layers.csv (layer definitions with 'layer', 'label', 'binary', and optional 'group')

    It accepts an arbitrary number of node and edge CSV files, merges node properties,
    and converts edge lists into a sparse adjacency matrix with integer data that encodes
    layers as powers of two (binary encoding).

    Assumptions
    -----------
    - All input CSVs have a header and no index column
    - Column mappings can be provided via JSON or dict in the config
    """
    def __init__(
            self,
            node_conf: Dict[str, Any] = dict(
                input_folder_prefix = "",
                files = [],
                colmap = "",
                sep = ";",
                main_file = 0,
                output = ""
            ),
            edge_conf: Dict[str, Any] = dict(
                input_folder_prefix = "",
                files = [],
                colmap = "",
                sep = ";",
                output = ""
            ),
            layer_conf: Dict[str, Any] = dict(
                raw_file = "",
                file = "",
                output = "",
                symmetrize = [],
                symmetrize_all = False,
                raw_sep=",",
                sep = ",",
                colors = ""
            ),
            output_folder: str = "",
            **kwargs: Any
        ) -> None:

        self.node_conf = node_conf
        self.edge_conf = edge_conf
        self.layer_conf = layer_conf
        self.output_folder = output_folder
        self.grouped = False


        # check if output folder is empty, if not, add / to end
        if self.output_folder != "":
            if not self.output_folder.endswith("/"):
                self.output_folder += "/"
        
        # other kwargs
        for (k, v) in kwargs.items():
         setattr(self, k, v)

    ###########################
    ##### LAYERS    ###########
    ###########################

    def _open_file(self, read_func, *args, **kwargs):
        success = 0
        iter = 0
        while not success and iter < 5:
            try:
                result = read_func(*args,**kwargs)
                success = 1
            except FileNotFoundError:
                iter +=1
        if success:
            print(f"\tRead file on attempt no. {iter+1}.")
        else:
            raise FileNotFoundError("Could not read file in 5 attempts.")
        return result
            
        
    def init_layers(self) -> None:
        """
        Initialize the layer dataframe.

        If a prepared layer file is provided in the config, it is read directly.
        Otherwise, a minimal layer definition is inferred or read from a raw file,
        then enriched with binary encodings, groups, and long labels.

        Side effects
        ------------
        - Sets self.layers (pd.DataFrame)
        - May create self.layers_ungrouped and self.layer_mapping if grouped
        - Persists the resulting CSV to layer_conf["output"] in output_folder
        """

        # if the layer file is not yet prepared
        if self.layer_conf["file"] == "":
            print("Trying to create enriched layer dataframe...")
            # there should at least be a very basic layer file (e.g. prepared from the edgelists, if nothing else)
            if self.layer_conf["raw_file"]=="":
                self.init_raw_layers_from_edges()
            else:
                print("\tReading raw layer input file...")
                self.layers = self._open_file(
                    pd.read_csv,
                    self.layer_conf["input_folder_prefix"] + self.layer_conf["raw_file"],
                    index_col=None,
                    header=0,
                    sep=self.layer_conf["raw_sep"]
                )
            # if we want to rename columns, there's either a JSON or a dict
            if self.layer_conf["colmap"] != "":
                print("\tRenaming layer dataframe columns...")
                if type(self.layer_conf["colmap"]) == str:
                    self.layer_conf["colmap"] = json.load(open(self.layer_conf["colmap"]))
                print(json.dumps(self.layer_conf["colmap"],indent="\t\t"))
                self.layers.rename(columns = self.layer_conf["colmap"], inplace = True)

            print("\tAdding binary representation, groups and long labels...")
            if self.grouped:
                grps = dict(zip(self.layers["group_layer"],self.layers["group_label"]))
                self.layers_ungrouped = self.layers
                self.layer_mapping = dict(zip(self.layers_ungrouped["layer"],self.layers_ungrouped["group_layer"]))
                self.layers = pd.DataFrame(np.array([list(grps.keys()),list(grps.values())]).T,columns = ["layer","label"])
            # creating different 2**i numbers for all layers for binary encoding
            self.layers["binary"] = self.layers.index.map(lambda i: int(2**i))
            if "group" not in self.layers:
                self.layers["group"] = self.layers["label"]
            # creating long labels for layers for visualizations etc.
            self.layers["label_long"] = self.layers.apply(lambda row: row['group']+": "+row["label"], axis=1)
            # layer_group -> integer representations: summing up all binary numbers corresponding to a certain group
            self.layer_group_binary = self.layers.groupby("group").sum()["binary"].map(int).to_dict()

            # adding group colors
            if self.layer_conf["colors"]!="":
                print("\tAdding colors...")
                if type(self.layer_conf["colors"]) == str:
                    self.layer_conf["colors"] = json.load(open(self.layer_conf["colors"]))
                # print(json.dumps(self.layer_conf["colors"],indent="\t\t"))
                self.layers["color"] = self.layers["group"].map(self.layer_conf["colors"])
            # exporting
            self.layer_conf["file"] = self.output_folder + self.layer_conf["output"]
            self.layers.to_csv(self.layer_conf["file"],index=False,header=True)
        
        print("Done.")
        print("Reading " + self.layer_conf["file"] + "...")
        self.layers = pd.read_csv(self.layer_conf["file"], index_col = None, header=0)
        print("Layer dataframe",self.layers.head(),self.layers.columns,end="\n")

    def init_raw_layers_from_edges(self) -> None:
        """
        Initialize a minimal layer dataframe from raw edge data.

        This reads the raw edgelist to detect distinct layer identifiers and
        constructs a basic dataframe with 'layer' and 'label'.
        """
        raise ValueError("To initialize layers from edgelist is not yet done.")
        # getting all layer types
        layers = self.edgelist["layer"].unique()
        # creating layer dataframe
        self.layers = pd.DataFrame({"label": layers, "layer": layers})

    ###########################
    ##### NODES    ############
    ###########################

    # Loading node attributes files
    # =============================
    def init_nodes(self) -> None:
        """
        Build the node attribute dataframe from configured node files.

        Reads and merges the configured node files, applies column mappings,
        and ensures an 'id' column exists (0..N-1). The resulting dataframe
        is assigned to self.nodes.

        Raises
        ------
        ValueError
            If no node files are configured.
        """
        print("Creating merged node attribute file...")

        if self.node_conf["colmap"] != "":
            if type(self.node_conf["colmap"]) == str:
                self.node_conf["colmap"] = json.load(open(self.node_conf["colmap"]))
        
        def get_node_file(p):
            # add input folder prefix from node_conf if given
            if self.node_conf["input_folder_prefix"] != "":
                if not self.node_conf["input_folder_prefix"].endswith("/"):
                    self.node_conf["input_folder_prefix"] += "/"
                p = self.node_conf["input_folder_prefix"] + p

            print(f"Reading file {p}...")
            if p.endswith('gz'):
                node_df = self._open_file(pd.read_csv,p,sep=self.node_conf["sep"],index_col=None,compression='gzip')#,nrows=10)
            elif p.endswith('sav'):
                node_df = self._open_file(pd.read_spss,p,usecols = self.node_conf["colmap"].keys(),convert_categoricals=False)
            else:
                node_df = self._open_file(pd.read_csv,p,sep=self.node_conf["sep"],index_col=None)#,nrows=10)
            if self.node_conf["colmap"] != "":
                for c in node_df.columns:
                    if c not in self.node_conf["colmap"]:
                        node_df.drop(c,axis=1,inplace=True)
                node_df.rename(columns = self.node_conf["colmap"],inplace=True)
            print(node_df.head())
            node_df.set_index("label",inplace=True)
            return node_df
        
        if self.node_conf["files"]!="":
            print("Merging all node files...")
            # TODO: check for joins, we might lose some lines here!
            nodes = pd.concat([get_node_file(f) for f in self.node_conf["files"]], axis=1)     

            # make label a column
            nodes.reset_index(inplace=True) 
            if "id" not in nodes.columns:
                # add ids if they don't exist
                nodes.reset_index(inplace=True)
                # renaming columns to human readable and consistency
                nodes.rename(columns={"index":"id"},inplace=True)

            self.nodes = nodes
            print("Initialized node dataframe.")
        else:
            raise ValueError("To initialize nodelist from edgelist is not yet done.")

    ###########################
    ##### EDGES    ############
    ###########################

    def init_edges(self) -> None:
        """
        Initialize empty adjacency and node mappings for edge loading.

        Side effects
        ------------
        - Sets self.nodemap and self.nodemap_back
        - Sets self.N (number of nodes)
        - Initializes self.A as an NxN csr_matrix with dtype uint64
        """
        # getting id <-> label mappings
        self.nodemap_back = dict(zip(self.nodes["id"], self.nodes['label']))
        self.nodemap = {v:k for k,v in self.nodemap_back.items()}

        self.N = len(self.nodemap.keys())
        print(f"N is {self.N}")
        self.A = csr_matrix((self.N,self.N), dtype=np.uint64)
    
    def adjacency_matrix(self, edgelist: pd.DataFrame, binary: int, symmetrize: bool = False) -> csr_matrix:
        """
        Construct a sparse adjacency matrix from an edgelist chunk.

        Parameters
        ----------
        edgelist : pd.DataFrame
            DataFrame with columns 'source' and 'target' (node labels).
        binary : int
            Power-of-two value corresponding to the current layer.
        symmetrize : bool, default False
            If True, enforce undirected edges by mirroring (i,j) and (j,i).

        Returns
        -------
        scipy.sparse.csr_matrix
            CSR adjacency matrix with data equal to `binary` for present edges.
        """
        # remapping labels to integer ids from 0 to N-1 to load into sparse CSR matrix
        i = pd.Series(map(lambda x: self.nodemap.get(x),edgelist["source"]))
        j = pd.Series(map(lambda x: self.nodemap.get(x),edgelist["target"]))

        in_nodelist = ~pd.isnull(i)&~pd.isnull(j)
        i = i[in_nodelist]
        j = j[in_nodelist]

        # if creating / enforcing undirected graph
        if symmetrize:
            sym_i = i+j
            sym_j = j+i
            i = sym_i
            del sym_i
            j = sym_j
            del sym_j
        
        A = csr_matrix((np.ones(len(i)),(i,j)), shape=(self.N,self.N), dtype = np.uint64)
        if symmetrize:
            A = csr_matrix(A>0, dtype=np.uint64)
        #  we multiply to 1/0 matrix by that number so that it corresponds to a certain edgetype
        A *= int(binary)

        return A
    
    def _print_size(self,var,name=""):
        """
        Prints variable size in GB to allow for memory control.
        """
        print(f"{name}: {round(sys.getsizeof(var)/1024**3,2)}GB")

    def read_all_edges(self) -> None:
        """
        Stream and accumulate all edges across configured files into self.A.

        For each layer, chunks of the corresponding edge file(s) are read,
        mapped to node IDs, converted to CSR blocks with the layer's binary
        code, optionally symmetrized, and added to the full adjacency matrix.

        Notes
        -----
        - If layer_conf["symmetrize_all"] is True, all layers are treated as undirected.
        - Column mappings in edge_conf["colmap"] are applied per chunk for memory efficiency.
        """

        # check if symmetrize_all in layer_conf is str
        if type(self.layer_conf["symmetrize_all"]) == str:
            # if "true" or "false", set to boolean
            if self.layer_conf["symmetrize_all"].lower() == "true":
                self.layer_conf["symmetrize_all"] = True
            elif self.layer_conf["symmetrize_all"].lower() == "false":
                self.layer_conf["symmetrize_all"] = False
            else:
                raise ValueError("symmetrize_all in layer_conf is not a boolean value.")
        
        # if symmetrize_all is True, add all layers to symmetrize list
        if self.layer_conf["symmetrize_all"]:
            self.layer_conf["symmetrize"] = self.layers["layer"].tolist()

        if self.grouped:
            l_list = self.layers_ungrouped["layer"]
        else:
            l_list = self.layers["layer"]

        # get chunks of the edgelist
        if self.edge_conf["colmap"] != "":
            if type(self.edge_conf["colmap"]) == str:
                self.edge_conf["colmap"] = json.load(open(self.edge_conf["colmap"]))

        if self.edge_conf["input_folder_prefix"] != "":
            if not self.edge_conf["input_folder_prefix"].endswith("/"):
                self.edge_conf["input_folder_prefix"] += "/"
                for i in range(len(self.edge_conf["files"])):
                    self.edge_conf["files"][i] = self.edge_conf["input_folder_prefix"] + self.edge_conf["files"][i]

        # START HEAVY PART
        chunksize = int(2e8)

        # read all edge files
        for ef in self.edge_conf["files"]:
            print(f"Reading edge file {ef}...")
            # loading file in chunks
            print("\tLoading file in chunks...")
            tic  = time()
            iter = 0
            with self._open_file(pd.read_csv,ef, sep=self.edge_conf["sep"],header=0,chunksize=chunksize) as reader:
                # for all chunks
                for chunk in reader:
                    iter += 1
                    self._print_size(chunk,"\tSize of current chunk")
                    print(f"\tChunk no. {iter}")
                    if self.edge_conf["colmap"]!="":
                        for k,v in self.edge_conf["colmap"].items():
                            if v == None and k in chunk.columns:
                                chunk.drop(k,inplace=True,axis=1)
                            else:
                                chunk.rename(columns = {k:v},inplace=True)

                    # make sure node labels are integers for the mapping to work
                    if chunk["source"].dtype!=int:
                        chunk["source"] = chunk["source"].map(int)
                    if chunk["target"].dtype!=int:
                        chunk["target"] = chunk["target"].map(int)

                    # check for layers in chunk and add them to adjacency matrix
                    for l in l_list:
                        if not self.grouped:
                            layer = l
                        else:
                            layer = self.layer_mapping[l]
                        # layer name
                        name = self.layers.set_index("layer").loc[layer]["label_long"]
                        binary = self.layers.set_index("layer").loc[layer]["binary"]
                        selection = chunk["layer"]==l
                        num_edges = (selection).sum()
                        # contains values of binary_layer / 0
                        if num_edges>0:
                            A = self.adjacency_matrix(chunk[selection], binary, symmetrize=layer in self.layer_conf["symmetrize"])
                            print(f"\tAdding {A.nnz} edges to layer {layer}.")
                            self.A += A
                            print("\tNumber of nonzero edges in full adjacency matrix:",self.A.nnz)
                            #temp = dict(zip(zip(*A.nonzero()),A.data))
                            #print("temp",temp)
                            #self.A_dict = {
                            #    key: self.A_dict.get(key,0) | temp.get(key,0) for key in set(self.A_dict.keys()) | temp.keys()
                            #}
                            #print("A_dict",self.A_dict)
                            # cleaning memory of large unnecessary objects
                            del A#,temp
            toc = time()
            print(f"\tFinished with chunk in {round(toc-tic,1)}s.")
            tic = toc

        #data = list(self.A_dict.values())
        #i = [elem[0] for elem in self.A_dict.keys()]
        #j = [elem[1] for elem in self.A_dict.keys()]
        #self.A = csr_matrix((data,(i,j)),shape=(self.N,self.N),dtype='int8')
        #del self.A_dict,i,j,data


    def init_all(self) -> None:
        """
        Execute the full preparation pipeline based on the config.

        Steps: init_layers() → init_nodes() → init_edges() → read_all_edges().
        If the 'save' flag is set on the instance, save_all() is called at the end.
        """
        self.init_layers()
        print(f"LAYERS: {self.layers.shape[0]}")
        self.init_nodes()
        print(f"NODES: {self.nodes.shape[0]}")
        self.init_edges()
        self.read_all_edges()
        print(f"EDGES: {self.A.nnz}")

        if "save" in self.__dict__:
            if self.save:
                self.save_all()

    def save_layer_df(self, output: str) -> None:
        """
        Save the layer dataframe to CSV.

        Parameters
        ----------
        output : str
            Output CSV path.
        """
        print("Saving layer dataframe...")
        self.layers.to_csv(output,index=False,header=True)
        print("Done.")
    
    def save_node_df(self, output: str) -> None:
        """
        Save the node dataframe to compressed CSV (gzip).

        Parameters
        ----------
        output : str
            Output CSV.GZ path.
        """
        print("Saving node dataframe...")
        self.nodes.to_csv(output,index=False,header=True,compression="gzip")
        print("Done.")

    def save_edge_npz(self, output: str) -> None:
        """
        Save the adjacency matrix to NPZ.

        Parameters
        ----------
        output : str
            Output NPZ path.
        """
        print("Saving edge adjacency matrix...")
        save_npz(output, self.A)
        print("Done.")

    def save_all(self, overwrite: bool = False) -> None:
        """
        Save layers, nodes, and edges to the configured output folder.

        Parameters
        ----------
        overwrite : bool, default False
            If False, refuse to overwrite existing output files.
        """
        # if overwrite in attributes, set overwrite
        if "overwrite" in self.__dict__:
            overwrite = self.overwrite
        # check if output_folder is given and give warning if not
        if self.output_folder== "":
            print("WARNING: No output folder given, saving to current directory.")
        else:
            # check if output folder has an ending / and add it if not
            if not self.output_folder.endswith("/"):
                self.output_folder += "/"
            # check if output folder exists, if not, create it
            if not os.path.isdir(self.output_folder):
                os.mkdir(self.output_folder)


        print("Writing node, layer and edge files...")

        if self.layer_conf["output"]=="": self.layer_conf["output"] = "layers.csv"
        # check if the above file exists, if overwrite=False, raise error
        if os.path.isfile(self.output_folder + self.layer_conf["output"]) and not overwrite:
            raise ValueError("Layer file already exists, please specify overwrite=True in config file.")
        print("Writing layer file to " + self.output_folder + self.layer_conf["output"] + "...")
        self.save_layer_df(self.output_folder + self.layer_conf["output"])
        print("Done")
        
        if self.node_conf["output"]=="": self.node_conf["output"] = "nodes.csv.gz"
        # check if the above file exists, if overwrite=False, raise error
        if os.path.isfile(self.output_folder + self.node_conf["output"]) and not overwrite:
            raise ValueError("Node file already exists, please specify overwrite=True in config file.")
        print("Writing node file to " + self.output_folder + self.node_conf["output"] + "...")
        self.save_node_df(self.output_folder + self.node_conf["output"])
        print("Done.")
        
        if self.edge_conf["output"]=="": self.edge_conf["output"] = "edges.npz"
        # check if the above file exists, if overwrite=False, raise error
        if os.path.isfile(self.output_folder + self.edge_conf["output"]) and not overwrite:
            raise ValueError("Edge file already exists, please specify overwrite=True in config file.")
        print("Writing edge file to " + self.output_folder + self.edge_conf["output"] + "...")
        self.save_edge_npz(self.output_folder + self.edge_conf["output"])
        print("Done.")

