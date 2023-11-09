"""
This script performs the data preparation steps and metadata enrichment for
a multilayer network.
"""

import pandas as pd
from scipy.sparse import csr_matrix, save_npz
import numpy as np

import json
import gc

# check whether we are in repo root, if not, change working directory
import os
if os.getcwd().endswith("src"):
    os.chdir("..")

print(f"We are currently working in the {os.getcwd()} directory.")

class RawCSVtoMLN:
    """
    This class converts CSV nodelist and edgelist to fast readable input for the MLN class.

    It accepts an arbitrary number of nodelist and edgelist CSV input files, reads through them,
    and merges the node properties, and converts the edgelists to a scipy.sparse adjacency matrix
    with integer elements that have a binary encoding for the layers.

    Every input CSV is assumed to have a header and no index col.
    """
    def __init__(
            self,
            node_conf = dict(
                input_folder_prefix = "",
                files = [],
                colmap = "",
                sep = ";",
                main_file = 0,
                output = ""
            ),
            edge_conf = dict(
                input_folder_prefix = "",
                files = [],
                colmap = "",
                sep = ";",
                output = ""
            ),
            layer_conf = dict(
                input_folder_prefix = "",
                raw_file = "",
                file = "",
                output = "",
                symmetrize = [],
                symmetrize_all = False,
                raw_sep=",",
                sep = ",",
                colors = ""
            ),
            **kwargs
        ):
        self.node_conf = node_conf
        self.edge_conf = edge_conf
        self.layer_conf = layer_conf
        for (k, v) in kwargs.items():
         setattr(self, k, v)

    ###########################
    ##### LAYERS    ###########
    ###########################
        
    def init_layers(self):
        """
        Either read layer file, or create rich layer dataframe from bare minimum input.
        """
        # if the layer file is not yet prepared
        if self.layer_conf["file"] == "":
            print("Trying to create enriched layer dataframe...")
            # there should at least be a very basic layer file (e.g. prepared from the edgelists, if nothing else)
            if self.layer_conf["raw_file"]=="":
                self.init_raw_layers_from_edges()
            else:
                print("\tReading raw layer input file...")
                self.layers = pd.read_csv(
                    self.layer_conf["raw_file"],
                    index_col=None,
                    header=0,
                    sep=self.layer_conf["raw_sep"]
                )
            # if we want to rename columns, there's either a JSON or a dict
            if self.layer_conf["colmap"] != "":
                print("\tRenaming layer dataframe columns...")
                if type(self.layer_conf["colmap"]) == str:
                    self.layer_conf["colmap"] = json.load(open(self.layer_conf["colmap"]))
                self.layers.rename(columns = self.layer_conf["colmap"])

            print("\tAdding binary representation, groups and long labels...")
            # creating different 2**i numbers for all linktypes for binary encoding
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
            self.layers.to_csv(self.layer_conf["output"],index=False,header=True)
            self.layer_conf["file"] = self.layer_conf["output"]
        
        # print("Done.")
        # print("Reading " + self.layer_conf["file"] + "...")
        # self.layers = pd.read_csv(self.layer_conf["file"], index_col = None, header=0)
        # print("Layer dataframe",self.laers.head(),self.layers.columns,end="\n")

    def init_raw_layers_from_edges(self):
        """
        This function reads the raw edgelist file and creates a layer dataframe
        from the different linktypes.
        """
        # getting all layer types
        layers = self.edgelist["layer"].unique()
        # creating layer dataframe
        self.layers = pd.DataFrame({"label": layers, "layer": layers})

    ###########################
    ##### NODES    ############
    ###########################

    # Loading node attributes files
    # =============================
    def init_nodes(self):
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
                node_df = pd.read_csv(p,sep=self.node_conf["sep"],index_col=None,compression='gzip')#,nrows=10)
            else:
                node_df = pd.read_csv(p,sep=self.node_conf["sep"],index_col=None)#,nrows=10)
            if self.node_conf["colmap"] != "":
                for k,v in self.node_conf["colmap"].items():
                    if k in node_df and v == None:
                        node_df.drop(k,axis=1,inplace=True)
                node_df.rename(columns = self.node_conf["colmap"],inplace=True)
            print(node_df.head())
            node_df.set_index("label",inplace=True)
            return node_df
        
        if self.node_conf["files"]!="":
            print("Merging all node files...")
            nodes = pd.concat([get_node_file(f) for f in self.node_conf["files"]], axis=1)     

            # make label a column
            nodes.reset_index(inplace=True) 
            # add ids
            nodes.reset_index(inplace=True)
            # renaming columns to human readable and consistency
            nodes.rename(columns={"index":"id"},inplace=True)

            self.nodes = nodes
            print("Initialized node dataframe.")
        else:
            source = self.edgelist["source"].unique()
            target = self.edgelist["target"].unique()
            nodes = pd.DataFrame({"label": np.unique(np.concatenate((source,target)))})
            nodes.reset_index(inplace=True)
            nodes.rename(columns={"index":"id"},inplace=True)
            self.nodes = nodes

    ###########################
    ##### EDGES    ############
    ###########################

    def read_edgelist(self):
        """
        This function reads the edgelist file and creates a pandas.DataFrame
        with the edgelist.

        It is not memory efficient if the edgelists are very large, it reads everything.
        """

        if self.edge_conf["colmap"] != "":
            if type(self.edge_conf["colmap"]) == str:
                self.edge_conf["colmap"] = json.load(open(self.edge_conf["colmap"]))

        if self.edge_conf["input_folder_prefix"] != "":
            if not self.edge_conf["input_folder_prefix"].endswith("/"):
                self.edge_conf["input_folder_prefix"] += "/"
                for i in range(len(self.edge_conf["files"])):
                    self.edge_conf["files"][i] = self.edge_conf["input_folder_prefix"] + self.edge_conf["files"][i]

        edgelists = []

        for ef in self.edge_conf["files"]:
            print(f"Reading edge file {ef}...")
            # loading file
            print("\tLoading file...")
            edgelist = pd.read_csv(ef, sep=self.edge_conf["sep"],header=0)#,nrows=1000000)
            if self.edge_conf["colmap"]!="":
                for k,v in self.edge_conf["colmap"].items():
                    if v == None and k in edgelist.columns:
                        edgelist.drop(k,inplace=True,axis=1)
                    else:
                        edgelist.rename(columns = {k:v},inplace=True)
            edgelists.append(edgelist)
            print("\tDone.")

        self.edgelist = pd.concat(edgelists,ignore_index=True)

    def init_edges(self):
        # getting id <-> label mappings
        self.nodemap_back = dict(zip(self.nodes["id"], self.nodes['label']))
        self.nodemap = {v:k for k,v in self.nodemap_back.items()}

        self.N = len(self.nodemap.keys())
        print(f"N is {self.N}")
        self.A = csr_matrix((self.N,self.N), dtype='int')
    
    def adjacency_matrix(self, edgelist, binary, symmetrize=False):
        """
        This function creates the adjacency matrix representation of a graph
        based on a pandas.DataFrame edgelist. The edgelist should be a plain array.

        Returns
        -------

        A : scipy.sparse.csr_matrix
            sparse adjacency matrix
        """
        # remapping labels to integer ids from 0 to N-1 to load into sparse CSR matrix
        print("\t\tMapping edgelist.")
        i = edgelist["source"].map(self.nodemap).tolist()
        j = edgelist["target"].map(self.nodemap).tolist()

        # if creating / enforcing undirected graph
        if symmetrize:
            print("\t\tSymmetrizing connections...")
            sym_i = i+j
            sym_j = j+i
            i = sym_i
            del sym_i
            j = sym_j
            del sym_j
        
        print("\t\tCreating adjacency matrix.")
        A = csr_matrix((np.ones(len(i)),(i,j)), shape=(self.N,self.N), dtype =  'int')
        if symmetrize:
            A = csr_matrix(A>0, dtype='int')
        #  we multiply to 1/0 matrix by that number so that it corresponds to a certain edgetype
        A *= binary

        return A

    def read_all_edges(self):
        """
        This function loads all edgelists for the
        different linktypes of the different layers, and subsequently
        adds 2**i to the adjacency matrix for the edges, where 2**i corresponds
        to the type of the edge from self.layers["binary"].
        In the end, the function saves the scipy.sparse.csr_matrix type
        adjacency matrix to the given location
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

        for layer in self.layers["layer"]:
            # linktype name
            name = self.layers.set_index("layer").loc[layer]["label_long"]
            print(f"\tReading layer {name} with code {layer}...")
            binary = self.layers.set_index("layer").loc[layer]["binary"]
            selection = self.edgelist["layer"]==layer
            num_edges = (selection).sum()
            print(f"\tStarting adjacency function for {num_edges} edges...")
            # contains values of binary_linktype / 0
            if num_edges>0:
                A = self.adjacency_matrix(self.edgelist[selection], binary, symmetrize=layer in self.edge_conf["symmetrize"])
                self.A += A
                print(f"Adding {A.nnz} edges.")
                # cleaning memory of large unnecessary objects
                del A
                gc.collect()
            print("\tDone.")
        gc.collect()

    def init_all(self):
        self.read_edgelist()
        self.init_layers()
        print(f"LAYERS: {self.layers.shape[0]}")
        self.init_nodes()
        print(f"NODES: {self.nodes.shape[0]}")
        self.init_edges()
        print(f"EDGES: {self.edgelist.shape[0]}")
        self.read_all_edges()
        print(f"EDGES: {self.A.nnz}")

        if "save" in self.__dict__:
            if self.save:
                self.save_all()

    def save_layer_df(self, output):
        print("Saving layer dataframe...")
        self.layers.to_csv(output,index=False,header=True)
        print("Done.")
    
    def save_node_df(self, output):
        print("Saving node dataframe...")
        self.nodes.to_csv(output,index=False,header=True,compression="gzip")
        print("Done.")

    def save_edge_npz(self, output):
        print("Saving edge adjacency matrix...")
        save_npz(output, self.A)
        print("Done.")

    def save_all(self, output_folder = "", overwrite = False):
        # check if output_folder is given and give warning if not
        if output_folder == "":
            print("WARNING: No output folder given, saving to current directory.")
        # check if output folder has an ending / and add it if not
        if not output_folder.endswith("/"):
            output_folder += "/"
        # check if output folder exists, if not, create it
        if not os.path.isdir(output_folder):
            os.mkdir(output_folder)


        print("Writing node, layer and edge files...")

        if self.layer_conf["output"]=="": self.layer_conf["output"] = "layers.csv"
        # check if the above file exists, if overwrite=False, raise error
        if os.path.isfile(output_folder + self.layer_conf["output"]) and not overwrite:
            raise ValueError("Layer file already exists, please specify overwrite=True in config file.")
        print("Writing layer file to " + output_folder + self.layer_conf["output"] + "...")
        self.save_layer_df(output_folder + self.layer_conf["output"])
        print("Done")
        
        if self.node_conf["output"]=="": self.node_conf["output"] = "nodes.csv.gz"
        # check if the above file exists, if overwrite=False, raise error
        if os.path.isfile(output_folder + self.node_conf["output"]) and not overwrite:
            raise ValueError("Node file already exists, please specify overwrite=True in config file.")
        print("Writing node file to " + output_folder + self.node_conf["output"] + "...")
        self.save_node_df(output_folder + self.node_conf["output"])
        print("Done.")
        
        if self.edge_conf["output"]=="": self.edge_conf["output"] = "edges.npz"
        # check if the above file exists, if overwrite=False, raise error
        if os.path.isfile(output_folder + self.edge_conf["output"]) and not overwrite:
            raise ValueError("Edge file already exists, please specify overwrite=True in config file.")
        print("Writing edge file to " + output_folder + self.edge_conf["output"] + "...")
        self.save_edge_npz(output_folder + self.edge_conf["output"])
        print("Done.")

