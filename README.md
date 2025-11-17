# mlnlib

Memory-efficient Python library for working with large multilayer networks on computer with limited number of cores.

Authors:
* Eszter Bokányi
* Rachel de Jong
* Yuliia Kazmina

Contact:
`e.bokanyi@liacs.leidenuniv.nl`

The code is distributed under the MIT Licence and should be properly attributed and cited upon reuse - see LICENCE.md for details.

## MultiLayerNetwork class

The `MultiLayerNetwork` class from `src/mln.py` contains methods and attributes to work with a large
multilayer network using different edge types and layers efficiently.

The network has to be unweighted, but can be directed.

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
form 2\**i. For example, if both type i and type j edge is present
between two people, then the corresponding value in self.A would be
2\**i+2\**j. It means that we can test for a certain edgetype using
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
from mlnlib.mln import MultiLayeredNetwork

# read the whole network
popnet = MultiLayeredNetwork(
    adjacency_file = 'path_to/adjacency.npz',
    node_attribute_file = 'path_to/attributes.csv.gz'
)

# FROM MEMORY
# import custom class for the network
from mlnlib.mln import MultiLayeredNetwork

# read the whole network
popnet = MultiLayeredNetwork(
    adjacency_matrix = A, # NxN scipy.sparse.csr_matrix
    node_attribute_file = df # pd.DataFrame containing N rows in the order of the matrix
)

# FROM LIBRARY
# import custom class for the network
from mlnlib.mln import MultiLayeredNetwork

# read the whole network
popnet = MultiLayeredNetwork(
    from_library="full"
)
```
