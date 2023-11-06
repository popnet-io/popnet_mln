# POPNET

Library for working with large multilayer networks by the POPNET group.

Authors:
* Eszter Bokányi
* Rachel de Jong
* Bart de Zoete
* Yuliia Kazmina

Contact:
`e.bokanyi@uva.nl`

## MultiLayeredNetwork class

The `MultiLayeredNetwork` class from `src/mln.py` contains methods and attributes to work with a large
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

## Dummy data

Using the `DummyNetwork` class from `src/dummy_network.py`, it is possible to generate a simplified version of
a possible register-based multilayer social network for testing purposes. The network is based on four
generations of family relationships, and a corresponding household, school, work, and neighbor layer.

The generating rules are the following, variable names refer to dictionary keys from `config` that has to
be given upon initialization. Nodes are labelled by intergers starting from 0, edges are directed and are represented
by `(source, target, edgetype)` triplets.

1. There are four generations / agegroups: great-grandparents (0), grandparents (1), parents (2), and their children (3)
2. In generation 0, we create a fixed number of nodes (`size_first_gen`), that we pair into marriages.
3. In each subsequent generation (1,2,3), we draw the number of children for each marriage according to the values for the previous generations given in `child_probabilities`.
4. We then create marriages between the members of the current generation.
5. We generate every other family relation according to the usual rules (e.g. grandparents, cousins, in-laws etc.).
6. With probabilities given in `death_probability`, people die in each generation. This means that we remove the dead nodes and the
7. corresponding edges.
8. We then create households by assuming that grandparents (generation 1) either live alone or with their partner
and that parents live with their children (generations 2 and 3)
9. On a (0,1)x(0,1) square, we assign a random coordinate to each household.
10. If a household pair is closer than a certain threshold (`neighbor_threshold`) in this square, we make the members neighbors of each other
11. We draw random coordinates for a predefined number of schools (`number_of_chools`) in the previous square, we assign children (generation 3) to the closest school.
12. We draw random coordinates for a predefined number of workplaces in the previous square (`number_of_workplaces`) and we assign parents (generation 2) to the closest workplace.

A sample config looks like the following, dictionary keys correspond to the generations:
```
config = {
    "size_first_gen" : 50,
    "child_probabilities" : {
        0 : [0.05,0.1,0.2,0.45,0.2],
        1 : [0.05,0.2,0.3,0.4,0.05],
        2 : [0.05,0.3,0.4,0.2,0.05] 
    },
    "death_probability" : {
        0: 1,
        1: 0.1,
        2: 0.01,
        3: 0
    },
    "marriage_probabilities" : {
        0: 1,
        1: 0.9,
        2: 0.8,
        3: 0
    },
    "neighbor_threshold" : 0.05,
    "number_of_schools" : 3,
    "number_of_workplaces" : 7
}
```

For a test code and a sample figure, visit `notebook/test_dummy_network.ipynb`.
