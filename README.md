# mlnlib

Memory-efficient Python tools for working with large multilayer networks on machines with limited memory/cores.

Authors:
* Eszter Bokányi
* Rachel de Jong
* Yuliia Kazmina

Contact: `e.bokanyi@liacs.leidenuniv.nl`

License: MIT (see `LICENSE.md`). Please attribute appropriately when reusing.

## Overview

`mlnlib` provides the `MultiLayerNetwork` class for building, loading, manipulating, and exporting multilayer networks where edges can exist on multiple layers. Layers are encoded efficiently in a single sparse adjacency matrix using a binary encoding (powers of two).

Key components stored in a `MultiLayerNetwork` instance:
- `nodes`: pandas DataFrame (or polars when enabled) with at least columns `label` and `id`
- `A`: scipy.sparse.csr_matrix with binary layer encoding in its data
- `layers`: pandas DataFrame with at least columns `layer`, `label`, and `binary`; optionally `group`

## Install

From PyPI:

```bash
pip install mlnlib
```

## Loading data

There are three supported ways to initialize a `MultiLayerNetwork` (class defined in `mlnlib/mln.py`).

1) Load a prepared library from disk

Folder layout must contain at least:
- `edges.npz` — NxN csr_matrix with binary encoded layers
- `nodes.csv.gz` or `nodes_{pd.__version__}.pkl` — nodes DataFrame with `label` and `id`
- `layers.csv` — layer definitions with `layer`, `label`, `binary` (and optional `group`)

```python
from mlnlib import MultiLayerNetwork

mln = MultiLayerNetwork(
        load_from_library=True,
        library_path="/path/to/library"
)
```

2) Build from raw CSVs via a config

Uses `RawCSVtoMLN` behind the scenes (see `src/mlnlib/preparation.py` and `test/config.json` for an example schema).

```python
from mlnlib import MultiLayerNetwork

mln = MultiLayerNetwork(
        load_from_config=True,
        config_path="config.json"
)
```

3) Provide in-memory objects

```python
from mlnlib import MultiLayerNetwork
from scipy.sparse import csr_matrix
import pandas as pd

# nodes must have columns: label, id (IDs are 0..N-1)
nodes = pd.DataFrame({
        "label": ["a","b","c"],
        "id": [0,1,2]
})

# layers dataframe: at least layer, label, binary
layers = pd.DataFrame({
        "layer": [0,1],
        "label": ["friendship","kinship"],
        "binary": [1,2]
})

# A is NxN csr_matrix whose data encodes layers as sums of powers of two
A = csr_matrix((3,3), dtype="int64")

mln = MultiLayerNetwork(nodes=nodes, edges=A, layers=layers)
```

## Binary layer encoding

Each layer is assigned a power-of-two value: 1, 2, 4, 8, ... When an edge exists on multiple layers, the corresponding entry in `A` stores the sum, enabling fast bitwise checks. For example, if `A[i,j] = 7`, then layers 0, 1, and 2 are present because `7 = 1 + 2 + 4`. Checking for layer 1 is a cheap mask: `7 & 2 == 2`.

Binary encoding is stored in `self.layers`. 

## Common tasks

- Convert node identifiers
    - `mln.to_id("person_123")` or `mln.to_label(42)`
- Filter by nodes/layers/groups
    - `mln.get_filtered_network(nodes_selected=[...], layers_selected=[...], groups_selected=[...], node_type="label", layer_type="label")`
- Per-layer adjacency
    - `mln.get_layer_adjacency_matrix(layer="friendship", layer_type="label", store=True, dtype="int8")`
- Supra-adjacency matrix
    - `mln.get_supra_adjacency_matrix(dtype="int8")`
- Edgelist export
    - `mln.get_edgelist(edge_attribute="label")`  # options: "binary", "layer", "label", "weight", or None
- Degrees and clustering
    - `mln.get_degrees([...])`, `mln.get_clustering_coefficient([...])`
- Excess closure
    - `mln.get_excess_closure(selected_nodes=[...], selected_layers=[...], layer_type="label")`
- Convert to graph libraries
    - `mln.to_igraph(directed=True, edge_attributes=True, node_attributes=False, edge_attribute_type="binary")`
    - `mln.to_networkx(directed=True, edge_attributes=True, node_attributes=False, edge_attribute_type="binary")`

## Saving and exporting

Save a prepared library for fast future loads:

```python
mln.save(path="/path/to/library", overwrite=False)
# creates edges.npz, nodes.csv.gz (or pickle), layers.csv
```

Export individual pieces:
- Edges: `mln.export_edges("edges.npz" | "edges.csv" | "edges.csv.gz")`
- Nodes: `mln.export_nodes("nodes.csv" | "nodes.csv.gz")`
- Layers: `mln.export_layers("layers.csv")`
- GraphML: `mln.save_to_graphml("network.graphml", edge_attribute_type="binary")`

## Data structures

- Nodes (`mln.nodes`): DataFrame with at least `label` and `id`. You may add arbitrary node attributes. Preferred mapping helpers are `to_id()` and `to_label()`.
- Layers (`mln.layers`): DataFrame with columns:
    - `layer`: arbitrary ID, can be of any type
    - `label`: human-readable name
    - `label_long` : longer description
    - `binary`: power-of-two encoding
    - `group` (optional): used for grouped aggregations
- Adjacency (`mln.A`): csr_matrix whose data contains the sum of layer powers-of-two. Use `mln.get_binary_adjacency()` to collapse to 0/1.

## API Reference

### Core Methods

#### Node ID mapping (preferred interface)
- `to_id(labels: Any | List[Any]) -> int | List[int]`  
  Convert node label(s) to integer ID(s).

- `to_label(ids: int | List[int]) -> Any | List[Any]`  
  Convert integer node ID(s) to label(s).

#### Filtering and subgraphs
- `get_filtered_network(nodes_selected=None, layers_selected=[], groups_selected=[], node_type="label", layer_type="label", keep_node_alignment=False) -> MultiLayerNetwork`  
  Return a filtered subnetwork based on node and/or layer selection.

- `get_egonetwork(ego_label, depth=1, return_list=False, ignore_limit=False) -> MultiLayerNetwork | List`  
  Extract ego network centered on a node at specified depth.

#### Layer-specific adjacency
- `get_layer_adjacency_matrix(layer, layer_type="layer", store=False, dtype="int64") -> csr_matrix`  
  Create binary adjacency matrix for a single layer or group.

- `get_binary_adjacency(dtype="int64") -> csr_matrix`  
  Convert multilayer adjacency to binary (0/1) format, collapsing all layer information.

- `get_supra_adjacency_matrix(dtype="int64") -> csr_matrix`  
  Build supra-adjacency matrix (L*N × L*N) with inter-layer couplings.

- `get_grouped_mln() -> MultiLayerNetwork`  
  Return network with layers aggregated by group (requires `group` column in `layers`).

- `clear_layer_adjacency_matrices()`, `clear_group_adjacency_matrices()`, `clear_all_adjacency_matrices()`  
  Clear cached adjacency matrices.

#### Edge lists and conversions
- `get_edgelist(edge_attribute="binary") -> DataFrame`  
  Convert sparse adjacency matrix to edgelist. Options: "binary", "layer", "label", "weight", or None.

- `convert_layer_representation(layer, input_type="layer", output_type="binary") -> Any | List[Any]`  
  Convert between layer representations: "layer", "label", "binary".

- `convert_layer_binary_to_list(num, output_type="layer") -> List`  
  Decode binary integer to list of active layers.

#### Graph library exports
- `to_igraph(directed=True, edge_attributes=True, node_attributes=False, replace_igraph=False, edge_attribute_type="binary") -> ig.Graph`  
  Create igraph object. Options for `edge_attribute_type`: "binary", "weight".

- `to_networkx(directed=True, edge_attributes=True, node_attributes=False, edge_attribute_type="binary", layer_type="layer", ignore_limit=False) -> nx.Graph | nx.DiGraph | None`  
  Create NetworkX object (limited by `nx_node_limit` unless `ignore_limit=True`).

- `save_to_graphml(file_name, directed=True, edge_attributes=True, node_attributes=False, overwrite=False, edge_attribute_type="binary")`  
  Save as GraphML for external tools (Gephi, etc.).

#### Metrics and analysis
- `get_degrees(selected_nodes=[]) -> Dict[Any, int]`  
  Calculate degree for selected nodes across all layers.

- `get_clustering_coefficient(selected_nodes=[], batchsize=100000) -> Dict[Any, float]`  
  Compute clustering coefficient for selected nodes.

- `get_excess_closure(selected_nodes=[], node_type="label", selected_layers=[], layer_type="layer", batchsize=100000) -> Dict[str, Dict]`  
  Calculate excess closure metric (returns both clustering coefficient and excess closure).

#### Node operations
- `add_nodes(nodes_df: pd.DataFrame)`  
  Add new nodes without connections (expands adjacency matrix).

- `remove_nodes(nodes: List[Any])`  
  Remove nodes and their edges; reindexes remaining nodes.

#### Aggregation
- `create_affiliation_matrix(key, affil_edgelist: List[Tuple])`  
  Create bipartite affiliation matrix for node-to-entity relationships.

- `get_aggregated_network(aggregation_column=None, keep_layers=False, convert_A="none") -> MultiLayerNetwork`  
  Aggregate network by node attribute (e.g., region, age group).

#### Persistence
- `save(path="", overwrite=False, **kwargs)`  
  Save MLN library (edges.npz, nodes.csv.gz, layers.csv, optional codebook).

- `export_edges(file_name)`, `export_nodes(file_name)`, `export_layers(file_name)`, `export_codebook(file_name)`  
  Export individual components.

- `load(path) -> Tuple[DataFrame, csr_matrix, DataFrame]`  
  Load MLN components from disk.

#### Utilities
- `report_time(message="", init=False)`  
  Helper for timing measurements (requires `verbose=True`).

### RawCSVtoMLN (preparation.py)

Utility class for converting raw CSV node/edge files into MLN library format.

- `__init__(node_conf, edge_conf, layer_conf, output_folder, **kwargs)`  
  Configure input/output paths and column mappings.

- `init_all()`  
  Run full pipeline: init_layers() → init_nodes() → init_edges() → read_all_edges() → save_all() (if configured).

- `save_all(overwrite=False)`  
  Save nodes.csv.gz, edges.npz, and layers.csv to output folder.

See `test/config.json` for configuration schema.

## Notes

- NetworkX conversion limits very large graphs (`nx_node_limit` in code). Use igraph for larger networks.
- The package supports pandas by default and optional polars interop for CSV loading and some operations.
- The network can be directed; layer information is encoded in the adjacency matrix when `adjacency_element="binary"`.

## Citation

If you use this library in academic work, please cite the POPNET/PLANET‑NL project (https://planetnl.org) and this repository. See `LICENSE.md` for details.
