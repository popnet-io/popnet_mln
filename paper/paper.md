---
title: "mlnlib: A portable and memory-efficient multilayer network toolkit"
tags:
  - Python
  - network science
  - multilayer networks
  - social networks
  - sparse matrices
  - population-scale networks
  - register-based networks
authors:
  - name: Eszter Bokányi
    corresponding: true
    affiliation: 1, 2
  - name: Rachel de Jong
    affiliation: 1
  - name: Yuliia Kazmina
    affiliation: 2
  - name: Frank Takes
    affiliation: 1
affiliations:
  - name: PLANET-NL, Leiden Institute of Advanced Computer Science, Leiden University, The Netherlands
    index: 1
  - name: PLANET-NL, Department of Political Science, University of Amsterdam, The Netherlands
    index: 2
date: 25 July 2026
bibliography: paper.bib
---

# Summary

Recently, it became possible to study the social network structure of entire societies by deriving social connections from central administrative registers of countries [@bokanyi2023anatomy; @panayiotou2025swedish; @cremers2025temporal]. These networks are often large with tens of millions of nodes and billions of edges, moreover, edges can be of multiple different types represented as layers [@kivela2014multilayer; @boccaletti2014structure]. The size and multilayer nature of these data pose challenges for efficient storage and manipulation, especially in constrained research environments where these privacy-sensitive datasets are usually available.


With `mlnlib`, we provide a Python package for memory-efficient and lightweight storage and manipulation of large multilayer networks. The package enables efficient layer-aware filtering and conversion operations while keeping the user-facing workflow simple and compatible with standard Python data science tools. The package provides the `MultiLayerNetwork` class for loading, querying, aggregating, exporting, and converting multilayer networks, and the `RawCSVtoMLN` class for preparing an intermediate file format for fast loading from raw CSV sources.

# Statement of need

One common strategy for multilayer networks is to store relation types as a generic edge attribute and repeatedly filter edge tables during analysis. This approach is often slow for large data and becomes impractical in constrained research environments where users cannot rely on specialized hardware or software infrastructure (for example, cluster schedulers, high-performance databases, custom compilers, or distributed systems).
Storing multilayer edges in a large supra-adjacency matrix  is another option, but it can be memory-inefficient when the number of layers is large and the data are sparse.
Finally, most graph libraries are not designed to handle large tabular node attributes alongside multilayer edge data. These are common in population-scale social networks where the registers also allow including demographic attributes of people (e.g., information on age, gender, education, or income).

`mlnlib` addresses this gap by focusing on an in-memory representation that is portable and practical in standardized computing environments. The design goals are:

1. Efficiently store multilayer edges encoded into a single sparse adjacency matrix.
2. Enable fast querying by layer combinations through bitwise operations on encoded edge values.
3. Support node-based querying using tabular node attributes handled through familiar dataframe workflows.
4. Interoperate with downstream graph analysis libraries.
5. Minimizing software or hardware dependencies to ensure broad accessibility and ease of installation.

This scope is intentionally narrow: `mlnlib` does not aim to replace comprehensive mulilayer graph analytics libraries. Instead, it provides an efficient intermediate representation and data-engineering layer before handoff to packages such as NetworkX and igraph [@networkx; @igraph]. Moreover, it can be easily requested as a standard Python package in managed computing environments, and it does not require specialized infrastructure for installation or execution.

# State of the field

The multilayer network engineering ecosystem faces significant maturity and scalability challenges [@panayiotou2024challenges]. Current multilayer network libraries vary widely in their support for core functionality, and many struggle with memory efficiency and scalability when handling large, sparse networks with contemporary data volumes.

General-purpose graph libraries such as NetworkX [@networkx] and igraph [@igraph] are widely used and provide rich algorithmic toolkits. However, multilayer handling or tabular storage of node attributes is not the first priority of these packages, therefore, related operations are not implemented or not efficient due to lack of explicit multilayer support and lack of dataframe integration for node properties.

The library best representing theoretical multilayer network concepts [@kivela2014multilayer] is `pymnet` [@nurmi2024pymnet], which provides a comprehensive set often tools for multilayer network analysis. However, it is not designed for large-scale data and can be memory-inefficient, representing multilayer edges in a supra-adjacency matrix that becomes prohibitive for large and sparse networks. Moreover, it does not support tabular node attributes, which are common in population-scale social networks.

`multinet` (also known as `uunet`) is an R package whose Python API is also available on pypi, and which provides a rich toolkit for standard multilayer network metrics and analysis operations. It supports multilayer workflows including community detection and centrality measures. However, like `pymnet`, `multinet` is primarily oriented toward comprehensive analysis rather than efficient storage and preprocessing. Loading large multilayer networks can be slow, and the package does not support tabular node attributes [@panayiotou2024challenges].

`py3plex` [@skrlj2019py3plex] is a lightweight Python library focused on visualization and analysis of multilayer networks. It provides tools for intra- and inter-layer visualization, and supports common multilayer operations including aggregation, slicing, indexing, and traversal. The library also incorporates node embeddings to accelerate layout computation. While useful for visualization and exploratory analysis, `py3plex` is not specifically optimized for memory-efficient storage of very large sparse networks or integration with tabular node attributes in data-preprocessing workflows.

`MuxViz` [@dedomerico2015muxviz] is an interactive visualization and analysis tool designed to facilitate understanding of multilayer network structure and dynamics. It combines network analysis algorithms with advanced visualization techniques to explore both structural properties and dynamical processes on multilayer networks. However, `MuxViz` is primarily designed as an interactive analysis tool rather than as a data engineering solution; it is better suited for exploratory analysis and understanding network structure at a small scale.

`Threadle` [@nordlund2025threadle] is a high-performance multilayer and multimode network data management system designed explicitly for population-scale administrative registers. Built in .NET, it addresses similar data-engineering challenges as `mlnlib`, including memory-efficient storage of multilayer edges, built-in node attribute management, moreover, support for both 1-mode and 2-mode (bipartite) relations. Some design features of Threadle draw on design insights from our `mlnlib` implementation. However, Threadle targets primarily the R ecosystem through its `threadleR` interface.

The gap that `mlnlib` addresses is distinctly different from the strengths of existing tools. While contemporary Python multilayer network libraries focus on analysis, visualization, and algorithmic exploration, they are not designed to efficiently handle the data-engineering challenges posed by population-scale networks with billions of edges, multiple layers, and extensive tabular node attributes, particularly in constrained computing environments where specialized infrastructure is unavailable. The engineering ecosystem struggles with memory-inefficient representations (such as supra-adjacency matrices), slow loading times for large sparse networks, and lack of support for integrating tabular node attributes. `mlnlib` fills this gap by providing a lightweight, portable intermediate representation that combines sparse matrix encoding of multilayer edges with familiar dataframe-based node attribute handling. This design prioritizes efficient data preparation and preprocessing over comprehensive graph algorithms, creating a practical bridge between raw data and established graph analysis libraries in resource-constrained research environments.

# Software design

The core `MultiLayerNetwork` object stores three components:

1. Node table: a stadard `polars` or `pandas` dataframe with at least `label` (unique string identifier) and integer `id` (unique interger identifier from 0 to N-1 where N is the number of nodes) columns, plus optional attributes.
2. Edge matrix: a `scipy.sparse.csr_matrix` where each nonzero integer encodes active layers using powers of two.
3. Layer table: metadata for each layer including at least `id` (unique integer identifier) and `label` (unique string identifier).

If layer $l$ is assigned code $2^l$, then an edge present on multiple layers stores the sum of those codes. For example, value $7$ encodes layers $0$, $1$, and $2$ because $7=1+2+4$. Presence of layer $k$ is tested by bit masking, i.e., $(x \& 2^k)=2^k$.

This representation supports fast extraction of layer-specific adjacency matrices, combined-layer filtering without scanning string-valued edge attributes, and easy conversion to binary, weighted, or labeled edgelist outputs. The class provides exports to `igraph`, `networkx`, and GraphML, enabling downstream analysis with established graph libraries.

For data preparation, `RawCSVtoMLN` builds the required node, edge, and layer files from configurable raw CSV inputs, allowing a reproducible setup in data pipelines.

# Research impact statement

The package was developed in the context of the POPNET/PLANET-NL project, where population-scale register and social data require repeated multilayer slicing and aggregation. In this context, an important requirement is reproducible execution on constrained computing environments, rather than dependence on specialized infrastructure.

By combining sparse encodings for multilayer edges with dataframe-native node attributes, `mlnlib` supports workflows that bridge tabular preprocessing and graph analysis. This is especially useful when researchers need to iterate between node attribute filters and multilayer edge selection before applying downstream methods.

The toolkit has been used in multiple recent studies on population-scale social structure, mobility, inequality, migration attitudes, and online-versus-register-based social networks [@bokanyi2026fragmentation; @kazmina2025mobility; @menyhert2025connectivity; @debel2025kinship; @kazmina2024contactthreat; @kazmina2024socioeconomic; @bokanyi2023anatomy]. The population-scale networks containing all family, household, neighbor, work, and school ties of the entire Netherlands between 2009 and 2021 have been made available in the intermediate format directly readable by the `mlnlib` package [@bokanyi2025planetnl].

# AI usage disclosure

Generative AI assistance was used during software and manuscript preparation. Specifically, GitHub Copilot was used for code documentation generation, general code tidying, and editing/correcting manuscript text. All AI-assisted outputs were reviewed, edited, and validated by the human authors, who made the core design and research decisions.

# Acknowledgements

This software was developed in the context of the POPNET/PLANET-NL project (https://planetnl.org).

# References
