---
title: "mlnlib: Memory-efficient multilayer network analysis for population-scale social networks"
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
  - name: Leiden Institute of Advanced Computer Science, Leiden University, The Netherlands
    index: 1
  - name: Department of Political Science, University of Amsterdam, The Netherlands
    index: 2
date: 15 June 2026
bibliography: paper.bib
---

# Summary

Population-scale social networks, often constructed from administrative registers and linked digital traces, enable the study of social structure at the level of entire societies. In parallel, many empirical applications require multilayer representations in which ties among the same entities are observed across multiple relation types, data sources, or time slices [@kivela2014multilayer; @boccaletti2014structure].

`mlnlib` is a Python package for memory-efficient storage and manipulation of large multilayer networks. It represents multilayer edge presence in a single sparse adjacency matrix using binary layer encoding, which enables efficient layer-aware filtering and conversion operations while keeping the user-facing workflow simple and compatible with standard Python data science tools. The package provides the `MultiLayerNetwork` class for loading, slicing, exporting, and converting multilayer networks, and the `RawCSVtoMLN` class for preparing multilayer data from raw CSV sources.

# Statement of need

A common but inefficient strategy for multilayer data is to store relation type as a generic edge attribute and repeatedly filter edge tables during analysis. This approach is often slow for large data and can become impractical in constrained research environments where users cannot rely on specialized infrastructure (for example, cluster schedulers, high-performance databases, custom compilers, or distributed systems).

`mlnlib` addresses this gap by focusing on an in-memory representation that is portable and practical in standardized computing environments. The design goals are:

1. Efficiently store sparse multilayer edges when the number of layers is modest relative to the number of nodes and edges.
2. Enable fast edge-based slicing by layer combinations through bitwise operations on encoded edge values.
3. Support node-based slicing using tabular node attributes handled through familiar dataframe workflows.
4. Interoperate with downstream graph-analysis ecosystems without reimplementing all network science methods.

This scope is intentionally narrow: `mlnlib` does not aim to replace comprehensive graph analytics libraries. Instead, it provides an efficient intermediate representation and data-engineering layer before handoff to packages such as NetworkX and igraph [@networkx; @igraph].

# State of the field

General-purpose graph libraries such as NetworkX [@networkx] and igraph [@igraph] are widely used and provide rich algorithmic toolkits. However, multilayer handling is often application-specific and left to custom user data models. Existing multilayer frameworks in the literature emphasize conceptual formalisms [@kivela2014multilayer; @boccaletti2014structure], but many applied workflows still rely on ad hoc implementations for storage and slicing.

`mlnlib` contributes a practical engineering solution: a sparse-matrix-centered multilayer representation that integrates with `numpy`, `scipy`, `pandas`, and `polars` [@numpy; @scipy; @pandas; @polars], while preserving direct export paths to established graph tools.

# Design and implementation

The core `MultiLayerNetwork` object stores three components:

1. Node table: a dataframe with at least `label` and integer `id` columns, plus optional attributes.
2. Edge matrix: a `scipy.sparse.csr_matrix` where each nonzero integer encodes active layers using powers of two.
3. Layer table: metadata for each layer including id, label, binary code, and optional groups.

If layer $l$ is assigned code $2^l$, then an edge present on multiple layers stores the sum of those codes. For example, value $7$ encodes layers $0$, $1$, and $2$ because $7=1+2+4$. Presence of layer $k$ is tested by bit masking, i.e., $(x \& 2^k)=2^k$.

This representation supports:

1. Fast extraction of layer-specific adjacency matrices.
2. Combined-layer filtering without scanning string-valued edge attributes.
3. Conversion to binary, weighted, or labeled edge-list outputs.
4. Export to `igraph`, `networkx`, and GraphML.

For data preparation, `RawCSVtoMLN` builds the required node, edge, and layer objects from configurable raw CSV inputs, allowing reproducible setup in data pipelines.

# Use cases and impact

The package was developed in the context of the POPNET/PLANET-NL project, where population-scale register and social data require repeated multilayer slicing and aggregation. In this context, an important requirement is reproducible execution on managed computing environments with heterogeneous constraints, rather than dependence on specialized infrastructure.

By combining sparse encodings for multilayer edges with dataframe-native node attributes, `mlnlib` supports workflows that bridge tabular preprocessing and graph analysis. This is especially useful when researchers need to iterate between node-attribute filters and multilayer edge selection before applying downstream methods.

The toolkit has been used in multiple recent studies on population-scale social structure, mobility, inequality, migration attitudes, and online-versus-register social networks [@bokanyi2026fragmentation; @kazmina2025mobility; @menyhert2025connectivity; @debel2025kinship; @kazmina2024contactthreat; @socnet_pii2024; @bokanyi2023anatomy].

# Acknowledgements

This software was developed in the context of the POPNET/PLANET-NL project (https://planetnl.org).

# References
