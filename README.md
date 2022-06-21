# popnet_mln
Library for working with large multilayer networks by the POPNET group.

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

A sample config looks like the following:
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
