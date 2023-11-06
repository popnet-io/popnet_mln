import numpy as np
import pandas as pd
from itertools import product
from random import choice
from itertools import product
from scipy.spatial import KDTree, distance_matrix
import pydot

import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
import matplotlib.pyplot as plt

class DummyNetwork:
    """
    This class generates a dummy POPNET network built over a simulated kinship structure.
    The configuration dictionary sets the parameters of the simulation.
    The resulting nodelist and edgelist can be exported with `self.export_network()` into
    a node and edge pandas DataFrame.
    """
    def __init__(self, **config):
        # how many people to start the simulation from
        self.size_first_gen = config["size_first_gen"]
        # 4 generations are stored in this dict
        self.generations = {i:[] for i in range(4)}
        # 4 generations of marriages are stored in this dict
        self.marriages = {i:[] for i in range(4)}
        # probabilities of having n number of children in each generation, already cumsummed for child number generation
        self.child_probabilities = {gen: np.cumsum(config["child_probabilities"][gen]) for gen in config["child_probabilities"]}
        self.marriage_probabilities = config["marriage_probabilities"]
        # probabilities of death 
        self.death_probability = config["death_probability"]
        # neighbor threshold
        self.neighbor_threshold = config["neighbor_threshold"]
        # how many schools are there in the dummy network
        self.number_of_schools = config["number_of_schools"]
        # how many workplaces are there in the dummy network
        self.number_of_workplaces = config["number_of_workplaces"]

        # contains edges of the parent-child tree
        self.parent_child_tree = []
        # can look up the marriage from which the chlid is coming from
        self.child_parent_lookup = {}

        # placeholder for generating node_ids
        self.last_node_id = 0
        # storage for nodes
        self.nodes = {}
        # storage for edges
        self.edges = set()

        # creating a generation, then marrying them
        for gen in range(4):
            self.create_generation(gen)
            self.create_marriages(gen)

        # generating family edges based on parent-child tree
        self.create_edges_family()
        # removing people that are already dead
        self.remove_dead()
        # create households
        self.create_edges_household()
        # add coordinates to the created households
        self.create_household_coords()
        # calculate neighbors
        self.create_edges_neighbors()
        # calculate school affiliation
        self.create_edges_school()
        # calculate work affiliation
        self.create_edges_work()

    def create_generation(self, gen):
        """
        Take the marriages of the previous generation, choose a child number for each
        of the marriages based on the generation-based child number distribution, then
        create the next generation and store parent-child register.
        """
        # assure that it's even
        if self.size_first_gen % 2 == 1:
            self.size_first_gen += 1
        if gen == 0:
            for n in range(self.size_first_gen):
                self.nodes[self.last_node_id] = {'gender': n%2, 'generation': gen}
                self.generations[gen].append(self.last_node_id)
                self.last_node_id += 1

                if n%2==1:
                    self.marriages[gen].append((n-1,n))

        else:
            child_numbers = np.digitize(np.random.rand(len(self.marriages[gen-1])),self.child_probabilities[gen-1])
            total_child_numbers = sum(child_numbers)
            gender = np.digitize(np.random.rand(total_child_numbers),[0,0.5,1])-1

            counter = 0
            for m,cn in zip(self.marriages[gen-1],child_numbers):
                for k in range(cn):
                    self.generations[gen].append(self.last_node_id)
                    self.nodes[self.last_node_id] = {'generation' : gen, 'gender': gender[counter]}

                    self.parent_child_tree.append((m[0],self.last_node_id))
                    self.parent_child_tree.append((m[1],self.last_node_id))
                    self.child_parent_lookup[self.last_node_id] = m

                    self.last_node_id += 1
                    counter += 1


    def create_marriages(self, gen):
        """
        Given the marriage probability within a generation, "pair" up male and female
        nodes, and store the result.
        """
        if gen != 0:
            p = self.marriage_probabilities.get(gen)

            # storage for nonmarried people
            nonmarried_females = [n for n in self.generations[gen] if self.nodes[n]['gender'] == 0]
            nonmarried_males = [n for n in self.generations[gen] if self.nodes[n]['gender'] == 1]

            # number of marriages
            num_marriages = int(p*min(len(nonmarried_males),len(nonmarried_females)))

            # either we reach the desired number, or we cannot do that
            for i in range(num_marriages):
                hit = 0
                counter = 0
                while hit == 0 and counter<1000 and len(nonmarried_females)>0 and len(nonmarried_males)>0:
                    m = choice(nonmarried_males)
                    f = choice(nonmarried_females)
                    if self.child_parent_lookup[m]!=self.child_parent_lookup[f]:
                        if m<f:
                            self.marriages[gen].append((m,f))
                        else:
                            self.marriages[gen].append((f,m))
                        nonmarried_males.remove(m)
                        nonmarried_females.remove(f)
                        hit = 1
                    counter += 1

    def create_edges_family(self):
        """
        Based on the parent-child register and marriages, generate all family links.
        """
        for gen in self.marriages:
            for e in self.marriages[gen]:
                self.edges.add((e[0],e[1],'partner'))
                self.edges.add((e[1],e[0],'partner'))

        for e in self.parent_child_tree:
            self.edges.add((e[0],e[1],'parent'))
            self.edges.add((e[1],e[0],'child'))

        for n1,n2 in product(self.nodes.keys(), self.nodes.keys()):
            if n1 in self.child_parent_lookup and n2 in self.child_parent_lookup and n1<n2:
                if self.child_parent_lookup[n1] == self.child_parent_lookup[n2]:
                    self.edges.add((n1,n2,'sibling'))
                    self.edges.add((n2,n1,'sibling'))

        for e1,e2 in product(self.edges,self.edges):
            if e1[0]!=e2[1] and e1[1] == e2[0]:
                if  e1[2] == "parent" and e2[2] == "parent":
                    self.edges.add((e1[0],e2[1],'grandparent'))
                    self.edges.add((e2[1],e1[0],'grandchild'))
                if e1[2] == "sibling" and e2[2] == "parent":
                    self.edges.add((e1[0],e2[1],'aunt / uncle'))
                    self.edges.add((e2[1],e1[0],'niece / nephew'))
                if e1[2] == "partner" and e2[2] == "child":
                    self.edges.add((e1[0],e2[1],'mother / father-in-law'))
                    self.edges.add((e2[1],e1[0],'daughter / son-in-law'))
                if e1[2] == "sibling" and e2[2] == "partner":
                    self.edges.add((e1[0],e2[1],'sister / brother-in-law'))
                if e1[2] == "partner" and e2[2] == "sibling":
                    self.edges.add((e1[0],e2[1],'sister / brother-in-law'))
                    
        for e1,e2 in product(self.edges,self.edges):
            if e1[0]!=e2[1]  and e1[1] == e2[0]:
                if e1[2] == "niece / nephew" and e2[2] == "parent":
                    self.edges.add((e1[0],e2[1],'cousin'))
                    self.edges.add((e2[1],e1[0],'cousin'))

    def remove_dead(self):
        """
        In each generation, according to the preset death probabilities, remove
        nodes and corresponding edges from all containers. Store the old variables
        for safety purposes.
        """
        self.to_remove = {}
        for gen in self.death_probability:
            self.to_remove[gen] = list(np.array(self.generations[gen])[np.where(np.random.rand(len(self.generations[gen]))<self.death_probability[gen])[0]])

        # save 
        self.all_nodes = self.nodes
        self.all_family_edges = self.edges
        self.all_parent_child_tree = self.parent_child_tree
        self.all_marriages = self.marriages

        self.to_remove_total = [n for gen in self.to_remove for n in self.to_remove[gen]]
        self.nodes = {k:self.nodes[k] for k in self.nodes if k not in self.to_remove_total}
        self.edges = set([e for e in self.edges if e[0] in self.nodes and e[1] in self.nodes])
        self.parent_child_tree = [p for p in self.parent_child_tree if p[0] in self.nodes and p[1] in self.nodes]
        self.marriages = {gen:[m for m in self.marriages[gen] if m[0] in self.nodes and m[1] in self.nodes] for gen in self.marriages}

    def create_edges_household(self):
        """
        Generate households based on the following rules:
            - generation 1: married people live together, nonmarried ones on their own
            - generation 2 and 3: parents and children live together
        """
        temp = [n for gen in self.marriages for m in self.marriages[gen] for n in m]
        nonmarried_nodes = [n for n in self.nodes if n not in temp]

        self.households = [[n] for n in nonmarried_nodes]
        for gen in self.marriages:
            for m in self.marriages[gen]:
                if gen==1:
                    self.households.append(list(m))
                if gen==2:
                    children = list(set([e[1] for e in self.edges if e[2]=="parent" and (e[0]==m[0] or e[0]==m[1])]))
                    self.households.append([*m,*children])
                    
        self.household_dict = {}
        for i,h in enumerate(self.households):
            for n in h:
                self.household_dict[n] = i
                
        for h in self.households:
            for n1,n2 in product(h,h):
                if n1!=n2:
                    self.edges.add((n1,n2,'household'))
                    self.edges.add((n2,n1,'household'))

    def create_household_coords(self):
        """
        For each household, assign random coordinates on the (0,1)x(0,1) square.
        """
        # adding random positions to households
        # this should better reflect parent-child distance distributions, but for now, it will be OK
        self.household_coords = np.random.rand(len(self.households),2)

    def create_edges_neighbors(self):
        """
        Calculating distances between all households, connecting household members
        as neighbors below self.neighbor_threshold distance.
        """
        D = distance_matrix(self.household_coords,self.household_coords)
        
        for h1,h2 in zip(*np.where((D<self.neighbor_threshold))):
            if h1<h2:
                for n1,n2 in product(self.households[h1],self.households[h2]):
                    self.edges.add((n1,n2,'neighbor'))
                    self.edges.add((n2,n1,'neighbor'))

    def create_edges_school(self):
        """
        Assign random coordinates to a predefined number of schools.
        Then assign each child (generation 3) to the closest school.
        """
        # how many schools are there
        num_schools = self.number_of_schools

        # random school coordinates
        school_coords = np.random.rand(num_schools,2)
        # the household coordinates of the children
        child_coords = self.household_coords[[self.household_dict[n] for n in self.generations[3] if n not in self.to_remove_total],:]

        # spatial lookup for closest school
        kdtree = KDTree(school_coords)
        _, school = kdtree.query(child_coords)

        # child -> school dict
        temp = list(zip([n for n in self.generations[3] if n not in self.to_remove_total], school))
        # storage for later easier SQL database generation
        self.school_dict = {k:v for k,v in temp}

        # if the schools are the same, add an edge for each child
        for e1,e2 in product(temp,temp):
            if e1 < e2:
                if e1[1] == e2[1]:
                    self.edges.add((e1[0],e2[0],'school'))
                    self.edges.add((e2[0],e1[0],'school'))

    def create_edges_work(self):
        """
        Assign random coordinates to a predefined number of workplaces.
        Then assign each adult (generation 2) to the closest workplace.
        """
        
        # how many workplaces are there
        num_wp = self.number_of_workplaces

        # random work coordinates
        work_coords = np.random.rand(num_wp,2)
        # the household coordinates of the employees
        employee_coords = self.household_coords[[self.household_dict[n] for n in self.generations[2] if n not in self.to_remove_total],:]

        # spatial lookup for closest workplace
        kdtree = KDTree(work_coords)
        _, work = kdtree.query(employee_coords)

        # employee -> workplace dict
        temp = list(zip([n for n in self.generations[2] if n not in self.to_remove_total], work))
        # storage for later easier SQL database generation
        self.work_dict = {k:v for k,v in temp}

        # if the workplaces are the same, add an edge for each employee
        for e1,e2 in product(temp,temp):
            if e1 < e2:
                if e1[1] == e2[1]:
                    self.edges.add((e1[0],e2[0],'work'))
                    self.edges.add((e2[0],e1[0],'work'))

    def create_family_tree_positions(self):
        """
        Convert the network to DOT and compute hierarchical layout for the parent-child register
        for visualization purposes.
        """
        Tdot = pydot.Dot("dummy_net",graph_type="digraph",strict=True)

        Tdot.set_node_defaults(shape="point",width=0.1)
        Tdot.set_edge_defaults(arrowsize=0.3,color="grey")

        for n in self.all_nodes:
            Tdot.add_node(pydot.Node(str(n)))

        for e in self.all_parent_child_tree:
            Tdot.add_edge(pydot.Edge(src=str(e[0]),dst=str(e[1])))

        for gen in self.generations:
            rank = "same"
            if gen == 0:
                rank = "source"
            elif gen == 3:
                rank = "sink"
            s = pydot.Subgraph(rank=rank)
            for n in self.generations[gen]:
                s.add_node(pydot.Node(str(n)))
            Tdot.add_subgraph(s)

        Tdot.write_plain('temp.csv')
        temp = pd.read_csv('temp.csv',skiprows=1,nrows=len(self.all_nodes.keys()),sep=" ",header=None)
        temp.columns = columns=["type","node","x","y","w","h","l","f","s","c","ec"]
        return dict(zip(temp["node"],zip(temp["x"],temp["y"])))

    def draw_family_tree(self, additional_edges = "partner", additional_edges_color = "color"):
        """
        Draw DOT network 
        """
        Tall = nx.from_edgelist([e for e in self.all_parent_child_tree],create_using = nx.DiGraph())
        for n in self.all_nodes:
            if n not in Tall.nodes():
                Tall.add_node(n)
        T = nx.from_edgelist([e for e in self.parent_child_tree],create_using = nx.DiGraph())
        for n in self.nodes:
            if n not in T.nodes():
                T.add_node(n)
        pos = self.create_family_tree_positions()
        fig,ax = plt.subplots(1,1,figsize=(30,5))
        nx.draw(Tall,pos,ax=ax,node_color='lightgrey',edge_color='lightgrey')
        nx.draw(T,pos,ax=ax)
        nx.draw_networkx_labels(T,pos,font_color='white')
        ax.axis('off')

    def export_network(self):
        edgelist = pd.DataFrame(list(self.edges))
        edgelist.columns = ['source','target','edgetype']
        nodelist = pd.DataFrame(self.nodes).T.reset_index().rename({'index':'node_id'},axis=1)
        
        return nodelist, edgelist