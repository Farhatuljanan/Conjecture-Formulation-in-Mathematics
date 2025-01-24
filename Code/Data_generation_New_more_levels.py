# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 19:11:12 2024

@author: fjanan
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 18:28:52 2024

@author: fjanan
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 16:22:01 2024

@author: fjanan
"""

import networkx as nx
import matplotlib.pyplot as plt
import itertools
import random
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from collections import Counter
import pandas as pd

def plot_graph(G):
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color='lightblue', arrowsize=20)
    plt.show()


def generate_random_graph(seed_number): 
 seed_value = seed_number
 np.random.seed(seed_value)
 level1_nodes = [f'A{i}' for i in range(1, random.choice([2,3,4,5,6,7]))]
 np.random.seed((seed_value+1))
 level2_nodes = [f'B{i}' for i in range(1, random.choice([5,7,9,11,13,15,17]))]
 np.random.seed((seed_value+1))
 level3_nodes = [f'C{i}' for i in range(1, random.choice([5,7,9,11,13,15,17]))]
 #level4_nodes = [f'C{i}' for i in range(1, random.choice([5,7,9,11,13]))]
 G = nx.DiGraph()
 G.add_nodes_from('S')
 G.add_nodes_from(level1_nodes)
 G.add_nodes_from(level2_nodes)
 G.add_nodes_from(level3_nodes)
 #G.add_nodes_from(level4_nodes)
 G.add_nodes_from('T')
 
# Generate all possible edges from level 1 to level 2 and level 2 to level 3
 edges_source_to_level1= list(itertools.product('S', level1_nodes))
 edges_level1_to_level2 = list(itertools.product(level1_nodes, level2_nodes))
 edges_level2_to_level3 = list(itertools.product(level2_nodes, level3_nodes))
 #edges_level3_to_level4 = list(itertools.product(level3_nodes, level4_nodes))
 edges_level3_to_sink= list(itertools.product(level3_nodes, 'T'))
 fifty_percent_l1_l2=len(edges_level1_to_level2)/2
 fifty_percent_l2_l3=len(edges_level2_to_level3)/2
 #fifty_percent_l3_l4=len(edges_level3_to_level4)/2
 random_sample_l1_l2=random.sample(edges_level1_to_level2, int(fifty_percent_l1_l2))
 random_sample_l2_l3=random.sample(edges_level2_to_level3, int(fifty_percent_l2_l3))
 #random_sample_l3_l4=random.sample(edges_level3_to_level4, int(fifty_percent_l3_l4))
 lenArcs_l1_l2=len(random_sample_l1_l2)
 lenArcs_l2_l3=len(random_sample_l2_l3)
 #lenArcs_l3_l4=len(random_sample_l3_l4)
 G.add_edges_from(random_sample_l1_l2)
 G.add_edges_from(random_sample_l2_l3)
 G.add_edges_from(edges_source_to_level1)
 #G.add_edges_from(random_sample_l3_l4)
 G.add_edges_from(edges_level3_to_sink)
# Capacity for first level
 num_nodes_l1 = len(level1_nodes)
 num_nodes_l2 = len(level2_nodes)
 num_nodes_l3 = len(level3_nodes)
 #num_nodes_l4 = len(level4_nodes)
 G.nodes['S']['Capacity']=10
 G.nodes['T']['Capacity']=10
 ##Count the number of overlapping arcs
 overlap_nodes_l1_l2=[]
 for i in list(range(0,len(random_sample_l1_l2))):
      target_nodes=random_sample_l1_l2[i][1]
      overlap_nodes_l1_l2.append(target_nodes)
 count_l1_l2 = Counter(overlap_nodes_l1_l2)
 index_count_l1_l2 = [{'index': i, 'key': key, 'value': value} for i, (key, value) in enumerate(count_l1_l2.items())]
 overlap_l1_l2=0
 for k in list(range(0,len(count_l1_l2))):
      if index_count_l1_l2[k]['value']>1 :
          overlap_l1_l2=overlap_l1_l2+index_count_l1_l2[k]['value']
          
 overlap_nodes_l2_l3=[]
 for i in list(range(0,len(random_sample_l2_l3))):
      target_nodes=random_sample_l2_l3[i][1]
      overlap_nodes_l2_l3.append(target_nodes)
 count_l2_l3 = Counter(overlap_nodes_l2_l3)
 index_count_l2_l3 = [{'index': i, 'key': key, 'value': value} for i, (key, value) in enumerate(count_l2_l3.items())]
 overlap_l2_l3=0
 for k in list(range(0,len(count_l2_l3))):
      if index_count_l2_l3[k]['value']>1 :
          overlap_l2_l3=overlap_l2_l3+index_count_l2_l3[k]['value']

 if num_nodes_l1 == 1:
    for i in level1_nodes:
       G.nodes[i]['Capacity'] = 10
 elif num_nodes_l1 == 2:
    for i in level1_nodes:
       G.nodes[i]['Capacity'] = 5
 elif num_nodes_l1 == 3:
    for i in level1_nodes:
       G.nodes[i]['Capacity'] = 3.33
 elif num_nodes_l1 == 4:
    for i in level1_nodes:
       G.nodes[i]['Capacity'] = 2.5
 elif num_nodes_l1 == 5:
    for i in level1_nodes:
       G.nodes[i]['Capacity'] = 2.0
 else:
    for i in level1_nodes:
       G.nodes[i]['Capacity'] = 1.67
 capacity_l2 = [2.5, 1.67, 1.25, 1, 0.83, 0.714, 0.625]
 if num_nodes_l2 == 4:
    for i in level2_nodes:
        G.nodes[i]['Capacity'] = capacity_l2[0]
 elif num_nodes_l2 == 6:
    for i in level2_nodes:
        G.nodes[i]['Capacity'] = capacity_l2[1]
 elif num_nodes_l2 == 8:
    for i in level2_nodes:
        G.nodes[i]['Capacity'] = capacity_l2[2]
 elif num_nodes_l2 == 10:
    for i in level2_nodes:
        G.nodes[i]['Capacity'] = capacity_l2[3]
 elif num_nodes_l2 == 12:
    for i in level2_nodes:
        G.nodes[i]['Capacity'] = capacity_l2[4]
 elif num_nodes_l2 == 14:
    for i in level2_nodes:
        G.nodes[i]['Capacity'] = capacity_l2[5]
 else:
    for i in level2_nodes:
        G.nodes[i]['Capacity'] = capacity_l2[6]
 if num_nodes_l3 == 4:
    for i in level3_nodes:
        G.nodes[i]['Capacity'] = capacity_l2[0]
 elif num_nodes_l3 == 6:
    for i in level3_nodes:
        G.nodes[i]['Capacity'] = capacity_l2[1]
 elif num_nodes_l3 == 8:
    for i in level3_nodes:
        G.nodes[i]['Capacity'] = capacity_l2[2]
 elif num_nodes_l3 == 10:
    for i in level3_nodes:
        G.nodes[i]['Capacity'] = capacity_l2[3]
 elif num_nodes_l3 == 12:
    for i in level3_nodes:
        G.nodes[i]['Capacity'] = capacity_l2[4]
 elif num_nodes_l3 == 14:
    for i in level3_nodes:
        G.nodes[i]['Capacity'] = capacity_l2[5]
 else:
    for i in level3_nodes:
        G.nodes[i]['Capacity'] = capacity_l2[6]
 
 for i, j in G.edges:
    G[i][j]['Capacity'] = 10
 return G, num_nodes_l1, num_nodes_l2, num_nodes_l3,lenArcs_l1_l2,lenArcs_l2_l3, overlap_l1_l2, overlap_l2_l3
Initial=generate_random_graph(100000)
Initial_graph=Initial[0]
total_run=3000
Variables=np.zeros((total_run,8))
Variables[0,0]=Initial[1] #number of nodes in the first level
Variables[0,1]=Initial[2]  # number of nodes in the second level
Variables[0,2]=Initial[3]  # number of nodes in the third level
Variables[0,3]=Initial[4]  #number of arcs from l1 to l2
Variables[0,4]=Initial[5]  #number of arcs from l2 to l3
Variables[0,5]=Initial[6]  #number of overlapping arcs from l1 to l2
Variables[0,6]=Initial[7]  #number of overlapping arcs from l2 to l3
Stored_graphs=[]
Stored_graphs.append(Initial_graph)
def is_graph_unique(new_graph, stored_graphs):
    new_graph_edges = set(new_graph.edges())
    for stored_graph in stored_graphs:
        stored_graph_edges = set(stored_graph.edges())
        if new_graph_edges == stored_graph_edges:
            return False
    return True
iter_=0
while iter_<=(total_run-2):
    seed_number=iter_+1
    new_ = generate_random_graph(seed_number)
    new_graph=new_[0]
    if is_graph_unique(new_graph, Stored_graphs):
        Stored_graphs.append(new_graph)
        iter_=iter_+1
        Variables[iter_,0]=new_[1]
        Variables[iter_,1]=new_[2]
        Variables[iter_,2]=new_[3]
        Variables[iter_,3]=new_[4]
        Variables[iter_,4]=new_[5]
        Variables[iter_,5]=new_[6]
        Variables[iter_,6]=new_[7]
outcome_maxflow=np.zeros((total_run,1))
for i in list(range(0,total_run)):
   A=Stored_graphs[i]
   G_transformed = nx.DiGraph()
   for node in A.nodes:
    # Split node into node_in and node_out
     G_transformed.add_edge(f"{node}_in", f"{node}_out")
     G_transformed[f"{node}_in"][f"{node}_out"]['Capacity']=A.nodes[node]['Capacity']
   for u, v, data in A.edges(data=True):
     G_transformed.add_edge(f"{u}_out", f"{v}_in", **data)

   
   for u, v, data in G_transformed.edges(data=True):
    capacity = data.get('Capacity', 'No capacity set')
    print(f"Edge {u} -> {v}: Capacity = {capacity}")  
   source='S_in'
   sink='T_out'
   m = gp.Model("Network_Dual_LB")
   theta = m.addVars(G_transformed.edges, vtype=GRB.CONTINUOUS, name="theta")
   pi = m.addVars(G_transformed.nodes,  vtype=GRB.CONTINUOUS, name="pi")
   mu = m.addVars(G_transformed.edges,  vtype=GRB.CONTINUOUS, name="mu")
  

   #Objective function: Min-cut (Dual of max flow)
   m.setObjective(gp.quicksum(G_transformed[i][j]['Capacity'] *  mu [i, j] for i, j in G_transformed.edges), GRB.MINIMIZE)

   # Constraints: Ensure connectivity
   m.addConstrs(pi[i]-pi[j]+theta[i, j]>=0 for i, j in G_transformed.edges)
   m.addConstr(pi[sink]-pi[source]>=1)
   m.addConstrs(mu[i,j]- theta[i,j] >= 0 for  i, j in G_transformed.edges)
   m.optimize()
   outcome_maxflow[i]= m.objVal  
   paths = list(nx.all_simple_paths(G_transformed, source='S_in', target='T_out'))
   mu_values={i: mu[i].X for i in mu}
   min_cut=0
   index_mu=[{'index': i, 'key':key, 'value':value} for i, (key, value) in enumerate(mu_values.items())]
   for k in list(range(0,len(G_transformed.edges))):
       min_cut=min_cut+ index_mu[k]['value']
   Variables[i,7]=min_cut
   #for path in paths:
    #print("Path:", " -> ".join(path))
    #path_capacities = [G_transformed[u][v]['Capacity'] for u, v in zip(path, path[1:])]
    #print("Capacities along the path:", path_capacities)
   
    

unique_values = []
[unique_values.append(x) for x in outcome_maxflow if x not in unique_values]
merged= np.hstack((Variables,outcome_maxflow))
merged_pd=pd.DataFrame(merged)
file_path='Dataset_unique_103.xlsx'
merged_pd.to_excel(file_path, index=False)

