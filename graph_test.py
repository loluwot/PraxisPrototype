import networkx as nx

ex = nx.petersen_graph()
pos = nx.nx_agraph.graphviz_layout(ex)
print(pos)