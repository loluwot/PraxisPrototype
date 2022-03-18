from classes import *
import matplotlib.pyplot as plt
import random
N_NODES = 20
s = System(n_nodes=N_NODES, node_clustering=100)

for _ in range(5):
    #add random requests for testing
    while not s.add_request(random.randint(0, N_NODES - 1), random.randint(0, N_NODES - 1), dict([(x, 1*(random.random()<0.5)) for x in ['a', 'b', 'c']]), priority=random.randint(0, 5)):
        pass
print(s.requests)
recommended = s.recommend_requests(5, 3, remove=True)
s.print_route_information(recommended)
s.plot_nodes()
s.plot_path(recommended)
print()
recommended = s.recommend_requests(5, 3, remove=True)
s.plot_path(recommended)
s.print_route_information(recommended)
print()
recommended = s.recommend_requests(5, 3, remove=True)
s.plot_path(recommended)
s.print_route_information(recommended)
# print(s.requests)
plt.show()