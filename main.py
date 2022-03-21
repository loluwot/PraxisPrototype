from classes import *
import matplotlib.pyplot as plt
import random
N_NODES = 10
s = System(n_nodes=N_NODES, node_clustering=100)

# for _ in range(5):
#     #add random requests for testing
#     while not s.add_request(random.randint(0, N_NODES - 1), random.randint(0, N_NODES - 1), dict([(x, 1*(random.random()<0.5)) for x in ['a', 'b', 'c']]), priority=random.randint(0, 5)):
#         pass

# s.requests = [(0, 3, 8, Counter({'a': 1, 'b': 0, 'c': 0})), (1, 3, 9, Counter({'a': 1, 'c': 1, 'b': 0})), (1, 0, 6, Counter({'c': 1, 'a': 0, 'b': 0})), (4, 2, 4, Counter({'a': 1, 'b': 0, 'c': 0})), (2, 1, 0, Counter({'a': 1, 'b': 1, 'c': 1}))]
s.requests = [(0, 7, 2, Counter({'c': 1, 'a': 0, 'b': 0})), (1, 3, 5, Counter({'a': 1, 'b': 0, 'c': 0})), (3, 0, 9, Counter({'c': 1, 'a': 0, 'b': 0})), (4, 6, 4, Counter({'c': 1, 'a': 0, 'b': 0})), (4, 8, 1, Counter({'b': 1, 'c': 1, 'a': 0}))]
# s.requests = [(0, 7, 2, Counter({'c': 1, 'a': 0, 'b': 0})), (1, 3, 5, Counter({'a': 1, 'b': 0, 'c': 0})), (4, 6, 4, Counter({'c': 1, 'a': 0, 'b': 0})), (4, 8, 1, Counter({'b': 1, 'c': 1, 'a': 0}))]
# print(s.requests)
recommended = s.recommend_requests(topn=5, cache=3, remove=True)
# s.print_route_information(recommended)
s.plot_nodes()
s.plot_path(recommended)
# plt.show()
print()
recommended = s.recommend_requests(topn=5, cache=3, remove=True)
if recommended:
    s.plot_path(recommended)
# s.print_route_information(recommended)
# plt.show()
print()
recommended = s.recommend_requests(topn=5, cache=3, remove=True)
if recommended:
    s.plot_path(recommended)
# s.print_route_information(recommended)
# print(s.requests)
plt.show()