from collections import defaultdict, Counter
from re import M
from aiohttp import request
import numpy as np
import random
import math
import heapq
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import matplotlib.pyplot as plt

class Node:
    def __init__(self, location, source=False, inventory=None) -> None:
        self.location = location
        self.source = source
        if not source:
            self.inventory = defaultdict(lambda: 0) if inventory is None else defaultdict(lambda: 0, inventory)
        else:
            self.inventory = defaultdict(lambda: math.inf)
        
NAMES = ['Warehouse', 'Restaurant']
DIST_TYPE = 2 #2 norm or 1 norm
class System:
    def __init__(self, 
        food_types=('a', 'b', 'c'), 
        node_clustering=4, 
        source_ratio=0.5, 
        n_nodes=10,
        initial_inventory=(2, 2, 2),
        d_random=10) -> None:

        self.food_types = food_types
        self.node_clustering = node_clustering
        self.n_nodes = n_nodes
        self.nodes = [dict(), dict()] #sinks, sources
        self.count = [0, 0]
        #node generation
        for _ in range(n_nodes):
            S = random.random() < source_ratio
            node = Node(np.rint(np.random.randn(2)*math.sqrt(node_clustering/2)), source=S, inventory=dict(zip(food_types, initial_inventory)))
            self.nodes[S][f'{NAMES[S]} {self.count[S]}'] = node
            self.count[S] += 1
        
        #construct dist matrix
        self.dist_matrix = defaultdict(lambda: defaultdict(lambda: math.inf))
        self.combined_dict = {**self.nodes[0], **self.nodes[1]}
        self.index_to_node = list(self.combined_dict.keys())
        self.node_to_index = dict(zip(self.index_to_node, range(len(self.index_to_node))))

        for from_i in range(n_nodes):
            for to_i in range(n_nodes):
                self.dist_matrix[from_i][to_i] = round(np.linalg.norm(self.combined_dict[self.index_to_node[from_i]].location - self.combined_dict[self.index_to_node[to_i]].location, ord=DIST_TYPE))
                #probably use something like google distmatrix api to find dist irl
        
        # print(self.dist_matrix)

        #will act as a priority queue
        self.requests = []
        self.cached = []

    def convert_to_index(self, x):
        if type(x) == str:
            return self.node_to_index[x]
        return x

    def convert_to_str(self, x):
        if type(x) == int:
            return self.index_to_node[x]
        return x

    def convert_to_node(self, x):
        return self.combined_dict[self.convert_to_str(x)]

    def add_request(self, source, dest, amounts, priority=0): #maybe calculate priority based on combination of user inputted variable, expiry date, etc.. lower is more urgent        
        if source == dest:
            print('Invalid request, source == dest.')
            return 0
        if not all([self.convert_to_node(source).inventory[k] >= v for k, v in amounts.items()]):
            print('Invalid request, not enough inventory.')
            return 0
        
        heapq.heappush(self.requests, (priority, self.convert_to_index(source), self.convert_to_index(dest), Counter(amounts)))
        return 1

    def plot_nodes(self):
        for k, node in self.combined_dict.items():
            # print(node.location.tolist())
            plt.plot(*node.location.tolist(), 'bo')
            if self.node_to_index[k] == 0:
                plt.plot(*node.location.tolist(), 'ro')   
    
    def plot_path(self, route):
        route_ids, _ = zip(*route[0])
        # print(route_ids)
        R_NODES = 0.3
        points = [self.combined_dict[self.index_to_node[ids]].location for ids in route_ids]
        def sign(x):
            return abs(x)/x if x != 0 else 0
        [plt.arrow(points[i][0], points[i][1], (points[i+1] - points[i])[0]*(1-R_NODES/np.linalg.norm(points[i+1] - points[i])), (points[i+1] - points[i])[1]*(1-R_NODES/np.linalg.norm(points[i+1] - points[i])), length_includes_head=True, head_width=0.3) for i in range(len(points) - 1)]
        # plt.plot(x, y)

    def recommend_requests(self, topn=10, cache=3, cur_location=0, remove=True):
        #caches "cache" number of paths that are non intersecting; that way recommendations work in system with multiple drivers
        if self.cached:
            if remove:
                popped = self.cached.pop(0)
                solved_reqs = popped[1]
                qs = []
                for _ in range(min(len(self.requests), topn)):
                    v = heapq.heappop(self.requests)
                    if v[1:-1] not in solved_reqs:
                        qs.append(v)
                [heapq.heappush(self.requests, x) for x in qs] #repush
                return popped
            else:
                return self.cached[0]
        fulfill = heapq.nsmallest(topn, self.requests)
        #or tools stuff
        
        index = self.n_nodes
        assignment_to_node = dict([(i, i) for i in range(self.n_nodes)]) 
        node_to_assignments = dict([(i, [i]) for i in range(self.n_nodes)]) 
        assignment_count = dict([(i, 0) for i in range(self.n_nodes)])

        augmented_dist_matrix = defaultdict(lambda: defaultdict(lambda: math.inf))
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                augmented_dist_matrix[i][j] = self.dist_matrix[i][j]

        def assignment_helper(v, index):
            assignment_count[v] += 1
            cur_v = v
            while assignment_count[v] > len(node_to_assignments[v]):
                assignment_to_node[index] = v
                node_to_assignments[v].append(index)
                #new node should have distance 0
                #new node should have equivalent connections to other nodes
                for i in range(index):
                    augmented_dist_matrix[index][i] = augmented_dist_matrix[v][i]
                    augmented_dist_matrix[i][index] = augmented_dist_matrix[i][v]

                augmented_dist_matrix[index][index] = 0 
                augmented_dist_matrix[v][index] = 0
                augmented_dist_matrix[index][v] = 0 

                cur_v = index
                index += 1
            return index, cur_v
        
        tups = []
        occured = set()
        for _, *tup, amount in fulfill:
            index, tup_new1 = assignment_helper(tup[0], index)
            index, tup_new2 = assignment_helper(tup[1], index)
            occured.add(tup_new1)
            occured.add(tup_new2)
            tups.append((tup_new1, tup_new2, amount))
        
        manager = pywrapcp.RoutingIndexManager(len(augmented_dist_matrix),
                                           cache, cur_location) #assume all vehicles start and end at common location; not necessarily a good assumption if drivers can start from home, etc..
        routing = pywrapcp.RoutingModel(manager)
        routing.AddDisjunction([manager.NodeToIndex(node) for node in range(index) if node not in occured], -1) #nodes not in requests should not appear at all
        # routing.AddDisjunction([manager.NodeToIndex(node) for node in range(index) if node in occured], 30000) #nodes not in requests should not appear at all

        def distance_callback(from_index, to_index):
            n1, n2 = [manager.IndexToNode(x) for x in [from_index, to_index]]
            return augmented_dist_matrix[n1][n2]
        
        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        dimension_name = 'Distance'
        routing.AddDimension(
            transit_callback_index,
            0,  # no slack
            300000,  # vehicle maximum travel distance, set to reasonable value
            True,  # start cumul to zero
            dimension_name)
        distance_dimension = routing.GetDimensionOrDie(dimension_name)
        distance_dimension.SetGlobalSpanCostCoefficient(1000)
        trimmed_requests = defaultdict(lambda: defaultdict(lambda: Counter()))
        # print(tups)
        starter = set()
        for *tup, amount in tups:
            starter.add(tup[0])
            trimmed_requests[tup[0]] = (tup[1], amount)
            trimmed_requests[tup[1]] = (tup[0], Counter(dict([(k, -v) for k, v in amount.items()])))
            pickup_index, delivery_index = [manager.NodeToIndex(x) for x in tup]
            # print(pickup_index, delivery_index)
            routing.AddPickupAndDelivery(pickup_index, delivery_index)
            routing.solver().Add(
            routing.VehicleVar(pickup_index) == routing.VehicleVar(
                delivery_index))
            routing.solver().Add(
                distance_dimension.CumulVar(pickup_index) <=
                distance_dimension.CumulVar(delivery_index))
            
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION)
        search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
        search_parameters.time_limit.seconds = 5
        search_parameters.log_search = False
        solution = routing.SolveWithParameters(search_parameters)

        # print(solution)
        routes = [[] for _ in range(cache)]
        satisfied_requests = [[] for _ in range(cache)]

        for v_id in range(cache):
            index = routing.Start(v_id)
            while not routing.IsEnd(index):
                N = manager.IndexToNode(index)
                if N in starter:
                    satisfied_requests[v_id].append((assignment_to_node[N], assignment_to_node[trimmed_requests[N][0]]))
                
                if len(routes[v_id]) == 0 or assignment_to_node[N] != routes[v_id][-1][0]:
                    routes[v_id].append([assignment_to_node[N], trimmed_requests[N][1]])
                else:
                    routes[v_id][-1][1].update(trimmed_requests[N][1])
                index = solution.Value(routing.NextVar(index))

            N = manager.IndexToNode(index)
            if len(routes[v_id]) == 0 or assignment_to_node[N] != routes[v_id][-1][0]:
                routes[v_id].append([assignment_to_node[N], trimmed_requests[N][1]])
            else:
                routes[v_id][-1][1].update(trimmed_requests[N][1])

            routes[v_id] = [routes[v_id][0]] + list(filter(lambda x: any(x[1].values()), routes[v_id][1:-1])) + [routes[v_id][-1]] 

        # print(satisfied_requests)
        routes = list(zip(routes, satisfied_requests))
        routes = list(filter(lambda x: len(x[0]) > 2, routes))
        if remove:
            self.cached = routes[1:]
            # print(routes[0][1])
            solved_reqs = routes[0][1]
            qs = []
            for _ in range(min(topn, len(self.requests))):
                v = heapq.heappop(self.requests)
                if v[1:-1] not in solved_reqs:
                    qs.append(v)
            [heapq.heappush(self.requests, x) for x in qs] #repush
            
        return routes[0]

    def print_route_information(self, x):
        for i, tup in enumerate(x[0]):
            print(f"{'Start at' if i == 0 else 'Goto'} {self.index_to_node[tup[0]]}.")
            if len(tup[1]) > 0:
                print('\n'.join([f"{'Pickup' if amt > 0 else 'Dropoff'} {abs(amt)} units of {k}." for k, amt in tup[1].items() if amt != 0]))
            