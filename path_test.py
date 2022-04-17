from collections import defaultdict, Counter
from typing import Dict, List
import numpy as np
import random
import math
import heapq
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import matplotlib.pyplot as plt
import json
import itertools
import re
from string import ascii_lowercase

# from torch import R
from helpers import format_helper, null_coalesce
import networkx as nx
import matplotlib.pyplot as plt
import requests
from urllib.parse import quote_plus
import googlemaps


class ComparableCounter(Counter):
    def __lt__(self, other):
        return list(self.values()) < list(other.values())     

def min_ncars(dists, n_cars=1):
    if n_cars == 1:
        return sum(dists)
    return max(sum((sort := sorted(dists))[:len(dists) + 1 - n_cars]), min_ncars(sort[len(dists) + 1 -n_cars:], n_cars=n_cars-1))

def recommend_requests(requests, n_nodes, dist_matrix, n_cars=1, topn=10, cache=3, cur_location=0, remove=False, request_ids=None):
    #caches "cache" number of paths that are non intersecting; that way recommendations work in system with multiple drivers
    #assume cars can start anywhere? (maybe a later addition)
    fulfill = requests
    # sorted_req = sorted(self.requests)
    #default to taking topn requests if no indices provided
    #or tools stuff
    index = n_nodes
    assignment_to_node = dict([(i, i) for i in range(n_nodes)]) 
    node_to_assignments = dict([(i, [i]) for i in range(n_nodes)]) 
    assignment_count = dict([(i, 0) for i in range(n_nodes)])
    assignment_count[cur_location] += 1
    
    augmented_dist_matrix = defaultdict(lambda: defaultdict(lambda: math.inf))
    for i in range(n_nodes):
        for j in range(n_nodes):
            augmented_dist_matrix[i][j] = dist_matrix[i][j]

    def assignment_helper(v, index):
        assignment_count[v] += 1
        cur_v = v
        while assignment_count[v] > len(node_to_assignments[v]):
            assignment_to_node[index] = v
            #new node should have distance 0
            #new node should have equivalent connections to other nodes
            for v_alt in node_to_assignments[v]:
                # print('ALT:', v_alt, 'REAL:', v)
                for i in range(index):
                    augmented_dist_matrix[index][i] = augmented_dist_matrix[v_alt][i]
                    augmented_dist_matrix[i][index] = augmented_dist_matrix[i][v_alt]
                augmented_dist_matrix[v_alt][index] = 0
                augmented_dist_matrix[index][v_alt] = 0 
            node_to_assignments[v].append(index)
            augmented_dist_matrix[index][index] = 0 
            cur_v = index
            index += 1
        return index, cur_v

    tups = []
    occured = set()
    for _, req in fulfill:
        index, tup_new1 = assignment_helper(req.source, index)
        index, tup_new2 = assignment_helper(req.dest, index)
        # print('FOR REQUEST BETWEEN', req.source, req.dest)
        # print(tup_new1)
        # print(tup_new2)
        occured.add(tup_new1)
        occured.add(tup_new2)
        tups.append((tup_new1, tup_new2, req.amounts))

    # print('TUPS', tups)
    # print('FINAL INDEX', index)
    # print('ASSIGNMENTS', node_to_assignments)
    # print('PARENT NODE', assignment_to_node)
    # print('OCCURENCES', occured)
    # print('\n'.join([' '.join([str(augmented_dist_matrix[i][j]) for j in range(index)]) for i in range(index)]))
    # print(any([augmented_dist_matrix[i][j] == math.inf for i in range(index) for j in range(index)]))

    manager = pywrapcp.RoutingIndexManager(index,
                                        cache, cur_location) #assume all vehicles start and end at common location; not necessarily a good assumption if drivers can start from home, etc..
    # manager = pywrapcp.RoutingIndexManager(10,
    #                                    3, 0) #assume all vehicles start and end at common location; not necessarily a good assumption if drivers can start from home, etc..
    routing_parameters = pywrapcp.DefaultRoutingModelParameters()
    # routing_parameters.solver_parameters.trace_propagation = True
    # routing_parameters.solver_parameters.trace_search = True

    routing = pywrapcp.RoutingModel(manager, routing_parameters)
    # print([manager.NodeToIndex(node) for node in range(index) if node not in occured])
    L = [manager.NodeToIndex(node) for node in range(index) if node not in occured]
    if L:
        routing.AddDisjunction(L, -1) #nodes not in requests should not appear at all
    # routing.AddDisjunction([manager.NodeToIndex(node) for node in range(index) if node in occured], 30000) #nodes in requests should be able to be dropped at a cost


    def distance_callback(from_index, to_index):
        n1, n2 = [manager.IndexToNode(x) for x in [from_index, to_index]]
        # print('CALLED BACK DISTANCE', augmented_dist_matrix[n1][n2])
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
    distance_dimension.SetGlobalSpanCostCoefficient(10)
    trimmed_requests = defaultdict(lambda: defaultdict(lambda: ComparableCounter()))
    # print(tups)
    starter = set()
    for *tup, amount in tups:
        starter.add(tup[0])
        trimmed_requests[tup[0]] = (tup[1], amount)
        trimmed_requests[tup[1]] = (tup[0], ComparableCounter(dict([(k, -v) for k, v in amount.items()])))
        pickup_index, delivery_index = [manager.NodeToIndex(x) for x in tup]
        # print(pickup_index, delivery_index)
        # print('INDICES', pickup_index, delivery_index)
        routing.AddPickupAndDelivery(pickup_index, delivery_index)
        routing.solver().Add(
        routing.VehicleVar(pickup_index) == routing.VehicleVar(
            delivery_index))
        routing.solver().Add(
            distance_dimension.CumulVar(pickup_index) <=
            distance_dimension.CumulVar(delivery_index))
    # print('TRIMMED REQUESTS', trimmed_requests)
    # print('GOT HERE')
    # print(pywrapcp.DefaultRoutingSearchParameters())
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION)
    search_parameters.local_search_metaheuristic = (
    # routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    routing_enums_pb2.LocalSearchMetaheuristic.AUTOMATIC)
    search_parameters.time_limit.seconds = 10
    # search_parameters.solution_limit = 1
    search_parameters.log_search = True
    solution = routing.SolveWithParameters(search_parameters)
    # print(routing.status())
    # print(solution)

    routes = [[] for _ in range(cache)]
    distances = [[] for _ in range(cache)]
    satisfied_requests = [[] for _ in range(cache)]

    for v_id in range(cache):
        idx = routing.Start(v_id)
        # print(index)
        dist = 0
        last_node = manager.IndexToNode(idx)
        while not routing.IsEnd(idx):
            N = manager.IndexToNode(idx)
            # print(N)
            dist += augmented_dist_matrix[N][last_node]
            if N in starter:
                satisfied_requests[v_id].append((assignment_to_node[N], assignment_to_node[trimmed_requests[N][0]]))
            
            if len(routes[v_id]) == 0 or assignment_to_node[N] != routes[v_id][-1][0]:
                # print('TUP', N, assignment_to_node[N], list(trimmed_requests[N][1].values()))
                routes[v_id].append([assignment_to_node[N], trimmed_requests[N][1].copy()])
            else:
                # print('TUP', N, assignment_to_node[N], list(trimmed_requests[N][1].values()))
                routes[v_id][-1][1].update(trimmed_requests[N][1])
            # print(routing.NextVar(idx))
            last_node = N
            idx = solution.Value(routing.NextVar(idx))
        # print(routes[v_id])
        N = manager.IndexToNode(idx)
        dist += augmented_dist_matrix[N][last_node]
        if len(routes[v_id]) == 0 or assignment_to_node[N] != routes[v_id][-1][0]:
            routes[v_id].append([assignment_to_node[N], trimmed_requests[N][1]])
        else:
            routes[v_id][-1][1].update(trimmed_requests[N][1])
        routes[v_id] = [routes[v_id][0]] + list(filter(lambda x: any(x[1].values()), routes[v_id][1:-1])) + [routes[v_id][-1]] 
        distances[v_id] = dist
        # print('VEHICLE CHANGE -----')
    # print(satisfied_requests)
    routes = list(zip(routes, satisfied_requests))
    # print('ROUTES', [list(zip(*routes[i][0]))[0] for i in range(cache)])
    # for route in [list(zip(*routes[i][0]))[0] for i in range(cache)]:
    #     for i in range(len(route) - 1):
    #         self.cache_graph.add_edge(self.convert_to_str(route[i]), self.convert_to_str(route[i+1]))
    routes = list(filter(lambda x: len(x[0]) > 2, routes))
    # print(routes)
    # print(distances)
    
    return min_ncars(distances, n_cars=n_cars)# - sum([dist_matrix[(z := list(zip(*routes[i][0]))[0])[1]][0] + dist_matrix[z[-2]][0] for i in range(len(routes))])

from string import ascii_lowercase, ascii_uppercase
from random import randint
# from classes import *

class Request:
    def __init__(self, source, dest, amounts) -> None:
        self.source = source
        self.dest = dest
        self.amounts = amounts

# test = [50, 40, 30]
# print(min_ncars(test, n_cars=2))
N_TRIALS = 100

def naive_l(path, dist_matrix):
    last = 0
    naive = sum([dist_matrix[last][req.source] + dist_matrix[req.source][(last := req.dest)] for req in path])
    naive += dist_matrix[ex_requests[-1].dest][0]
    return naive

for n_cars in range(1, 4):
    res = [0, 0]
    for _ in range(N_TRIALS):
        n_requests = 20
        def rand_without(ma, x):
            while (y := random.randint(0, ma)) == x:
                pass
            return y

        dist_matrix = defaultdict(lambda: defaultdict(lambda: math.inf))
        # food_types = [FoodType(food_type=random.choice(ascii_lowercase), expiry_date=sorted([None if not x else x for x in random.sample(range(10), 2)], key=lambda x: x if x is not None else (0 if random.random() < 0.5 else math.inf))) for _ in range(3)]
        food_types = random.sample(ascii_lowercase, 3)
        n_nodes = 10
        ex_requests = [Request((used:=random.randint(0, n_nodes-1)), rand_without(n_nodes-1, used), dict([(food_type, randint(0, 3)) for food_type in food_types])) for _ in range(n_requests)]
        used_nodes = list(set(itertools.chain.from_iterable([[req.source, req.dest] for req in ex_requests]))) + [0]
        coord_assignment = dict([(k, np.random.randn(2)/math.sqrt(2)*10) for k in range(n_nodes)])
        for node in used_nodes:
            for node2 in used_nodes:
                dist_matrix[node][node2] = np.linalg.norm(coord_assignment[node] - coord_assignment[node2])

        reqs = list(zip([0]*len(ex_requests), ex_requests))
        net_dist = recommend_requests(reqs, n_nodes, dist_matrix, n_cars=n_cars)
        # naive = sum([dist_matrix[0][req.source] + dist_matrix[req.source][req.dest] + dist_matrix[0][req.dest] for req in ex_requests])
        naive = []
        for car in range(n_cars-1):
            naive.append(naive_l(ex_requests[car*(n_requests)//n_cars:(car+1)*(n_requests)//n_cars], dist_matrix))
        naive.append(naive_l(ex_requests[(n_cars-1)*(n_requests)//n_cars:], dist_matrix))
        naive = min_ncars(naive, n_cars=n_cars)
        
        print('USING PATHS:', net_dist)
        print('NAIVE:', naive)
        res[0] += net_dist
        res[1] += naive
    print([res[i]/N_TRIALS for i in range(2)])
    input()