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

DISTMATRIX_APIKEY = 'AIzaSyC_SpwD0cp9H7IoY5Pnnl6dTDqkispU6E0'
# STATICMAPS_APIKEY = 'AIzaSyCPa7ywj1-5mj08sfVJTLsEjtwDnJYaviQ'
CLIENT = googlemaps.Client(DISTMATRIX_APIKEY)
# CLIENT2 = googlemaps.Client(STATICMAPS_APIKEY)
GOOGLE_API_LINK = 'https://maps.googleapis.com/maps/api/distancematrix/json?units=metric&origins={}&destinations={}&key={}'

class ComparableCounter(Counter):
    def __lt__(self, other):
        return list(self.values()) < list(other.values())     

class FoodType:
    @classmethod
    def fromstr(cls, query_s):
        print(json.loads(query_s))
        return cls(**json.loads(query_s))

    def __init__(self, **args) -> None:
        for k, v in args.items():
            setattr(self, k, v)
        # setattr(self, 'food_type', getattr(self, 'food_type').replace('Andy ', ''))
        self.rawargs = args
        # self.filters = dict([(k, str(v)) for k, v in args.items()])

    def __str__(self) -> str:
        return json.dumps(self.rawargs)

    def __hash__(self) -> int:
        return hash(str(self))

    def __eq__(self, __o: object) -> bool:
        return hash(self) == hash(__o)

    def readable_raw(self) -> List:
        L = []
        for prop in dir(self):
            s = format_helper(prop, getattr(self, prop))
            if s:
                L.append((prop, s))
        return L

    def readable(self) -> str:
        return ', '.join(list(zip(*self.readable_raw()))[1])

class Node:
    def __init__(self, **args) -> None:
        # self.location = location
        # self.server_id = server_id
        # self.source = source
        # self.name = name
        for k, v in args.items():
            setattr(self, k, v)
            
        # print(args)
        # if not source:
        #     self.inventory = defaultdict(lambda: 0) if inventory is None else defaultdict(lambda: 0, inventory)
        # else:
        #     self.inventory = defaultdict(lambda: math.inf)
        
    def get_inv(self, food_types):
        return dict(zip(food_types, [query_warehouse(str(food_type), self.server_id) for food_type in food_types]))

    def __hash__(self) -> int:
        return hash(self.server_id)

    def __eq__(self, __o: object) -> bool:
        return hash(self) == hash(__o)

    def json(self):
        return {'location': self.location, 'id': self.server_id, 'source': self.source, 'name': self.name}

class Request:
    def __init__(self, source, dest, amounts, request_id) -> None:
        self.source = source
        self.dest = dest
        self.amounts = amounts
        self.request_id = request_id

    def __eq__(self, __o: object) -> bool:
        return (self.request_id == __o if not hasattr(__o, 'request_id') else self.request_id == __o.request_id)

    def json(self):
        return {'source': self.source, 'dest': self.dest, 'inventory': [{'food_type': str(k), 'amount': v} for k, v in self.amounts.items()]}

    def __lt__(self, other):
        return self.request_id < other.request_id

class FakeServer:
    def __init__(self) -> None:
        self.id_to_node = dict()
        self.nodes = []
        self.available_nid = 0
        self.requests = [] #request
        self.food_types = [FoodType(food_type=random.choice(ascii_lowercase), expiry_date=sorted([None if not x else x for x in random.sample(range(10), 2)], key=lambda x: x if x is not None else (0 if random.random() < 0.5 else math.inf))) for _ in range(3)]

    def add_node(self, location, source=False):
        new_node = Node(location=location, server_id=self.available_nid, source=source, name=f'TEST NAME {self.available_nid}')
        self.nodes.append(new_node)
        self.id_to_node[self.available_nid] = new_node
        self.available_nid += 1

    def add_request(self, source_id, dest_id, amounts, flip=False):
        not_used_id = 0
        for req in self.requests:
            if req.request_id + 1 not in self.requests:
                not_used_id = req.request_id + 1
                break
        if not flip:
            self.requests.append(Request(source_id, dest_id, amounts, not_used_id))
        else:
            self.requests.append(Request(dest_id, source_id, amounts, not_used_id))

    def add_random_nodes(self, n_nodes):
        node_clustering = 20
        for _ in range(n_nodes):
            self.add_node(list(np.rint(np.random.randn(2)*node_clustering/math.sqrt(2))), source=random.random() > 0.5)
        
    def add_random_requests(self, n_requests):
        for _ in range(n_requests):
            self.add_request((u := random.randint(1, self.available_nid - 1)), random.randint(0, u-1), dict([(food_type, random.randint(0, 3)) for food_type in self.food_types]), flip=random.random() > 0.5)
            print(self.requests[-1].request_id)
        
server = FakeServer()
server.add_random_nodes(6)

SERVER_URL = 'https://ajwmagnuson.pythonanywhere.com/extdata/'
#SERVER SIDE QUERIES
#IMPLEMENT WHEN FUNCTIONAL

def query_warehouse_id(warehouse_id):
    # return server.id_to_node[warehouse_id].json() #{'location', 'id', 'source'}
    res = requests.get(SERVER_URL, {'q': 'get_location', 'id': warehouse_id})
    return res.json()

def query_warehouse(query, warehouse_id) -> int:
    return math.inf

def query_requests() -> List:
    res = requests.get(SERVER_URL, {'q': 'get_requests'})
    return [req['pk'] for req in res.json()]
    # return [request.request_id for request in server.requests] #should return list of ids

def query_request_id(request_id) -> Dict:
    # print('QUERYING REQUEST_ID', request_id)
    # return server.requests[server.requests.index(request_id)].json() #{'source', 'dest', 'inventory': [{'food_type', 'amount'}]}

    res = requests.get(SERVER_URL, {'q': 'get_request', 'id': request_id})
    return res.json()

def del_request_id(request_id):
    pass
    # server.requests.remove(request_id)

def get_nodes() -> List:
    res = requests.get(SERVER_URL, {'q': 'get_locations'})
    return [req['pk'] for req in res.json()]

NAMES = ['Warehouse', 'Restaurant']
DIST_TYPE = 2 #2 norm or 1 norm

def dist_func(location1, location2):
    if location1 == location2:
        return 0
    # print(location1)
    # print(location2)
    # res = requests.get(GOOGLE_API_LINK.format(quote_plus(location1 + ' ON'), quote_plus(location2 + ' ON'), DISTMATRIX_APIKEY))
    # # print(res.json())
    # return res.json()['rows'][0]['elements'][0]['distance']['value']/1000

    origins = [location1 + ' ON']
    destinations = [location2 + ' ON']
    matrix = CLIENT.distance_matrix(origins, destinations)
    return matrix['rows'][0]['elements'][0]['distance']['value']/1000

    # return 
    # return round(np.linalg.norm(np.array(list(map(float, location1))) - np.array(list(map(float, location2))), ord=DIST_TYPE))

class System:
    def __init__(self) -> None:
        self.food_types = []
        self.nodes = [dict(), dict()] #sinks, sources
        self.count = [0, 0]
        #node generation
        # for i in range(n_nodes):
        #     S = random.random() < source_ratio if source_ratio < 1 else i < source_ratio
        #     node = Node(np.rint(np.random.randn(2)*node_clustering/math.sqrt(2)), source=S, inventory=dict(zip(food_types, initial_inventory)))
        #     self.nodes[S][f'{NAMES[S]} {self.count[S]}'] = node
        #     self.count[S] += 1
    
        #defaults
        self.dist_matrix = defaultdict(lambda: defaultdict(lambda: math.inf))
        # self.combined_dict = {**self.nodes[0], **self.nodes[1]}
        self.combined_dict = dict()
        self.index_to_node = []
        self.node_to_index = dict()
        self.requests = []
        self.cached = []
        self.cached_requests = None
        self.request_ids = []
        self.id_to_request = dict()
        self.n_nodes = 0
        self.cache_graph = nx.Graph()

        starting_nodes = get_nodes()
        for node in starting_nodes:
            self.add_node_from_id(node)
        #construct dist matrix
        # for from_i in range(n_nodes):
        #     for to_i in range(from_i+1):
        #         v = round(np.linalg.norm(self.combined_dict[self.index_to_node[from_i]].location - self.combined_dict[self.index_to_node[to_i]].location, ord=DIST_TYPE))
        #         self.dist_matrix[from_i][to_i] = v
        #         self.dist_matrix[to_i][from_i] = v
        #         #trivial dist matrix
        #         # self.dist_matrix[from_i][to_i] = 0 if from_i == to_i else 10
        #         #probably use something like google distmatrix api to find dist irl

    def add_node(self, **args): #source is donor, sink is warehouse
        # print('ADDING NODE', server_id)
        node = Node(**args)
        #updating name dicts and counts
        S = int(args['source'])
        node_name = f'{NAMES[S]} {args["server_id"]}' if 'name' not in args else args['name']
        if node_name in self.index_to_node:
            return self.node_to_index[node_name]
        self.nodes[S][node_name] = node
        self.combined_dict[node_name] = node
        self.count[S] += 1
        #updating conversions
        self.index_to_node.append(node_name)
        last = len(self.index_to_node) - 1
        self.node_to_index[node_name] = last
        #updating dist matrix
        for index in range(len(self.index_to_node)):
            # self.dist_matrix[index][last] = (dist := dist_func(self.convert_tos_node(index).location, args['location']))
            self.dist_matrix[index][last] = (dist := dist_func(self.convert_to_node(index).address, args['address']))
            self.dist_matrix[last][index] = dist
        self.n_nodes += 1
        self.cache_graph.add_node(node_name)
        for x in self.cache_graph.nodes():
            if x != node_name:
                self.cache_graph.add_edge(node_name, x)
        return last

    def add_node_from_id(self, wid):
        prop_dict = query_warehouse_id(wid)
        return self.add_node(**prop_dict, server_id=wid) #assume it at least has name, location and source

    def convert_to_index(self, x):
        if type(x) == str:
            return self.node_to_index[x]
        return x

    def convert_to_str(self, x):
        if type(x) == int:
            return self.index_to_node[x]
        return x

    def convert_to_node(self, x) -> Node:
        return self.combined_dict[self.convert_to_str(x)]

    def add_request(self, source, dest, amounts, request_id, priority=0): #maybe calculate priority based on combination of user inputted variable, expiry date, etc.. lower is more urgent        
        if source == dest:
            print('Invalid request, source == dest.')
            return 0
        cur_inv = self.convert_to_node(source).get_inv(list(amounts.keys()))
        if not all([cur_inv[k] >= v for k, v in amounts.items()]):
            print('Invalid request, not enough inventory.')
            return 0
        # print('SYSTEM ADDING REQUEST WITH ID', request_id)
        #creates request object and pushes it into queue
        heapq.heappush(self.requests, (req := (priority, Request(self.convert_to_index(source), self.convert_to_index(dest), ComparableCounter(amounts), request_id))))
        self.id_to_request[request_id] = req
        return 1

    def remove_request(self, request_id):
        # print('SYSTEM REQUEST-ID TO REMOVE', request_id)
        _, requests = zip(*self.requests)
        del_request_id(request_id)
        self.request_ids.remove(request_id)
        # print('SYSTEM NEW IDS', self.request_ids)
        del self.requests[requests.index(request_id)]
        del self.id_to_request[request_id]

    def update_requests(self): #periodically fetch new requests from server
        request_ids = query_requests()
        # print('SYSTEM NEW REQUEST_IDS', request_ids)
        unique_ids = set(request_ids)
        unique_selfids = set(self.request_ids)
        if unique_ids != unique_selfids:
            new_requests = list(unique_ids - unique_selfids)
            # print('SYSTEM New requests', new_requests)
            removed_requests = list(unique_selfids - unique_ids)
            # print('SYSTEM Removed requests', removed_requests)
            conv_requests = [query_request_id(request_id) for request_id in new_requests]
            location_ids = [(self.add_node_from_id(q['source']), self.add_node_from_id(q['dest'])) for q in conv_requests]
            food_inventory = [dict([(FoodType.fromstr(json.dumps({'food_type':inv_item['name'], 'expiry_date': inv_item['expiry_date']})), inv_item['amount']) for inv_item in q['inventory']]) for q in conv_requests]
            net_food_types = itertools.chain.from_iterable([inv.keys() for inv in food_inventory])
            self.food_types.extend(set(net_food_types) - set(self.food_types))
            res = [self.add_request(source, dest, amounts, request_id) for (source, dest), amounts, request_id in zip(location_ids, food_inventory, new_requests)]
            [self.remove_request(request_id) for request_id in removed_requests]
            if not all(res):
                print("Warning: some requests could not be added due to errors.")
            self.request_ids = request_ids # FIX LATER; CANT ADD ALL IDS SINCE SOME MIGHT HAVE ERRORED
    # def plot_nodes(self):
    #     for k, node in self.combined_dict.items():
    #         # print(node.location.tolist())
    #         plt.plot(*node.location.tolist(), 'bo')
    #         if self.node_to_index[k] == 0:
    #             plt.plot(*node.location.tolist(), 'ro')   
    
    # def plot_path(self, route):
    #     route_ids, _ = zip(*route[0])
    #     # print(route_ids)
    #     R_NODES = 0.3
    #     points = [self.combined_dict[self.index_to_node[ids]].location for ids in route_ids]
    #     [plt.arrow(points[i][0], points[i][1], (points[i+1] - points[i])[0]*(1-R_NODES/np.linalg.norm(points[i+1] - points[i])), (points[i+1] - points[i])[1]*(1-R_NODES/np.linalg.norm(points[i+1] - points[i])), length_includes_head=True, head_width=0.3) for i in range(len(points) - 1)]
    #     # plt.plot(x, y)

    def satisfy_path(self, idx):
        solved_reqs = self.cached[idx][0][1]
        removed_requests = []
        route = self.cached[idx][0][0]
        # for i in range(len(route) - 1):
        #     try:
        #         self.cache_graph.remove_edge(self.convert_to_str(route[i][0]), self.convert_to_str(route[i+1][0]))
        #     except nx.NetworkXError:
        #         pass
        for i, (_, request) in enumerate(self.cached_requests):
            # print(request)
            if (request.source, request.dest) in solved_reqs:
                self.remove_request(request.request_id)
                removed_requests.append(i)
        for i in removed_requests[::-1]:
            del self.cached_requests[i]
            # pass
        del self.cached[idx]


    def recommend_requests(self, topn=10, cache=3, cur_location=0, remove=False, request_ids=None):
        #caches "cache" number of paths that are non intersecting; that way recommendations work in system with multiple drivers
        #assume cars can start anywhere? (maybe a later addition)
        fulfill = heapq.nsmallest(topn, self.requests) if request_ids is None else [self.id_to_request[rid] for rid in request_ids]
        if self.cached and self.cached_requests == fulfill:
            return self.cached[0]

        # sorted_req = sorted(self.requests)
        #default to taking topn requests if no indices provided
        if not fulfill:
            print('No requests')
            return None, None

        #or tools stuff
        index = self.n_nodes
        assignment_to_node = dict([(i, i) for i in range(self.n_nodes)]) 
        node_to_assignments = dict([(i, [i]) for i in range(self.n_nodes)]) 
        assignment_count = dict([(i, 0) for i in range(self.n_nodes)])
        assignment_count[cur_location] += 1
        
        augmented_dist_matrix = defaultdict(lambda: defaultdict(lambda: math.inf))
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                augmented_dist_matrix[i][j] = self.dist_matrix[i][j]

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
            print('FOR REQUEST BETWEEN', req.source, req.dest)
            print(tup_new1)
            print(tup_new2)
            occured.add(tup_new1)
            occured.add(tup_new2)
            tups.append((tup_new1, tup_new2, req.amounts))

        # print('TUPS', tups)
        print('FINAL INDEX', index)
        print('ASSIGNMENTS', node_to_assignments)
        print('PARENT NODE', assignment_to_node)
        print('OCCURENCES', occured)
        print('\n'.join([' '.join([str(augmented_dist_matrix[i][j]) for j in range(index)]) for i in range(index)]))
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
        print(routing.status())
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
        print('ROUTES', [list(zip(*routes[i][0]))[0] for i in range(cache)])
        # for route in [list(zip(*routes[i][0]))[0] for i in range(cache)]:
        #     for i in range(len(route) - 1):
        #         self.cache_graph.add_edge(self.convert_to_str(route[i]), self.convert_to_str(route[i+1]))
        
        routes = list(filter(lambda x: len(x[0]) > 2, routes))
        self.cached = list(zip(routes, distances))
        self.cached_requests = fulfill
    
        return routes[0], distances[0]

    def print_route_information(self, x):
        s = ''
        if not x:
            return s
        # print(x)
        for i, tup in enumerate(x[0]):
            # print('TUPLES', tup)
            # print('NODE INDEX', tup[0])
            s += f"{'Start at' if i == 0 else 'Go to'} {self.index_to_node[tup[0]]}.\n"
            if len(tup[1]) > 0:
                s += '\n'.join([f"{'Pickup' if amt > 0 else 'Dropoff'} " +  f"{abs(amt)} kg units of {k.readable()}." for k, amt in tup[1].items() if amt != 0])
                s += '\n'
        return s

    def raw_route_info(self, x):
        s = []
        if not x:
            return s
        
        raw_labels = defaultdict(lambda : None)

        for i, tup in enumerate(x[0]):
            # print('TUPLES', tup)
            # print('NODE INDEX', tup[0])
            s.append((f"{'Start at' if i == 0 else 'Go to'} {self.index_to_node[tup[0]]} (labelled {null_coalesce(raw_labels[self.index_to_node[tup[0]]], i)}).", ))
            raw_labels[self.index_to_node[tup[0]]] = null_coalesce(raw_labels[self.index_to_node[tup[0]]], i)
            if len(tup[1]) > 0:
                s.extend([(f"{'Pickup' if amt > 0 else 'Dropoff'} {abs(amt)} kg of {{{k.food_type}}}.", k) for k, amt in tup[1].items() if amt != 0])
        return s
            
    def print_requests(self):
        s = ''
        sortedreq = sorted(self.requests)
        print(sortedreq)
        for i in range(len(self.requests)):
            amount_s = '\n'.join([f'Amount of {str(food)}: {sortedreq[i][3][food]} kg' for food in self.food_types])
            s += f"Source: {self.convert_to_str(sortedreq[i][1])}\nDest: {self.convert_to_str(sortedreq[i][2])}\nAmounts:\n{amount_s}\n"
        s += '\n'
        return s

    def display_graph(self):
        nx.draw(self.cache_graph)
        plt.show()

AVG_SPEED = 10
STD_SPEED = 1
LOADING_PAUSE = 10
PAUSE_STD = 1
class DeliveryDriver:
    def __init__(self, system : System, path) -> None:
        self.progress = 0
        self.cur_path = path
        self.distance = system.dist_matrix[path[0][0]][path[1][0]]
        self.cur_time = 0
        self.speeds = []
        self.system = system
        self.pause = random.gauss(LOADING_PAUSE, PAUSE_STD)
        
    def update(self):
        if self.pause > 0:
            self.pause -= 1
            return 0
            
        self.progress += (speed := random.gauss(AVG_SPEED, STD_SPEED))
        
        if self.progress >= self.distance:
            completed = self.cur_path.pop(0)
            if len(self.cur_path) <= 1:
                print(f'Completed all deliveries in path.')
                return 1
            self.distance = self.system.dist_matrix[self.cur_path[0][0]][self.cur_path[1][0]]
            self.progress = 0
            self.speeds = []
            self.pause = random.gauss(LOADING_PAUSE, PAUSE_STD)
            print(f'Completed delivery from {completed[0]} to {self.cur_path[0][0]}')
            return -1
        
        self.speeds.append(speed)
        self.cur_time += 1
        return 0

    def get_eta(self):
        THRESHOLD = 3
        avg = np.mean(self.speeds) if len(self.speeds) != 0 else AVG_SPEED
        std = np.std(self.speeds) if len(self.speeds) > THRESHOLD else AVG_SPEED/2
        return [self.cur_time + (self.distance - self.progress)/(avg + sign*std) for sign in [1, -1]] 

    
    