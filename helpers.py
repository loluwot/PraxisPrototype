ATTRIBUTE_TO_NAME = {'food_type': 'Food Type', 'expiry_date': 'Expiry Date'}
ATTRIBUTE_TO_IMAGE = {'food_type': 'apple', 'expiry_date': 'date'}

FORMATS = ['{}: less than {}', '{}: more than {}', '{}: from {} to {}']
def safe_index(L, val):
    try:
        return L.index(val)
    except ValueError:
        return -1

def format_helper(prop, r_tup):
    if prop not in ATTRIBUTE_TO_NAME:
        return ''
    if type(r_tup) == tuple or type(r_tup) == list: 
        # r_tup = r_tup[::-1]
        return FORMATS[safe_index(r_tup, None)].format(*([ATTRIBUTE_TO_NAME[prop]] + [r_val for r_val in r_tup if r_val is not None]))
    return f'{ATTRIBUTE_TO_NAME[prop]}: {r_tup}'

from typing import Tuple
import networkx as nx
import math

def embed_graph(G, width=300, height=300, padding=10):
    pos = nx.nx_agraph.graphviz_layout(G)
    # print(pos.values())
    valx, valy = zip(*pos.values())
    mtup = [(min(val), max(val)) for val in [valx, valy]]
    dims = [width, height]
    return dict(zip(pos.keys(), [[(z - mtup[i][0])/(mtup[i][1] - mtup[i][0])*(dims[i] - padding*2) + padding for i, z in enumerate(val)] for val in pos.values()]))

def normalize(vec : Tuple):
    dist = math.hypot(*vec)
    return [v/dist for v in vec]

def perp_vector(vec : Tuple):
    if not all(vec):
        return [0 if i != vec.index(0) else 1 for i in range(2)]
    return normalize([1, -vec[0]/vec[1]])

def null_coalesce(v1, v2):
    return (L := [v1, v2])[(0 if None not in L else (1 - L.index(None)))]

