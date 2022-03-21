from more_itertools import last
from classes import *
import matplotlib.pyplot as plt
import random
import dearpygui.dearpygui as dpg


N_NODES = 0
CLUSTERING = 0
ITEMS = ['a', 'b', 'c']
system = None
def set_config():
    global N_NODES, CLUSTERING, system
    NEW_NNODES = dpg.get_value("n_nodes")
    NEW_CLUSTERING = dpg.get_value("clustering")
    if NEW_NNODES != N_NODES and NEW_CLUSTERING != CLUSTERING:
        N_NODES = NEW_NNODES
        CLUSTERING = NEW_CLUSTERING
        system = System(n_nodes=N_NODES, node_clustering=CLUSTERING, food_types=ITEMS)
        dpg.set_value("config_notif", "Set parameters.")
        dpg.configure_item("config_notif", show=True)
    else:
        dpg.set_value("config_notif", "Already set.")
cur_request = [[], [None, None, defaultdict(lambda: 0)]]
def start():
    global cur_request
    def link_callback(sender, app_data):
        global cur_request
        # app_data -> (link_id1, link_id2)
        name_source, food_type = app_data[0].split('_')
        amt = dpg.get_value(f'amount_{app_data[0]}')
        name_receive = app_data[1].split('_')[0]
        if amt == 0:
            return
        x = dpg.add_node_link(app_data[0], app_data[1], parent=sender, tag=f'{app_data[0]}|{app_data[1]}')
        if cur_request[0]:
            # print([name_source, name_receive])
            # print(cur_request[1][:2])
            if [name_source, name_receive] != cur_request[1][:2]:
                # print([req for req in cur_request[0]])
                [dpg.delete_item(req) for req in cur_request[0]]
                cur_request[0] = []
        cur_request[0].append(x)
        cur_request[1][0] = name_source
        cur_request[1][1] = name_receive
        cur_request[1][2][food_type] = amt
        # print(cur_request)
    # callback runs when user attempts to disconnect attributes
    def delink_callback(sender, app_data):
        # app_data -> link_id
        global cur_request
        app_data_s = app_data.split('|')
        name_source, food_type = app_data_s[0].split('_')
        name_receive = app_data_s[1].split('_')[0]
        dpg.delete_item(app_data)
        cur_request[1][2][food_type] = 0
        if all([cur_request[1][2][food] == 0 for food in ITEMS]):
            cur_request = [[], [None, None, defaultdict(lambda: 0)]]

    def update_requests():
        global system
        recommended = system.recommend_requests(topn=min(5, len(system.requests)), cache=min(3, len(system.requests)), remove=False)
        v = system.print_route_information(recommended)
        dpg.set_value('requests_recommendation', v)
        pass


    def submit():
        global system, cur_request
        dpg.configure_item("submit_display", show=True)
        if cur_request[1][0]:
            dpg.set_value("submit_display", "Request added.")
            system.add_request(*cur_request[1])
            [dpg.delete_item(connection) for connection in cur_request[0]]
            cur_request = [[], [None, None, defaultdict(lambda: 0)]]
            update_requests()
        else:
            dpg.set_value("submit_display", "Request could not be added.")

    if system is not None:
        dpg.add_window(label='Request Adder', tag='requests_add', width=600, height=400, pos=(200, 0))
        dpg.add_window(label='Request Recommender', tag='requests_accept', width=400, height=400, pos=(800, 0))
        dpg.add_text("Requests", parent='requests_add')
        dpg.add_text("", parent='requests_accept', tag='requests_recommendation')
        # dpg.add_combo(label='Food Type', items=ITEMS, tag='food', parent='requests_add')
        dpg.add_button(label="Submit", tag='submit', parent='requests_add')
        dpg.set_item_callback("submit", submit)
        dpg.add_text("Request added", parent='requests_add', show=False, tag='submit_display')

        # dpg.add_input_int(label='Food Amount', tag='amount', parent='requests_add', min_value=0, max_value=999, step=1, min_clamped=True)
        dpg.add_node_editor(parent='requests_add', tag='locations', callback=link_callback, delink_callback=delink_callback)
        N_WARE = len(system.nodes[0])
        PADDING = (N_WARE + 1)//2 - 1
        GUESSED_HEIGHT = 100
        for i, (name, node) in enumerate(system.nodes[1].items()):
            with dpg.node(label=name.replace('Restaurant', 'Rest.'), parent='locations', pos=((i//(300//GUESSED_HEIGHT))*100, (i % (300//GUESSED_HEIGHT))*GUESSED_HEIGHT+10)):
                for item in ITEMS:
                    with dpg.node_attribute(label='Output', attribute_type=dpg.mvNode_Attr_Output, tag=f'{name}_{item}'):
                        dpg.add_input_int(label=f"{item}", tag=f'amount_{name}_{item}', min_value=0, max_value=999, step=1, min_clamped=True, width=70)
        for i, (name, node) in enumerate(system.nodes[0].items()):
            with dpg.node(label=name, parent='locations', pos=(450 - (PADDING - i//2)*110, (i % 2)*100+10)):
                with dpg.node_attribute(label="Inventory", attribute_type=dpg.mvNode_Attr_Input, tag=f'{name}_input', shape=dpg.mvNode_PinShape_TriangleFilled):
                    dpg.add_text("Input")
                with dpg.node_attribute(label="Inventory", attribute_type=dpg.mvNode_Attr_Static):
                    dpg.add_text("Output")
                for item in ITEMS:
                    with dpg.node_attribute(label="Inventory", attribute_type=dpg.mvNode_Attr_Output, tag=f'{name}_{item}'):
                        # dpg.add_text(f"Item {item}: {node.inventory[item]}", tag=f'inventory_{name}_{item}')
                        dpg.add_input_int(label=f"{item}", tag=f'amount_{name}_{item}', min_value=0, max_value=node.inventory[item], step=1, min_clamped=True, max_clamped=True, width=70)
    # pass

dpg.create_context()
dpg.create_viewport(title='Feed it Forward Prototype', width=1200, height=500)
with dpg.window(label="Settings", tag='settings', width=200, height=200):
    dpg.add_text("Number of Nodes")
    dpg.add_slider_int(label="", default_value=5, max_value=10, min_value=5, clamped=True, tag='n_nodes')
    dpg.add_text("Clustering")
    dpg.add_slider_float(label="", default_value=100, max_value=200, min_value=10, clamped=True, tag='clustering')
    dpg.add_button(label="Set configuration", tag='config')
    dpg.add_button(label="Start", tag='start')
    dpg.add_text("Set parameters.", show=False, tag='config_notif')
    dpg.set_item_callback("config", set_config)
    dpg.set_item_callback("start", start)

dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()





# for _ in range(5):
#     #add random requests for testing
#     while not s.add_request(random.randint(0, N_NODES - 1), random.randint(0, N_NODES - 1), dict([(x, 1*(random.random()<0.5)) for x in ['a', 'b', 'c']]), priority=random.randint(0, 5)):
#         pass

# # s.requests = [(0, 3, 8, Counter({'a': 1, 'b': 0, 'c': 0})), (1, 3, 9, Counter({'a': 1, 'c': 1, 'b': 0})), (1, 0, 6, Counter({'c': 1, 'a': 0, 'b': 0})), (4, 2, 4, Counter({'a': 1, 'b': 0, 'c': 0})), (2, 1, 0, Counter({'a': 1, 'b': 1, 'c': 1}))]
# # s.requests = [(0, 7, 2, Counter({'c': 1, 'a': 0, 'b': 0})), (1, 3, 5, Counter({'a': 1, 'b': 0, 'c': 0})), (3, 0, 9, Counter({'c': 1, 'a': 0, 'b': 0})), (4, 6, 4, Counter({'c': 1, 'a': 0, 'b': 0})), (4, 8, 1, Counter({'b': 1, 'c': 1, 'a': 0}))]
# # s.requests = [(0, 7, 2, Counter({'c': 1, 'a': 0, 'b': 0})), (1, 3, 5, Counter({'a': 1, 'b': 0, 'c': 0})), (4, 6, 4, Counter({'c': 1, 'a': 0, 'b': 0})), (4, 8, 1, Counter({'b': 1, 'c': 1, 'a': 0}))]
# # print(s.requests)
# recommended = s.recommend_requests(topn=5, cache=3, remove=True)
# # s.print_route_information(recommended)
# s.plot_nodes()
# s.plot_path(recommended)
# # plt.show()
# print()
# recommended = s.recommend_requests(topn=5, cache=3, remove=True)
# if recommended:
#     s.plot_path(recommended)
# # s.print_route_information(recommended)
# # plt.show()
# print()
# recommended = s.recommend_requests(topn=5, cache=3, remove=True)
# if recommended:
#     s.plot_path(recommended)
# # s.print_route_information(recommended)
# # print(s.requests)
# plt.show()