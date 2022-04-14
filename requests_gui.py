from os import remove
import dearpygui.dearpygui as dpg
from string import ascii_uppercase, ascii_lowercase
import random
from random import randint
from collections import defaultdict

from matplotlib.pyplot import get
from classes import *
from helpers import ATTRIBUTE_TO_IMAGE, embed_graph, normalize, perp_vector
from gui_helpers import *
from googlemaps.maps import StaticMapMarker
from googlemaps.maps import StaticMapPath

food_types = [FoodType(food_type=random.choice(ascii_lowercase), expiry_date=sorted([None if not x else x for x in random.sample(range(10), 2)], key=lambda x: x if x is not None else (0 if random.random() < 0.5 else math.inf))) for _ in range(3)]
# print([food_type.readable() for food_type in food_types])
n_requests = 30
# ex_requests = [Request((used:=random.choice(ascii_uppercase)), random.choice(list(set(ascii_uppercase) - set(used))), dict([(food_type, randint(0, 3)) for food_type in food_types]), rid) for rid in range(n_requests)]
# selected_requests = []
N_OPTIONS = 3

class RequestHandler:
    def __init__(self, system) -> None:
        self.system = system
        # self.TOTAL_FONTS = set()
        # self.COLORS = {'white': (236, 239, 241), 'green': (165, 214, 167)}
        # self.THEMES = set()
        self.ITEM_STATE = defaultdict(lambda : False)
        self.rows = []
        self.selected_requests = []
        self.row_to_id = dict()
        self.id_to_row = dict()

    def checkbox_callback(self, sender, app_data):
        # item_name = sender.replace('checkbox', 'grouped')
        item_id = int(sender.replace('checkbox', ''))
        # set_color(item_name, list(COLORS.keys())[int(app_data)])
        print('callback due to checkbox')
        # dpg.highlight_table_row('request_table', self.rows.index(item_id) + 1, list(COLORS.values())[int(app_data)])
        if app_data:
            self.selected_requests.append(item_id)
        else:
            try:
                self.selected_requests.remove(item_id)
            except ValueError:
                pass

    def highlight_update(self):
        # [dpg.highlight_table_row('request_table', self.rows.index(item_id) + 1, list(COLORS.values())[1]) for item_id in self.selected_requests]
        pass
    def checkall_callback(self, sender, app_data):
        for idx in self.rows:
            dpg.set_value(f'checkbox{idx}', app_data)
            self.checkbox_callback(f'checkbox{idx}', app_data)

    def select_path_callback(self, sender, app_data, user_data):
        self.system.satisfy_path(user_data)
        print(self.selected_requests)
        # self.update_paths() <TAG SO UNCOMMENTING IS EASIER>
        self.update_requests()
        # self.selected_requests = []
        self.highlight_update()
        print(self.selected_requests)
        pass

    def update_paths(self):
        for i in range(N_OPTIONS):
            dpg.configure_item(f'pathdisplay{i}', show=False)
            # dpg.set_value(f'pathdetailed{i}', '')
            for child in get_children(f'pathdetailed{i}'):
                dpg.delete_item(child)
            try:
                dpg.delete_item(f'temp{i}')
            except:
                pass
        for i, (route, dist) in enumerate(self.system.cached):
            dpg.configure_item(f'pathdisplay{i}', show=True)
            dpg.add_text('Instructions:', parent=f'pathdetailed{i}')
            set_font(dpg.last_item(), 13, bold=True)
            for tup in self.system.raw_route_info(route):
                if len(tup) > 1:
                    text, k = tup
                    res = format_tooltip(text, parent=f'pathdetailed{i}')
                    with res[k.food_type]:
                        display_details(k)
                else:
                    dpg.add_text(tup[0], parent=f'pathdetailed{i}', bullet=True)
            with dpg.group(horizontal=True, parent=f'pathdetailed{i}'):
                dpg.add_text('Distance:')
                set_font(dpg.last_item(), 13, bold=True)
                dpg.add_text(f'{dist:.2f}km')
            self.draw_route(route, i, parent=f'pathdisplay{i}', before=f'pathbutton{i}')
        self.clear_canvas()
        self.draw_graph(parent='graph_canvas')

        # dpg.delete_item('graph_canvas')

    def submit_callback(self, sender, app_data):
        #generate paths
        # requests = [self.system.requests[i] for i in selected_requests]
        # requests = [self.remove_request(i, from_self=True) for i in self.selected_requests]
        # print(self.selected_requests)
        temp = dpg.add_loading_indicator(parent='path_window')
        self.system.recommend_requests(request_ids=[self.row_to_id[i] for i in self.selected_requests], cache=N_OPTIONS)
        # print('ROUTE', a)
        # self.update_paths() <TAG SO UNCOMMENT IS EASIER>
        dpg.delete_item(temp)
        # for i in range(N_OPTIONS):
        #     dpg.configure_item(f'pathdisplay{i}', show=False)
        #     dpg.set_value(f'pathdetailed{i}', '')

        # for i, (route, dist) in enumerate(self.system.cached):
        #     dpg.configure_item(f'pathdisplay{i}', show=True)
        #     dpg.set_value(f'pathdetailed{i}', f'{self.system.print_route_information(route)}\nDistance: {dist}')
            
        # self.selected_requests = []
        
        # print(requests)

    def expand(self, sender, app_data, n1, n2):
        print('USER DATA', app_data)
        # item_name = app_data[1].replace('grouped', 'detailed')
        item_name = app_data[1].replace(n1, n2)
        print('ITEM NAME', item_name)
        if not hasattr(self, n1):
            setattr(self, n1, defaultdict(lambda : False))
        if not app_data[0]:
            getattr(self, n1)[item_name] = not getattr(self, n1)[item_name]
        print('FLIP', getattr(self, n1)[item_name])
        dpg.configure_item(item_name, show=getattr(self, n1)[item_name])

    def create_request_box(self, request, i, parent=None):
        # source, dest, amounts = tup
        source = self.system.convert_to_str(request.source)
        dest = self.system.convert_to_str(request.dest)
        # source = request.source
        # dest = request.dest
        amounts = request.amounts
        rid = request.request_id

        self.id_to_row[rid] = i
        self.row_to_id[i] = rid
        default_args = {'tag': f'row{i}', 'filter_key': f'{source} to {dest}'}
        with dpg.table_row(**default_args) if parent is None else dpg.table_row(**default_args, parent=parent) as row:
            with dpg.table_cell():
                # with dpg.group(tag=f'grouped{i}', horizontal=True, user_data={'n1': 1}):
                #     dpg.add_text(f"{source}", tag=f'text{i}1')
                #     with dpg.drawlist(width=20, height=20):
                #         dpg.draw_image('r_icon', (2, 2), (20, 20))
                #     dpg.add_text(f"{dest}", tag=f'text{i}2')
                # set_font(f'grouped{i}', 16)
                # set_color(f'grouped{i}', 'white')
                # dpg.bind_item_handler_registry(f"grouped{i}", "whandler")
                dpg.add_text(f'{source} to {dest}', show=False)
                with dpg.collapsing_header(label=f'{source} to {dest}') as head:
                    set_font(head, 15)
                    with dpg.group(tag=f'detailed{i}'):
                        for ii, (k, v) in enumerate(amounts.items()):
                            if not v:
                                continue
                            display_details(k, v)
                            if ii != len(amounts.items()) - 1:
                                dpg.add_separator()
                    set_font(f'detailed{i}', 13)
                # print([dpg.get_item_type(x) for x in dpg.get_item_children(f'detailed{i}')[1]])
            with dpg.table_cell():
                dpg.add_checkbox(tag=f'checkbox{i}', callback=self.checkbox_callback)

    def create_request_table(self, parent=None):
        with dpg.table(header_row=False, tag='request_table', scrollY=True, pad_outerX=True, height=300) if parent is None else dpg.table(header_row=False, tag='request_table', scrollY=True, borders_innerH=True, parent=parent):#, row_background=True):
            # for _ in range(2):
            dpg.add_table_column(width_stretch=True)
            dpg.add_table_column(width_fixed=True, width=50)
            with dpg.table_row(filter_key=None, tag='header_row'):
                with dpg.table_cell():
                    dpg.add_text("Requests", tag='header1')
                    set_font('header1', 18)
                with dpg.table_cell():
                    dpg.add_spacer(height=0)
                    dpg.add_checkbox(tag='checkall', callback=self.checkall_callback)
                    dpg.add_spacer(height=0)
            for i, (_, request) in enumerate(self.system.requests):
                self.create_request_box(request, i)
                self.rows.append(i)
        dpg.highlight_table_row('request_table', 0, [96, 125, 139, 100])
    
    def create_expand_handler(self, name, n1, n2):
        with dpg.item_handler_registry(tag=name) as handler:
            dpg.add_item_clicked_handler(callback=lambda sender, app_data: self.expand(sender, app_data, n1, n2))

    def update_requests(self):
        add_ids = set(self.system.request_ids) - set(self.row_to_id.values())
        remove_ids = set(self.row_to_id.values()) - set(self.system.request_ids)
        # print('GUI UPDATE ADD', add_ids)
        # print('GUI UPDATE REMOVE', remove_ids)
        [self.add_request(self.system.id_to_request[rid]) for rid in add_ids]
        [self.remove_request(rid) for rid in remove_ids] 

    def add_request(self, request):
        new_idx = 0
        for idx in sorted(self.rows):
            if idx+1 not in self.rows:
                new_idx = idx+1
                break
        self.rows.append(new_idx)
        self.create_request_box(request[1], new_idx, parent='request_table')

    def remove_request(self, i, from_self=False):
        if not from_self:
            i = self.id_to_row[i]
        print('ROW ID REMOVED', i)
        self.selected_requests.remove(i)
        # print(self.rows)
        dpg.delete_item(f'row{i}')
        if from_self:
            self.system.remove_request(self.row_to_id[i]) #some sort of reporting mechanism
        self.rows.remove(i)
        rid = self.row_to_id[i]
        del self.row_to_id[i]
        del self.id_to_row[rid]

    def filter_callback(self, s, a, u):
        dpg.configure_item('header_row', filter_key=a)
        dpg.set_value(u, dpg.get_value(s))

    def draw_nodes(self, parent=None):
        pos = embed_graph(self.system.cache_graph)
        for k, v in pos.items():
            x = dpg.draw_circle(v, 10, fill=[100, 100, 100, 100], parent=(0 if parent is None else parent))

    def multi_edge_to_bezier(self, edge, num, width=40):
        midpoint = np.array([sum([edge[j][i] for j in range(2)])/2 for i in range(2)])
        vector = [edge[1][i] - edge[0][i] for i in range(2)]
        perp = np.array(perp_vector(vector))

        start = midpoint - perp*width/2
        step_vec = perp*width/(num + 1)
        start += step_vec
        return start, step_vec

    def bezier_tangent(self, edge, middle):
        control = 2*np.array(middle) - np.array(edge[0])/2 - np.array(edge[1])/2
        return [normalize((control - edge[i]).tolist()) for i in range(2)]

    def draw_edges(self, edges, color='turquoise', parent=None):
        condensed = Counter(map(lambda x: tuple(sorted(x)), edges))
        # print(condensed)
        cur_loc = dict([(k, list(self.multi_edge_to_bezier(k, v))) for k, v in condensed.items()])
        for edge in edges:
            k_edge = tuple(sorted(edge))
            middle_point = cur_loc[k_edge][0].tolist()
            cur_loc[k_edge][0] += cur_loc[k_edge][1]
            dpg.draw_bezier_quadratic(edge[0], middle_point, edge[1], parent=(0 if parent is None else parent), color=FULL_COLORS[color])
            tangent = self.bezier_tangent(edge, middle_point)[1]
            dpg.draw_arrow((np.array(edge[1]) + np.array(tangent)*10).tolist(), (np.array(edge[1]) + np.array(tangent)*12).tolist(),  parent=(0 if parent is None else parent), color=FULL_COLORS[color])
    
    def get_edges(self, route):
        # self.draw_map(route)
        pos = embed_graph(self.system.cache_graph)
        return [(tuple(pos[self.system.convert_to_str(route[0][i][0])]), tuple(pos[self.system.convert_to_str(route[0][i+1][0])])) for i in range(len(route[0]) - 1)]

    def clear_canvas(self, canvas='graph_canvas'):
        # canvas_conv = dpg.get_item_configuration(canvas)
        # dpg.draw_rectangle((0, 0), (canvas_conv['width'], canvas_conv['height']), parent=canvas, fill=[0, 0, 0, 255])
        for child in itertools.chain.from_iterable(dpg.get_item_children(canvas).values()):
            dpg.delete_item(child)
    
    def draw_graph(self, parent=None):
        # default_args = dict(width=300, height=300, tag='graph_canvas')
        # with dpg.drawlist(**default_args) if parent is None else dpg.drawlist(**default_args, parent=parent):
        #     # print(self.system.cache_graph)
        #     pos = embed_graph(self.system.cache_graph)
        #     for k, v in pos.items():
        #         dpg.draw_circle(v, 10, fill=[100, 100, 100, 100])
        self.draw_nodes(parent=parent)
        edges = []
        for route, _ in self.system.cached:
            edges.extend(self.get_edges(route))
        self.draw_edges(edges, parent=parent)

    def draw_route(self, route, id, parent=None, before=None):
        path = [(self.system.convert_to_node(route[0][i][0]).address + ' ON') for i in range(len(route[0]) - 1)]
        response = CLIENT.static_map(
            size=(300, 300),
            maptype="roadmap",
            format="png",
            scale=2,
            path=StaticMapPath(points=path, weight=5, color='red', geodesic=True),
            markers=[StaticMapMarker(locations=[address], label=f'{i}') for i, address in enumerate(path)],
        )
        with open(f'temp{id}.png', 'wb') as f:
            f.write(b''.join([x for x in response]))
        if f'im{id}' in dpg.get_aliases():
            dpg.remove_alias(f'im{id}')
        load_image(f'temp{id}.png', f'im{id}')
        with dpg.drawlist(width=300, height=300, parent=(0 if parent is None else parent), before=(0 if before is None else before), tag=f'temp{id}'):
            dpg.draw_image(f'im{id}', (0, 0), (300, 300))
    # def draw_map(self, parent=None):
    #     for i, (route, _) in enumerate(self.system.cached):
    #         self.draw_route(route, i, parent=parent)
    #     pass

    def update_system(self, s, a, u):
        self.system.update_requests()
        self.update_requests()

    def start_requests(self):
        with dpg.theme() as theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_TableRowBg, (44, 62, 80), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_TableRowBgAlt, (255, 255, 255), category=dpg.mvThemeCat_Core)
                # dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 0, category=dpg.mvThemeCat_Core)
        
        with dpg.theme(tag="button_theme"):
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, hsv_to_rgb(4/7.0, 0.6, 0.4))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, hsv_to_rgb(4/7.0, 0.8, 0.6))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, hsv_to_rgb(4/7.0, 0.7, 0.5))
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 3, 3)
        dpg.bind_theme('button_theme')
        # dpg.bind_theme(theme)

        load_image('request_icon.png', 'r_icon')
        load_image('apple.png', 'apple')
        load_image('date.png', 'date')
        load_image('amount.png', 'amount')

        with dpg.window(label="Requests Display", autosize=True, width=500, tag='request_window'):
            dpg.add_input_text(label="Filter (inc, -exc)", user_data='request_table', callback=self.filter_callback)
            set_font(dpg.last_item(), 13)
            self.create_request_table()
            # dpg.bind_item_theme(dpg.last_item(), 'button_theme')
            with dpg.child_window(border=False, height=50, no_scrollbar=True):
                dpg.add_spacer(height=1)
                # with dpg.group(horizontal=True):
                #     # dpg.add_spacer(width=3)
                dpg.add_button(label='Select requests', tag='button1', width=-1, callback=self.submit_callback)
                dpg.add_button(label='Update', tag='button2', width=-1, callback=self.update_system)
                dpg.add_spacer(height=1)
                set_font('button1', 15)
                set_font('button2', 15)

        with dpg.window(label='Path Display', height=200, width=400, pos=(500, 0), min_size=[400, 0], no_close=True, autosize=True, tag='path_window'):
            # with dpg.child(autosize_x=True, autosize_y=True):
            with dpg.table(header_row=False):
                for _ in range(1):
                    dpg.add_table_column(width_stretch=True)
                for idx in range(N_OPTIONS):
                    with dpg.table_row(tag=f'row_{idx}'):
                        with dpg.table_cell():
                            with dpg.collapsing_header(label=f'Path {idx+1}', tag=f'pathdisplay{idx}', show=False) as pathx:
                                set_font(pathx, 15)
                                with dpg.group(tag=f'pathdet{idx}'):
                                    dpg.add_group(tag=f'pathdetailed{idx}')
                                    dpg.add_button(label='Select this path', tag=f'pathbutton{idx}', user_data=idx, callback=self.select_path_callback)
                                # set_font(f'pathdisplay{idx}', 18)
                                set_font(f'pathdet{idx}', 13)
                                set_font(f'pathbutton{idx}', 15)
                                # dpg.bind_item_handler_registry(f"pathname{idx}", "whandler2")
                        # dpg.add_separator()
        # with dpg.window(label='Bird\'s Eye View', height=400, width=400, pos=(500, 300), no_close=True, tag='delivery_window'):
        #     with dpg.drawlist(width=300, height=300, tag='graph_canvas'):
        #         # print(self.system.cache_graph)
        #         # pos = embed_graph(self.system.cache_graph)
        #         # for k, v in pos.items():
        #         #     dpg.draw_circle(v, 10, fill=[100, 100, 100, 100])
        #         self.clear_canvas()
        #         self.draw_graph()

dpg.create_context()
dpg.add_font_registry(tag='font_registry')
dpg.create_viewport(title='RN4402 Feed it Forward Prototype', x_pos = 0, y_pos = 0)

class FakeSystem:
    def __init__(self, requests) -> None:
        self.requests = requests

    def remove_request(self, rid):
        print(self.requests)
        self.requests.remove(rid)

s = System()
r = RequestHandler(s)
# server.add_random_requests(10)
# s.update_requests()
r.start_requests()

dpg.setup_dearpygui()
dpg.show_viewport()
# dpg.toggle_viewport_fullscreen()
# dpg.start_dearpygui()


frame_count = 0
UPDATE_TIME = 60*20
while dpg.is_dearpygui_running():
    # insert here any code you would like to run in the render loop
    # you can manually stop by using stop_dearpygui()
    if not (frame_count % UPDATE_TIME):# and frame_count:
        s.update_requests()
        # print('N_NODES', s.n_nodes)
        # print(s.id_to_request)
        r.update_requests()
        # new_requests = [((used:=random.choice(ascii_uppercase)), random.choice(list(set(ascii_uppercase) - set(used))), dict([(food_type, randint(0, 3)) for food_type in food_types])) for _ in range(5)]
        # [r.add_request(request) for request in new_requests]
        # print(f"frame {frame_count}")
    dpg.render_dearpygui_frame()
    frame_count += 1
dpg.destroy_context()