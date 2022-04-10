from os import remove
import dearpygui.dearpygui as dpg
from string import ascii_uppercase, ascii_lowercase
import random
from random import randint
from collections import defaultdict

from matplotlib.pyplot import get
from classes import *
from helpers import ATTRIBUTE_TO_IMAGE
from gui_helpers import *
food_types = [FoodType(food_type=random.choice(ascii_lowercase), expiry_date=sorted([None if not x else x for x in random.sample(range(10), 2)], key=lambda x: x if x is not None else (0 if random.random() < 0.5 else math.inf))) for _ in range(3)]
# print([food_type.readable() for food_type in food_types])
n_requests = 30
ex_requests = [Request((used:=random.choice(ascii_uppercase)), random.choice(list(set(ascii_uppercase) - set(used))), dict([(food_type, randint(0, 3)) for food_type in food_types]), rid) for rid in range(n_requests)]
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
        dpg.highlight_table_row('request_table', self.rows.index(item_id) + 1, list(COLORS.values())[int(app_data)])
        if app_data:
            self.selected_requests.append(item_id)
        else:
            try:
                self.selected_requests.remove(item_id)
            except ValueError:
                pass

    def highlight_update(self):
        [dpg.highlight_table_row('request_table', self.rows.index(item_id) + 1, list(COLORS.values())[1]) for item_id in self.selected_requests]

    def checkall_callback(self, sender, app_data):
        for idx in self.rows:
            dpg.set_value(f'checkbox{idx}', app_data)
            self.checkbox_callback(f'checkbox{idx}', app_data)

    def select_path_callback(self, sender, app_data, user_data):
        self.system.satisfy_path(user_data)
        print(self.selected_requests)
        self.update_paths()
        self.update_requests()
        # self.selected_requests = []
        self.highlight_update()
        print(self.selected_requests)
        pass

    def update_paths(self):
        for i in range(N_OPTIONS):
            dpg.configure_item(f'pathdisplay{i}', show=False)
            dpg.set_value(f'pathdetailed{i}', '')

        for i, (route, dist) in enumerate(self.system.cached):
            dpg.configure_item(f'pathdisplay{i}', show=True)
            dpg.set_value(f'pathdetailed{i}', f'{self.system.print_route_information(route)}\nDistance: {dist}')

    def submit_callback(self, sender, app_data):
        #generate paths
        # requests = [self.system.requests[i] for i in selected_requests]
        # requests = [self.remove_request(i, from_self=True) for i in self.selected_requests]
        # print(self.selected_requests)
        temp = dpg.add_loading_indicator(parent='path_window')
        self.system.recommend_requests(request_ids=[self.row_to_id[i] for i in self.selected_requests], cache=N_OPTIONS)
        dpg.delete_item(temp)
        # print('ROUTE', a)
        self.update_paths()
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
                            L = k.readable_raw()
                            for prop, x in L:
                                with dpg.group(horizontal=True):
                                    with dpg.drawlist(width=16, height=16):
                                        dpg.draw_image(ATTRIBUTE_TO_IMAGE[prop], (2, 2), (16, 16))
                                    dpg.add_text(x)
                            with dpg.group(horizontal=True):
                                with dpg.drawlist(width=20, height=20):
                                    dpg.draw_image('amount', (0, 0), (20, 20))
                                dpg.add_text(f'Amount: {v}')
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

    def start_requests(self):
        with dpg.theme() as theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_TableRowBg, (44, 62, 80), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_TableRowBgAlt, (255, 255, 255), category=dpg.mvThemeCat_Core)
                # dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 0, category=dpg.mvThemeCat_Core)
        
        with dpg.theme(tag="button_theme"):
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, hsv_to_rgb(3/7.0, 0.6, 0.4))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, hsv_to_rgb(3/7.0, 0.8, 0.6))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, hsv_to_rgb(3/7.0, 0.7, 0.5))
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 3, 3)
        dpg.bind_theme('button_theme')
        # dpg.bind_theme(theme)

        load_image('request_icon.png', 'r_icon')
        load_image('apple.png', 'apple')
        load_image('date.png', 'date')
        load_image('amount.png', 'amount')
        with dpg.window(label="Requests Display", autosize=True, width=600, tag='request_window'):
            dpg.add_input_text(label="Filter (inc, -exc)", user_data='request_table', callback=self.filter_callback)
            set_font(dpg.last_item(), 13)
            self.create_request_table()
            # dpg.bind_item_theme(dpg.last_item(), 'button_theme')
            with dpg.child_window(border=False, height=50, no_scrollbar=True):
                dpg.add_spacer(height=1)
                # with dpg.group(horizontal=True):
                #     # dpg.add_spacer(width=3)
                dpg.add_button(label='Submit', tag='button1', width=-1, callback=self.submit_callback)
                dpg.add_spacer(height=1)
                set_font('button1', 15)
        with dpg.window(label='Path Display', height=200, width=400, pos=(600, 0), min_size=[400, 0], no_close=True, autosize=True, tag='path_window'):
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
                                    dpg.add_text('', tag=f'pathdetailed{idx}', wrap=0)
                                    dpg.add_button(label='Select this path', user_data=idx, callback=self.select_path_callback)
                                # set_font(f'pathdisplay{idx}', 18)
                                set_font(f'pathdet{idx}', 13)
                                # dpg.bind_item_handler_registry(f"pathname{idx}", "whandler2")
                        # dpg.add_separator()
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
r.start_requests()

dpg.setup_dearpygui()
dpg.show_viewport()
# dpg.toggle_viewport_fullscreen()
# dpg.start_dearpygui()

server.add_random_requests(20)
frame_count = 0
UPDATE_TIME = 60*5
while dpg.is_dearpygui_running():
    # insert here any code you would like to run in the render loop
    # you can manually stop by using stop_dearpygui()
    if not (frame_count % UPDATE_TIME) and frame_count:
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