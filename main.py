from classes import *
import matplotlib.pyplot as plt
import random
import dearpygui.dearpygui as dpg
import requests

SERVER_URL = ''
def get_inventory(warehouse_id, view='food'):
    req_dict = {'id': warehouse_id, 'view': view}
    res = requests.get(SERVER_URL, req_dict)
    if not res.ok:
        raise ValueError("Server down")
    return res.json()

def get_food_from_id(food_id):
    res = requests.get(SERVER_URL, {'food_id':food_id})
    if not res.ok:
        raise ValueError("Server down")
    return res.json() #type, expiry date, allergens?, parent pallet id

def get_pallet_from_id(pallet_id):
    res = requests.get(SERVER_URL, {'pallet_id':pallet_id})
    if not res.ok:
        raise ValueError("Server down")
    return res.json() #list of food ids, location, parent warehouses

def get_request_list():
    res = requests.get(SERVER_URL)
    if not res.ok:
        raise ValueError("Server down")
    return res.json() #list of request ids

def get_request_from_id(request_id):
    res = requests.get(SERVER_URL, {'request_id', request_id})
    if not res.ok:
        raise ValueError("Server down")
    return res.json() #request information (list of food, warehouse1, warehouse2)

def update_inventory(request_id):
    res = requests.post(SERVER_URL, {'request_id', request_id})
    if not res.ok:
        raise ValueError("Server down")
    return res


#requests are path -> [(node id, items)...]

test_system = System(source_ratio=3, n_nodes=9)

dpg.create_context()
dpg.create_viewport(title='RN4402 Feed it Forward Prototype', x_pos = 0, y_pos = 0)

with dpg.window(label="Example Window"):
    dpg.add_text("Hello, world")
    dpg.add_button(label="Save")
    dpg.add_input_text(label="string", default_value="Quick brown fox")
    dpg.add_slider_float(label="float", default_value=0.273, max_value=1)

dpg.setup_dearpygui()
dpg.show_viewport()
dpg.toggle_viewport_fullscreen()
dpg.start_dearpygui()
dpg.destroy_context()




#What this app should do
#Periodically get requests from central server (replicates actual functionality of being one app)
#Show requests as they travel (Jagger view)
#Recommend some paths given current request list (Delivery driver view)




