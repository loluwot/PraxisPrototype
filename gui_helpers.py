from json import tool
import dearpygui.dearpygui as dpg
import itertools
from helpers import ATTRIBUTE_TO_IMAGE

TOTAL_FONTS = set()
COLORS = {'white': (236, 239, 241, 0), 'bluegrey': (104, 159, 56,100)}
FULL_COLORS = {'turquoise': (26, 188, 156, 100), 'green': (46, 204, 113, 100), 'blue': (52, 152, 219), 'blue1': (30, 136, 229, 100)}
THEMES = set()

def hsv_to_rgb(h, s, v):
    if s == 0.0: return (v, v, v)
    i = int(h*6.) # XXX assume int() truncates!
    f = (h*6.)-i; p,q,t = v*(1.-s), v*(1.-s*f), v*(1.-s*(1.-f)); i%=6
    if i == 0: return (255*v, 255*t, 255*p)
    if i == 1: return (255*q, 255*v, 255*p)
    if i == 2: return (255*p, 255*v, 255*t)
    if i == 3: return (255*p, 255*q, 255*v)
    if i == 4: return (255*t, 255*p, 255*v)
    if i == 5: return (255*v, 255*p, 255*q)


def load_image(image_name, tag):
    width, height, channels, data = dpg.load_image(image_name)
    with dpg.texture_registry():
        dpg.add_static_texture(width, height, data, tag=tag)

def get_children(item):
    return [x for x in itertools.chain.from_iterable(dpg.get_item_children(item).values())]

def set_font(item, size, bold=False):
    if dpg.is_item_container(item):
        for children in [x for x in itertools.chain.from_iterable(dpg.get_item_children(item).values()) if 'mvText' in dpg.get_item_type(x)]:
            set_font(children, size, bold=bold)
    name = f'{"bold" if bold else "regular"}{size}'
    if name not in TOTAL_FONTS:
        # dpg.add_font(f'Roboto/Roboto-{"Bold" if bold else "Regular"}.ttf', size, parent='font_registry', tag=name)
        dpg.add_font(f'Proxima/Proxima Nova {"Bold" if bold else "Reg"}.ttf', size, parent='font_registry', tag=name)
        # dpg.add_font(f'FontsFree-Net-proxima_nova_reg-webfont.ttf', size, parent='font_registry', tag=name)
        TOTAL_FONTS.add(name)
    dpg.bind_item_font(item, name)

def set_color(item, color):
    if color not in THEMES:
        with dpg.theme(tag=color):
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_Text, COLORS[color], category=dpg.mvThemeCat_Core)
        THEMES.add(color)
    dpg.bind_item_theme(item, color)

import re
def format_tooltip(line, parent=0):
    other = re.split('{.+?}', line)
    tooltips = [res[1:-1] for res in re.findall('{.+?}', line)]
    L = [None for _ in range(len(other) + len(tooltips))]
    L[::2] = other
    L[1::2] = tooltips
    tooltip_ids = dict()
    with dpg.group(horizontal=True, parent=parent, horizontal_spacing=0):
        for i, s in enumerate(L):
            t = dpg.add_text(s, bullet=not i)
            if i % 2:
                dpg.add_text(' (Hover for more details)')
                toolid = dpg.tooltip(t)
                set_font(t, 13, bold=True)
                tooltip_ids[s] = toolid
    return tooltip_ids

def display_details(food_type, amount=None):
    L = food_type.readable_raw()
    for prop, x in L:
        with dpg.group(horizontal=True):
            with dpg.drawlist(width=16, height=16):
                dpg.draw_image(ATTRIBUTE_TO_IMAGE[prop], (2, 2), (16, 16))
            dpg.add_text(x)
    if amount is not None:
        with dpg.group(horizontal=True):
            with dpg.drawlist(width=20, height=20):
                dpg.draw_image('amount', (0, 0), (20, 20))
            dpg.add_text(f'Amount: {amount} kg')
# format_tooltip('a {tooltip} {here}')
    