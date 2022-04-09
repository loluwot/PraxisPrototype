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
        return FORMATS[safe_index(r_tup, None)].format(*([ATTRIBUTE_TO_NAME[prop]] + [r_val for r_val in r_tup if r_val is not None]))
    return f'{ATTRIBUTE_TO_NAME[prop]}: {r_tup}'
