import requests

LINK = 'https://api.edamam.com/api/food-database/v2/'
APP_ID = 'f62e12b4'
API_KEY = '42b014ebe865c54314aecc08397faaf5'
REQ1 = f'{LINK}{{req_type}}?app_id={APP_ID}&app_key={API_KEY}'

FILTERED_LIST = set(['VEGAN', 'VEGETARIAN', 'PESCATARIAN', 'DAIRY_FREE', 'GLUTEN_FREE', 'WHEAT_FREE', 'EGG_FREE', 'MILK_FREE', 'PEANUT_FREE', 'TREE_NUT_FREE', 'SOY_FREE', 'FISH_FREE', 'SHELLFISH_FREE', 'PORK_FREE', 'RED_MEAT_FREE', 'CRUSTACEAN_FREE', 'CELERY_FREE', 'MUSTARD_FREE', 'SESAME_FREE', 'LUPINE_FREE', 'MOLLUSK_FREE', 'ALCOHOL_FREE', 'KOSHER'])

def labels_from_food(name):
    res = requests.get(REQ1.format(req_type='parser'), {'ingr': name}).json()
    # print(res['parsed'])
    ingredients = [food['food']['foodId'] for food in res['parsed']]
    req = {
        'ingredients':[{'quantity': 1, 'measureURI': '', 'foodId':ingr} for ingr in ingredients]
        }
    res = requests.post(REQ1.format(req_type='nutrients'), json=req)
    # print(res.json())
    return set(res.json()['healthLabels']) & FILTERED_LIST

import openfoodfacts
def barcode_to_labels(barcode):
    product = openfoodfacts.products.get_product(str(barcode))
    return labels_from_food(product['product']['product_name'])


print(labels_from_food('banana'))
print(barcode_to_labels(3017620422003))