from PIL import Image, ImageOps
import numpy as np
for name in ['apple', 'amount', 'date']:
    im = Image.open(f'{name}.png')
    data = np.array(im)   # "data" is a height x width x 4 numpy array
    red, green, blue, alpha = data.T # Temporarily unpack the bands for readability

    # Replace white with red... (leaves alpha values alone...)
    white_areas = (red == 255) & (blue == 255) & (green == 255)
    data[..., :-1][white_areas.T] = (0, 0, 0)
    im_invert = Image.fromarray(data)
    im_invert.save(f'{name}_invert.png')