import numpy as np

#                      Nothing           Road       Building
COLORS = np.array(((255, 255, 255), (255, 0, 0), (255, 255, 0)))

def classes_to_image(classes, n_classes=3):
    'Converts a WxH array output by the model into an RGB image'

    W, H = classes.shape

    img = COLORS[classes.astype('uint8')].reshape((W, H, n_classes))

    return img.astype('uint8')

def high_dimensional_to_rgb(img):
    'Converts a KxWxH array, usually coming from a satellite image with many bands, into an RGB image'

    return np.moveaxis(img[:3, :, :], 0, 2).astype('uint8')
