import numpy as np

#                      Nothing           Road       Building
COLORS = np.array(((255, 255, 255), (255, 0, 0), (255, 255, 0)))

def classes_to_image(classes):
    'Converts a 3xWxH array output by the model into an RGB image'

    C, W, H = classes.shape
    assert C == 3
    classes = np.argmax(classes, axis=0).reshape((-1,))
    img = COLORS[classes].reshape((W, H, C))

    return img.astype('uint8')
