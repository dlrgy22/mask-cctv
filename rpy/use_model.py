from keras.models import load_model
from keras.preprocessing import image
import numpy as np

def standard_scaler(x_data):
    mean = np.mean(x_data, axis = 0)
    std = np.std(x_data, axis = 0)
    scale_data = (x_data - mean) / std
    return scale_data

white_model = load_model('white_mask.h5')
img = image.load_img('./sv_img/face.jpg', target_size=(128, 128))
img_tensor = image.img_to_array(img)
img_tensor = img_tensor.reshape(-1, 128, 128, 3)
print(img_tensor.shape)

scale_img = standard_scaler(img_tensor)
pred = white_model.predict_classes(scale_img)
print(pred)