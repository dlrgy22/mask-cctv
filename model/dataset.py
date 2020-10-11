from keras.preprocessing import image
from sklearn.model_selection import train_test_split
import numpy as np
import os

def make_dataset(path,x_data, y_data):
    img_list = os.listdir(path)
    for img_name in img_list:
        if img_name == '.DS_Store':
            continue
        img_path = path + '/' + img_name

        img = image.load_img(img_path, target_size=(250, 250))
        img_tensor = image.img_to_array(img)
        x_data.append(img_tensor)
        if path == './mask_p':
            y_data.append([1, 0])
        else:
            y_data.append([0, 1])

x_data = []
y_data = []
path = './mask_p'
make_dataset('./mask_p', x_data, y_data)
make_dataset('./nomask_p', x_data, y_data)
x_data = np.array(x_data)
y_data = np.array(y_data)
x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.1, shuffle = True, random_state=1234)
np.save('x_train.npy', x_train)
np.save('x_val.npy', x_val)
np.save('y_train.npy', y_train)
np.save('y_val.npy', y_val)