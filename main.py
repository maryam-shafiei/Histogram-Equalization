import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.144])


def create_histogram(img_arr):
    return np.bincount(img_arr.flatten(), minlength=256)


def calc_cumsum(histogram_arr):
    return np.cumsum(histogram_arr)


def create_mapping(histogram_arr, img_h, img_w):
    return np.ceil(255 * calc_cumsum(histogram_arr) / (img_h * img_w)).astype(np.uint8)


def apply(mapping_arr, img_arr):
    img_list = list(img_arr.flatten())
    new_img_list = [mapping_arr[i] for i in img_list]
    return Image.fromarray(np.reshape(np.asarray(new_img_list), img_arr.shape))


image_address = input('Please enter the address of image: ')
img = Image.open(image_address)
img_w, img_h = img.size
gray_array = np.rint(rgb2gray(np.array(img))).astype(int)
#gray_array = np.asarray(img_gray)

histogram = create_histogram(gray_array)
print(calc_cumsum(histogram))
mapping = create_mapping(histogram, img_h, img_w)
app_img = apply(mapping, gray_array)
app_img.save("output.png")

x_axis = np.arange(256)
fig = plt.figure()
fig.add_subplot(1, 2, 1)
plt.title("Before")
plt.bar(x_axis, calc_cumsum(histogram))
#plt.bar(x_axis, histogram)
fig.add_subplot(1, 2, 2)
plt.title("After")
plt.bar(x_axis, calc_cumsum(create_histogram(np.asarray(app_img))), color="red")
#plt.bar(x_axis, create_histogram(np.asarray(app_img)), color="red")
plt.show()
