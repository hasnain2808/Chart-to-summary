import tensorflow as tf
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
import keras
# plt.ion()
model = keras.models.load_model(
    'my_model.h5',
) 

size = 48
dim = (size, size)

lis = os.listdir("data_test_source\\")

import numpy as np
import cv2

# Load an color image in grayscale

NCATS = {'dot_line': 0,
		 'hbar_categorical': 1,
		 'line': 2,
		 'pie': 3,
		 'vbar_categorical': 4}	
NCATS = {v:k for k,v in NCATS.items()}

for dire in list(lis):
	print(dire )
	lu = os.listdir("data_test_source\\" + dire + "\\")
	#print(lu)
	for i in list(lu):
		#print("data_test_source\\" + dire + "\\"+ i)
		img = cv2.imread("data_test_source\\" + dire + "\\"	+ i)
		#print(img)
		imgdis=img
		img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
		imgforpred = np.array(img)
		imgforpred = np.moveaxis(imgforpred, -1, 0)
		imgforpred = np.expand_dims(imgforpred, axis=0)
		pred = model.predict(imgforpred)
		print('predicted label : ', NCATS[np.argmax(pred)])
		cv2.imshow('image',cv2.resize(imgdis, (512,512), interpolation = cv2.INTER_AREA))
		cv2.waitKey(0)


		# time.sleep(10)
		cv2.destroyAllWindows()

# for i in list(lis):
#     image = mpimg.imread("data_test\\" + i)
#     # plt.show()
#     print(i)
#     plt.imshow(image)
#     plt.show(block=False)
#     time.sleep(10)
#     plt.close('all')



# plt.close('all')

#     # for i in range(1,4):
#     #  PATH = "kodim01.png"
#     #  N = "%02d" % i
#     #  print PATH.replace("01", N)
#     #  image = mpimg.imread(PATH) # images are color images
#     #  plt.show()
#     #  plt.imshow(image)