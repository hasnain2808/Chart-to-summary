import tensorflow as tf
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
# plt.ion()
model = tf.keras.models.load_model(
    'vgg16epoch50tr79te76',
) 

size = 256
dim = (size, size)

lis = os.listdir("data_test\\")

import numpy as np
import cv2

# Load an color image in grayscale

NCATS = { 0 :'AreaGraph',1 : 'BarGraph',2 : 'LineGraph',
        3 : 'Map',4 : 'ParetoChart', 5:  'PieChart',
         6 : 'RadarPlot', 7:  'ScatterGraph',8 :'Table',
           9:'VennDiagram'}

for i in list(lis):
    #print(i)
    img = cv2.imread("data_test\\" + i)
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    imgforpred = np.array(img)
    imgforpred = np.expand_dims(imgforpred, axis=0)
    pred = model.predict(imgforpred)
    print('predicted label : ', NCATS[np.argmax(pred)])
    cv2.imshow('image',img)
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