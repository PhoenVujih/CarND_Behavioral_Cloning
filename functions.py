import numpy as np
import pandas as pd
import cv2
import matplotlib.image as mpimg
from sklearn.utils import shuffle

# Define the input size of the model
img_h, img_w, img_d = 160, 320, 3


def preprocess(image):
    """
    Crop, resize and convert the colorspace of the image. 
    """
    # Crop the image to remove the sky and the hood of the car
    img_crop = image[60:-25, :, :]
    # Resize the image to 160x320x3 to recover the resolution and make the features more evident
    img_resize = cv2.resize(img_crop, (img_w, img_h), cv2.INTER_AREA)
    # Convert the image to YUV color space (Ref: Nvidia Paper)
    img_yuv = cv2.cvtColor(img_resize, cv2.COLOR_RGB2YUV)
    return img_yuv


def read_data(sample):
    """
    Read data in the normal way: 
    read the image from centeral camera and the steering angle without revision
    """
    img = mpimg.imread(sample.center)
    ang = sample.steering
    return img, ang


def rand_read_data(sample):
    """
    Read data in random ways: 
    1. Left: 1/6
    2. Flipped left: 1/6
    3. Right: 1/6
    4. Flipped right: 1/6
    5. Center: 1/6
    6. Flipped center: 1/6
    """
    # Randomly choose the camera (left, center, right)
    ran_num = np.random.rand()
    if ran_num<0.33:
        img = mpimg.imread(sample.left)
        ang = sample.steering + 0.2
    elif (ran_num<0.66) & (ran_num>0.33):
        img = mpimg.imread(sample.right)
        ang = sample.steering - 0.2   
    else:
        img = mpimg.imread(sample.center)
        ang = sample.steering

    # Randomly flip the image
    if np.random.rand()<0.5:
        img = cv2.flip(img, 1)
        ang = -ang

    # Randomly adjust the brightness
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv[:,:,2] =  hsv[:,:,2] * (0.4 * np.random.rand() + 0.8)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return img, ang


def batch_generator(samples, is_training, batch_size=100):
    imgs = np.empty([batch_size, img_h, img_w, img_d])
    angs = np.empty(batch_size)
    while True:
        # Shuffle the sample list
        samples = shuffle(samples)
        # Select the first 100 samples as a batch
        for i in range(batch_size):
            sample = samples.iloc[i]
            if is_training & (np.random.rand()<0.75):
                img, ang = rand_read_data(sample)
            else:
                img, ang = read_data(sample)
            imgs[i] = preprocess(img)
            angs[i] = ang
        yield imgs, angs
