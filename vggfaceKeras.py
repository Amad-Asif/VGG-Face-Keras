# -*- coding: utf-8 -*-
from keras.models import Model, Sequential
from keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
import cv2
import numpy as np
from keras.preprocessing.image import load_img, save_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
import matplotlib.pyplot as plt


'''
VGG-Face model implementation using Keras
'''

'''
We can think of filters as feature detectors.  the significance # of feature detectors intuitively is
the number of features (like edges, lines, object parts etc...) that the network can potentially learn.
Also note that each filter generates a feature map. Feature maps allow you to learn the explanatory
factors within the image so the more of the the more the network learns

The number of filters is the number of neurons, since each neuron performs a different convolution on
the input to the layer (more precisely, the neurons' input weights form convolution kernels).

A feature map is the result of applying a filter (thus, you have as many feature maps as filters), and
its size is a result of window/kernel size of your filter and stride.

2 different convolutional filters are applied to the input image, resulting in 2 different feature maps
(the output of the filters). Each pixel of each feature map is an output of the convolutional layer.
'''

'''
the number of filters are incremented so as to be able to properly encode the increasingly richer and
richer representations as the signal moves up the representational hierarchy in order to avoid the
bottleneck effect.

It increases the representational power of the network. Since initial layers learn only primitive
regularities in the data you do not need to have so much units there. However as you go through the
upper layers many more abstractions keen to be observed. To compensate this increase of abstraction,
you need to increase number units too.

if you have 28x28 input images and a convolutional layer with 20 7x7 filters and stride 1, you will get
20 22x22 feature maps at the output of this layer. Note that this is presented to the next layer as a
volume with width = height = 22 and depth = num_channels = 20.
'''

#Create a Sequential Model
model = Sequential()
#1*1 Zero padding
#64 filters of size 3*3
#Input resolution of image is 224*224
# 64 3*3 filters
# Conv -> Conv -> MaxPool
model.add(ZeroPadding2D((1, 1), input_shape=(224, 224, 3)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

# Conv -> Conv -> MaxPool
# increase the number of filters, keeping the filter size same
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2, 2)))

# Conv -> Conv -> Conv -> MaxPool
# increase the number of filters, keeping the filter size same
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

# Conv -> Conv -> Conv -> MaxPool
# increase the number of filters, keeping the filter size same
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

# Conv -> Conv -> Conv -> MaxPool
# Keep the number of filters same, keeping the filter size same
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))


'''
The idea of dropout is simplistic in nature. This layer drops out a random set of activations in
that layer by setting them to zero. it forces the network to be redundant. By that I mean the network
should be able to provide the right classification or output for a specific example even if some of
the activations are dropped out. It makes sure that the network isn’t getting too fitted to the
training data and thus helps alleviate the overfitting problem. An important note is that this layer
is only used during training and not during test time.
'''

'''
1x1 convolution leads to dimension reduction. For example, an image of 200 x 200 with 50 features
on convolution with 20 filters of 1x1 would result in size of 200 x 200 x 20.

A 1x1 convolution simply maps an input pixel with all it's channels to an output pixel, not looking at
anything around itself. It is often used to reduce the number of depth channels, since it is often
very slow to multiply volumes with extremely large depths.

In GoogLeNet architecture, 1x1 convolution is used for two purposes
To make network deep by adding an “inception module”
1) To reduce the dimensions inside this “inception module”.
2) To add more non-linearity by having ReLU immediately after every 1x1 convolution.
1x1 convolutions (in yellow), are specially used before 3x3 and 5x5 convolution to reduce the
dimensions.

1x1 Convolution with higher strides leads to even more redution in data by decreasing resolution,
while losing very little non-spatially correlated information.
Replace fully connected layers with 1x1 convolutions as Yann LeCun believes they are the same

example -
6 * 6 * 3 (Image)  convolution  1 * 1 * 3 (Filter) =          6 * 6 * 1 (output)
Depth - 3                     Depth - 3, 1 - 1 Filter            Depth - 1
Here the width and height of the image remains unchanged while the depth is 1
By increasing the number of filters we can control the depth of the output

6 * 6 * 3 (Image)   convolution    1 * 1 * 3 (2 - Filters applied) =     6 * 6 * 2 (output)
Depth - 3                           Depth - 3, 1 - 2 Filters                Depth - 2

In summary, 1x1 convolutions serve as a means to
control the depth of the input volume as it is passed to the next layer, either decrease it, or
increase it, or just add a non-linearity when it doesn’t alter the depth. This control is achieved by
the choosing the appropriate number of filters. We can control the other two dimensions - width and
height by the filter sizes and padding parameters, or use pooling to reduce width and height.
In the case when it is reduces the dimensions, it is a means to reduce computations - can be an order
of magnitude less
'''

# Conv -> Dropout -> Conv -> Dropout -> Conv -> Flatten -> Sofmax Activation

model.add(Convolution2D(4096, (7, 7), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(4096, (1, 1), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(2622, (1, 1)))
model.add(Flatten())
model.add(Activation('softmax'))

'''
We will load the pretrained weights
'''
from keras.models import model_from_json
model.load_weights('vgg_face_weights.h5')

"""
we’ll use previous layer of the output layer for representation.
The following usage will give output of that layer.
"""
# layers[-2] represents the second last layer of the model
vgg_face_descriptor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)

'''
We can represent images 2622 dimensional vector as below
VGG model expects 224x224x3 sized input images. Here, 3rd dimension refers to number of channels or
RGB colors
'''

# Method to resize the image as the model expects 224x224x3 images


def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img


"""
We’ve represented input images as vectors. We will decide both pictures are same person or not based
on comparing these vector representations. Now, we need to find the distance of these vectors. There
are two common ways to find the distance of two vectors: cosine distance and euclidean distance.
Cosine distance is equal to 1 minus cosine similarity. No matter which measurement we adapt, they all
serve for finding similarities between vectors.
"""


def findCosineSimilarity(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))


def findEuclideanDistance(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance

'''
We’ve represented images as vectors and find the similarity measures of two vectors. If both images
are same person, then measurement should be small. Otherwise, the measurement should be large if two
images are different person. Here, epsilon value states threshold.
'''

'''
Way to find represenation of the image
img1_representation = vgg_face_descriptor.predict(preprocess_image('1.png'))[0, :]
img2_representation = vgg_face_descriptor.predict(preprocess_image('2.png'))[0, :]
'''
epsilon_cos = 0.40 #cosine similarity
epsilon_euc = 120 #euclidean distance

def verify_face(img1, img2):
    img1_representation = vgg_face_descriptor.predict(preprocess_image(img1))[0, :]
    img2_representation = vgg_face_descriptor.predict(preprocess_image(img2))[0, :]

    cosine_similarity = findCosineSimilarity(img1_representation, img2_representation)
    euclidean_distance = findEuclideanDistance(img1_representation, img2_representation)

    print("Cosine Similarity", cosine_similarity)
    print("Euclidean Distance", euclidean_distance)

verify_face("images/2.png", "images/7.png")