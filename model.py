# Imports
import numpy as np
import pandas
import os
import cv2
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Dense
from keras.layers.core import Flatten, Dropout, Activation
from keras.optimizers import Adam
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

##### Load data #####
driving_log = pandas.read_csv('driving_log.csv', delimiter=',', 
	names=['CENTER', 'LEFT', 'RIGHT', 'STEERING_ANGLE', 'THROTTLE', 'BREAK', 'SPEED'])
driving_log = driving_log[['CENTER', 'STEERING_ANGLE']]
#print(driving_log.head(n=10))

X_train = []
y_train = []
for index, row in driving_log.iterrows():
	X_train.append(cv2.imread(row['CENTER']))
	y_train.append(row['STEERING_ANGLE'])
X_train = np.array(X_train)
y_train = np.array(y_train)
print(X_train.shape)
print(y_train.shape)

##### Split Data #####
# Shuffle and separate training and test data
X_train, y_train = shuffle(X_train, y_train)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2)

##### Variables #####
input_shape = X_train.shape[1:]

pool_size = (2,2)
strides = (1,1)

n_filters1 = 6
filter1 = 7

n_filters2 = 12
filter2 = 6

n_filters3 = 24
filter3 = 5

n_filters4 = 36
filter4 = 4

n_filters5 = 48
filter5 = 3

keep_prob = 0.5
output_shape1 = 100
output_shape2 = 50
output_shape3 = 10

batch_size = 32
epochs = 10
learning_rate = 0.001

##### Create Model #####
model = Sequential()

# Convolutional Layer 1 (160 x 320 x 3) -> (164 x 314 x 6) -> (82 x 157 x 6)
model.add(Convolution2D(n_filters1, filter1, filter1, 
	border_mode='valid',
	subsample=strides, 
	input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size,
	strides=strides, 
	border_mode='valid'))

# Convolutional Layer 2 (82 x 157 x 6) -> (77 x 152 x 12) -> (39 x 76 x 12)
model.add(Convolution2D(n_filters2, filter2, filter2, 
	border_mode='valid',
	subsample=strides))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size, 
	strides=strides, 
	border_mode='valid'))

# Convolutional Layer 3 (39 x 76 x 12) -> (35 x 72 x 24) -> (18 x 36 x 24)
model.add(Convolution2D(n_filters3, filter3, filter3, 
	border_mode='valid',
	subsample=strides))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size, 
	strides=strides, 
	border_mode='valid'))

# Convolutional Layer 4 (18 x 36 x 24) -> (15 x 33 x 36) -> (7 x 17 x 36)
model.add(Convolution2D(n_filters4, filter4, filter4, 
	border_mode='valid',
	subsample=strides))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size, 
	strides=strides, 
	border_mode='valid'))

# Convolutional Layer 5 (7 x 17 x 36) -> (5 x 15 x 48) -> (3 x 8 x 48)
model.add(Convolution2D(n_filters5, filter5, filter5, 
	border_mode='valid',
	subsample=strides))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size, 
	strides=strides, 
	border_mode='valid'))

# Flatten
model.add(Flatten())

# Fully Connected Layer 1
model.add(Dense(output_shape1, input_shape=(1152,)))
model.add(Activation('relu'))
model.add(Dropout(keep_prob))

# Fully Connected Layer 2
model.add(Dense(output_shape2))
model.add(Activation('relu'))
model.add(Dropout(keep_prob))

# Fully Connected Layer 3
model.add(Dense(output_shape3))

##### Train the model #####
adam = Adam(lr=learning_rate)

model.compile(optimizer=adam, 
	loss='mean_squared_error', 
	metrics=['accuracy'])

model.fit(X_train, y_train, 
	batch_size=batch_size, 
	nb_epoch=epochs, 
	validation_split=0.2,
	shuffle=True)









