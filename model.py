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
import pdb

##### Load data #####
driving_log = pandas.read_csv('driving_log.csv', delimiter=',', 
	names=['CENTER', 'LEFT', 'RIGHT', 'STEERING_ANGLE', 'THROTTLE', 'BREAK', 'SPEED'])
driving_log = driving_log[['CENTER', 'STEERING_ANGLE']]
print(driving_log.head(n=10))

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

##### Generate More Data #####
datagen = ImageDataGenerator()

##### Variables #####
input_shape = X_train.shape[1:]

pool_size = (2,2)
strides = (1,1)

n_filters1 = 3
filter1 = 5

n_filters2 = 6
filter2 = 5

n_filters3 = 12
filter3 = 5

n_filters4 = 24
filter4 = 3

n_filters5 = 36
filter5 = 3

keep_prob = 0.5
output_shape1 = 100
output_shape2 = 50
output_shape3 = 10
output_shape4 = 1

batch_size = 128
epochs = 3
learning_rate = 0.001

##### Create Model #####
model = Sequential()

# Convolutional Layer 1 (160 x 320 x 3) -> (156 x 316 x 3) -> (78 x 158 x 3)
model.add(Convolution2D(n_filters1, filter1, filter1, 
	border_mode='valid',
	subsample=strides, 
	input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size,
	strides=strides, 
	border_mode='valid'))

# Convolutional Layer 2 (78 x 158 x 3) -> (74 x 154 x 6) -> (37 x 77 x 6)
model.add(Convolution2D(n_filters2, filter2, filter2, 
	border_mode='valid',
	subsample=strides))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size, 
	strides=strides, 
	border_mode='valid'))

# Convolutional Layer 3 (37 x 77 x 6) -> (33 x 73 x 12) -> (17 x 37 x 12)
model.add(Convolution2D(n_filters3, filter3, filter3, 
	border_mode='valid',
	subsample=strides))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size, 
	strides=strides, 
	border_mode='valid'))

# Convolutional Layer 4 (17 x 37 x 12) -> (15 x 35 x 24) -> (8 x 18 x 24)
model.add(Convolution2D(n_filters4, filter4, filter4, 
	border_mode='valid',
	subsample=strides))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size, 
	strides=strides, 
	border_mode='valid'))

# Convolutional Layer 5 (8 x 18 x 24) -> (6 x 16 x 36) -> (3 x 8 x 36)
model.add(Convolution2D(n_filters5, filter5, filter5, 
	border_mode='valid',
	subsample=strides))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size, 
	strides=strides, 
	border_mode='valid'))

# Flatten 3 x 8 x 36 = 864
model.add(Flatten())

# Fully Connected Layer 1
model.add(Dense(output_shape1, input_shape=(864,)))
model.add(Activation('relu'))
model.add(Dropout(keep_prob))

# Fully Connected Layer 2
model.add(Dense(output_shape2))
model.add(Activation('relu'))
model.add(Dropout(keep_prob))

# Fully Connected Layer 3
model.add(Dense(output_shape3))
model.add(Activation('relu'))
model.add(Dropout(keep_prob))

# Output Layer
model.add(Dense(output_shape4))

##### Train the model #####
adam = Adam(lr=learning_rate)

model.compile(loss='mean_squared_error',
	optimizer=adam,
	metrics=['accuracy'])

model.fit(X_train, y_train, 
	batch_size=batch_size, 
	nb_epoch=epochs, 
	validation_split=0.2,
	shuffle=True)

model.save('./model.h5')
print('Model saved')







