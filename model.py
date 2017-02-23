import numpy as np
import pandas
import os
# Silence overly verbose TF logging
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import cv2
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ELU , Dense
from keras.layers.core import Lambda, Flatten, Dropout, Activation
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from skimage.exposure import adjust_gamma
import pdb
import math
import json
import progressbar
from keras.models import model_from_json
import argparse
import random
from shutil import copyfile

###############################
########## Arguments ##########
###############################

parser = argparse.ArgumentParser(description='Remote Driving')
parser.add_argument('-m', '--model', type=str,
	help='Path to model definition json. Model weights should be on the same path.')
parser.add_argument('-w', '--weights', type=str,
	help='Path to weights file. (Optional)')
parser.add_argument('-o', '--output', type=str,
	help='Output file for model. Model weights will be saved to the same path.',
	default='model.json')
parser.add_argument('-d','--debug', action='store_true',
	help='Enable Debug Mode')
parser.add_argument('-c','--csv', type=str,
	help='Path to CSV for Fine Tuning mode.')
args = parser.parse_args()

###############################
########## Variables ##########
###############################

FINE_TUNING = True if args.model else False
DEBUG = args.debug == True
CSV_PATH = args.csv if args.csv else 'driving_log.csv'

# Multiply steering angle for left and right camera images
STEERING_CORRECTION=.08
# Probability for loading side camera images
SIDE_IMAGE_KEEP = .8
# Probability for flipping images
FLIP_IMAGE_KEEP = .4

IMG_WIDTH = 200
IMG_HEIGHT = 66
IMG_DEPTH = 3
INPUT_SIZE = (IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH)

# Number of epochs without improvement for early stopping
PATIENCE = 3

STRIDES = (2,2)
STRIDES2 = (1,1)

N_FILTERS_1 = 24
FILTER_1 = 5

N_FILTERS_2 = 36
FILTER_2 = 5

N_FILTERS_3 = 48
FILTER_3 = 5

N_FILTERS_4 = 64
FILTER_4 = 3

N_FILTERS_5 = 64
FILTER_5 = 3

DROPOUT_RATE = 0.25

OUTPUT_SHAPE_1 = 100
OUTPUT_SHAPE_2 = 50
OUTPUT_SHAPE_3 = 10
OUTPUT_SHAPE_4 = 1

BATCH_SIZE = 128
EPOCHS = 30

if FINE_TUNING:
	LEARNING_RATE = 0.00001
else:
	LEARNING_RATE = 0.001

adam = Adam(lr=LEARNING_RATE)

###############################
########## Load data ##########
###############################

driving_log = pandas.read_csv(CSV_PATH, delimiter=',', 
	names=['CENTER', 'LEFT', 'RIGHT', 'STEERING_ANGLE', 'THROTTLE', 'BREAK', 'SPEED'])
driving_log = driving_log[['CENTER', 'LEFT', 'RIGHT', 'STEERING_ANGLE']]

X_train = []
y_train = []
print('loading images...')

with progressbar.ProgressBar(max_value=len(driving_log)) as bar:
	for index, row in driving_log.iterrows():
		X_train.append([row['CENTER'],row['LEFT'],row['RIGHT']])
		y_train.append(row['STEERING_ANGLE'])
		bar.update(index)

if DEBUG:
	print("Number of training samples before splitting: " + str(len(X_train) * 4))
	print("Number of labels: " + str(len(y_train)))

print("Number of training samples before splitting: " + str(len(X_train) * 4))
print("Number of labels: " + str(len(y_train)))

################################
########## Split Data ##########
################################

# Shuffle and separate training and validation data
X_train, y_train = shuffle(X_train, y_train)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

if DEBUG:
	print("Number of training samples after splitting: " + len(X_train))
	print("Number of validation samples: " + len(X_val))

# Variables that use len(X_train)
if FINE_TUNING:
	epoch_sample_multiplier = 10
else:
	epoch_sample_multiplier =  math.floor(2 * len(X_train)/BATCH_SIZE)

###################################
########## Generate Data ##########
###################################

def process_img(img):
	img = img[60:126,60:260]
	img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
	img = cv2.GaussianBlur(img, (3,3), 0)
	return img

def flip_img():
	flip = cv2.flip(img, 1)
	x_flip = np.array([flip])
	y_flip = y * -1.0
	x_chunk = np.append(x_chunk, x_flip, axis=0)
	y_chunk = np.append(y_chunk, y_flip)
	return x_chunk, y_chunk

def load_side_img(side):
	side_img = cv2.imread(x_input[i_train][side])
	if side == 1:
		side_steering = steering_center + STEERING_CORRECTION
	else:
		side_steering = steering_center - STEERING_CORRECTION
	side_img = process_img(side_img)
	x_chunk = np.append(x_chunk, side_img, axis=0)
	y_chunk = np.append(y_chunk, side_steering)
	return x_chunk, y_chunk

def generate_data(x_input,y_input):
	indices = []
	switch = [0, 1]

	while True:
		x_chunk = np.array([]).reshape((0,) + INPUT_SIZE)
		y_chunk = np.array([])
		while len(x_chunk) < (BATCH_SIZE):
			if len(indices) == 0:
				indices = np.arange(0, len(x_input) -1)
			index_p = np.random.choice(indices)
			i_train = indices[index_p]
			np.delete(indices,index_p)
			img = cv2.imread(x_input[i_train][0])
			img = process_img(img)
			x = np.array([img])
			y = y_input[i_train]
			steering_center = float(y)
			if y == 0:
				non_zero = random.choice(switch)
				if non_zero == 0:
					x_chunk = np.append(x_chunk, x, axis=0)
					y_chunk = np.append(y_chunk, y)
			if (len(x_chunk) != BATCH_SIZE) and should_keep(FLIP_IMAGE_KEEP):
				x_chunk, y_chunk = flip_img()
			if (len(x_chunk) != BATCH_SIZE) and should_keep(SIDE_IMAGE_KEEP):
				x_chunk, y_chunk = load_side_img(1)
			if (len(x_chunk) != BATCH_SIZE) and should_keep(SIDE_IMAGE_KEEP):
				x_chunk, y_chunk = load_side_img(2)

		yield x_chunk, y_chunk

######################################
########## Helper Functions ##########
######################################

# Calculate size of each layer and return input size for fully connected layer #1
def calc_input_size():
	c1_out = []
	c1_out.append(math.ceil((IMG_HEIGHT - FILTER_1 + 1) / STRIDES[0]))
	c1_out.append(math.ceil((IMG_WIDTH - FILTER_1 + 1) / STRIDES[0]))
	c1_out.append(N_FILTERS_1)

	c2_out = []
	c2_out.append(math.ceil((c1_out[0] - FILTER_2 + 1) / STRIDES[0]))
	c2_out.append(math.ceil((c1_out[1] - FILTER_2 + 1) / STRIDES[0]))
	c2_out.append(N_FILTERS_2)

	c3_out = []
	c3_out.append(math.ceil((c2_out[0] - FILTER_3 + 1) / STRIDES[0]))
	c3_out.append(math.ceil((c2_out[1] - FILTER_3 + 1) / STRIDES[0]))
	c3_out.append(N_FILTERS_3)

	c4_out = []
	c4_out.append(math.ceil((c3_out[0] - FILTER_4 + 1) / STRIDES2[0]))
	c4_out.append(math.ceil((c3_out[1] - FILTER_4 + 1) / STRIDES2[0]))
	c4_out.append(N_FILTERS_4)

	c5_out = []
	c5_out.append(math.ceil((c4_out[0] - FILTER_5 + 1) / STRIDES2[0]))
	c5_out.append(math.ceil((c4_out[1] - FILTER_5 + 1) / STRIDES2[0]))
	c5_out.append(N_FILTERS_5)

	if DEBUG == True:
		print("Conv1 Shape: ({0} * {1} * {2})".format(c1_out[0],c1_out[1],c1_out[2]) )
		print("Conv2 Shape: ({0} * {1} * {2})".format(c2_out[0],c2_out[1],c2_out[2]) )
		print("Conv3 Shape: ({0} * {1} * {2})".format(c3_out[0],c3_out[1],c3_out[2]) )
		print("Conv4 Shape: ({0} * {1} * {2})".format(c4_out[0],c4_out[1],c4_out[2]) )
		print("Conv5 Shape: ({0} * {1} * {2}) = {3}".format(c5_out[0],c5_out[1],c5_out[2], (c5_out[0] * c5_out[1] * c5_out[2])))

	result = c5_out[0] * c5_out[1] * c5_out[2]

	return result

# Determine Keep probability for loading images
def should_keep(prob):
	return np.random.random_sample() < prob

##################################
########## Create Model ##########
##################################

model = Sequential()

# Normalize Data
model.add(Lambda(lambda x: x/255.0 -0.5,
	input_shape=INPUT_SIZE,
	output_shape=INPUT_SIZE))

# Convolutional Layer 1 
model.add(Convolution2D(N_FILTERS_1, FILTER_1, FILTER_1, 
	border_mode='valid',
	subsample=STRIDES, 
	input_shape=INPUT_SIZE))
model.add(Activation('relu'))

# Convolutional Layer 2 
model.add(Convolution2D(N_FILTERS_2, FILTER_2, FILTER_2, 
	border_mode='valid',
	subsample=STRIDES))
model.add(Activation('relu'))

# Convolutional Layer 3 
model.add(Convolution2D(N_FILTERS_3, FILTER_3, FILTER_3, 
	border_mode='valid',
	subsample=STRIDES))
model.add(Activation('relu'))

# Convolutional Layer 4 
model.add(Convolution2D(N_FILTERS_4, FILTER_4, FILTER_4, 
	border_mode='valid',
	subsample=STRIDES2))
model.add(Activation('relu'))

# Convolutional Layer 5 
model.add(Convolution2D(N_FILTERS_5, FILTER_5, FILTER_5, 
	border_mode='valid',
	subsample=STRIDES2))
model.add(Activation('relu'))
model.add(Dropout(DROPOUT_RATE))

# Flatten 
model.add(Flatten())

# Fully Connected Layer 1
model.add(Dense(OUTPUT_SHAPE_1, input_shape=(calc_input_size(),)))
model.add(Activation('relu'))

# Fully Connected Layer 2
model.add(Dense(OUTPUT_SHAPE_2))
model.add(Activation('relu'))

# Fully Connected Layer 3
model.add(Dense(OUTPUT_SHAPE_3))
model.add(Activation('relu'))

# Output Layer
model.add(Dense(OUTPUT_SHAPE_4))

######################################
########## Fine-tuning mode ##########
######################################

# Freeze all but the last convolutional layer and the fully connected layers.
# This is to allow fine tuning conv5 without causing overfitting, as a very limited data set is being used.
# Technique described in https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html	

if FINE_TUNING:
	print("Training (Fine Tuning Mode Enabled)")

	weights_file = args.weights if args.weights else args.model.replace('json', 'h5')
	model.load_weights(weights_file)
	for layer in model.layers[:2]:
		layer.trainable = False

	model.compile(adam, "mse", metrics=['accuracy'])

#####################################
########## Train the model ##########
#####################################

else:
	model.compile(loss='mean_squared_error',
		optimizer=adam,
		metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss',
	min_delta=0,
	patience=PATIENCE,
	verbose=1,
	mode='auto')

checkpoint = ModelCheckpoint(os.path.dirname(args.output) +"\weights.{epoch:02d}-{val_loss:.2f}.h5",
	verbose=0, 
	save_best_only=False, 
	save_weights_only=True, 
	mode='auto', 
	period=1)

model.fit_generator(generate_data(X_train, y_train),
	validation_data=generate_data(X_val, y_val),
	callbacks=[early_stopping, checkpoint],
	samples_per_epoch=epoch_sample_multiplier * BATCH_SIZE,
	nb_epoch=EPOCHS,
	nb_val_samples=len(X_val))

print("Saving model...")
if not FINE_TUNING:
	model_json = model.to_json()
	with open(args.output, "w") as json_file:
	    json.dump(model_json, json_file)
else:
	copyfile(args.model, args.output)

model.save(args.output.replace('json','h5'))
print('Model saved')


