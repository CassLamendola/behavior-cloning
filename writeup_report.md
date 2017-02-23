# Behavior Cloning
## Behavior Cloning Project Goals
* Collect driving data from a Udacity simulator
* Build a Convolutional Neural Network (CNN) with Keras
* Train the model to predict steering angles from images
* Test the model in the simulator

## Rubric Points 
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation. 
----------------------------------------------------------------------------
### Files Submitted & Code Quality
#### 1. Submission includes required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py - to create and train the model
* drive.py - for driving the car in autonomous mode
* model.h5 & model.json - containing a trained CNN
* writeup\_report.md summarizing the results

#### 2. Submission includes fucntional code

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing:

```
python drive.py model.json
```
#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy
#### 1. An appropriate model architecture has been employed

The model consists of a CNN with 5 convolutional layers, 3 hidden fully connected layers, and a final output layer. The first 3 convolutional layers use 5x5 filter size and a stride of 2. The 4th and 5th convolutional layers use 3x3 filters and a stride of 1. The depths of the convolutions range from 24 to 64. (model.py lines 257 - 312)

The model includes ReLu layers to introduce nonlinearity and the data is normalized in the model using Keras Lambda layer. (line 260)

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers on the last convolutional layer to reduce overvitting. (model.py line 294)

#### 3. Model parameter tuning

The model used an adam optimizer with a learning rate of 0.001 for training and 0.00001 for fine-tuning the model. (model.py line 96 - 101)

#### 4. Appropriate training data

Training data was chosen to keep the vehicle on the road. A combination of driving in the center of the lane and recovering from the left and right sides of the road was used.

### Architecture and Training Documentation
#### 1. Solution design approach

The overall strategy for the model architecture was to start with a simple CNN with a few layers and add complexity as needed to improve accuracy and decrease loss. Initially, the model had 3 convolutional layers with a stride of 1 and max pooling after each layer.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that both were very low and often wouldn't improve at all after the first epoch. The best accuracy achieved with this model was less than 50%.

Eventually, the model was changed to resemble the NVIDIA architecture as descrived in [this](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) paper. I thought this model would be appropriate because it was created to do the same task. At first, only the number of layers was adjusted. I only changed one part of the architecture at a time to see what would improve the model. Later, the max pooling was ommitted and the convolutions instead used a stride of 2 to reduce size. Eventually, I ended up with a model that was identical to the NVIDIA model.

To combat overfitting, I added dropout on the last convolutional layer.

The final step was to run the simulator to see how well the car was driving around track one. The first few times, the steering angle didn't change and the vehicle drove right off the road. At this point, more data was collected (about 30,000 frames) and the model began to predict different angles. There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I collected even more 'recovery' data. I placed the car near the edge of the lane and recorded the car driving back to center. 

After that, every model I trained came so close to finishing the track, but crossed the line in one or two places. At this point I decided to implement fine-tuning. [This](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html) was a helpful resource. I collected data at the point of the track where the vehicle was crossing the line and fine-tuned on that data.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final model architecture

The final model architecture (model.py lines 257 - 312) consists of a convolution neural network with the following layers and layer sizes:

* Convolutional Layer 1 -> (66 x 200 x 3) -> (31 x 98 x 24)
* Convolutional Layer 2 -> (31 x 98 x 6) -> (14 x 47 x 36)
* Convolutional Layer 3 -> (14 x 47 x 12) -> (5 x 22 x 48)
* Convolutional Layer 4 -> (5 x 22 x 24) -> (3 x 20 x 64)
* Convolutional Layer 5 -> (3 x 20 x 36) -> (1 x 18 x 64)
* Fully Connected Layer 1 -> 1 x 18 x 64 = 1152 -> 100
* Fully Connected Layer 2 -> 100 -> 50
* Fully Connected Layer 3 -> 50 -> 10
* Output Layer -> 10 -> 1

Each layer (other than the output layer) is followed by ReLu activation.

#### 3. Creation of the training set & training process

![alt text]()








