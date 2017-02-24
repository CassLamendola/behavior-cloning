# Behavior Cloning
## Use Deep Learning to Clone Driving Behavior

### Overview

In this project, I used what I've learned about deep neural networks and convolutional neural networks to clone driving behavior. I created a model based on an [paper from NVIDIA](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) using Keras.

Udacity provided a simulator which I used to steer a car around a track for data collection. I trained a neural network with image data and steering angles from the simulator. Then I used this model to drive the car autonomously around the track.

For a description of the steps I took to meet the requirements in the project [rubric](https://review.udacity.com/#!/rubrics/432/view) and an example of my output, see my writeup [here](https://github.com/CassLamendola/behavior-cloning/blob/master/writeup_report.md).

Here is a list of the files I've included in my project:

[model.py](https://github.com/CassLamendola/behavior-cloning/blob/master/model.py) (script used to create and train the model)
[drive.py](https://github.com/CassLamendola/behavior-cloning/blob/master/drive.py) (script to drive the car)
model.h5 (a trained Keras model)
model.json (an outline of the model)
[writeup.md](https://github.com/CassLamendola/behavior-cloning/blob/master/writeup_report.md) (a report writeup file)