## Behavioral Cloning Project

The goals / steps of this project are the following:

* Use the simulator to collect data of good driving behavior
* Build a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[recordingerror]: ./writeup_images/recording_fails.png "Problem recording to directory"
[center]: ./writeup_images/center.png "Image from center camera"
[left]: ./writeup_images/left.png "Image from left camera"
[right]: ./writeup_images/right.png "Image from right camera"
[centerflipped]: ./writeup_images/center_flipped.png "Image from center camera, flipped left<->right"
[cameraangles]: ./writeup_images/cameraangles.png "Diagram of why a correction must be applied to left and right camera images"


### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files: 

* **`model.py`** contains the script to create and train the model 
* **`drive.py`** for driving the car in autonomous mode 
* **`model.h5`** contains a trained convolution neural network 
* **`writeupreport.md`** summarizes the results
* **`video.mp4`** video of the car driving around the first track, with steering data supplied by **`model.h5`**

#### 2. Submission includes functional code

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5```you should see the car drive around the track autonomously without leaving the road.

#### 3. Submission code is usable and readable

The `model.py` file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I used the [NVIDIA Neural Network](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) architecture which creates steering data using camera input.

#### 2. Attempts to reduce overfitting in the model

To reduce overfitting, data was split into 80% training and 20% validation.  Overfitting was insignificant using the augmented data set. Validation loss was similar to test set loss.  I also included a dropout layer to reduce overfitting.


#### 3. Model parameter tuning

The model used an Adam optimizer, so the learning rate was not tuned manually. A correction angle was added to the driving angle to pair with a corresponding camera image.

I trained the model with correction angles between 0.6 and 0.8 and ended up using .65 as my final value.  At higher values, the car overcorrected on slight curves, though it did perform well on the tighter turns. The car also shows the ability to recover when it approaches the side of the road. 

#### 4. Appropriate training data

For details about how I created the training data, see ***"Creation of the Training Set"*** below.

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

To get the data pipeline working I used a simple 1 layer fully connected network, using only the center camera.

To test performance, I trained LeNet using data only from the center camera.
Performace was inconsistent when driving the car onto the bridge.

My first layer is a cropping layer which removes features such as the car hood and the sky and trees above the horizon, features which may confuse the model.


I augmented the training data by adding images from the left and right cameras and flipped images from the center camera.

I implemented Python generators to serve training and validation data to `model.fit_generator()`, making `model.py` run much faster. However, the car still failed at the tight turns after the bridge.

Next, I implemented the [NVIDIA Neural Network](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) architecture, a network specifically for training self driving cars using camera input.


For the correction angle, higher values resulted in fast, overcompensating turns. Training with lower values such as .65  reduced swerving, although slower insufficient turning through the tight curves.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.


#### 2. Final Model Architecture

Here is a visualization of the architecture layers and sizes.  Because of the large number of parameters in the first fully connected layer, I added a dropout layer after it to reduce overfitting.


| Layer                         |     Description                       |
|:---------------------:|:---------------------------------------------:|
| Input                 | 160x320x3 RGB image                                      A
| Cropping              | Crop top 50 pixels and bottom 20 pixels; output shape = 90x320x3 |
| Normalization         | Each new pixel value = old pixel value/255 - 0.5      |
| Convolution 5x5       | 5x5 kernel, 2x2 stride, 24 output channels, output shape = 43x158x24  |
| RELU                  |                                                       |
| Convolution 5x5       | 5x5 kernel, 2x2 stride, 36 output channels, output shape = 20x77x36   |
| RELU                  |                                                       |
| Convolution 5x5       | 5x5 kernel, 2x2 stride, 48 output channels, output shape = 8x37x48    |
| RELU                  |                                                       |
| Convolution 5x5       | 3x3 kernel, 1x1 stride, 64 output channels, output shape = 6x35x64    |
| RELU                  |                                                       |
| Convolution 5x5       | 3x3 kernel, 1x1 stride, 64 output channels, output shape = 4x33x64    |
| RELU                  |                                                       |
| Flatten               | Input 4x33x64, output 8448    |
| Fully connected       | Input 8448, output 100        |
| Dropout               | Set units to zero with probability 0.5 |
| Fully connected       | Input 100, output 50          |
| Fully connected       | Input 50, output 10           |
| Fully connected       | Input 10, output 1 (labels)   |




#### 3. Creation of the Training Set & Training Process


I created `driving_log.csv` by driving two laps. Each line of `driving_log.csv` corresponded to one sample, consisting of a relative path to center, left, and right camera images, steering angle, throttle, brake, and speed.

For each data sample, I used all three images and augmented the data with a flipped version of the center camera image.  

**Center camera view**

![center camera][center]



**Corresponding LEFT camera view**

![left camera][left]

**Corresponding RIGHT camera view**

![right camera][right]

**Flipped center image**

![center flipped][centerflipped]

As the car drives in the simulator, it receives data from the center camera only.
The left and right cameras give 
views of what the center camera
would see should the car veer to either side, and the car should correct itself accordingly. So after adding the left and right camera images to the training and validation sets, we should
assign angles to the images that represent what the steering angle should be if the center camera were sees 
the image recorded by the left or right camera, or what the car should do if the center camera finds itself positioned where the left or right cameras are positioned.

If a car centered on the road swerves, causing the center camera to now occupy the spot formerly occupied by the **left** camera, a driving correction angle should be **added** (clockwise) to stay on the road.  Swerving to the **right** would require a correction angle to be **subtracted** (counter-clockwise). 

In the diagram below, a line from the left camera's position
to the same destination is further to the right (clockwise), while a line from the right camera's position to the same destination is 
further to the left (counter-clockwise). 

![angle corrections][cameraangles]

Adding the left and right images paired with their corrected angles to the training set
will help the car recover when veering to either side. 

The angle for each flipped image is the "mirror image" or  **negative** of the current driving angle. The original track is counterclockwise with predominantly left turns. Flipping the center camera image and pairing it with a corresponding flipped angle adds supplies right turn data, which helps the model generalize.

Images were read from files and flipped images were added with a Python generator.  Aside from shuffling the data, the generator also added lines to the 
file that stored image locations along with angle data `driving_log.csv` in batches of 32, and supplied 
data to `model.fit_generator()` in batches of 128 (each line of `driving_log.csv` was used to provide 
a center, left, right and flipped center image). 


The data set consisted of 8036 samples, each of which had a path to a center, left, and right image.
`sklearn.model_selection.train_test_split()` was used to split off 20% of the samples to use for validation.
For each sample, the center-flipped image was created on the fly within the generator.
Therefore, my network was trained on a total of 
`floor(8036x0.8) x 4 = 25,712` image+angle pairs, and validated on a total of `ceil(8036x0.2) x 4 = 6432` image+angle pairs.

A separate generator was created for training data and validation data. The training generator provided images and angles from the training set, and the validation generator provided images and data derived from the validation set.

I trained for 5 epochs using an Adams optimizer, which was probably more epochs than necessary, but I wanted to be sure the validation error was plateauing. I also monitered the validation error to make sure it was not increasing for later epochs.

