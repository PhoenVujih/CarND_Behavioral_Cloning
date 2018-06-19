# **Behavioral Cloning**


---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/model.png "model"
[image2]: ./examples/origin.png "origin"
[image3]: ./examples/crop.png "crop"
[image4]: ./examples/flipped.png "flip"
[image5]: ./examples/resize.png "resize"
[image6]: ./examples/bright.png "bright"
[image7]: ./examples/placeholder_small.png "Flipped Image"



---

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The model I used in the project is originally inspired by the Nvidia Model. The model consists of 5 convolution layers and 4 fully connect layers. Notice that the original Nvidia Model contains 5 fully connect layers and I delete the one with the most weights and slightly adjust it.

The first reason why I adjusted it is that the situation of the simulator is quite simple and 4 fc layers are enough for it. Secondly, less parameters may help avoid overfitting of the model.  

The model includes ELU layers to introduce nonlinearity (code from line 23 to line 33), and the data is normalized in the model using a Keras lambda layer (code line 21).

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 29 and 31).

I first added only one dropout layer in the model and find the validation error increased while the training error decreased. I then added another dropout layer to further against the overfitting and find it worked well. The keep probability for the dropout layer is 0.5.

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an Adam optimizer with an original learning rate 0.0001, so the learning rate was not tuned manually (model.py line 36).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. Moreover, I also drive for several reversal loops the obtain the opposite data.

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first try was to employ the model developed be Nvidia with an input size of 66x200x3. To reduce the number of parameters of the model and avoid overfitting, I deleted the fc layer with the most weights.

During the training, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. So I added two dropout layers to combat the overfitting and finally the model worked well.

The final step was to run the simulator to see how well the car was driving around track one. The car ran well for the first try while it swung slightly after the turning. After several tries, I found that the car ran better if I resize the cropped image to the original size: 160x320x3 but not 66x200x3. The possible reason for it may be that the vertically enlarged image is easier for the model to capture the proper features inside.

At the end of the process, the vehicle is able to drive autonomously around the track without swinging.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-37) consisted of a convolution neural network with the following layers and layer sizes:

![alt text][image1]

#### 3. Creation of the Training Set

To capture good driving behavior, I first recorded five laps on track one using center lane driving and recovering from the left and right side at the during the corner (that's also why my car ran along the outer side of the corner when it turned, it copied my habit on driving). To balanced the data, I also drive the car for 3 reversal laps.

To make use of the images captured from the left and right camera, I added a bias on the steering angle when the left or right images were chosen. For the left image, the bias is +0.2 while for the right image, it's -0.2

To further generalize the model, I randomly flipped the image and the angle. It could further help balance the data and avoid the bias of the car during driving. Moreover, I randomly adjust the brightness of the colors in the image by editing the image in HSV color space which may help generalize the model.

##### Original image

![alt text][image2]

##### Flipped image
![alt text][image4]

##### Brightness adjusted image
![alt text][image6]

Finally, the images were cropped, resized and convert to YUV color space (Ref: Nvidia Paper) as the input of the model.

##### Cropped image
![alt text][image3]

##### Resized image
![alt text][image5]

#### 4. Batch Generation

After the collection process, I had more than 27000 of data points. I then use a generator to generate the batch for training (functions.py line 68 to 83).

The generator create the batch of the size 100 by randomly pick the data from the train dataset. It contains 6 kinds of data:

Data type |Fraction of the batch | Brightness adjusted
-|-|-
left image| 12.5% | 50%
flipped left image| 12.5% | 50%
right image| 12.5% | 50%
flipped right image| 12.5% | 50%
center image (normal)| 37.5% | 16.7%
flipped center image| 12.5% | 50%

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. To save the best model trained, I use the checkpoint to save the model with minimum validation error in 50 epochs. I used an adam optimizer so that manually training the learning rate wasn't necessary.
