# Behaviorial Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
In this project, A deep convolutional neural network was built to clone driving behavior. The model use an end-to-end strategy which means the image as the input and steering angle as the output.

The following is the video of the performance of the model.

[![model run](https://www.youtube.com/watch?v=YtueTZdf1sY/0.jpg)](https://www.youtube.com/watch?v=YtueTZdf1sY)

For more details about the data collection, data processing and model training, please refer to [writeup_report.md](./writeup_report.md)


## Dependencies
This lab requires:

* Tensorflow
* Keras
* Pandas
* Numpy
* OpenCV
* Sklearn


The simulator can be downloaded from the classroom. In the classroom, we have also provided sample data that you can optionally use to help train your model.

### How to use it with your own dataset

Check the code in `model.py`, change the name of the csv file to your own and run it. It is assumed that you've already correct the filepath of the images if you collect data and train the model in the different platform.

Usage of `drive.py` requires you have saved the trained model as an h5 file, i.e. `model.h5`. See the [Keras documentation](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model) for how to create this file using the following command:
```sh
model.save(filepath)
```

Once the model has been saved, it can be used with drive.py using this command with the help of the simulator:

```sh
python drive.py model.h5
```
