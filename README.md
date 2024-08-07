# Chest X-ray Classification
 Image classification of chest X-ray exams using CNN and Resnet-50

## Authors 
 Names: Thomaz Justo, Yuri Kiefer Alves, Lucas Schneider, and Thomas Sponchiado

## Introduction

 This report presents the development of a machine learning pipeline for the task of image classification using convolutional neural networks (CNNs) and Resnet-50. A dataset containing radiographs of COVID-19 patients was chosen.

 The objective of this work is to explore machine learning techniques and understand the inherent challenges in image classification.

 Throughout the process, steps were taken to model the problems, implement CNN & Resnet-50 using Keras, train the neural networks with the respective datasets, optimize hyperparameters, and analyze the obtained results.

## Methodology

 To develop the machine learning pipeline, several steps were followed:

 - Selection of datasets and their preprocessing
 - Initial network architecture definition
 - Definition of activation and error functions
 - Selection of optimizers
 
## Dataset

 To train the neural networks, the "COVID-19 Radiography Database" dataset was used, found on the Kaggle repository (https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database). It contains 21,165 images with a resolution of 299x299 pixels and features X-ray images along with segmentation masks for the lungs in each image.

 The labels used by the dataset to classify the images are: COVID, Lung_Opacity, Viral Pneumonia, and Normal. This separation is embedded in the filename of each image, and this will be the classification objective of the CNN and the Resnet-50 model used.

## Setup

 - To use the algorithm, download the dataset available at: https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database
 - Move the downloaded .zip file to the folder containing the .py file to be accessed by the algorithm
 - If not installed, install the following dependencies:
 ```sh
 keras
 numpy
 matplotlib
 sklearn
 os
 cv2
 seaborn
 ZipFile
 tensorflow
 ```

## Models Used

 ### Data Preprocessing

  The input data for the model can be either the image pixels or features extracted from the images.

 In this case, only the image pixels were used as input for the neural networks.

 The preprocessing converted all images to grayscale, reducing the processing complexity from 3 channels to one, and all were resized to 224x224 to be compatible with both the CNN and the pre-trained Resnet-50.

 The classes were extracted directly from the image by their folder names and underwent label preprocessing, replacing them with numbers to be used in the neural networks.

 ### Image Collection from the Dataset

  The images from the dataset were organized into folders, each containing photos of the respective pathology. Therefore, it is necessary to extract these images from the folders and insert them into a list, where they can be reorganized and assigned a class for future classification.

 ### Class Collection from the Images

  After organizing the dataset, it can be split into training and testing sets to generate metrics correctly later.

 The image classes were converted from strings to numbers so that the CNN could identify these labels.

 ### Index Randomization

  To better separate the dataset into training and testing sets, the image indices and classes were randomly reorganized. After that, the data were split into training and testing sets to be tested on the CNN and Resnet.

## CNN Architecture

### Data Input

Convolutional neural networks (CNNs) are deep learning algorithms widely used for image classification and recognition with little or sometimes no preprocessing.

Like MLPs, they consist of neurons that capture and process information from the previous layer and pass it on to the next layer of neurons. However, the model scans the images with a kernel and performs convolution on the values.

In the network used, the data input consists of the pixels of each image in 224x224 grayscale format.

### Convolution Layers

The following layers are hidden convolution layers that extract features and patterns from the input image using kernels (filters), performing the dot product of the image values to form a new image.

CNNs have a certain arbitrariness in the number of convolution layers used, so this value can be adjusted to better suit the proposed problem.

### Pooling Layers

Next, there are pooling layers, which aim to reduce the amount of information in the image to simplify its processing.
Methods used can be Max pooling, where the highest point in the neighborhood is chosen to be represented in the reduced image, or Average pooling, where the average values of the neighborhood are calculated and this value represents the simplified image.

### MLP Classification Layer

At the end of the model, fully connected neuron layers (MLPs) are used to classify the image and present the classification found by the model.

### Activation Functions

Used for complex algorithms due to their non-linearity, where linear algorithms would not be able to perform the task.

These functions perform mathematical operations to activate or deactivate neurons in a layer based on information from previous layers, causing the output to be slightly different from the previous epoch but without losing the good results generated by altering the weights of the neurons.

The most used functions are: Linear, Sigmoid, Tanh, ReLU, and Softmax.

### Error Functions

These measure the differences between the model's predicted results and the actual results obtained from the dataset.

Functions such as categorical crossentropy or sparse categorical crossentropy are used for multiclass problems, while binary crossentropy is used for binary categories, and squared or absolute error is used for regression problems.

For multiclass functions, this loss is calculated based on the number of references in the batch and the number of correct responses predicted by the model.

### Optimizers

These are algorithms that assist the learning process by adjusting the weights of neurons in the model layers and can alter the learning rate to become slower as it converges to the optimal point, preventing it from overshooting the ideal point and generating new errors.

Examples include Adam, Adagrad, and RMSprop, which have the same idea of altering the learning rate based on past gradients but with slight differences in how they handle very small learning rates.

Examples like SGD and Momentum update the weights based on gradients but in slightly different ways, making one faster than the other, but increasing noise.

## CNN Model Training

During CNN training, the model receives the images in the predetermined format, performs convolutions with each image, and makes decisions based on a number of references.

The process is repeated for the number of epochs determined at the beginning of the test, and at each epoch, the weights of the neurons are adjusted by the optimizer, also determined at the beginning of the test, to reduce the model's losses in image prediction.

Tests were performed for each network, exploring at least three variations of: number of layers, number of neurons per layer, types of activation functions, error functions, and optimizers. This way, at least 30 different experiments are planned.

### Results

We obtained good results using the CNN, with an accuracy of 93% and a loss of 0.33.

## Resnet-50 Architecture

From the CNN model, other models can be inferred, such as Resnet-50 (Residual Network), which is very similar to the previous one, except for the number of convolution layers, which in this case are 48 layers. Adding a Max pooling layer and an Average pooling layer makes up 50 layers, giving the model its name.

This model is the successor to the Resnet-34 network (itself a successor to the VGG 16 and 19 models), which had 34 layers with neurons of different weights and the ability to skip neuron layers when necessary, becoming a residual network rather than a standard convolutional network.

The network has two design rules. The first states that the number of filters in each layer is the same, and the second states that if the feature map number is halved, the filters must compensate by doubling in size to maintain complexity in each layer.

To assist the learning process, this network can be imported as pre-trained on standard datasets, with only adjustments to the final layers or fine-tuning the model.

The model for classifying images from the dataset used in this work was pre-trained with the Imagenet dataset images, and to use grayscale images like the above model, the initial layer of the model was replaced with one that accepts images with only one color channel. A dense layer was added at the end of the model for classification, where Resnet fine-tuning will be performed.

## Resnet-50 Training

Like the CNN, training is based on the number of epochs, the number of references for decision-making, and different optimizers, which were also tested with more than 30 different configurations.

### Results

We sought a model with better accuracy using a low number of epochs and found a model with results above 90.9% accuracy and 0.24 loss.
Using 50 epochs did not yield better results than the original. Here, we obtained 90.6% accuracy and 0.34 loss.

## Conclusion & References & Video

The use of CNNs for classifying images from the dataset was effective in terms of both time and accuracy, reaching over 90% accuracy in 10 epochs, with a loss below 0.4 on the test dataset.

Compared to Resnet-50, after adjustments, the CNN showed better results in terms of accuracy and loss reduction. However, it requires more training, which in more complex situations or with larger amounts of training images can become a bottleneck for the system and delay project delivery.

Resnet performed efficiently by requiring little training, suggesting that pre-training the network, even with completely different images, is beneficial for creating generic filters and understanding patterns more quickly, achieving close to 90% accuracy in just 2 epochs. However, it required a large number of epochs to reach 93% accuracy, as the best model showed.

A problem noted is the difficulty in classifying some images. The hypothesis is that these errors occurred due to the large difference in the number of samples in the dataset, where almost half of it is in only one class. However, this problem did not significantly affect the model's ability to classify.

Presentation video: https://youtu.be/jJI6fs9V7w4?si=zC1o56KrbsCVcJnv

## References

CNN:

https://keras.io/examples/timeseries/timeseries_classification_from_scratch/

https://keras.io/examples/vision/mnist_convnet/

https://keras.io/examples/vision/

Resnet-50:

https://datagen.tech/guides/computer-vision/resnet-50/

https://keras.io/api/applications/resnet/

## Meta

Thomas Sponchiado Pastore
Thomaz Justo
Yuri Kiefer Alves
Lucas Schneider

## Contribuição
1. Fork it (<https://github.com/thom01s/Chest-X-ray-Classification/fork>)
3. Create your feature branch
4. Commit your changes
5. Push to the branch
6. Create a new Pull Request
