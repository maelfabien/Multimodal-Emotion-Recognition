# Facial Emotion Recognition

The aim of this section is to explore facial emotion recognition techniques from a live webcam video stream.

![image](video_app.png)

## Data

The data set used for training is the **Kaggle FER2013** emotion recognition data set : https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data

The data consists of 48x48 pixel grayscale images of faces. The faces have been automatically registered so that the face is more or less centered and occupies about the same amount of space in each image. The task is to categorize each face based on the emotion shown in the facial expression in to one of seven categories (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral).

![alt text](Images/Read_Images/bar_plot.png)

## Requirements

```
Python : 3.6.5
Tensorflow : 1.10.1
Keras : 2.2.2
Numpy : 1.15.4
OpenCV : 4.0.0
```

## Files

The different files that can be found in this repo :
- `Images` : All pictures used in the ReadMe, in the notebooks or save from the models
- `Resources` : Some resources that have been used to build the notebooks
- `Notebooks` : Notebooks that have been used to obtain the final model
- `Model`: Contains the pre-trained models for emotion recognition, face detection and facial landmarks
- `Python` : The code to launch the live facial emotion recognition

Among the notebooks, the role of each notebook is the following :
- `01-Pre-Processing.ipynb` : Transform the initial CSV file into train and test data sets
- `02-HOG_Features.ipynb` : A manual extraction of features (Histograms of Oriented Gradients, Landmarks) and SVM
- `03-Pre-Processing-EmotionalDAN.ipynb` : An implementation of Deep Alignment Networks to extract features
- `04-LGBM.ipynb` : Use of classical Boosting techniques on top on flatenned image or auto-encoded image
- `05-Simple_Arch.ipynb` : A simple Deep Learning Architecture
- `06-Inception.ipynb` : An implementation of the Inception Architecture
- `07-Xception.ipynb` : An implementation of the Xception Architecture
- `08-DeXpression.ipynb` : An implementation of the DeXpression Architecture
- `09-Prediction.ipynb` : Live Webcam prediction of the model
- `10-Hybrid.ipynb` : A hybrid deep learning model taking both the HOG/Landmarks model and the image

## Video Processing

#### Pipeline

The video processing pipeline was built the following way :
- Launch the webcam
- Identify the face by Histogram of Oriented Gradients
- Zoom on the face
- Dimension the face to 48 * 48 pixels
- Make a prediction on the face using our pre-trained model
- Also identify the number of blinks on the facial landmarks on each picture

#### Model

The model we have chosen is an **XCeption** model, since it outperformed the other approaches we developed so far. We tuned the model with :
- Data augmentation
- Early stopping
- Decreasing learning rate on plateau
- L2-Regularization
- Class weight balancing
- And kept the best model

As you might have understood, the aim was to limit overfitting as much as possible in order to obtain a robust model.

- To know more on how we prevented overfitting, check this article : https://maelfabien.github.io/deeplearning/regu/
- To know more on the **XCeption** model, check this article : https://maelfabien.github.io/deeplearning/xception/

![image](Images/Read_Images/model_fit.png)

The XCeption architecture is based on DepthWise Separable convolutions that allow to train much fewer parameters, and therefore reduce training time on Colab's GPUs to less than 90 minutes.

![image](Images/Read_Images/video_pipeline2.png)

When it comes to applying CNNs in real life application, being able to explain the results is a great challenge. We can indeed  plot class activation maps, which display the pixels that have been activated by the last convolution layer. We notice how the pixels are being activated differently depending on the emotion being labeled. The happiness seems to depend on the pixels linked to the eyes and mouth, whereas the sadness or the anger seem for example to be more related to the eyebrows.

![image](Images/Read_Images/light.png)

## Performance

The set of emotions we are trying to predict are the following :
- Happiness
- Sadness
- Fear
- Disgust
- Surprise
- Neutral
- Anger

The models have been trained on Google Colab using free GPUs.

|       Features                          |   Accuracy    |
|-----------------------------------------|---------------|
| LGBM on flat image                      |     --.-%     |
| LGBM on auto-encoded image              |     --.-%     |
| SVM on HOG Features                     |     32.8%     |
| SVM on Facial Landmarks features        |     46.4%     |
| SVM on Facial Landmarks and HOG features|     47.5%     |
| SVM on Sliding window Landmarks & HOG   |     24.6%     |
| Simple Deep Learning Architecture       |     62.7%     |
| Inception Architecture                  |     59.5%     |
| Xception Architecture                   |     64.5%     |
| DeXpression Architecture                |     --.-%     |
| Hybrid (HOG, Landmarks, Image)          |     45.8%     |

# Download the model and the data
| Modality | Data | Processed Data (for training) | Pre-trained Model | Colab Notebook | Other |
|:--------:|:----:|:-----------------------------:|:-----------------:|:--------------:|:-----:|
| Video | [here](https://drive.google.com/file/d/1hWqVdOYNvCuioiDk-CBgMtKOgl05aA--/view?usp=sharing) | [X-train](https://drive.google.com/file/d/14xs-0nZNQuuMdtTOwqcJQm_GZ_rTO8mB/view?usp=sharing) [y-train](https://drive.google.com/file/d/1EX5KkPquwpHD9ZKpTxGhk3_RFmVDD8bf/view?usp=sharing) [X-test](https://drive.google.com/file/d/1TFH3kvGDS0iWjqKYo3lZuIu65I9h0LYr/view?usp=sharing) [y-test](https://drive.google.com/file/d/1HTzGc_J4kTQRFvLIvcMQA3mt6PnyNT53/view?usp=sharing) |  [Weights](https://drive.google.com/file/d/1-L3LnxVXv4vByg_hqxXMZPvjKSQ12Ycs/view?usp=sharing) [Model](https://drive.google.com/file/d/1_dpHN9L6hsQYzTX2zk9K5JF2CZ1FOcZh/view?usp=sharing) | [Colab Notebook](https://colab.research.google.com/drive/1dV1IvYLV24vXGvyzMFNAA18csu8btV2-) | [Face Detect Model](https://drive.google.com/file/d/18YMrAStwXbN-aPZ45ylNrdAXQQPJx0Hd/view?usp=sharing) |

# Live prediction

Since the input data is centered around the face, making a live prediction requires :
- identifying the faces
- then, for each face :
  - zoom on it
  - apply grayscale
  - reduce dimension to match input data

The face identification is done using a pre-trained Histogram of Oriented Gradients model. For further information, check the following article :
https://maelfabien.github.io/tutorials/face-detection/#b-the-integral-image

The treatment of the image is done through OpenCV

*1. Read the initial image*

![alt text](Images/Read_Images/face_1.png)

*2. Apply gray filter and find faces*

![alt text](Images/Read_Images/face_2.png)

*3. Zoom and rescale each image*

![alt text](Images/Read_Images/face_5.png)

Live prediction Illustration :

![alt text](Images/Read_Images/Mon-film-7_1.gif)

## Sources
- Visualization : https://github.com/JostineHo/mememoji/blob/master/data_visualization.ipynb
- State of the art Architecture : https://github.com/amineHorseman/facial-expression-recognition-using-cnn
- Eyes Tracking : https://www.pyimagesearch.com/2017/04/24/eye-blink-detection-opencv-python-dlib/
- Face Alignment : https://www.pyimagesearch.com/2017/05/22/face-alignment-with-opencv-and-python/
- C.Pramerdorfer,  and  M.Kampel.Facial  Expression  Recognition  using  Con-volutional  Neural  Networks:  State  of  the  Art.  Computer  Vision  Lab,  TU  Wien. https://arxiv.org/pdf/1612.02903.pdf
- A Brief Review of Facial Emotion Recognition Based
on Visual Information : https://www.mdpi.com/1424-8220/18/2/401/pdf
- Going deeper in facial expression recognition using deep neural networks : https://ieeexplore.ieee.org/document/7477450
- Emotional Deep Alignment Network paper : https://arxiv.org/abs/1810.10529
- Emotional Deep Alignment Network github : https://github.com/IvonaTau/emotionaldan
- HOG, Landmarks and SVM : https://github.com/amineHorseman/facial-expression-recognition-svm
