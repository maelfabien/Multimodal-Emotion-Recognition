# Real-Time Multimodal Emotion Recognition

<img alt="GitHub followers" src="https://img.shields.io/github/followers/maelfabien.svg?style=social"> <img alt="GitHub contributors" src="https://img.shields.io/github/contributors-anon/maelfabien/Multimodal-Emotion-Recognition.svg"> <img alt="GitHub commit activity" src="https://img.shields.io/github/commit-activity/y/maelfabien/Multimodal-Emotion-Recognition.svg"> <img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/3.svg">

Table of Content :
- [I. Context](https://github.com/maelfabien/Multimodal-Emotion-Recognition/blob/master/README.md#i-context)
- [II. Data Sources](https://github.com/maelfabien/Multimodal-Emotion-Recognition/blob/master/README.md#ii-data-sources)
- [III. Downloads](https://github.com/maelfabien/Multimodal-Emotion-Recognition/blob/master/README.md#iii-download)
- [IV. Methodology](https://github.com/maelfabien/Multimodal-Emotion-Recognition/blob/master/README.md#iv-methodology)
  - [a. Text Processing](https://github.com/maelfabien/Multimodal-Emotion-Recognition/blob/master/README.md#a-text-processing)
  - [b. Audio Processing](https://github.com/maelfabien/Multimodal-Emotion-Recognition/blob/master/README.md#b-audio-processing)
  - [c. Video Processing](https://github.com/maelfabien/Multimodal-Emotion-Recognition/blob/master/README.md#c-video-processing)
  - [d. Ensemble Model](https://github.com/maelfabien/Multimodal-Emotion-Recognition/blob/master/README.md#d-ensemble-model)
- [V. How to use it ?](https://github.com/maelfabien/Multimodal-Emotion-Recognition/blob/master/README.md#v-how-to-use-it-)
- [VI. Demonstration](https://github.com/maelfabien/Multimodal-Emotion-Recognition/blob/master/README.md#vi-demonstration)
- [VII. Research Paper](https://github.com/maelfabien/Multimodal-Emotion-Recognition/blob/master/README.md#vii-research-paper)
- [VIII. Deployment](https://github.com/maelfabien/Multimodal-Emotion-Recognition/blob/master/README.md#viii-deployment)

In this project, we are exploring state of the art models in multimodal sentiment analysis. We have chosen to explore text, sound and video inputs and develop an ensemble model that gathers the information from all these sources and displays it in a clear and interpretable way.

## I. Context

Affective computing is a field of Machine Learning and Computer Science that studies the recognition and the processing of human affects.
Multimodal Emotion Recognition is a relatively new discipline that aims to include text inputs, as well as sound and video. This field has been rising with the development of social network that gave researchers access to a vast amount of data.


## II. Data Sources
We have chosen to diversify the data sources we used depending on the type of data considered. All data sets used are free of charge and can be directly downloaded.
- For the text input, we are using an annotated essay text corps provided by ...
- For sound data sets, we are using the **Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)**. This database contains 7356 files (total size: 24.8 GB). The database contains 24 professional actors (12 female, 12 male), vocalizing two lexically-matched statements in a neutral North American accent. Speech includes calm, happy, sad, angry, fearful, surprise, and disgust expressions, and song contains calm, happy, sad, angry, and fearful emotions. Each expression is produced at two levels of emotional intensity(normal, strong), with an additional neutral expression. All conditions are avail-able in three modality formats: Audio-only (16bit, 48kHz .wav), Audio-Video(720p H.264, AAC 48kHz, .mp4), and Video-only (no sound).” https://zenodo.org/record/1188976#.XCx-tc9KhQI
- For the video data sets, we are using the popular **FER2013** Kaggle Challenge data set. The data consists of 48x48 pixel grayscale images of faces. The faces have been automatically registered so that the face is more or less centered and occupies about the same amount of space in each image. The data set remains quite challenging to use, since there are empty pictures, or wrongly classified images. https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data

## III. Download

| Modality | Data | Processed Data (for training) | Pre-trained Model | Colab Notebook |Other |
| --- | --- | --- | --- | --- | --- |
| Text | [here](https://drive.google.com/file/d/1hWqVdOYNvCuioiDk-CBgMtKOgl05aA--/view?usp=sharing) | --- |  --- | --- | --- |
| Audio | [here](https://drive.google.com/file/d/1OL2Kx9dPdeZWoue6ofHcUNs5jwpfh4Fc/view?usp=sharing) | [X-train](https://drive.google.com/file/d/1oCY8NpLfHsyCvqCDnRmTdsrDJDWg72Rj/view?usp=sharing) [y-train](https://drive.google.com/file/d/1y5j43I09Xe6RHK8BsHP8_ZNkUuTehhgY/view?usp=sharing) [X-test](https://drive.google.com/file/d/1MN1Fxc_sDR1ZDQmPdFMwlnhP4qn9d8bT/view?usp=sharing) [y-test](https://drive.google.com/file/d/1ovvCXumkEP1oLxErgMgyIg1Z1Eih430W/view?usp=sharing)| [Weights] [Model] | [Colab Notebook](https://colab.research.google.com/drive/1EY8m7uj3BzU-OsjAPGBqoapw1OSUHhum)  | --- |
| Video | [here](https://drive.google.com/file/d/1hWqVdOYNvCuioiDk-CBgMtKOgl05aA--/view?usp=sharing) | [X-train](https://drive.google.com/file/d/14xs-0nZNQuuMdtTOwqcJQm_GZ_rTO8mB/view?usp=sharing) [y-train](https://drive.google.com/file/d/1EX5KkPquwpHD9ZKpTxGhk3_RFmVDD8bf/view?usp=sharing) [X-test](https://drive.google.com/file/d/1TFH3kvGDS0iWjqKYo3lZuIu65I9h0LYr/view?usp=sharing) [y-test](https://drive.google.com/file/d/1HTzGc_J4kTQRFvLIvcMQA3mt6PnyNT53/view?usp=sharing) |  [Weights](https://drive.google.com/file/d/1-L3LnxVXv4vByg_hqxXMZPvjKSQ12Ycs/view?usp=sharing) [Model](https://drive.google.com/file/d/1_dpHN9L6hsQYzTX2zk9K5JF2CZ1FOcZh/view?usp=sharing) | [Colab Notebook](https://colab.research.google.com/drive/1dV1IvYLV24vXGvyzMFNAA18csu8btV2-) | [Face Detect Model](https://drive.google.com/file/d/18YMrAStwXbN-aPZ45ylNrdAXQQPJx0Hd/view?usp=sharing)
| Ensemble | [here](https://drive.google.com/file/d/1hWqVdOYNvCuioiDk-CBgMtKOgl05aA--/view?usp=sharing) | --- |  --- | --- | --- |

## IV. Methodology
Our aim is to develop a model able to provide a live sentiment analysis with a visual user interface.Therefore, we have decided to separate two types of inputs :
- Textual input, such as answers to questions that would be asked to a person from the platform
- Video input from a live webcam or stored from an MP4 or WAV file, from which we split the audio and the images

### a. Text Processing

![image](/Presentation/Images/text_pipeline.png)

### b. Audio Processing

#### Pipeline

The speech emotion recognition pipeline was built the following way :
- Voice recording
- Audio signal discretization
- Log-mel-spectrogram extraction
- Split spectrogram using a rolling window
- Make a prediction using our pre-trained model

#### Model

The model we have chosen is a **Time distributed ConvNet**, it combines convolutional and recurrent neural network. To limit overfitting, we tuned the model with :
- Audio data augmentation
- Early stopping
- L2-Regularization
- And kept the best model

![image](/Presentation/Images/sound_pipeline.png)

### c. Video Processing

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
- To know more on the XCeption model, check this article : https://maelfabien.github.io/deeplearning/xception/

![image](/Presentation/Images/model_fit.png)

The XCeption architecture is based on DepthWise Separable convolutions that allow to train much fewer parameters, and therefore reduce training time on Colab's GPUs to less than 90 minutes.

![image](/Presentation/Images/video_pipeline2.png)

### d. Ensemble Model

![image](/Presentation/Images/ensemble_pipeline.png)

## V. How to use it ?

The project currently is under the form of a set of notebooks for each modality. The combination of the video, sound and text analysis can be found in the "Common" section. We will soon be publishing *.py* files as well as detailed explanations on the requirements.

- Run the file `live_face.py` to run the facial emotion recognition.

## VI. Demonstration

Here is a quick demo of the facial emotion recognition algorithm from a webcam stream :

![image](/Presentation/Images/Mon-film-7_1.gif)

## VII. Research Paper

If you are interested in the research paper we are working on currently, feel free to check out this link :
https://www.overleaf.com/read/xvtrrfpvzwhf

## VIII. Deployment

The app will soon be available based on this template : https://github.com/render-examples/fastai-v3
