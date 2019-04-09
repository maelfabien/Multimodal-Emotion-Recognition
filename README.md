# Real-Time Multimodal Emotion Recognition

<img alt="GitHub followers" src="https://img.shields.io/github/followers/maelfabien.svg?style=social"> <img alt="GitHub contributors" src="https://img.shields.io/github/contributors-anon/maelfabien/Multimodal-Emotion-Recognition.svg"> <img alt="GitHub commit activity" src="https://img.shields.io/github/commit-activity/y/maelfabien/Multimodal-Emotion-Recognition.svg"> <img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/3.svg">

In this project, we are exploring state of the art models in multimodal sentiment analysis. We have chosen to explore text, sound and video inputs and develop an ensemble model that gathers the information from all these sources and displays it in a clear and interpretable way. 

## I. Context

Affective computing is a field of Machine Learning and Computer Science that studies the recognition and the processing of human affects. 
Multimodal Emotion Recognition is a relatively new discipline that aims to include text inputs, as well as sound and video. This field has been rising with the development of social network that gave researchers access to a vast amount of data.


## II. Data Sources
We have chosen to diversify the data sources we used depending on the type of data considered. All data sets used are free of charge and can be directly downloaded.
- For the text input, we are using an annotated essay text corps provided by ...
- For sound data sets, we are using the Ryerson Audio-Visual Database ofEmotional Speech and Song (RAVDESS).”The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)contains 7356 files (total size: 24.8 GB). The database contains 24 professionalactors (12 female, 12 male), vocalizing two lexically-matched statements in aneutral North American accent. Speech includes calm, happy, sad, angry, fearful,surprise, and disgust expressions, and song contains calm, happy, sad, angry, andfearful emotions. Each expression is produced at two levels of emotional intensity(normal, strong), with an additional neutral expression. All conditions are avail-able in three modality formats: Audio-only (16bit, 48kHz .wav), Audio-Video(720p H.264, AAC 48kHz, .mp4), and Video-only (no sound).” https://zenodo.org/record/1188976#.XCx-tc9KhQI
- For the video data sets, we are using the popular FER2013 Kaggle Challenge data set. The data consists of 48x48 pixel grayscale images of faces. The faceshave been automatically registered so that the face is more or less centered andoccupies about the same amount of space in each image. The data set remainsquite challenging to use, since there are empty pictures, or wrongly classified images. https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data

|Data|Link|
|Video|[here](https://drive.google.com/file/d/1hWqVdOYNvCuioiDk-CBgMtKOgl05aA--/view?usp=sharing)|


## III. Methodology
Our aim is to develop a model able to provide a live sentiment analysis with avisual user interface.Therefore, we have decided to separate two types of inputs :
- Textual input, such as answers to questions that would be asked to a personfrom the platform
- Video input from a live webcam or stored from an MP4 or WAV file, fromwhich we split the audio and the images

### a. Text Processing 

![image](/Presentation/Images/text_pipeline.png)

### b. Audio Processing

![image](/Presentation/Images/sound_pipeline.png)

### c. Video Processing

![image](/Presentation/Images/video_pipeline.png)

## IV. How to use it ?
The project currently is under the form of a set of notebooks for each domain. The combination of the video, sound and text analysis can be found in the "Common" section. We will soon be publishing .py files as well as detailed explanations on the requirements. 

If you are interested in the research paper we are working on currently, feel free to check out this link :
https://www.overleaf.com/read/xvtrrfpvzwhf
