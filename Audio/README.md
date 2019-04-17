# Speech Emotion Recognition

The aim of this section is to explore speech emotion recognition techniques from an audio recording.

## Data

The data set used for training is the *Ryerson Audio-Visual Database of Emotional Speech and Song*: https://zenodo.org/record/1188976#.XA48aC17Q1J

 **RAVDESS** contains 24 professional actors (12 female, 12 male), vocalizing two lexically-matched statements in a neutral North American accent. Speech includes calm, happy, sad, angry, fearful, surprise, and disgust expressions, and song contains calm, happy, sad, angry, and fearful emotions. Each expression is produced at two levels of emotional intensity (normal, strong), with an additional neutral expression.

 ![image](Images/RAVDESS.png)

## Requirements

```
Python : 3.6.5
Scipy : 1.1.0
Scikit-learn : 0.20.1
Tensorflow : 1.12.0
Keras : 2.2.4
Numpy : 1.15.4
Pydub : 0.23.0
Ffmpeg : 4.0.2
```

## Files

The different files that can be found in this repo :
- `Model` : Saved models (SVM and TimeDistributed CNNs)
- `Notebook` : All notebooks (preprocessing and model training)
- `Python` : Personal audio library
- `Images`: Set of pictures saved from the notebooks and final report
- `Resources` : Some resources on Speech Emotion Recognition

Notebooks provided on this repo:
- `01 - Preprocessing[SVM].ipynb` : Signal preprocessing and feature extraction from time and frequency domain (global statistics) to train SVM classifier.
- `02 - Train [SVM].ipynb` : Implementation and training of SVM classifier for Speech Emotion Recognition
- `01 - Preprocessing[CNN-LSTM].ipynb` :  Signal preprocessing and log-mel-spectrogram extraction to train TimeDistributed CNNs
- `02 - Train [CNN-LSTM].ipynb` : Implementation and training of TimeDistributed CNNs classifier for Speech Emotion Recognition

## Signal preprocessing


## Performance


## Sources
