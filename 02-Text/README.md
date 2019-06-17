# Text-based Personality Traits Recognition

![image](/00-Presentation/Images/text_app.png)

In this section you will find all resources, models and Python scripts relative to text-based personality traits recognition.

## Introduction

Emotion recognition through text is a challenging task that goes beyond conventional sentiment analysis : instead of simply detecting neutral, positive or negative feelings from text, the goal is to identify a set of emotions characterized by a higher granularity. For instance, feelings like anger or happiness could be included in the classification. As recognizing such emotions can turn out to be complex even for the human eye, machine learning algorithms are likely to obtain mixed performances. It is important to note that nowadays, emotion recognition from facial expression tends to perform better than from textual expression. Indeed, many subtleties should be taken into account in order to perform an accurate detection of human emotions through text, context-dependency being one of the most crucial. This is the reason why using advanced natural language processing is required to obtain the best performance possible.

In the context of our study, we chose to use text mining in order not to detect regular emotions such as disgust or surprise, but to recognize personality traits based on the "Big Five" model in psychology. Even though emotion recognition and personality traits classification are two separate fields of studies based on different theoretical underpinnings, they use similar learning-based methods and literature from both areas can be interesting. The main motivation behind this choice is to offer a broader assessment to the user : as emotions can only be understood in the light of a person's own  characteristics, we thought that analyzing personality traits would provide a new key to understanding emotional fluctuations. Our final goal is to enrich the user experience and improve the quality of our analysis : any appropriate and complementary information deepening our understanding of the user's idiosyncrasies is welcome

Many psychology researchers (starting with D. W. Fiske [1949], then Norman [1963] and Goldberg [1981]), believe that it is possible to exhibit five categories, or core factors, that determine one's personality. The acronym OCEAN (for openness, conscientiousness, extraversion, agreeableness, and neuroticism) is often used to refer to this model. We chose to use this precise model as it is nowadays the most popular in psychology : while the five dimensions don't capture the peculiarity of everyone's personality, it is the theoretical framework most recognized by researchers and practitioners in this field.

Many linguistic-oriented tools can be used to derive a person's personality traits, for instance the individual's linguistic markers (obtained using text analysis, psycholinguistic databases and lexicons for instance). Since one of the earliest studies in this particular field [Mairesse et al., 2007], researchers have introduced multiple linguistic features and have shown correlations between them and the Big Five. These features could therefore have a non-negligible impact on classification performances, but as we stated before, we will mainly focus machine learning methods and leave out the linguistic modeling as it does not fit into the spectrum of our study.

Our main goal is to leverage on the use of statistical learning methods in order to build a tool capable of recognizing the personality traits of an individual given a text containing his answers to pre-established personal questions. Our first idea was to record a user's interview and convert the file from audio to text : in this way we would have been able to work with similar data for text, audio and video. Nevertheless, the good transcription of audio files to text requires the use of expensive APIs, and the tools available for free in the market don't provide sufficient quality. This is the reason why we chose to apply our personality traits detection model to short texts directly written by users : in this way we can easily target particular themes or questions and provide indications of the language level to use. As a result of this, we can make sure that the text data we use to perform the personality traits detection is consistent with the data used for training, and therefore ensure the highest possible quality of results.

## Data

| Data | Processed data for training | Processed data for testing | Pre-trained CNN/LSTM model|
|:----:|:---------------------------:|:--------------------------:|:-------------------------:|
| [Stream-of-consciousness](https://drive.google.com/file/d/1bbbn8kSBmcVObafdzAQEipRBc4SVqwtb/view?usp=sharing) | [X-train](https://drive.google.com/file/d/1sgxv40PkrzxqFCfoaVGyTxpSMvgzRyyx/view?usp=sharing) [y-train](https://drive.google.com/file/d/1iL4N_k2501fGb6WiDLECvDQaJrCVzOjV/view?usp=sharing) | [X-test](https://drive.google.com/file/d/1ez38Be4__hA0quIrcLkiSl_zAl_HXktL/view?usp=sharing) [y-test](https://drive.google.com/file/d/1G0Tm9Vq5UoJcdQw0q11xgV4dnrP8JiPI/view?usp=sharing) | [Weights](https://drive.google.com/file/d/1XpFAMykCdmphzMw9umS21f8ITf5QwVRg/view?usp=sharing) [Model](https://drive.google.com/file/d/1mXn3poSmg0chYGXKNB7gjFl50E10kuU2/view?usp=sharing) |

We are using data that was gathered in a study by Pennebaker and King [1999]. It consists of a total of 2,468 daily writing submissions from 34 psychology students (29 women and 5 men whose ages ranged from 18 to 67 with a mean of 26.4). The writing submissions were in the form of a course unrated assignment. For each assignment, students were expected to write a minimum of 20 minutes per day about a specific topic. The data was collected during a 2-week summer course between 1993 to 1996. Each student completed their daily writing for 10 consecutive days. Students’ personality scores were assessed by answering the Big Five Inventory (BFI) [John et al., 1991]. The BFI is a 44-item self-report questionnaire that provides a score for each of the five personality traits. Each item consists of short phrases and is rated using a 5-point scale that ranges from 1 (disagree strongly) to 5 (agree strongly). An instance in the data source consists of an ID, the actual essay, and five classification labels of the Big Five personality traits. Labels were originally in the form of either yes (‘y’) or no (‘n’) to indicate scoring high or low for a given trait. It is important to note that the classification labels have been applied according to answers to a rather short self-report questionnaire : there might be a non-negligible bias in the data due to both the relative simplicity of the BFI test compared to the complexity of psychological features, and the cognitive biases preventing users from providing a perfectly accurate assessment of their own characteristics.

## Files

The different files that can be found in this repo :
- `Model` : Saved models (SVM and CNN/LSTM)
- `Notebook` : A Jupyter notebook for preprocessing, training and visualization
- `Python` : .py files for loading/visualizing data, training and predicting
- `Images`: Set of pictures saved from the notebooks and final report
- `Resources` : Some resources on Personnality Traits Recognition

## Requirements
```
Python : 3.6.5
Scipy : 1.1.0
Scikit-learn : 0.20.0
Tensorflow : 1.12.0
Keras : 2.2.2
Numpy : 1.15.2
Nltk : 3.3.0
Gensim : 3.4.0
```

## Pipeline

![image](/00-Presentation/Images/text_pipeline.png)

The text-based personality recognition pipeline has the following structure :
- Text data retrieving
- Custom natural language preprocessing :
	- Tokenization of the document
	- Cleaning and standardization of formulations using regular expressions (for instance replacing "can't" by "cannot", "'ve" by "have")
	- Deletion of the punctuation
	- Lowercasing the tokens
	- Removal of predefined stopwords (such as 'a', 'an' etc.)
	- Application of part-of-speech tags on the remaining tokens
	- Lemmatization of tokens using part-of-speech tags for more accuracy.
	- Padding the sequences of tokens of each document to constrain the shape of the input vectors. The input size has been fixed to 300 : all tokens beyond this index are deleted. If the input vector has less than 300 tokens, zeros are added at the beginning of the vector in order to normalize the shape. The dimension of the padded sequence has been determine using the characteristics of our training data. The average number of words in each essay was 652 before any preprocessing. After the standardization of formulations, and the removal of punctuation characters and stopwords, the average number of words dropped to 168 with a standard deviation of 68. In order to make sure we incorporate in our classification the right number of words without discarding too much information, we set the padding dimension to 300, which is roughly equal to the average length plus two times the standard deviation.
- 300-dimension Word2Vec trainable embedding
- Prediction using our pre-trained model

## Model

We have chosen a neural network architecture based on both one-dimensional convolutional neural networks and recurrent neural networks.
The one-dimensional convolution layer plays a role comparable to feature extraction : it allows finding patterns in text data. The Long-Short Term Memory cell is then used in order to leverage on the sequential nature of natural language : unlike regular neural network where inputs are assumed to be independent of each other, these architectures progressively accumulate and capture information through the sequences. LSTMs have the property of selectively remembering patterns for long durations of time.
Our final model first includes 3 consecutive blocks consisting of the following four layers : one-dimensional convolution layer - max pooling - spatial dropout - batch normalization. The numbers of convolution filters are respectively 128, 256 and 512 for each block, kernel size is 8, max pooling size is 2 and dropout rate is 0.3.
Following the three blocks, we chose to stack 3 LSTM cells with 180 outputs each. Finally, a fully connected layer of 128 nodes is added before the last classification layer.

## Performance

We tried different baseline models in order to assess the performance of our final architecture. Here are the accuracies of the different models.

![image](/00-Presentation/Images/perf_text_final.png)
