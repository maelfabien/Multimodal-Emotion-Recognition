# Multimodal Emotion Recognition WebApp

We analye facial, vocal and textual emotions, using mostly deep learning based approaches. We deployed a web app using Flask :

![image](webapp.png)

## How to use it ?

To use the web app :
- Clone the project locally
- Go in the WebApp folder
- Run `$ pip install -r requirements.txt``
- Launch the app by running `python main.py`
- Go to http://127.0.0.1:5000/ (or follow the link given in your terminal

## How does it work ?

As stated in the project home page, we have defined and trained deep learning models to analyze emotions and psychological traits from video, audio and text inputs.
The user should be able to click on any of the sections, take the interview, and get a feedback on his performance compared to other people who already took the test.

## Getting the feedback

For both the text and the audio, a button will directly allow you to get feedback. A dashboard displays your performance compared to other candidates.

For the video, due to restrictions of Flask, we are recording the video input for 45 seconds. After this time, the image will freeze. Simply switch the URL to `/video_dash` instead of `/video1` in the URL bar to go to the dashboard.

## Organization

The organization of the project is the following :

- Models : All the pre-trained models used by the WebApp
- library : The Python scripts that run the emotion detection algorithms
- static :
  - CSS : The CSS style sheet and fixed images to display
  - JS : The JavaScript of the app (D3.js) and the databases that store the information
- templates : All the HTML pages of the project
- tmp : Temporary files (i.e. an image from video interview, an audio file or a PDF)
- main.py : The Flask page that calls the functions and redirects to HTML files
