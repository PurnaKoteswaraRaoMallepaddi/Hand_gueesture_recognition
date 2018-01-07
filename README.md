# Hand_guesture_recognition
guessing the orintation of the hand and out putting the voice messages that are to be conveyed through the hardware using rasberry pi which is under the process.

##Requirements
1. Anaconda - This installs python along with most popular python libraries including sklearn. If not already installed, install it from https://www.continuum.io/downloads . 
2. Keras- Python library for Deep laerning.
3. open cv - Python library for playing with the images.

##Preparing dataset:
- Put all the unlabelled audio files in a folder named `calls`, or any other folder and update the name of folder in `handguesture.py`.
-Then the preprocessed data will be created and stored in the `data.npy` file will be created along with the `lables.npy`
-Then the model will be created and stored in the format of  `.h5` file for the furthur use.

##Output format
-Out of all i have used 3 classes for the Convelutional model to  determine which class the image depend upon.
-The out put format is always a softmax vector of probabilities and the maximum one too.
-We can apply the out put aany where as of now i  am trying to apply it for understanding the signs of the dumb people.
