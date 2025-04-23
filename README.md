# Music Genre Classification in Python
I implemented a music genre classifer using a convolution neural network (CNN). I used the GTZAN
dataset, a popular music genre dataset that includes 1000 songs across 10 different
genres: blues, classical, country, disco, hip-hop, jazz, metal, pop, reggae, and rock.
I attained a maximum accuracy of <ins>0.798</ins>, a <ins>28.71% increase</ins> from the accuracy
using the orignial dataset. 

** Note: data files were excluded from the repo because they were too large **

<div>
  <img src="https://img.shields.io/badge/Python-blue">
  <img src="https://img.shields.io/badge/TensorFlow/Keras-blue"> 
  <img src="https://img.shields.io/badge/NumPy-blue"> 
  <img src="https://img.shields.io/badge/PyDub-blue"> 
  <img src="https://img.shields.io/badge/SoX-blue"> 
</div>


## Methodology
Given the wav files for each of the 10 genres, I converted the audio signals into 128x128x3
spectrogram images using the Sound eXchange (SoX) library. I then split the
spectrogram data, where 80% was used for training and 20% for validation. I utilized
three methods of data augmentation: loudness variation, pitch change, and random noise. For 
each method, I trained a CNN model and evaluated its accuracy score using TensorFlow/Keras.

## CNN Model
The model architecture includes five convolutional layers with ReLU activation followed by max-pooling layers for dimensionality reduction. 
Dropout layers with a dropout rate of 0.2 are incorporated after each max-pooling layer to prevent overfitting. 
The feature maps are flattened into a one-dimensional vector before passing through two fully connected layers, the first with 64 neurons and ReLU activation, 
the second with 10 neurons and softmax activation for multi-class classification.

The model was compiled using the Adam optimizer with a learning rate of 0.001, sparse categorical cross-entropy loss function, and accuracy as the metric for evaluation. 
The training was done over 100 epochs with a batch size of 32.

## Results
The accuracy score of the original dataset was 0.620. For the loudness
variation, pitch change, and random noise augmented datasets, the accuracies were 0.663, 0.633, and
0.798 respectively.






