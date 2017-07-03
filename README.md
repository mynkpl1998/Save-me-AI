# Save-me-AI
Implementation of Deep Q-Learning to Learn how to play a simple game written in python. 

Here is the demo computer playing the game

![image](https://raw.githubusercontent.com/mynkpl1998/Save-me-AI/master/data/ai_plays.gif)

# Overview
This project contains three major files, AI.py file creates the model and train the model to play the game, play_ai.py uses the saved pre-trained model to play the game. If you like play the game by your own you can play using game.py script.

# Installation Dependencies:

* Pygame
* Keras
* Tensorflow or Theano
* Numpy

# How to run ?
```
 git clone https://github.com/mynkpl1998/Save-me-AI.git
 cd Save-me-AI
 python play_ai.py
```

# How to train ?
```
 git clone https://github.com/mynkpl1998/Save-me-AI.git
 cd Save-me-AI
 python AI.py
```

# Network Architecture

 * Convert the screen images to grayscale
 * Stack the four frames together 
 * 3 Convolutional Neural Network without maxpool layers
 * 2 Fully Connected Dense layers
 
 # Hyperparameters
 
  * Learning Rate (Default : 1e-06)
  * Initial Epsilon (Default : 0.1)
  * Final Epsilon (Default : 0.0001)
