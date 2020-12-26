# CartPole-v1 with CNN using Keras

*Tensorflow*: [![PyPI](https://badge.fury.io/py/tensorflow.svg)](https://badge.fury.io/py/tensorflow)  
*Keras*: [![Keras](https://img.shields.io/pypi/pyversions/keras.svg?style=plastic)](https://badge.fury.io/py/keras)

**Problem definition**
<img align="center" src=https://i.imgur.com/IPlXZ5P.png width="600">


A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The system is controlled by applying a force of +1 or -1 to the cart. The pendulum starts upright, and the goal is to prevent it from falling over. A reward of +1 is provided for every timestep that the pole remains upright. The episode ends when the pole is more than 15 degrees from vertical, or the cart moves more than 2.4 units from the center.
CartPole-v1 defines "solving" as getting average reward of 195.0 over 100 consecutive trials.

## Model Architecture
Resolving CartPole using images involves using the $env.render(mode =$ 'rgb\_array'$)$ function to have a rendered frame image for each step of the episode.
Cartpole gym environment outputs 600x400 RGB arrays (600x400x3). That was way too many pixels with such simple task, more than I needed, so I converted the output to grayscale and I downsized it.
The result was something like this:
<img align="center" src=https://i.imgur.com/WOU6srq.png width="300">

**Architecture used:**



## Results

* CartPole-v1 trained with classic DNN using DQN without images:
* CartPole-v1 trained with CNN usign DQN:




