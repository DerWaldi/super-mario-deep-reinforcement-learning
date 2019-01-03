# Super Mario Bros AI using Deep Reinforcement Learning

## Overview

The aim of this project is to master the game of Super Mario Bros using a Deep Reinforcement Learning approach.<br />
Therefore a player agent with a Deep Q Learning Model was implemented.

## Installation

### Install required libraries
```
pip install -r requirements.txt
```

## Usage
### Step 1: Train Model
Adjust hyper parameters as needed in the "config.py" file.
```
python train.py
```
Observe trianing loss using tensorboard:
```
tensorboard --logdir="logs/"
```

### Step 2: Play 
```
python play.py
```

## Credits

Inspired by Deep Q Learning with Tensorflow and Space Invaders from Thomas SIMONINI:<br/>
https://www.youtube.com/watch?v=gCJyVX98KJ4

Using the Super Mario Bros gym from Kautenja:<br/>
https://github.com/Kautenja/gym-super-mario-bros
