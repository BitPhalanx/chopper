# Attempt to play Chopper Command using OpenAI library
# There are 2 versions of the game
#
# 1. RAM as input (ChopperCommand-ram-v0)
#      RAM of Atari 2600 consists of 128 bytes
#      AI nets score higher using this as input
# 2. Screen images as input (ChopperCommand-v0)
#      RGB image, array of shape (210, 160, 3)
#
# Each action is repeatedly performed for k frames,
# with k being uniformly sampled from {2,3,4}
#
# It seems that the highest scores were made using DQN,
# but not many used policy gradient methods. I will
# attempt to use policy gradient.


# Import OpenAI gym and other needed libraries
import gym
import tensorflow as tf
import numpy as np
import random
import math

# Pretend there's code here actually doing something ahha
# I'm still researching
