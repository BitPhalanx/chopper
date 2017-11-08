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

# def cnn_model():
  # We will create the model for our CNN here
  # The plan: Input layer takes in 105x80x3 = 25200
  # 

def choose_action():
  # Feed in probability and return an action 
  # Actions: up, down, left, right, shoot, nothing
  #           2     5     4      3      1        0
  # Fix this return after neural network is created
  return random.randint(1,6)

def main():
  # Prepare Chopper Command as env variable
  # and start access to images
  env = gym.make('ChopperCommand-v0')
  observation = env.reset()

  # observation now holds unsigned 8 bit int array
  # with shape (210, 160, 3). Let's half this for
  # our neural network to allow easier processing
  # by taking every other pixel
  reduced_observation = observation[::2, ::2, :]
  # reduced_observation is now shape (105,80,3)
  # Confirm reduced observation shape
  # print("Reduced observation shape: ", reduced_observation.shape)

  # Choosing to keep colors since enemies have different
  # colors. We can now feed this into our CNN.
  # Blue planes and white helicoptors = enemies
  # Black trucks = friendly

  # We need int32s, so cast to this type and double check
  # casted_observation = tf.cast(reduced_observation, tf.float32) / 255.0
  # casted_observation = tf.cast(reduced_observation, tf.float32)
  # print(observation)
  # print(casted_observation)

  # Reshape our array into 4-D tensor to allow input to NN  
  # input_layer = tf.reshape(reduced_observation, [-1,105,80,3])
  # print("Input Layer shape: ", input_layer.shape)

  #env.render allows us to render the graphics
  while True:
    observation, reward, done, info = env.step(choose_action())
    print(observation)
    env.render()
    # env.step()
    if done:
      # Compute weight changes and backpropagate here
      # Then reset the environment for another run.
      env.reset()

if __name__ == "__main__":
  main()
