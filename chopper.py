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
# import math
import time

def cnn_model(x):
  # We will create the model for our CNN here
  # Input layer takes in 104x80x3 = 25200
  with tf.name_scope('reshape'):
    x_image = tf.reshape(x, [-1, 104, 80, 3])

  # Conv 10x10 box across 3 color channels into 32 features
  with tf.name_scope('conv1'):
    W_conv1 = weight_variable([10,10,3,32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1)

  # Max pool to half size (52x40)
  with tf.name_scope('pool1'):
    h_pool1 = max_pool_2x2(h_conv1)

  # 2nd conv, 10x10 box from 32 to 64 features
  with tf.name_scope('conv2'):
    W_conv2 = weight_variable([10,10,32,64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

  # 2nd max pool, half size again (26x20)
  with tf.name_scope('pool2'):
    h_pool2 = max_pool_2x2(h_conv2)

  # Start fully connected layer, 26*20*64 = 33280 to 8192 fully connected
  with tf.name_scope('fc1'):
    W_fc1 = weight_variable([26*20*64, 8192])
    b_fc1 = bias_variable([8192])
    # Flatten max pool to enter fully connected layer
    h_pool2_flat = tf.reshape(h_pool2, [-1, 26*20*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  # Fully connected from 8192 to 6 (1 for each action possible)
  with tf.name_scope('fc2'):
    W_fc2 = weight_variable([8192, 6])
    b_fc2 = bias_variable([6])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

  return y_conv, keep_prob

def conv2d(x, W):
  # Return full stride 2d conv
  return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
  # 2x2 max pool
  return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

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
  # Remove odd number from first observation
  reduced_observation = reduced_observation[1:, :, :]
  # reduced_observation is now shape (104,80,3)
  # Confirm reduced observation shape
  print("Reduced observation shape: ", reduced_observation.shape)
  float_input = reduced_observation.astype(np.float32)
  # reduced_observation.view('<f4')
  y_conv, keep_prob = cnn_model(float_input)
  print("Keep_prob: ", keep_prob)
  print("Y_Conv: ", y_conv)

  # Choosing to keep colors since enemies have different
  # colors. We can now feed this into our CNN.
  # Blue planes and white helicoptors = enemies
  # Black trucks = friendly

  # Reshape our array into 4-D tensor to allow input to NN  
  # input_layer = tf.reshape(reduced_observation, [-1,105,80,3])
  # print("Input Layer shape: ", input_layer.shape)

  #env.render allows us to render the graphics
  while True:
    observation, reward, done, info = env.step(choose_action())
    # print(observation)
    print(reward)
    # info is an object from the dict class, holding an attribute
    # called ale.lives, returns number of lives we have left
    # print(info['ale.lives'])
    time.sleep(0.05)
    env.render()
    # env.step()
    if done:
      # Compute weight changes and backpropagate here
      # Then reset the environment for another run.
      env.reset()

if __name__ == "__main__":
  main()
