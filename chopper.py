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
# I will try to fit best episode per batch
# and do a ton of batches/episodes


# Import OpenAI gym and other needed libraries
import gym
import tensorflow as tf
import numpy as np
import random
import time

# Execution parameters, we can toy with these to optimize
bn_is_training = False # Tells batch norm if we're training or not
lr = 1E-3
render_graphics = True
slowdown_dbg = False
epsilon_greedy = 1.0
epsilon_decay = 0.999
num_of_batches = 500
episodes_per_batch = 10

def cnn_model():
  # Batch Norm HyperParameters
  bn_scale = True
  input_tensor = tf.placeholder(tf.float32)
  train_tensor = tf.placeholder(tf.float32)

  # Input layer takes in 104x80x3 = 25200
  with tf.name_scope('reshape'):
    input_reshape = tf.reshape(input_tensor, [-1, 104, 80, 3])

  # Conv 3x3 box across 3 color channels into 32 features
  with tf.name_scope('conv1'):
    W_conv1 = weight_variable([3,3,3,32])
    b_conv1 = bias_variable([32])
    pre_bn_conv1 = conv2d(input_reshape, W_conv1) + b_conv1
    post_bn_conv1 = tf.contrib.layers.batch_norm(pre_bn_conv1, center = True, scale = bn_scale, is_training = bn_is_training, scope = 'bn1')
    h_conv1 = tf.nn.relu(post_bn_conv1)

  # Max pool to half size (52x40)
  with tf.name_scope('pool1'):
    h_pool1 = max_pool_2x2(h_conv1)

  # 2nd conv, 3x3 box from 32 to 64 features
  with tf.name_scope('conv2'):
    W_conv2 = weight_variable([3,3,32,64])
    b_conv2 = bias_variable([64])
    pre_bn_conv2 = conv2d(h_pool1, W_conv2) + b_conv2
    post_bn_conv2 = tf.contrib.layers.batch_norm(pre_bn_conv2, center = True, scale = bn_scale, is_training = bn_is_training, scope = 'bn2')
    h_conv2 = tf.nn.relu(post_bn_conv2)

  # 2nd max pool, half size again (26x20)
  with tf.name_scope('pool2'):
    h_pool2 = max_pool_2x2(h_conv2)

  # 3rd conv, 3x3 box from 64 to 128 features
  with tf.name_scope('conv3'):
    W_conv3 = weight_variable([3,3,64,128])
    b_conv3 = bias_variable([128])
    pre_bn_conv3 = conv2d(h_pool2, W_conv3) + b_conv3
    post_bn_conv3 = tf.contrib.layers.batch_norm(pre_bn_conv3, center = True, scale = bn_scale, is_training = bn_is_training, scope = 'bn3')
    h_conv3 = tf.nn.relu(post_bn_conv3)

  # 3rd max pool, half size last time (13x10)
  with tf.name_scope('pool3'):
    h_pool3 = max_pool_2x2(h_conv3)

  # First fully connected layer, 13*10*128 = 16640 to 512 fully connected
  with tf.name_scope('fc1'):
    W_fc1 = weight_variable([13*10*128, 512])
    b_fc1 = bias_variable([512])
    # Flatten max pool to enter fully connected layer
    h_pool3_flat = tf.reshape(h_pool3, [-1, 13*10*128])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

  with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  # Fully connected from 512 to 6 (1 for each action possible)
  with tf.name_scope('fc2'):
    W_fc2 = weight_variable([512, 6])
    b_fc2 = bias_variable([6])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    moveprobs = tf.nn.softmax(y_conv)
  
  # Optimizer and loss setup
  # TODO check if this is set up properly
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = moveprobs, labels = train_tensor))
  optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

  return moveprobs, input_tensor, keep_prob , train_tensor #, y_conv # Might be helpful for backprop?

def conv2d(x, W):
  # Return full stride 2d conv
  return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
  # 2x2 max pool
  return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def weight_variable(shape):
  initial = tf.contrib.layers.xavier_initializer()
  return tf.Variable(initial(shape))

def bias_variable(shape):
  initial = tf.contrib.layers.xavier_initializer()
  return tf.Variable(initial(shape))

def choose_action(moveprobs, input_epsilon_greedy):
  # Feed in probability and return an action 
  # Actions: up, down, left, right, shoot, nothing
  #           2     5     4      3      1        0
  # print(moveprobs[0][0])
  if np.random.uniform() <= input_epsilon_greedy:
    random_selection = random.randint(0,5)
    return random_selection
  else:
    index_of_max = moveprobs[0][0].argmax()
    return index_of_max
  
def prep_image(observation):
  # observation holds unsigned 8 bit int array
  # with shape (210, 160, 3). Half this
  reduced_observation = observation[::2, ::2, :]
  # Remove odd number from first observation
  reduced_observation = reduced_observation[1:, :, :]
  # reduced_observation is now shape (104,80,3)
  float_input = reduced_observation.astype(np.float32)
  return float_input
 
def main():
  # global allows us to modify epsilon_greedy
  global epsilon_greedy, bn_is_training
  # Start the game
  env = gym.make('ChopperCommand-v0')
  observation = env.reset()

  # Prepare our CNN model and get first image
  sess = tf.InteractiveSession()
  float_input = prep_image(observation)
  moveprobs, input_tensor, keep_prob, train_tensor = cnn_model()
  sess.run(tf.global_variables_initializer())

  # Prepare game management variables

  for batch in range(num_of_batches):
    # Store array of moves and final score for each episode
    nn_input_arr, chosen_act_arr, rewards_arr, total_episode_reward_arr = [], [], [], []

    for episode in range(episodes_per_batch):
      # Per episode game training management variables
      ep_nn_input_arr, ep_chosen_act_arr, ep_rewards_arr = [], [], []
      episode_running = True
      episode_reward = 0

      while episode_running:
        ep_nn_input_arr.append(float_input)
        output_actions = sess.run([moveprobs], feed_dict={input_tensor: float_input, keep_prob: 1.0})
        chosen_act = choose_action(output_actions, epsilon_greedy)
        ep_chosen_act_arr.append(chosen_act)
        observation, reward, done, info = env.step(chosen_act)
        ep_rewards_arr.append(reward)
        episode_reward = episode_reward + reward

        # Prepare next image for input into our graph
        float_input = prep_image(observation)

        if slowdown_dbg:
          # Slowdown to better see what's happening
          time.sleep(0.05)
        if render_graphics:
          env.render()
        if done:
          # Store episode list to batch list
          nn_input_arr.append(ep_nn_input_arr)
          chosen_act_arr.append(ep_chosen_act_arr)
          rewards_arr.append(ep_rewards_arr)
          # Then reset the environment for next episode.
          print("  Total reward this episode: {}".format(episode_reward))
          total_episode_reward_arr.append(episode_reward)
          episode_reward = 0 # Probably not needed
          observation = env.reset()
          float_input = prep_image(observation)
          episode_running = False
    # Results from this batch
    print("Batch #{} has been completed.".format(batch+1))
    index_best_eps = total_episode_reward_arr.index(max(total_episode_reward_arr))
    frames_of_best_eps = len(chosen_act_arr[index_best_eps])
    print("Best episode was #{}.".format(index_best_eps+1))
    print("Number of frames seen this episode was {}".format(frames_of_best_eps+1))
    # Backpropagate here. Choose to propagate the best episode from batch
    # Does setting bn_is_training to true really change it??? Hope so...
    bn_is_training = True
    # Generate one_hot for our network to train off of
    one_hot_train = np.eye(6)[chosen_act_arr[index_best_eps]]
    # Feed in and train #TODO check if training properly
    # Training with 50% dropout rate in hidden fully connected layer
    for trainloop in range(index_best_eps):
      _ = sess.run([moveprobs], feed_dict={input_tensor: nn_input_arr[trainloop],train_tensor: one_hot_train[trainloop], keep_prob: 0.5})

    # Also recalculate epsilon_greedy
    epsilon_greedy = epsilon_greedy * epsilon_decay
    print("Reduce epsilon_greedy to {}".format(epsilon_greedy))

    # Set batch norm to not training and generate more tests
    bn_is_training = False

  # Now that training batches are done, do test runs
  print("Batches complete, now we test the NN!")
  # Set batchnorm training to false
  bn_is_training = False
  for testcount in range(episodes_per_batch):
    print("Test run #{}".format(testcount+1))
    observation = env.reset()
    float_input = prep_image(observation)
    testloop = True
    episode_reward = 0
    while testloop:
      env.render()
      output_actions = sess.run([moveprobs], feed_dict={input_tensor: float_input, keep_prob: 1.0})
      chosen_act = choose_action(output_actions, 0.0)
      observation, reward, done, info = env.step(chosen_act)
      episode_reward += reward
      float_input = prep_image(observation)
      if done:
        testloop = False
    print("  Score: {}".format(episode_reward))

if __name__ == "__main__":
  main()
