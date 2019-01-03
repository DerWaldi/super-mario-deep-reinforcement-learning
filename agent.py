import tensorflow as tf
import numpy as np
import random
from tqdm import tqdm

from collections import deque # ordererd collection with ends

from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros # our super mario gyme
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

# hyper parameters
from config import *

from model import DQNetwork
from preprocess import preprocess_frame, stack_frames
from utils import Memory

from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros # our super mario gyme
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
        
class Agent:
    def __init__(self, level_name):  
        self.level_name = level_name  
        # setup environment
        self.env = gym_super_mario_bros.make(level_name)
        self.env = BinarySpaceToDiscreteSpaceEnv(self.env, SIMPLE_MOVEMENT)
        # one hot encoded version of our actions
        self.possible_actions = np.array(np.identity(self.env.action_space.n, dtype=int).tolist())

        # resest graph
        tf.reset_default_graph()
        
        # instantiate the DQNetwork
        self.DQNetwork = DQNetwork(state_size, action_size, learning_rate)
        
        # instantiate memory
        self.memory = Memory(max_size=memory_size)
        
        # initialize deque with zero images
        self.stacked_frames = deque([np.zeros((100, 128), dtype=np.int) for i in range(stack_size)], maxlen=4)

        for i in range(pretrain_length):    
            # If it's the first step
            if i == 0:
                state = self.env.reset()        
                state, self.stacked_frames = stack_frames(self.stacked_frames, state, True)

            # Get next state, the rewards, done by taking a random action
            choice = random.randint(1, len(self.possible_actions)) - 1
            action = self.possible_actions[choice]
            next_state, reward, done, _ = self.env.step(choice)

            # stack the frames
            next_state, self.stacked_frames = stack_frames(self.stacked_frames, next_state, False)

            # if the episode is finished (we're dead)
            if done:
                # we inished the episode
                next_state = np.zeros(state.shape)

                # add experience to memory
                self.memory.add((state, action, reward, next_state, done))

                # start a new episode
                state = self.env.reset()
                state, self.stacked_frames = stack_frames(self.stacked_frames, state, True)
            else:
                # add experience to memory
                self.memory.add((state, action, reward, next_state, done))

                # our new state is now the next_state
                state = next_state
       
        # saver will help us save our model
        self.saver = tf.train.Saver()

        # setup tensorboard writer
        self.writer = tf.summary.FileWriter("logs/")

        # losses
        tf.summary.scalar("Loss", self.DQNetwork.loss)
        
        self.write_op = tf.summary.merge_all()
    
    def predict_action(self, sess, explore_start, explore_stop, decay_rate, decay_step, state, actions):
        # first we randomize a number
        exp_exp_tradeoff = np.random.rand()

        explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)

        if explore_probability > exp_exp_tradeoff:
            # make a random action
            choice = random.randint(1, len(self.possible_actions)) - 1
            action = self.possible_actions[choice]
        else:
            # estimate the Qs values state
            Qs = sess.run(self.DQNetwork.output, feed_dict={self.DQNetwork.inputs_: state.reshape((1, *state.shape))})

            # take the biggest Q value (= best action)
            choice = np.argmax(Qs)
            action = self.possible_actions[choice]

        return action, choice, explore_probability
    
    def play_notebook(self):
        import matplotlib.pyplot as plt
        # imports to render env to gif
        from JSAnimation.IPython_display import display_animation
        from matplotlib import animation
        from IPython.display import display

        # http://mckinziebrandon.me/TensorflowNotebooks/2016/12/21/openai.html
        def display_frames_as_gif(frames):
            """
            Displays a list of frames as a gif, with controls
            """
            #plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi = 72)
            patch = plt.imshow(frames[0])
            plt.axis('off')

            def animate(i):
                patch.set_data(frames[i])

            anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
            display(display_animation(anim, default_mode='loop'))

        frames = []
        with tf.Session() as sess:
            total_test_rewards = []

            # Load the model
            self.saver.restore(sess, "models/{0}.cpkt".format(self.level_name))

            for episode in range(1):
                total_rewards = 0

                state = self.env.reset()
                state, self.stacked_frames = stack_frames(self.stacked_frames, state, True)

                print("****************************************************")
                print("EPISODE ", episode)

                while True:
                    # Reshape the state
                    state = state.reshape((1, *state_size))
                    # Get action from Q-network 
                    # Estimate the Qs values state
                    Qs = sess.run(self.DQNetwork.output, feed_dict = {self.DQNetwork.inputs_: state})

                    # Take the biggest Q value (= the best action)
                    choice = np.argmax(Qs)

                    #Perform the action and get the next_state, reward, and done information
                    next_state, reward, done, _ = self.env.step(choice)
                    frames.append(self.env.render(mode = 'rgb_array'))

                    total_rewards += reward

                    if done:
                        print ("Score", total_rewards)
                        total_test_rewards.append(total_rewards)
                        break


                    next_state, self.stacked_frames = stack_frames(self.stacked_frames, next_state, False)
                    state = next_state

            self.env.close()

        display_frames_as_gif(frames)
        
    def play(self):
        with tf.Session() as sess:
            total_test_rewards = []

            # Load the model
            self.saver.restore(sess, "models/{0}.cpkt".format(self.level_name))

            for episode in range(1):
                total_rewards = 0

                state = self.env.reset()
                state, self.stacked_frames = stack_frames(self.stacked_frames, state, True)

                print("****************************************************")
                print("EPISODE ", episode)

                while True:
                    # Reshape the state
                    state = state.reshape((1, *state_size))
                    # Get action from Q-network 
                    # Estimate the Qs values state
                    Qs = sess.run(self.DQNetwork.output, feed_dict = {self.DQNetwork.inputs_: state})

                    # Take the biggest Q value (= the best action)
                    choice = np.argmax(Qs)

                    #Perform the action and get the next_state, reward, and done information
                    next_state, reward, done, _ = self.env.step(choice)
                    self.env.render()

                    total_rewards += reward

                    if done:
                        print ("Score", total_rewards)
                        total_test_rewards.append(total_rewards)
                        break

                    next_state, self.stacked_frames = stack_frames(self.stacked_frames, next_state, False)
                    state = next_state
            self.env.close()
    
    def train(self):        
        with tf.Session() as sess:
            # initialize the variables
            sess.run(tf.global_variables_initializer())

            # initialize decay rate (that will be used to reduce epsilon)
            decay_step = 0

            for episode in range(total_episodes):
                # set step to 0
                step = 0

                # initialize rewards of episode
                episode_rewards = []

                # make a new episode and opserve the first state
                state = self.env.reset()

                # remember that stack frame function
                state, self.stacked_frames = stack_frames(self.stacked_frames, state, True)

                print("Episode:", episode)

                while step < max_steps:
                    step += 1
                    #print("step:", step)

                    # increase decay_step
                    decay_step += 1

                    # predict an action
                    action, choice, explore_probability = self.predict_action(sess,
                                                         explore_start, 
                                                         explore_stop, 
                                                         decay_rate, 
                                                         decay_step, 
                                                         state, 
                                                         self.possible_actions)

                    # perform the action and get the next_state, reward, and done information
                    next_state, reward, done, _ = self.env.step(choice)

                    if episode_render:
                        self.env.render()

                    # add the reward to total reward
                    episode_rewards.append(reward)

                    # the game is finished
                    if done:
                        print("done")
                        # the episode ends so no next state
                        next_state = np.zeros((110, 84), dtype=np.int)

                        next_state, self.stacked_frames = stack_frames(self.stacked_frames, next_state, False)

                        # set step = max_steps to end episode
                        step = max_steps

                        # get total reward of the episode
                        total_reward = np.sum(episode_rewards)

                        print("Episode:", episode, 
                              "Total reward:", total_reward, 
                              "Explore P:", explore_probability, 
                              "Training Loss:", loss)

                        #rewards_list.append((episode, total_reward))

                        # store transition <s_i, a, r_{i+1}, s_{i+1}> in memory
                        self.memory.add((state, action, reward, next_state, done))
                    else:
                        # stack frame of the next state
                        next_state, self.stacked_frames = stack_frames(self.stacked_frames, next_state, False)

                        # store transition <s_i, a, r_{i+1}, s_{i+1}> in memory
                        self.memory.add((state, action, reward, next_state, done))

                        # s_{i} := s_{i+1}
                        state = next_state

                    ### Learning part
                    # obtain random mini-batch from memory
                    batch = self.memory.sample(batch_size)
                    states_mb = np.array([each[0] for each in batch], ndmin=3)
                    actions_mb = np.array([each[1] for each in batch])
                    rewards_mb = np.array([each[2] for each in batch])
                    next_states_mb = np.array([each[3] for each in batch], ndmin=3)
                    dones_mb = np.array([each[4] for each in batch])

                    target_Qs_batch = []

                    # get Q values for next_state
                    Qs_next_state = sess.run(self.DQNetwork.output, feed_dict={self.DQNetwork.inputs_: next_states_mb})

                    # set Q_target = r if episode ends with s+1
                    for i in range(len(batch)):
                        terminal = dones_mb[i]

                    # if we are in a terminal state, only equals reward
                    if terminal:
                        target_Qs_batch.append(rewards_mb[i])
                    else:
                        target = rewards_mb[i] + gamma * np.max(Qs_next_state[i])
                        target_Qs_batch.append(target)

                    targets_mb = np.array([each for each in target_Qs_batch])

                    loss, _ = sess.run([self.DQNetwork.loss, self.DQNetwork.optimizer],
                                      feed_dict={self.DQNetwork.inputs_: states_mb, 
                                                 self.DQNetwork.target_Q: targets_mb, 
                                                 self.DQNetwork.actions_: actions_mb})

                    # write tf summaries
                    summary = sess.run(self.write_op, feed_dict={self.DQNetwork.inputs_: states_mb, 
                                                 self.DQNetwork.target_Q: targets_mb, 
                                                 self.DQNetwork.actions_: actions_mb})
                    self.writer.add_summary(summary, episode)
                    self.writer.flush()

                # save model every 5 episodes
                if episode % 5 == 0:
                    self.saver.save(sess, "models/{0}.cpkt".format(self.level_name))
                    print("Model Saved")