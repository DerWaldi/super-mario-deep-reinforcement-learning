### model hyperparameters
state_size = [100, 128, 4] # 4 stacked frames
action_size = 7 # 7 possible actions
learning_rate = 0.00025 # alpha (aka learning rate)

### training hyperparameters
total_episodes = 50 # total episodes for training
max_steps = 5000 # max possible steps in an episode
batch_size = 64

# exploration parameters
explore_start = 1.0 # exploration probability at start
explore_stop = 0.01 # minimum exploration probability
decay_rate = 0.00001 # exponential decay rate for exploration prob

# Q learning hyperparameters
gamma = 0.9 # discounting rate

### memory
pretrain_length = batch_size # number of experiences stored in the memory when initialized
memory_size = 1000000 # number of experiences the memory can keep

### preprocessing hyperparameters
stack_size = 4

## turn this to true if you want to render the environment during training
episode_render = True