import math
import random
from itertools import count

import gymnasium as gym
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

from skimage import transform

import matplotlib
import matplotlib.pyplot as plt

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display


# SumTree class used to store priorities and experiences for Prioritized Experience Replay
class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity   # Capacity of the tree (maximum number of experiences)
        self.tree = np.zeros(2 * capacity - 1)  # Initialize the tree with zeros
        self.data = np.zeros(capacity, dtype=object) # Initialize data array to store experiences
        self.size = 0
        self.data_pointer = 0

    def add(self, priority, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(tree_idx, priority)

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0
        if self.size < self.capacity:
            self.size += 1

    def update(self, tree_idx, priority):
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate(tree_idx, change)

    def _propagate(self, tree_idx, change):
        parent = (tree_idx - 1) // 2
        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def get_leaf(self, v):
        parent_idx = 0

        while True:
            left_child_idx = 2 * parent_idx + 1
            right_child_idx = left_child_idx + 1

            if left_child_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            else:
                if v <= self.tree[left_child_idx]:
                    parent_idx = left_child_idx
                else:
                    v -= self.tree[left_child_idx]
                    parent_idx = right_child_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    def total_priority(self):
        return self.tree[0]

    def __len__(self):
        return self.size

# Prioritized Experience Replay buffer using SumTree
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha):
        self.alpha = alpha  # Priority exponent
        self.tree = SumTree(capacity)

    def add(self, error, sample):
        priority = (abs(error) + 1e-5) ** self.alpha    # Compute priority based on TD error
        self.tree.add(priority, sample)

    def sample(self, batch_size, beta):
        batch = []
        idxs = []
        segment = self.tree.total_priority() / batch_size
        priorities = []

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)

            v = random.uniform(a, b)
            idx, priority, data = self.tree.get_leaf(v)

            priorities.append(priority)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total_priority()
        is_weights = np.power(self.tree.size * sampling_probabilities, -beta)
        is_weights /= is_weights.max()

        return idxs, batch, is_weights

    def update(self, idx, error):
        priority = (abs(error) + 1e-5) ** self.alpha
        self.tree.update(idx, priority)

    def __len__(self):
        return len(self.tree)

# Wrapper class to modify rewards and handle game-specific details
class GameWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.prev_lives = self.env.unwrapped.ale.lives()

    def step(self, action):
        observation, reward, done, info, _ = self.env.step(action)
        score = reward

        current_lives = self.env.unwrapped.ale.lives()
        if current_lives == 0:
            reward -= 10
        elif current_lives < self.prev_lives:
            reward -= 5

        self.prev_lives = current_lives

        if reward < 0:
            # Normalize negative rewards to the range [-1, 0)
            reward = (reward / -10) * -1
        elif reward == 0:
            # Keep zero rewards as 0
            reward = 0.0
        elif reward > 30:
            reward = 1.0
        else:
            # Normalize positive rewards to the range (0, 1]
            reward = reward / 30

        return observation, reward, done, info, score


# Function to preprocess the game state
def preprocess_state(state):
    gray = state.mean(axis=2)
    cropped_frame = gray[8:-12, 4:-12]
    normalized_frame = cropped_frame / 255.0
    preprocessed_frame = transform.resize(normalized_frame, [84, 84])
    return preprocessed_frame


# FrameStacker class to stack multiple frames
class FrameStacker:
    def __init__(self, num_frames):
        self.num_frames = num_frames
        self.frames = deque(maxlen=num_frames)

    def reset(self, initial_frame):
        for _ in range(self.num_frames):
            self.frames.append(initial_frame)
        return np.stack(self.frames, axis=0)

    def append(self, frame):
        self.frames.append(frame)
        return np.stack(self.frames, axis=0)

# Neural network model with Dueling architecture
class DuelingNetwork(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DuelingNetwork, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU()
        )

        # Calculate the number of features after convolution
        self._n_features = self._get_conv_output((in_channels, 84, 84))

        self.fc_value = nn.Sequential(
            nn.Linear(self._n_features, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        self.fc_advantage = nn.Sequential(
            nn.Linear(self._n_features, 256),
            nn.ReLU(),
            nn.Linear(256, out_channels)
        )
    def _get_conv_output(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, state):
        features = self.conv(state)
        features = features.view(features.size(0), -1)

        value = self.fc_value(features)
        advantage = self.fc_advantage(features)

        # Combine value and advantage streams
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values

# Dueling DQN Agent
class DuelingDQN:
    def __init__(self, state_size, action_size, device, epsilon_start=1.0, epsilon_min=0.1, epsilon_decay=0.99999,
                 buffer_size=100000, alpha=0.6, beta_start=0.4, beta_frames=1000000):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.gamma = 0.99  # Discount factor
        self.epsilon = epsilon_start  # Initial exploration rate
        self.epsilon_min = epsilon_min  # Minimum exploration rate
        self.epsilon_decay = epsilon_decay  # Decay rate for exploration
        self.update_rate = 10000  # Target network update frequency
        self.learning_rate = 0.0001  # Learning rate
        self.buffer = PrioritizedReplayBuffer(buffer_size, alpha)  # Experience replay buffer
        self.beta_start = beta_start  # Initial value of beta for importance sampling
        self.beta_frames = beta_frames  # Number of frames to linearly anneal beta to 1
        self.frame_idx = 1  # Frame index

        self.main_network = DuelingNetwork(state_size[0], action_size).to(device)
        self.target_network = DuelingNetwork(state_size[0], action_size).to(device)
        self.target_network.load_state_dict(self.main_network.state_dict())
        self.optimizer = optim.Adam(self.main_network.parameters(), lr=self.learning_rate)

    def store_transition(self, state, action, reward, next_state, done):
        error = self.compute_td_error(state, action, reward, next_state, done) # Calculate TD error
        self.buffer.add(error, (state, action, reward, next_state, done)) # Store transition with priority

    def epsilon_greedy(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return np.random.randint(self.action_size) # Explore: choose random action
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.main_network(state) # Exploit: choose action with highest Q-value
        return torch.argmax(q_values[0]).item()

    def train(self, batch_size):
        beta = min(1.0, self.beta_start + self.frame_idx * (1.0 - self.beta_start) / self.beta_frames) # Anneal beta
        idxs, minibatch, is_weights = self.buffer.sample(batch_size, beta) # Sample minibatch

        states, actions, rewards, next_states, dones = zip(*minibatch)
        states = torch.FloatTensor(states).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        is_weights = torch.FloatTensor(is_weights).to(self.device)

        current_qs = self.main_network(states) # Current Q-values
        next_qs = self.target_network(next_states) # Next Q-values

        target_qs = rewards + self.gamma * torch.max(next_qs, dim=1)[0] * (1 - dones) # Target Q-values
        target_qs = target_qs.unsqueeze(1)

        current_qs = current_qs.gather(1, torch.tensor(actions).unsqueeze(1).to(self.device)) # Gather Q-values for taken actions

        errors = torch.abs(current_qs - target_qs).detach().cpu().numpy() # Calculate TD errors

        for idx, error in zip(idxs, errors):
            self.buffer.update(idx, error)  # Update priorities in the replay buffer

        loss = (is_weights * nn.MSELoss(reduction='none')(current_qs, target_qs)).mean() # Compute weighted loss

        self.optimizer.zero_grad()
        loss.backward() # Backpropagation
        self.optimizer.step() # Gradient descent

        return loss.item() # Return loss value

    def update_target_network(self):
        self.target_network.load_state_dict(self.main_network.state_dict())

    def decay_epsilon(self, i):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay # Decay exploration rate

    def save_model(self, path):
        torch.save(self.main_network.state_dict(), path) # Save model weights

    def load_model(self, path):
        self.main_network.load_state_dict(torch.load(path)) # Load model weights

    def compute_td_error(self, state, action, reward, next_state, done):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        next_state = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        reward = torch.FloatTensor([reward]).to(self.device)
        done = torch.FloatTensor([done]).to(self.device)

        with torch.no_grad():
            current_q = self.main_network(state)[0][action] # Current Q-value
            if done:
                td_error = reward - current_q # TD error for terminal state
            else:
                next_q = torch.max(self.target_network(next_state)[0])  # Max Q-value for next state
                td_error = reward + self.gamma * next_q - current_q # TD error
        return td_error.item() # Return TD error


# Plotting function for training results
def plot_results(show_result=False, num_eps='10'):
    plt.figure(1)

    # Save the plot as an image
    if show_result:
        plt.savefig(f'training_results_{num_eps}.png')

    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    scores_t = torch.tensor(episode_scores, dtype=torch.float)
    # rewards_t = torch.tensor(episode_rewards, dtype=torch.float)
    if show_result:
        plt.title(f'Dueling Network - Training Results {num_eps} Episodes')
    else:
        plt.clf()
        plt.title('Dueling Network - Training...')
    plt.xlabel('Episode')
    plt.ylabel('Value')

    # Take 10 episode averages and plot them
    if len(durations_t) >= 10:
        durations_means = durations_t.unfold(0, 10, 1).mean(1).view(-1)
        durations_means = torch.cat((torch.zeros(9), durations_means))
        plt.plot(durations_means.numpy(), label='Duration')

        scores_means = scores_t.unfold(0, 10, 1).mean(1).view(-1)
        scores_means = torch.cat((torch.zeros(9), scores_means))
        plt.plot(scores_means.numpy(), label='Score')

        plt.legend()

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)

        else:
            display.display(plt.gcf())

# Plotting function for average loss per episode
def plot_loss(show_result=False):
    plt.figure(2)

    plt.clf()
    plt.title('Training Loss')
    plt.xlabel('Episode')
    plt.ylabel('Average Loss')
    plt.plot(avg_losses, label='Average Loss per Episode')

    plt.legend()

    plt.pause(0.001)
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

# Plotting function for total rewards per episode
def plot_total_rewards(show_result=False):
    plt.figure(3)

    plt.clf()
    plt.title('Total Rewards per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.plot(total_rewards, label='Total Reward per Episode')

    plt.legend()

    plt.pause(0.001)
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

# Main program

# Training / Testing
TRAINING = False

# Enviroment (Game) Settings
RENDER = True
DIFFICULTY = 1

if RENDER:
    env = gym.make("ALE/SpaceInvaders-v5", difficulty=DIFFICULTY, render_mode='human')
else:
    env = gym.make("ALE/SpaceInvaders-v5", difficulty=DIFFICULTY)

env = GameWrapper(env) # using GameWrapper to use custom reward policy

# HYPER-PARAMETERS
num_episodes = 1000
num_episodes_model = 2000  # which model to play? (only for testing)
num_games = 100  # how many times to play ? 
batch_size = 8   # batch size for the mini-sample for Deep Q-Learning
num_screens = 4  # how many of the last frames should the Network take as input

state_size = (num_screens, 84, 84)
action_size = env.action_space.n

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # use CUDA if available

dqn = DuelingDQN(state_size, action_size, device)
frame_stacker = FrameStacker(num_screens)

done = False  # initialazation for the done flag of the game
time_step = 0 # counter

if TRAINING:

    # lists used for storing data of the training process
    episode_durations = []
    episode_rewards = []
    episode_scores = []
    avg_losses = []
    total_rewards = []

    for i in range(num_episodes):
        Return = 0
        EpisodeScore = 0
        state, info = env.reset()
        state = preprocess_state(state)
        state = frame_stacker.reset(state)
        train_counter = 0
        episode_losses = []

        for t in count():
            if RENDER:
                env.render()

            time_step += 1

            if time_step % dqn.update_rate == 0:
                dqn.update_target_network()
                time_step = 1

            action = dqn.epsilon_greedy(state)
            next_state, reward, done, info, score = env.step(action)
            next_state = preprocess_state(next_state)
            next_state = frame_stacker.append(next_state)

            dqn.store_transition(state, action, reward, next_state, done)
            state = next_state
            Return += reward
            EpisodeScore += score

            if done:
                print('Episode: ', i, ', Return: ', Return, ', Score: ', EpisodeScore, 'Duration: ', t)
                episode_scores.append(EpisodeScore)
                episode_rewards.append(Return)
                episode_durations.append(t)
                total_rewards.append(Return)  # Track total rewards per episode
                plot_results(num_eps=str(num_episodes))
                plot_total_rewards()  # Plot total rewards
                break

            train_counter += 1

            if len(dqn.buffer) > batch_size and train_counter == 4:
                loss = dqn.train(batch_size)
                episode_losses.append(loss)
                plot_loss()
                train_counter = 0

            dqn.decay_epsilon(i)  # Decay epsilon after each step

        avg_loss = np.mean(episode_losses) if episode_losses else 0
        avg_losses.append(avg_loss)
        plot_loss()
        
        if (i + 1) % 100 == 0: # saving a model once per 100 episodes
            dqn.save_model(f'dueling_dqn_model_{i + 1}.pth')

    # Save the model after training
    dqn.save_model(f'dueling_dqn_model_{num_episodes}.pth')

    env.close()
    
    # Printing results
    print('Mean Score: ', np.mean(episode_scores))
    print('Max Score: ', np.max(episode_scores))

    # Plotting results    
    plot_results(num_eps=str(num_episodes), show_result=True)
    plot_loss(show_result=True)
    plot_total_rewards(show_result=True)  # Plot final total rewards

else: # Testing

    # Load the model for playing the game
    dqn.load_model(f'./space_invaders_experiment_3/dueling_dqn_model_{num_episodes_model}.pth')
    dqn.epsilon = 0.0  # No exploration when playing

    max_score = 0
    min_score = float('inf')
    total_scores = []

    for game in range(num_games):
        state, info = env.reset()
        state = preprocess_state(state)
        state = frame_stacker.reset(state)
        total_score = 0

        for t in count():
            if RENDER:
                env.render()

            action = dqn.epsilon_greedy(state)
            next_state, reward, done, info, score = env.step(action)
            next_state = preprocess_state(next_state)
            next_state = frame_stacker.append(next_state)

            state = next_state
            total_score += score

            if done:
                print(f'Game {game + 1} Over! Total Score: {total_score}')
                total_scores.append(total_score)
                if total_score > max_score:
                    max_score = total_score
                if total_score < min_score:
                    min_score = total_score
                break

        env.close()

    average_score = sum(total_scores) / len(total_scores)
    median_score = np.median(total_scores)

    # Printing results of Testing
    print(f'Min Score over {num_games} games: {min_score}')
    print(f'Max Score over {num_games} games: {max_score}')
    print(f'Median Score over {num_games} games: {median_score}')
    print(f'Average Score over {num_games} games: {average_score}')
