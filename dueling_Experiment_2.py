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

def preprocess_state(state):
    gray = state.mean(axis=2)
    cropped_frame = gray[8:-12, 4:-12]
    normalized_frame = cropped_frame / 255.0
    preprocessed_frame = transform.resize(normalized_frame, [84, 84])
    return preprocessed_frame


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

        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values


class DuelingDQN:
    def __init__(self, state_size, action_size, device, epsilon_start=1.0, epsilon_min=0.1, epsilon_decay=0.99):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.replay_buffer = deque(maxlen=10000)
        self.gamma = 0.95
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.update_rate = 1000
        self.learning_rate = 0.00025

        self.main_network = DuelingNetwork(state_size[0], action_size).to(device)
        self.target_network = DuelingNetwork(state_size[0], action_size).to(device)
        self.target_network.load_state_dict(self.main_network.state_dict())
        self.optimizer = optim.Adam(self.main_network.parameters(), lr=self.learning_rate)

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def epsilon_greedy(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return np.random.randint(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.main_network(state)
        return torch.argmax(q_values[0]).item()

    def train(self, batch_size):
        minibatch = random.sample(self.replay_buffer, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            next_state = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            reward = torch.FloatTensor([reward]).to(self.device)
            done = torch.FloatTensor([done]).to(self.device)
            if not done:
                with torch.no_grad():
                    target_q = reward + self.gamma * torch.max(self.target_network(next_state)[0])
            else:
                target_q = reward
            current_q = self.main_network(state)
            current_q[0][action] = target_q
            self.optimizer.zero_grad()
            loss = nn.MSELoss()(self.main_network(state), current_q)
            loss.backward()
            self.optimizer.step()

    def update_target_network(self):
        self.target_network.load_state_dict(self.main_network.state_dict())

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self, path):
        torch.save(self.main_network.state_dict(), path)

    def load_model(self, path):
        self.main_network.load_state_dict(torch.load(path))


def plot_results(show_result=False, num_eps='10'):
    plt.figure(1)

    # Save the plot as an image
    if show_result:
        plt.savefig(f'dueling_training_results_{num_eps}.png')

    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    scores_t = torch.tensor(episode_scores, dtype=torch.float)
    rewards_t = torch.tensor(episode_rewards, dtype=torch.float)
    if show_result:
        plt.title(f'Dueling Network - Training Results {num_eps} Episodes')
    else:
        plt.clf()
        plt.title('Dueling Network - Training...')
    plt.xlabel('Episode')
    plt.ylabel('Value')
    plt.plot(durations_t.numpy(), label='Duration')
    plt.plot(scores_t.numpy(), label='Score')
    plt.plot(rewards_t.numpy(), label='Reward')

    plt.legend()

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)

        else:
            display.display(plt.gcf())



# Main program

TRAINING = False
RENDER = True
env = gym.make("ALE/SpaceInvaders-v5", difficulty=1, render_mode='human')
# env = gym.make("ALE/SpaceInvaders-v5", difficulty=1)

num_episodes = 2000
num_episodes_model = 1000  # which model to play?
num_games = 100  # how many times to play ?
batch_size = 8
num_screens = 4

state_size = (num_screens, 84, 84)
action_size = env.action_space.n

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dqn = DuelingDQN(state_size, action_size, device)
frame_stacker = FrameStacker(num_screens)

done = False
time_step = 0

if TRAINING:

    episode_durations = []
    episode_rewards = []
    episode_scores = []

    for i in range(num_episodes):
        Return = 0
        EpisodeScore = 0
        state, info = env.reset()
        state = preprocess_state(state)
        state = frame_stacker.reset(state)

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
                plot_results(num_eps=str(num_episodes))
                break

            if len(dqn.replay_buffer) > batch_size:
                dqn.train(batch_size)

        dqn.decay_epsilon()  # Decay epsilon after each episode

        if (i + 1) % 100 == 0:
            dqn.save_model(f'dueling_dqn_model_{i + 1}.pth')

    # Save the model after training
    dqn.save_model(f'dueling_dqn_model_{num_episodes}.pth')

    print('Mean Score: ', np.mean(episode_scores))
    print('Max Score: ', np.max(episode_scores))

    env.close()

    plot_results(num_eps=str(num_episodes), show_result=True)

else:

    # Load the model for playing the game
    dqn.load_model(f'./space_invaders_experiments_1_and_2/dueling_dqn_model_{num_episodes_model}.pth')
    dqn.epsilon = 0.0  # No exploration when playing

    max_score = 0
    min_score = float('inf')
    total_scores = []

    num_games = 100

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

    print(f'Min Score over {num_games} games: {min_score}')
    print(f'Max Score over {num_games} games: {max_score}')
    print(f'Median Score over {num_games} games: {median_score}')
    print(f'Average Score over {num_games} games: {average_score}')
