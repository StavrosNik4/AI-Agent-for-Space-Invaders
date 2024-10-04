import random
from itertools import count

import gymnasium as gym
import numpy as np
from skimage import transform

random.seed(42)

# Function to preprocess the game state
def preprocess_state(state):
    gray = state.mean(axis=2)
    cropped_frame = gray[8:-12, 4:-12]
    normalized_frame = cropped_frame / 255.0
    preprocessed_frame = transform.resize(normalized_frame, [84, 84])
    return preprocessed_frame

RENDER = False

if RENDER:
    env = gym.make("ALE/SpaceInvaders-v5", difficulty=1, render_mode='human')
else:
    env = gym.make("ALE/SpaceInvaders-v5", difficulty=1)
env = GameWrapper(env)

num_games = 100

max_score = 0
min_score = float('inf')
total_scores = []

for game in range(num_games):
    state, info = env.reset()
    state = preprocess_state(state)

    total_score = 0

    for t in count():
        if RENDER:
            env.render()

        action = random.randint(0, 5)
        next_state, reward, done, info, score = env.step(action)
        next_state = preprocess_state(next_state)

        state = next_state
        total_score += reward

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