import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers
import random
from collections import deque
from TreasureMaze import TreasureMaze
from GameExperience import GameExperience

# Hyperparameters
learning_rate = 0.001
gamma = 0.95  # Discount factor
epsilon = 1.0  # Initial exploration rate
epsilon_min = 0.01  # Minimum exploration rate
epsilon_decay = 0.995  # Decay rate for epsilon
batch_size = 64
num_episodes = 500  # Adjust based on performance

# Initialize environment
env = TreasureMaze([
    [1., 0., 1., 1., 1., 1., 1., 1.],
    [1., 0., 1., 1., 1., 0., 1., 1.],
    [1., 1., 1., 1., 0., 1., 0., 1.],
    [1., 1., 1., 0., 1., 1., 1., 1.],
    [1., 1., 0., 1., 1., 1., 1., 1.],
    [1., 1., 1., 0., 1., 0., 0., 0.],
    [1., 1., 1., 0., 1., 1., 1., 1.],
    [1., 1., 1., 1., 0., 1., 1., 1.]
])

state_size = env.observe().shape[1]
action_size = 4  # Left, Up, Right, Down

# Define Deep Q-Network
def build_model():
    model = models.Sequential([
        layers.Dense(64, activation='relu', input_shape=(state_size,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(action_size, activation='linear')
    ])
    model.compile(loss='mse', optimizer=optimizers.Adam(learning_rate=learning_rate))
    return model

model = build_model()
experience = GameExperience(model, max_memory=2000, discount=gamma)

# Training loop
for episode in range(num_episodes):
    state = env.reset((0, 0))
    state = env.observe()
    total_reward = 0

    for step in range(100):  # Limit steps to prevent infinite loops
        # Choose action using epsilon-greedy strategy
        if np.random.rand() < epsilon:
            action = np.random.choice(action_size)  # Explore
        else:
            q_values = model.predict(state)
            action = np.argmax(q_values[0])  # Exploit
        
        # Perform action
        next_state, reward, status = env.act(action)
        total_reward += reward
        game_over = True if status != 'not_over' else False
        
        # Store experience in replay memory
        experience.remember([state, action, reward, next_state, game_over])
        state = next_state
        
        # Train model if enough samples are collected
        if len(experience.memory) > batch_size:
            inputs, targets = experience.get_data(batch_size)
            model.train_on_batch(inputs, targets)
        
        if game_over:
            break
    
    # Decay epsilon
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay
    
    print(f"Episode {episode+1}: Total Reward = {total_reward}")

# Save trained model
model.save("pirate_agent_model.h5")
print("Training completed and model saved!")
