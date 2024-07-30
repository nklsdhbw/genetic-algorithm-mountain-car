import numpy as np
import gymnasium as gym
import pickle
from geneticalgorithm import NeuralNetwork

# Environment
env = gym.make('MountainCar-v0', render_mode='human')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

def load_model(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def run_model(chromosome, model_type='linear'):
    state, _ = env.reset()
    total_reward = 0
    for _ in range(200):
        env.render()
        if model_type == 'linear':
            action = np.argmax(np.dot(state, chromosome))
        else:
            action_values = chromosome.forward(state)
            action = np.argmax(action_values)
        state, reward, done, _, _ = env.step(action)
        total_reward += reward
        if done:
            break
    env.close()
    return total_reward  # Total reward (negative value) reflects the number of steps

if __name__ == "__main__":
    print("Running the best linear model...")
    best_chromosome_linear = load_model('./models/best_chromosome_linear.pkl')
    total_reward_linear = run_model(best_chromosome_linear, model_type='linear')
    print('Total reward using the best linear model:', total_reward_linear)

    print("Running the best neural network model...")
    best_chromosome_nn = load_model('./models/best_chromosome_nn.pkl')
    total_reward_nn = run_model(best_chromosome_nn, model_type='nn')
    print('Total reward using the best neural network model:', total_reward_nn)