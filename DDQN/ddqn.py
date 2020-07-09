#Double Deep Q network in pytorch

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("tkagg")
import matplotlib.pyplot as plt
from collections import deque
import time
import random
import copy

#Hyperparameters
LEARNING_RATE = 1e-3
EPSILON = 0.05
GAMMA = 0.99
MAX_BUFFER_LENGTH = 5000
POL_CONST = 0.995
MAX_TRAJ_LENGTH = 10000
BATCH_SIZE = 8

class DQN:
    def __init__(self, env, hidden_size):
        self.env = env
        self.q = nn.Sequential(nn.Linear(self.env.observation_space.shape[0], hidden_size),
                                nn.ReLU(),
                                nn.Linear(hidden_size, self.env.action_space.n))
        self.target_q = copy.deepcopy(self.q)

    def select_action(self, observation, eps = EPSILON):
        """
        Epsilon greedy action selection
        """
        if np.random.uniform(0, 1) < eps:
            action = torch.tensor(self.env.action_space.sample(), dtype = torch.float32)
        else:
            q_values = self.q(observation)
            action = torch.argmax(q_values)

        return action


    def train(self, episodes, gamma = GAMMA, eps = EPSILON, max_buffer_length = MAX_BUFFER_LENGTH, max_traj_length = MAX_TRAJ_LENGTH, batch_size = BATCH_SIZE, polyak_const = POL_CONST, plot_rewards = True):
        """
        Trains the Double Deep Q network
        """
        epoch_rewards = []
        optimizer = torch.optim.Adam(self.q.parameters(), lr = 1e-3)
        for episode in range(episodes):
            replay_buffer = deque(maxlen = max_buffer_length)
            observation = self.env.reset()
            episode_rewards = []
            done = False

            for i in range(max_traj_length):
                #Play the game and collect experience
                observation = torch.tensor(observation, dtype = torch.float32)
                action = self.select_action(observation, eps = eps)
                next_observation, reward, done, _ = self.env.step(int(action.item()))
                episode_rewards.append(reward)
                next_observation = torch.tensor(next_observation, dtype = torch.float32)
                #action = torch.tensor(action, dtype = torch.float32)
                reward = torch.tensor(reward, dtype = torch.float32)
                done = torch.tensor(done, dtype = torch.float32)
                transition = (observation, action, reward, done, next_observation)
                replay_buffer.append(transition)
                observation = next_observation
                if done:
                    break

                loss = torch.tensor(0.0, dtype = torch.float32)
                if len(replay_buffer) >= BATCH_SIZE:
                    #Randomly sample from the replay buffer
                    list = random.sample(replay_buffer, BATCH_SIZE)
                    observations = torch.stack([row[0] for row in list])
                    actions = torch.stack([row[1].long() for row in list])
                    rewards = torch.stack([row[2] for row in list])
                    dones = torch.stack([row[3] for row in list])
                    next_observations = torch.stack([row[4] for row in list])


                    with torch.no_grad():
                        target_qvals = rewards + gamma * self.target_q(next_observations).gather(1, torch.argmax(self.q(observations), dim = 1).view(-1, 1)).squeeze(-1)
                    optimizer.zero_grad()
                    qvals = self.q(observations).gather(1, actions.view(actions.size(0), 1))
                    loss = nn.MSELoss()(target_qvals, qvals.view(-1,1))
                    loss.backward()
                    optimizer.step()

                    #Update using polyak averaging
                    with torch.no_grad():
                        for p_target, p in zip(self.target_q.parameters(), self.q.parameters()):
                            p_target.data.mul_(polyak_const)
                            p_target.data.add_((1 - polyak_const) * p.data)



            total_rewards = np.sum(episode_rewards)
            epoch_rewards.append(total_rewards)
            print(f"Episode {episode} Total Reward {total_rewards}")
            if len(replay_buffer) >= BATCH_SIZE:
                print(f"DQN Loss {loss.item()}")
            else:
                print("----Collecting experience----")
            self.env.close()
        if plot_rewards:
            plt.plot(epoch_rewards)
            plt.show()

    def save(self, save_dir):
        """
        Saves the trained q and target q networks in the specified directory
        """
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        torch.save(self.q.state_dict(), f"{save_dir}/q.pt")
        torch.save(self.q.state_dict(), f"{save_dir}/target_q.pt")
        print(f"Models saved at {save_dir}")

    def load(self, load_dir):
        """
        Loads the trained q and target q network from the specified directory
        """
        self.q.load_state_dict(torch.load(f"{load_dir}/q.pt"))
        self.target_q.load_state_dict(torch.load(f"{load_dir}/target_q.pt"))
        print(f"Models loaded from {load_dir}")

    def eval(self, episodes, render):
        """
        Evaluate the learned Q function approximation for a specified number of episodes
        """
        print(f"Evaluating for {episodes} episodes")
        start = time.time()
        total_rewards = []

        for i in range(episodes):
            observation = self.env.reset()
            episode_rewards = []
            done = False

            while not done:
                if render:
                    self.env.render()

                observation = torch.tensor(observation, dtype = torch.float32)
                action = self.select_action(observation)
                next_observation, reward, done, _ = self.env.step(int(action.item()))
                episode_rewards.append(reward)
                observation = next_observation

            total_rewards.append(np.sum(episode_rewards))
            print(f"Episode - {i} Total Reward - {total_rewards[-1]:.2f}")

        self.env.close()
        print(f"Evaluation Completed in {time.time() - start} seconds")
        print(f"Average episodic reward = {np.mean(total_rewards)}")


if __name__ == "__main__":
    import gym
    env = gym.make("CartPole-v0")
    #from pybullet_envs import bullet

    #env = bullet.racecarGymEnv.RacecarGymEnv(renders=False, isDiscrete=True)
    dqn_agent = DQN(env, 500)
    dqn_agent.train(3000, plot_rewards = True)
    #dqn_agent.save("./DDQN/models")
    #dqn_agent.load("./DDQN/models")
    dqn_agent.eval(10, render = True)
