# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import gym
import matplotlib.pyplot as plt
import numpy as np

env = gym.make('FrozenLake-v1')


def maxAction(Q, state, actions=[0, 1, 2, 3]):
    values = np.array([Q[state,a] for a in actions])
    action = np.argmax(values)

    return action
def get_state(observation):
    t = int(np.floor(observation/4))
    r = observation%4
    return (r, t)
#
env.reward_range = (-1,10)
env.spec.new_step_api = True

Q = {}
n_games = 100000
alpha = 0.1
gamma = 0.99
eps = 0.9
states=[]
for i in range(16):
#    for j in range(4):
        states.append(i)

for state in states:
    for action in range(4):
        Q[state, action] = 0

total_rewards = np.zeros(n_games)
eps_rewards = 0
env.reset()
for i in range(n_games):
    if i % 100 == 0:
        print('episode ', i, 'score ', eps_rewards, 'eps', eps)
    observation = env.reset()
    state = (observation)
    done = False
    action = env.action_space.sample() if np.random.random() < eps else maxAction(Q, state)
    
    eps_rewards = 0
    while not done:
        observation_, reward, done, info = env.step(action)
        
        state_ = (observation_)
#        print (state_)
        action_ = maxAction(Q, state_)
        if done and reward == 0:
            eps_rewards += -10
            Q[state, action] = Q[state,action] + alpha*(-1 + gamma*Q[state_,action_] - Q[state,action])
        if observation_ == 15:
#            print (action)
            eps_rewards += 20
            Q[state, action] = Q[state,action] + alpha*(20 + gamma*Q[state_,action_] - Q[state,action])
        eps_rewards += -2
        Q[state, action] = Q[state,action] + alpha*(-2 + gamma*Q[state_,action_] - Q[state,action])
        state = state_
        action = action_
        
    total_rewards[i] = eps_rewards
    eps = eps - 1 / n_games if eps > 0.1 else 0.1
mean_rewards = np.zeros(n_games)
for t in range(n_games):
    mean_rewards[t] = np.mean(total_rewards[max(0, t-50):(t+1)])
plt.plot(mean_rewards)
plt.show()


for i in range(1):
#    if i % 100 == 0:
#        print('episode ', i, 'score ', eps_rewards, 'eps', eps)
    observation = env.reset()
    state = (observation)
    done = False
    action = env.action_space.sample() if np.random.random() < eps else maxAction(Q, state)
    
    eps_rewards = 0
    while not done:
        observation_, reward, done, info = env.step(action)
        
        state_ = (observation_)
#        print (state_)
        action_ = maxAction(Q, state_)
        if done and reward == 0:
            eps_rewards += -10
#            Q[state, action] = Q[state,action] + alpha*(-1 + gamma*Q[state_,action_] - Q[state,action])
        if observation_ == 20:
            print (action)
            eps_rewards += 20
#            Q[state, action] = Q[state,action] + alpha*(20 + gamma*Q[state_,action_] - Q[state,action])
        eps_rewards += -2
#        Q[state, action] = Q[state,action] + alpha*(-2 + gamma*Q[state_,action_] - Q[state,action])
        state = state_
        action = action_
        
    total_rewards[i] = eps_rewards
    eps = eps - 2 / n_games if eps > 0.1 else 0.1
total_rewards = np.zeros(n_games)
mean_rewards = np.zeros(n_games)
for t in range(n_games):
    mean_rewards[t] = np.mean(total_rewards[max(0, t-50):(t+1)])
plt.plot(mean_rewards)
plt.show()
