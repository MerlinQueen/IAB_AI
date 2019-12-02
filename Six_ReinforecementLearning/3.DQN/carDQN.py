# %%
import numpy as np 
import gym
from keras.models import Sequential
from keras.layers import Dense,Activation,Flatten,Input
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

import tensorflow as tf 
tf.__version__
# %%
# 创建环境
ENV_NAME = 'CartPole-v0'
env = gym.make(ENV_NAME)
np.random.seed(0)
env.seed(0)
nb_actions = env.action_space.n


# %%
# 搭建神经网络
model  = Sequential()
model.add(Flatten(input_shape=(1,)+env.observation_space.shape))
model.add(Dense(16,activation='relu'))
model.add(Dense(nb_actions,activation='linear'))
model.summary()


# %%
policy = EpsGreedyQPolicy()
memory = SequentialMemory(limit=50000, window_length=1)
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this slows down training quite a lot. 
dqn.fit(env, nb_steps=1000, visualize=True, verbose=2)

# %%
dqn.test(env, nb_episodes=5, visualize=True)

# %%
