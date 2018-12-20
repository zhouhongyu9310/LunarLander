#!/bin/env python
import random
import gym
import numpy as np
from collections import deque
import sys
import os
### important !! for consistency, random seed all use 0 ###
env = gym.make('LunarLander-v2')
os.environ['PYTHONHASHSEED'] = '0'
os.environ["KERAS_BACKEND"] = "theano"  ## tersowflow cannot used for reproduciable
random.seed(0)
np.random.seed(0)
env.seed(0)
### finish consistency setting ###
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K


epis = 10000  ## test case
res = deque(maxlen=100) # save last 100

############### Following is the agent for test #############
class Agent:
	def __init__(self, state_size, action_size):
		self.state_size = state_size
		self.action_size = action_size
		self.model = self._build_model()
		self.target_model = self._build_model()
		self.epsilon = 1.0
		
		# set target model
		self.target_model.set_weights(self.model.get_weights())
	
	def _huber_loss(self, target, prediction):
		error = prediction - target
		return K.mean(K.sqrt(1+K.square(error))-1, axis=-1)

	def _build_model(self):
		model = Sequential()
		model.add(Dense(50, input_dim=self.state_size,activation='relu'))
		model.add(Dense(50,activation='relu'))
		model.add(Dense(self.action_size, activation='linear'))
		model.compile(loss=self._huber_loss,
			      optimizer=Adam(lr=0))
		return model

	def predict(self, state):
		act_values = self.model.predict(state)
		return np.argmax(act_values[0])  # returns action
	
	def load(self, name):
		self.model.load_weights(name)

	def save(self, name):
		self.model.save_weights(name)



#### Following is the experiments #####
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = Agent(state_size, action_size)
agent.load('lunar_ddq_final.h5')

done = False
total_step = 0
for e in range(epis):
	state = env.reset()
	state = np.reshape(state, [1, state_size])
	cout = 0
	total_reward = 0
	while True:
		action = agent.predict(state)
		next_state, reward, done, _ = env.step(action)
		next_state = np.reshape(next_state, [1, state_size])
		state = next_state
		total_reward += reward
		cout += 1
		
		if done:
			res.append(total_reward)
			print("episode: %d count: %d score: %.4f Ave:%.4f " %(e, cout, total_reward, sum(res)/len(res)))
			break
