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


### All parameters needed ###
### Hyper Parameters ##
epis = 1000
save_memory_len = 100000 ## finish tunning (1
gamma = 0.990 ### finish tunning (3
epsilon_decay = 0.993 ## finish tunning (2
learning_rate = 0.0003 ## finish tunning (2
min_sample_to_reply = 1000
batch_size = 30  ## finish tunning (1
update_target_freq = 500 ## finish tunning (4 

### parameter tunning ###
if '-epis' in sys.argv:
	epis = int(sys.argv[sys.argv.index('-epis')+1])
if '-memlen' in sys.argv:
	save_memory_len = int(sys.argv[sys.argv.index('-memlen')+1])
if '-gamma' in sys.argv:
	gamma = float(sys.argv[sys.argv.index('-gamma')+1])
if '-epsilon' in sys.argv:
	epsilon_decay = float(sys.argv[sys.argv.index('-epsilon')+1])
if '-alpha' in sys.argv:
	learning_rate = float(sys.argv[sys.argv.index('-alpha')+1])
if '-batch' in sys.argv:
	batch_size =  int(sys.argv[sys.argv.index('-batch')+1])
if '-upfreq' in sys.argv:
	update_target_freq = int(sys.argv[sys.argv.index('-upfreq')+1])

if len(sys.argv) == 1:
	print("Default Setting")

save_memory = deque(maxlen = save_memory_len)
print("All para: Epis %d memlen %d gamma %.3f epsilon %.3f alpha %.5f batch %d upfreq %d" %(epis, save_memory_len, gamma, epsilon_decay, learning_rate, batch_size, update_target_freq))
		
## average ##
last_n = 200
train_mode = False
if '-train' in sys.argv:
	train_mode = True
	last_n = 100

res = deque(maxlen=last_n)
end_reward = 215 ## last_n repeat episode need to have average greater than end_reward  
## End ##

### End of Parameters ###

############### Following is the agent #############

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
			      optimizer=Adam(lr=learning_rate))
		return model

	def predict(self, state):
		if np.random.rand() <= self.epsilon:
			return random.randrange(self.action_size)
		act_values = self.model.predict(state)
		return np.argmax(act_values[0])  # returns action

	def replay(self, batch_size):
		## vectorization ##
		if len(save_memory) > min_sample_to_reply:
			minibatch = np.array(random.sample(save_memory, batch_size))
			curs = np.concatenate(minibatch[:,0])
			nexs = np.concatenate(minibatch[:,3])
			curract = minibatch[:,1].astype(int)
			targ_qvalue = self.model.predict(curs)
			nextaction = np.argmax(self.model.predict(nexs), axis=1)
			ddtarg = self.target_model.predict(nexs)
			nexs_qvalue = ddtarg[np.arange(0,len(ddtarg)).astype(int), nextaction]
			done = minibatch[:,4].astype(int)
			targ_qvalue[np.arange(0,len(targ_qvalue)).astype(int), curract] = minibatch[:,2] + gamma *(1-done)*nexs_qvalue 

			hist = self.model.fit(curs, targ_qvalue, batch_size=batch_size, epochs=1, verbose=0)
			return hist.history['loss'][-1]
			#return 0.0
		else:
			return 0.0
	
	def load(self, name, eps):
		self.model.load_weights(name)
		self.epsilon = eps

	def save(self, name):
		self.model.save_weights(name)



#### Following is the experiments #####

state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = Agent(state_size, action_size)
done = False
total_step = 0
for e in range(epis):
	state = env.reset()
	state = np.reshape(state, [1, state_size])
	cout = 0
	total_reward = 0
	loss = 0
	while True:
		action = agent.predict(state)
		next_state, reward, done, _ = env.step(action)
		next_state = np.reshape(next_state, [1, state_size])
		save_memory.append((state, action, reward, next_state, done))
		state = next_state
		total_reward += reward
		if total_step % update_target_freq == 0:
			agent.target_model.set_weights(agent.model.get_weights())
		cout += 1
		total_step += 1
		loss += agent.replay(batch_size)
		
		if done:
			agent.epsilon *= epsilon_decay
			res.append(total_reward)
			print("episode: %d count: %d score: %.4f average_last%d: %.4f epis: %.4f loss: %.4f " %(e, cout, total_reward, last_n,sum(res)/len(res), agent.epsilon,loss))
			break

	if train_mode:
		if sum(res)/len(res) > end_reward:
			agent.save("lunar_ddq_final.h5")
			break

