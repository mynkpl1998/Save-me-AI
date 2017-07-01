import numpy as np
from scipy.misc import imresize
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam
from collections import deque
import matplotlib.pyplot as plt
import random
import gym

env = gym.make('FrozenLake-v0')

# Build the model
model = Sequential()
model.add(Dense(16,kernel_initializer="normal",batch_input_shape=(None,16)))
model.add(Activation('relu'))
model.add(Dense(4))
model.add(Activation('linear'))
model.compile(loss='mse',optimizer=Adam())
print(model.summary())


NUM_EPOCH_OBSERVE = 1000
NUM_EPOCH_TRAIN = 3000
LOSS_LIST = []
REWARD_LIST = []
QUEUE_MEMORY = 4000
INITIAL_EPSILON = 0.1
FINAL_EPSILON = 0.001
GAMMA = 0.99
EPSILON = INITIAL_EPSILON
experience = deque(maxlen=QUEUE_MEMORY)

for EPOCH in range(NUM_EPOCH_TRAIN+NUM_EPOCH_OBSERVE):
	s = env.reset()
	total_reward = 0
	GAME_OVER = False
	loss = 0
	while not GAME_OVER:
		if EPOCH <= NUM_EPOCH_OBSERVE:
			action = env.action_space.sample()
		else:
			if np.random.rand() <= EPSILON:
				action = env.action_space.sample()
				print('Random Action')
			else:
				q = model.predict(np.identity(16)[s:s+1])[0]
				action = np.argmax(q)
				#print('Model Action')

		s_1, reward, GAME_OVER, _ = env.step(action)
		experience.append((s,action,reward,GAME_OVER,s_1))

		total_reward += reward

		if EPOCH > NUM_EPOCH_OBSERVE:
			batch_indices = np.random.randint(low=0,high=len(experience),size=32)
			batch = [experience[i] for i in batch_indices]
			X = np.zeros((32,16))
			Y = np.zeros((32,4))
			for i in range(len(batch)):
				s_t, a_t, r_t, game_over, s_1 = batch[i]
				X[i] = s_t
				Y[i] = model.predict(np.identity(16)[s_t:s_t+1])[0]
				Q_sa = np.max(model.predict(np.identity(16)[s_1:s_1+1])[0])
				if game_over:
					Y[i,a_t] = r_t
				else:
					Y[i,a_t] = r_t + GAMMA*Q_sa
			loss += model.train_on_batch(X,Y)

	if EPSILON > FINAL_EPSILON and EPOCH > NUM_EPOCH_OBSERVE:
		EPSILON -= (INITIAL_EPSILON - FINAL_EPSILON)/(NUM_EPOCH_OBSERVE+NUM_EPOCH_TRAIN)

	print("EPISODE %d FINISHED "%(EPOCH+1))
	print("REWARD : %d"%(total_reward))
	print('EPSILON : %f'%(EPSILON))
	REWARD_LIST.append(total_reward)
	LOSS_LIST.append(loss)

print('Done')
plt.plot(REWARD_LIST)
plt.xlabel('Episodes')
plt.ylabel('Reward')
plt.show()

