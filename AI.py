from __future__ import division, print_function
from keras.models import Sequential
from keras.layers.core import Activation, Dense, Flatten
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam
from scipy.misc import imresize
import collections
import numpy as np
import os
# import wrapped game
import wrapped_game
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def preprocess_images(images):
	
	if images.shape[0] < 4:
		# single image case
		x_t = images[0]
		x_t = imresize(x_t,(62,80))
		x_t = x_t.astype('float')
		x_t /= 255.0
		s_t = np.stack((x_t,x_t,x_t,x_t),axis=2)
	
	else:
		# 4 images
		xt_list = []
		for i in range(images.shape[0]):
			x_t = imresize(images[i],(62,80))
			x_t = x_t.astype('float')
			x_t /= 255.0
			xt_list.append(x_t)
		s_t = np.stack((xt_list[0],xt_list[1],xt_list[2],xt_list[3]),axis=2)

	s_t = np.expand_dims(s_t,axis=0)
	return s_t

def get_next_batch(experience,model,num_actions,gamma,batch_size):
	batch_indices = np.random.randint(low=0,high=len(experience),size=batch_size)
	batch = [experience[i] for i in batch_indices]
	X = np.zeros((batch_size,62,80,4))
	Y = np.zeros((batch_size,num_actions))
	for i in range(len(batch)):
		s_t, a_t, r_t, s_tp1, game_over = batch[i]
		X[i] = s_t
		Y[i] = model.predict(s_t)[0]
		Q_sa = np.max(model.predict(s_tp1)[0])
		if game_over:
			Y[i,a_t] = r_t
		else:
			Y[i,a_t] = r_t + gamma*Q_sa
	return X,Y

# build the model
model = Sequential()
model.add(Conv2D(32,kernel_size=8,strides=4,padding="same",kernel_initializer="normal",input_shape=(62,80,4)))
model.add(Activation('relu'))
model.add(Conv2D(64,kernel_size=4,strides=2,kernel_initializer="normal",padding="same"))
model.add(Activation("relu"))
model.add(Conv2D(64,kernel_size=3,strides=1,kernel_initializer="normal",padding="same"))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(512,kernel_initializer="normal"))
model.add(Activation('relu'))
model.add(Dense(3,kernel_initializer="normal"))

model.compile(optimizer=Adam(lr=1e-6),loss="mse")

# initialize parametes
DATA_DIR = "data"
NUM_ACTIONS = 3
GAMMA = 0.99
INITIAL_EPSILON = 0.1
FINAL_EPSILON = 0.0001
MEMORY_SIZE = 50000
NUM_EPOCHS_OBSERVE = 100
NUM_EPOCHS_TRAIN = 3000

BATCH_SIZE = 32
NUM_EPOCHS = NUM_EPOCHS_OBSERVE+NUM_EPOCHS_TRAIN

# instantiate game and experience replay queue
game = wrapped_game.save_me()
experience = collections.deque(maxlen=MEMORY_SIZE)
num_games, num_wins = 0,0
epsilon = INITIAL_EPSILON

# loss and num_wins
loss_list = []
num_wins_list = []

for e in range(NUM_EPOCHS):
	game.reset()
	loss = 0.0

	# get first state 
	a_0 = 1 # (0 = left, 1 = stay, 2 = right)
	x_t, r_0, game_over = game.step(a_0)
	s_t = preprocess_images(x_t)

	while not game_over:
		s_tm1 = s_t

		# next action
		if e<= NUM_EPOCHS_OBSERVE:
			a_t = np.random.randint(low=0,high=NUM_ACTIONS,size=1)[0]
		else:
			if np.random.rand()<=epsilon:
				a_t = np.random.randint(low=0,high=NUM_ACTIONS,size=1)[0]
			else:
				q = model.predict(s_t)[0]
				a_t = np.argmax(q)

		# apply action, get reward
		x_t, r_t, game_over = game.step(a_t)
		s_t = preprocess_images(x_t)
		# if reward, increment num_wins
		if r_t == 1:
			num_wins += 1
		# store experience
		experience.append((s_tm1,a_t,r_t,s_t,game_over))

		if e>NUM_EPOCHS_OBSERVE:
			# finished observing now start training
			# get next batch
			X, Y = get_next_batch(experience,model,NUM_ACTIONS,GAMMA,BATCH_SIZE)
			loss+= model.train_on_batch(X,Y)

	# reduce epsilon gradually
	if epsilon > FINAL_EPSILON:
		epsilon -= (INITIAL_EPSILON - FINAL_EPSILON)/NUM_EPOCHS

	print("Epoch : %d | Loss: %f | Win Count : %d"%(e,loss,num_wins))
	loss_list.append(loss)
	num_wins_list.append(num_wins)

	if e%100 == 0:
		model.save(os.path.join(DATA_DIR,"RL_SAVE_ME_GAME_AI_MODEL.h5"),overwrite=True)

model.save(os.path.join(DATA_DIR,"RL_SAVE_ME_GAME_AI_MODEL.h5"),overwrite=True)

#plot the graphs
plt.plot(loss_list,label='loss')
plt.title('model loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.savefig(os.path.join(DATA_DIR,"loss"))
plt.gcf().clear()


plt.plot(num_wins_list,label='Wins')
plt.title('No of wins')
plt.xlabel('epoch')
plt.ylabel('wins')
plt.savefig(os.path.join(DATA_DIR,"wins"))
