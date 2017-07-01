# Author : Mayank Kumar Pal
# Email : mayank15147@iiitd.ac.in
import pygame
import random
import collections
import numpy as np
import os
import json

class save_me(object):

	def __init__ (self):

		pygame.init()
		pygame.key.set_repeat(100, 100)
		#set constans
		self.COlOR_BLACK = (0,0,0)
		self.COlOR_BLUE = (0,0,255)
		self.COLOR_CHOC = (139,69,19)
		self.COLOR_RED = (255,0,0)
		self.COLOR_GRAY = (128,128,128)
		self.COLOR_WHITE = (255,255,255)
		self.GAME_WIDTH = 310
		self.GAME_HEIGHT = 400
		self.PADDLE_WIDTH = 100
		self.PADDLE_VELOCITY = 102
		self.PADDLE_FLOOR = 380
		self.SQUARE_SIDE = 99
		self.SQUARE_VELOCITY = 18
		self.SQUARE_CIELING = 10
		self.PADDLE_HEIGHT = 20
		self.FONT_SIZE = 10
		self.MAX_NUM_TRIES = 1000
		self.CUSTOM_EVENT = pygame.USEREVENT + 1
		self.font = pygame.font.SysFont("Comic Sans MS",self.FONT_SIZE)
		
		self.frames = collections.deque(maxlen=4)
		self.num_tries = 0
		self.game_score = 0
		self.paddle_x = 104
		self.generate_cube_locations()
		self.cube1_y = self.SQUARE_CIELING
		self.cube2_y = self.SQUARE_CIELING
		self.GAME_OVER = False
		if os.path.isfile('data/high_score.json'):
			with open('data/high_score.json') as file:
				tmp = json.load(file)
				self.HIGH_SCORE = tmp['high_score']
		else:
			self.HIGH_SCORE = -999999


		self.screen = pygame.display.set_mode((self.GAME_WIDTH,self.GAME_HEIGHT))
		pygame.display.set_caption('SAVE ME')
		self.clock = pygame.time.Clock()

	
	def generate_cube_locations(self):
		self.loc1 = random.randint(1,3)
		self.loc2 = random.randint(1,3)

		self.cube1_x,self.cube1_col = self.map_location(self.loc1)
		self.cube2_x,self.cube2_col = self.map_location(self.loc2)



	def map_location(self,loc):
		if loc == 1:
			return (2,self.COlOR_BLUE)
		elif loc == 2:
			return (104,self.COLOR_CHOC)
		else:
			return (206,self.COLOR_GRAY)

	def player(self):

		# game loop
		while not self.GAME_OVER:

			pygame.event.pump()
			if self.num_tries >= self.MAX_NUM_TRIES:
				self.GAME_OVER = True
				break

			for event in pygame.event.get():
				if event.type == pygame.QUIT:
						self.GAME_OVER = True
				if event.type == pygame.KEYDOWN:
					if event.key == pygame.K_ESCAPE:
						self.GAME_OVER = True
				if event.type == pygame.KEYDOWN:
					if event.key == pygame.K_LEFT:
						self.paddle_x -= self.PADDLE_VELOCITY
						if self.paddle_x < 0:
							self.paddle_x = 2
				if event.type == pygame.KEYDOWN:
					if event.key == pygame.K_RIGHT:
						self.paddle_x += self.PADDLE_VELOCITY
						if self.paddle_x > self.GAME_WIDTH - self.PADDLE_WIDTH:
							self.paddle_x = 206


			self.cube2_y += self.SQUARE_VELOCITY
			self.cube1_y += self.SQUARE_VELOCITY
			score_text = self.font.render("Score : {:d},High Score : {:d}".format(self.game_score,self.HIGH_SCORE),True,self.COLOR_WHITE)
			
			self.screen.fill(self.COlOR_BLACK)
			self.screen.blit(score_text,((self.GAME_WIDTH - score_text.get_width()),0))
			cube1 = pygame.draw.rect(self.screen,self.cube1_col,pygame.Rect(self.cube1_x,self.cube1_y,self.SQUARE_SIDE,self.SQUARE_SIDE))
			cube2 = pygame.draw.rect(self.screen,self.cube2_col,pygame.Rect(self.cube2_x,self.cube2_y,self.SQUARE_SIDE,self.SQUARE_SIDE))
			paddle = pygame.draw.rect(self.screen,self.COLOR_RED,pygame.Rect(self.paddle_x,self.PADDLE_FLOOR,self.PADDLE_WIDTH,self.PADDLE_HEIGHT))
			
			if self.cube2_y >= self.GAME_HEIGHT - self.PADDLE_HEIGHT:
				if cube2.colliderect(paddle) or cube1.colliderect(paddle):
					self.game_score -= 0
				else:
					self.game_score += 1
				self.generate_cube_locations()
				self.cube2_y = self.SQUARE_CIELING
				self.cube1_y = self.SQUARE_CIELING
				self.num_tries += 1
			
			pygame.display.flip()
			
			# save last 4 frames
			self.frames.append(pygame.surfarray.array2d(self.screen))
			self.clock.tick(30) # 30 frames per second

			# save last 4 frames
			S = np.array(self.frames)
			with open("data/game_screenshots.npy","wb") as fscreen:
				np.save(fscreen,S)

		if self.game_score > self.HIGH_SCORE:
			dummy = {}
			dummy['high_score'] = self.game_score
			with open('data/high_score.json','wb') as file:
				json.dump(dummy,file)

if __name__ == "__main__":
	game = save_me()
	game.player()
