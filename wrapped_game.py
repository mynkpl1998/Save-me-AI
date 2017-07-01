# Author : Mayank Kumar Pal
# Email : mayank15147@iiitd.ac.in
import pygame
import random
import collections
import numpy as np
import os


class save_me(object):

    def __init__(self):
        #os.environ['SDL_VIDEODRIVER'] = 'dummy'
        pygame.init()
        pygame.key.set_repeat(10, 100)
        self.COlOR_BLACK = (0, 0, 0)
        self.COlOR_BLUE = (0, 0, 255)
        self.COLOR_CHOC = (139, 69, 19)
        self.COLOR_RED = (255, 0, 0)
        self.COLOR_GRAY = (128, 128, 128)
        self.COLOR_WHITE = (255, 255, 255)
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
        self.MAX_NUM_TRIES = 1
        self.CUSTOM_EVENT = pygame.USEREVENT + 1
        self.font = pygame.font.SysFont("Comic Sans MS", self.FONT_SIZE)

    def reset(self):

        self.frames = collections.deque(maxlen=4)
        self.num_tries = 0
        self.game_score = 0
        self.reward = 0
        self.paddle_x = 104
        self.generate_cube_locations()
        self.cube1_y = self.SQUARE_CIELING
        self.cube2_y = self.SQUARE_CIELING
        self.GAME_OVER = False

        self.screen = pygame.display.set_mode(
            (self.GAME_WIDTH, self.GAME_HEIGHT))
        pygame.display.set_caption('SAVE ME')
        self.clock = pygame.time.Clock()

    def generate_cube_locations(self):
        self.loc1 = random.randint(1, 3)
        self.loc2 = random.randint(1, 3)

        self.cube1_x, self.cube1_col = self.map_location(self.loc1)
        self.cube2_x, self.cube2_col = self.map_location(self.loc2)

    def map_location(self, loc):
        if loc == 1:
            return (2, self.COlOR_BLUE)
        elif loc == 2:
            return (104, self.COLOR_CHOC)
        else:
            return (206, self.COLOR_GRAY)

    def step(self, action):

        pygame.event.pump()
        # if action = 0 , move left
        # if action = 1 , stay
        # if action = 2 , move right
        if action == 0:
            self.paddle_x -= self.PADDLE_VELOCITY
            if self.paddle_x < 0:
                self.paddle_x = 2
        elif action == 2:
            self.paddle_x += self.PADDLE_VELOCITY
            if self.paddle_x > self.GAME_WIDTH - self.PADDLE_WIDTH:
                self.paddle_x = 206

        else:
            pass

        self.screen.fill(self.COlOR_BLACK)
        #self.screen.blit(score_text,((self.GAME_WIDTH - score_text.get_width()),0))
        self.cube2_y += self.SQUARE_VELOCITY
        self.cube1_y += self.SQUARE_VELOCITY
        cube1 = pygame.draw.rect(self.screen, self.cube1_col, pygame.Rect(
            self.cube1_x, self.cube1_y, self.SQUARE_SIDE, self.SQUARE_SIDE))
        cube2 = pygame.draw.rect(self.screen, self.cube2_col, pygame.Rect(
            self.cube2_x, self.cube2_y, self.SQUARE_SIDE, self.SQUARE_SIDE))
        paddle = pygame.draw.rect(self.screen, self.COLOR_RED, pygame.Rect(
            self.paddle_x, self.PADDLE_FLOOR, self.PADDLE_WIDTH, self.PADDLE_HEIGHT))

        self.reward = 0
        if self.cube2_y >= self.GAME_HEIGHT - self.PADDLE_HEIGHT:
            if cube2.colliderect(paddle) or cube1.colliderect(paddle):
                self.reward = -1
            else:
                self.reward = 1

            self.generate_cube_locations()
            self.cube2_y = self.SQUARE_CIELING
            self.cube1_y = self.SQUARE_CIELING
            self.num_tries += 1

        pygame.display.flip()

        # save last 4 frames
        self.frames.append(pygame.surfarray.array2d(self.screen))

        if self.num_tries >= self.MAX_NUM_TRIES:
            self.GAME_OVER = True

        self.clock.tick(30)  # 30 frames per second

        return self.get_frames(), self.reward, self.GAME_OVER

    def get_frames(self):
        return np.array(list(self.frames))


if __name__ == "__main__":
    game = save_me()

    NUM_EPOCH = 10
    for e in range(NUM_EPOCH):
        print("Epoch : {:d}".format(e))
        game.reset()
        input_t = game.get_frames()
        game_over = False
        while not game_over:
            action = np.random.randint(0, 3, size=1)[0]
            input_tp1, reward, game_over = game.step(action)
            print(action, reward, game_over)
