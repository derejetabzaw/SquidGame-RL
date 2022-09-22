import sys
import enum 
from tkinter import CENTER
import random
import pygame 
from pygame.locals import *
import numpy as np
from PIL import ImageGrab
from PIL import Image
import cv2 
import io
import os 
import time
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = (30, 30)
import seaborn as sns
import pandas as pd
import numpy as np
from random import randint
import os
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from keras.callbacks import TensorBoard
from collections import deque
import random
import pickle
import json
import keyboard
# from game import gamstate

vec = pygame.math.Vector2
ACC = 0.2
FRIC = -0.5
curator_red = pygame.image.load("resources/curater/redlight.png")
curator_green = pygame.image.load("resources/curater/greenlight.jpg")
curator_turning = pygame.image.load("resources/curater/turning.jpg")
RED_STATE_TIME_RANGE = (1000,6000)
GREEN_STATE_TIME_RANGE = (400,3000)
TURNING_STATE_TIME_RANGE = (150,400)

class Player(pygame.sprite.Sprite):
    def __init__(self,x,y,width,height):
        super().__init__()
        self.x = x
        self.y = y
        self.width = width 
        self.height = height 
        self.surf = pygame.Surface((width,height))
        self.surf.fill((128,255,40))
        self.rect = self.surf.get_rect(center = (x,y))
        self.pos = vec((x,y))
        self.rect.midbottom = self.pos 
        self.vel = vec(0,0)
        self.acc = vec(0,0)
    def move(self):
        self.acc = vec(0,0)
        pressed_keys = pygame.key.get_pressed()
        if pressed_keys[K_UP]:
            self.acc.y = ACC
        self.acc.y += self.vel.y * FRIC
        self.vel += self.acc
        self.pos -= self.vel + 0.5 * self.acc
        self.rect.midbottom = self.pos

    def display(self,displaye_surface):
        displaye_surface.blit(self.surf , self.rect)

    def agent_move(self):
        self.acc = vec(0,0)
        self.acc.y = ACC
        self.acc.y += self.vel.y * FRIC
        self.vel += self.acc
        self.pos -= self.vel + 0.5 * self.acc
        self.rect.midbottom = self.pos
    
    def agent_stop(self):
        self.rect.midbottom = self.pos

class FinishLine(pygame.sprite.Sprite):
    def __init__(self,x,y,width,height):
        super().__init__()
        self.surf = pygame.Surface((width,height))
        self.surf.fill((255,0,0))
        self.rect = self.surf.get_rect(center = (x,y))
    def display(self,display):
        display.blit(self.surf, self.rect)

    def check_game_won(self,player_pos):
        if player_pos < self.rect.top:
            return True

class GameState(enum.Enum):
    ACTIVE = 1
    WON = 2
    LOST = 3


class CuratorState(enum.Enum):
    RED = 1 
    GREEN = 2
    TURNING = 3

class Curator(pygame.sprite.Sprite):
    def __init__(self,x,y,width,height,player:Player):
        super().__init__()
        self.x = x
        self.y = y
        self.width = width 
        self.height = height
        self.surf = pygame.Surface((width,height))
        self.rect = self.surf.get_rect(center = (x,y))
        self.state = CuratorState.GREEN
        self.last_state = CuratorState.TURNING 
        self.state_time_start = pygame.time.get_ticks()
        self.state_wait_time = random.uniform(GREEN_STATE_TIME_RANGE[0],GREEN_STATE_TIME_RANGE[1])
        self.player = player
        self.last_player_pos_y = player.rect.bottom
    def update(self):
        time_elapsed = (pygame.time.get_ticks() - self.state_time_start)
        if time_elapsed > self.state_wait_time:
            if self.state == CuratorState.GREEN:
                self.last_state = CuratorState.GREEN
                self.state = CuratorState.TURNING
                self.state_wait_time = random.uniform(TURNING_STATE_TIME_RANGE[0], TURNING_STATE_TIME_RANGE[1])
            elif self.state == CuratorState.TURNING and self.last_state == CuratorState.GREEN:
                self.last_state = CuratorState.TURNING
                self.state = CuratorState.RED
                self.state_wait_time = random.uniform(RED_STATE_TIME_RANGE[0],RED_STATE_TIME_RANGE[1])
                self.last_player_pos_y = self.player.rect.bottom
            elif self.state == CuratorState.TURNING and self.last_state == CuratorState.RED:
                self.last_state = CuratorState.TURNING
                self.state = CuratorState.GREEN 
                self.state_wait_time = random.uniform(GREEN_STATE_TIME_RANGE[0],GREEN_STATE_TIME_RANGE[1])
            elif self.state == CuratorState.RED:
                self.last_state = CuratorState.RED
                self.state = CuratorState.RED 
                self.state = CuratorState.TURNING
                self.state_wait_time = random.uniform(TURNING_STATE_TIME_RANGE[0],TURNING_STATE_TIME_RANGE[1])
            self.state_time_start = pygame.time.get_ticks()

    def display(self,display):
        if self.state == CuratorState.GREEN:
            display.blit(curator_green,self.rect)
        elif self.state == CuratorState.TURNING:
            display.blit(curator_turning,self.rect)
        elif self.state == CuratorState.RED:
            display.blit(curator_red,self.rect)

    def detect_movement(self):
        if self.state == CuratorState.RED:
            if self.player.rect.bottom < self.last_player_pos_y:
                return True
        False


#game parameters
ACTIONS = 2 # possible actions: Move Forward , do nothing
GAMMA = 0.99 # decay rate of past observations original 0.99
OBSERVATION = 50000. # timesteps to observe before training
EXPLORE = 100000  # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.1 # starting value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
FRAME_PER_ACTION = 1
LEARNING_RATE = 1e-4
img_rows , img_cols = 40,20
img_channels = 4 #We stack 4 frames

def buildmodel():
    model = Sequential()
    model.add(Conv2D(32, (8, 8), strides=(4, 4), padding='same',input_shape=(img_cols,img_rows,img_channels)))  #20*40*4
    model.add(Activation('relu'))
    model.add(Conv2D(64, (4, 4), strides=(2, 2), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same'))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(ACTIONS))
    adam = Adam(lr=LEARNING_RATE)
    model.compile(loss='mse',optimizer=adam)
    return model

print (buildmodel().summary())