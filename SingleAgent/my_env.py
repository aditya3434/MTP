import gym
from gym.spaces import Box
import numpy as np
import pygame

FPS = 100
GAP = 100
COLLISION_DIST = 55
OFFROAD_DIST = 35

class Vehicle():

    def __init__(self, x_initial=0, y_initial=0, v_initial=0, angle_initial=0):
        self.x = x_initial
        self.y = y_initial
        self.v = v_initial
        self.angle = angle_initial
        self.acc = 0
        self.color = (255, 0, 0)

    def take_action(self, action, dt):

        if action is None:
            return

        self.acc = action[0]
        self.angle += 5*action[1]*dt
        self.v += 5*self.acc*dt
        self.x += 5*self.v*np.cos(self.angle*np.pi/180)*dt
        self.y += 5*self.v*np.sin(self.angle*np.pi/180)*dt

def dist_euclid(v1, v2):
    return np.sqrt((v1.x-v2.x)**2+(v1.y-v2.y)**2)

class myEnv(gym.Env):

    def __init__(self):
        self.action_space = Box(-1, 1, shape=(1,), dtype=np.float32)
        self.observation_space = Box(-np.inf, np.inf, shape=(3,), dtype=np.float32)

        self.back_car = Vehicle(x_initial=0, y_initial=400)
        self.front_car = Vehicle(x_initial=2*GAP, y_initial=400)

        self.ego_car = Vehicle(x_initial=GAP, y_initial=400)

        pygame.init()
        self.clock = pygame.time.Clock()
        self.dt = self.clock.tick(FPS)/1000
        self.screen = pygame.Surface((800, 600))

    def step(self, action):

        reward = 0
        done = False

        action[0] = np.clip(action[0], 0, 1)

        if (action[0] <= 0 and self.ego_car.v <= 0):
            action[0] *= -1

        #print(f'self.dt =============== {self.dt}')

        self.ego_car.take_action([action[0], 0], self.dt)
        self.back_car.take_action([0.2, 0], self.dt)
        self.front_car.take_action([0.2, 0], self.dt)

        # Reaching goal location 
        if (self.ego_car.x >= 950):
            reward += 100
            done = True
        
        '''if abs(self.ego_car.y-400) > OFFROAD_DIST:
            reward -= 100
            done = True'''

        back_dist = dist_euclid(self.ego_car, self.back_car)
        front_dist = dist_euclid(self.ego_car, self.front_car)

        if (back_dist < COLLISION_DIST or front_dist < COLLISION_DIST):
            reward -= 100
            done = True

        dist_error = abs(front_dist-back_dist)

        #reward += (self.ego_car.x//200)*10

        # Reward for movement
        if(self.ego_car.v > 1):
            reward += 1
        
        if(back_dist < 60 or front_dist < 60):
            reward -= 1

        '''if (dist_error >= 2):
            reward += 1'''

        obs = self.feature_scaling(np.hstack((self.ego_car.x, back_dist, front_dist)))

        return np.array(obs, dtype=np.float32), reward, done, None

    def reset(self):
        self.back_car = Vehicle(x_initial=0, y_initial=400)
        self.front_car = Vehicle(x_initial=2*GAP, y_initial=400)
        self.ego_car = Vehicle(x_initial=GAP, y_initial=400)

        self.state_max = np.hstack((1000, 2*GAP, 2*GAP))
        self.state_min = np.hstack((0, 0, 0))

        return np.array([0, 0.5, 0.5], dtype=np.float32)

    def draw(self):

        pygame.draw.rect(self.screen, (125, 125, 125), [0, 400, 1000, 75])
        finish = pygame.image.load('../finish.jpg')
        finish = pygame.transform.rotate(finish, 90)
        finish = pygame.transform.scale(finish, (30, 75))
        self.screen.blit(finish, (950, 400))

        sprite = pygame.image.load('../car.png')
        sprite = pygame.transform.rotate(sprite, -90)
        sprite = pygame.transform.scale(sprite, (80, 80))

        self.screen.blit(sprite, (self.back_car.x, self.back_car.y))
        self.screen.blit(sprite, (self.front_car.x, self.front_car.y))

        sprite = pygame.transform.rotate(sprite, (-1)*self.ego_car.angle)

        self.screen.blit(sprite, (self.ego_car.x, self.ego_car.y))
        

    def render(self):
        self.screen = pygame.display.set_mode((1000, 800))
        self.screen.fill((255, 255, 255))
        self.draw()
        pygame.display.flip()

    def feature_scaling(self, state):
        return (state - self.state_min) / (self.state_max - self.state_min)