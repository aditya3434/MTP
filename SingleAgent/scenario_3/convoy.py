import gym
from gym.spaces import Box
from math import copysign
import numpy as np
import pygame

FPS = 100
GAP = 100
COLLISION_DIST = 55
OFFROAD_DIST = 35
X_INITIAL = 450
Y_INITIAL = 512
V_INITIAL = 20

# Environment variables

gravity_acc = 9.8   # Acceleration due to gravity
Cr = 0.001           # Coefficient of rolling friction
Cd = 0.32           # Drag coefficient
air_density = 1.3   # Air density

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
        self.angle += 20*action[1]*dt
        self.v += 5*self.acc*dt
        self.x += self.v*np.cos(self.angle*np.pi/180)*dt
        self.y -= self.v*np.sin(self.angle*np.pi/180)*dt

class ComplexVehicle(Vehicle):

    def __init__(self, x_initial=0, y_initial=0, v_initial=0, angle_initial=0):
        super().__init__(x_initial, y_initial, v_initial, angle_initial)

        self.mass = 1600         # Car mass
        self.car_area = 2.4      # Car front area
        

    def take_action(self, action, dt):

        if action is None:
            return

        self.acc = action[0]
        self.angle += 20*action[1]*dt

        v_mps = self.v*(5/18)

        Fr = self.mass*gravity_acc*Cr*copysign(1, self.v)           # Frictional force
        Fa = 0.5*air_density*Cd*self.car_area*abs(v_mps)*v_mps      # Air resistance

        a_net = self.acc-(Fr+Fa)/self.mass                          # Net acceleration 

        self.v += 5*a_net*dt
        self.x += self.v*np.cos(self.angle*np.pi/180)*dt
        self.y -= self.v*np.sin(self.angle*np.pi/180)*dt

def dist_euclid(v1, v2):
    return np.sqrt((v1.x-v2.x)**2+(v1.y-v2.y)**2)

class convoyEnv(gym.Env):

    def __init__(self):
        self.action_space = Box(0, 1, shape=(1,), dtype=np.float32)
        self.observation_space = Box(-np.inf, np.inf, shape=(3,), dtype=np.float32)

        self.back_car = ComplexVehicle(x_initial=X_INITIAL, y_initial=Y_INITIAL, v_initial=V_INITIAL)
        self.front_car = ComplexVehicle(x_initial=X_INITIAL+2*GAP, y_initial=Y_INITIAL, v_initial=V_INITIAL)
        self.ego_car = ComplexVehicle(x_initial=X_INITIAL+GAP, y_initial=Y_INITIAL, v_initial=V_INITIAL)

        pygame.init()
        self.clock = pygame.time.Clock()
        self.dt = self.clock.tick(FPS)/1000
        self.screen = pygame.Surface((800, 600))

    def step(self, action, flag):

        reward = 0
        done = False

        action[0] = np.clip(action[0], 0, 1)

        auto_action = self.best_action(self.back_car)
        self.back_car.take_action(auto_action, self.dt)

        pid_action = self.pid_controller()

        if flag and abs(action[0]-pid_action) >= 0.3:
            action[0] = pid_action

        self.ego_car.take_action([0, action[0]], self.dt)
        
        auto_action = self.best_action(self.front_car)
        self.front_car.take_action(auto_action, self.dt)

        # Reaching goal location 
        if (self.ego_car.y <= 250):
            #print("Target reached!")
            reward += 100
            done = True

        if (self.ego_car.x >= 800):
            #print("Wrong turn!")
            reward -= 100
            done = True
        
        if self.ego_car.y <= 450 and self.ego_car.x <= 650:
            #print("Offroad!")
            reward -= 100
            done = True

        back_dist = dist_euclid(self.ego_car, self.back_car)
        front_dist = dist_euclid(self.ego_car, self.front_car)

        if (back_dist < COLLISION_DIST or front_dist < COLLISION_DIST):
            #print("Collision!")
            reward -= 100
            done = True

        if self.ego_car.y-self.front_car.y < 0 and self.ego_car.x >= 650:
            #print("Overtaken!")
            reward -= 100
            done = True

        # Reward for movement

        if self.ego_car.angle <= 3 and self.ego_car.x < 650:
            reward += 0.02

        if self.ego_car.angle <= self.front_car.angle and abs(self.ego_car.angle-self.front_car.angle) <= 3 and self.ego_car.x >= 650:
            reward += 0.005*(90-abs(90-self.ego_car.angle))

        obs = self.feature_scaling(np.hstack((self.ego_car.angle, back_dist, front_dist)))

        return np.array(obs, dtype=np.float32), reward, done, None

    def reset(self):
        self.back_car = ComplexVehicle(x_initial=X_INITIAL, y_initial=Y_INITIAL, v_initial=V_INITIAL)
        self.front_car = ComplexVehicle(x_initial=X_INITIAL+2*GAP, y_initial=Y_INITIAL, v_initial=V_INITIAL)
        self.ego_car = ComplexVehicle(x_initial=X_INITIAL+GAP, y_initial=Y_INITIAL, v_initial=V_INITIAL)

        self.state_max = np.hstack((90, 2*GAP, 2*GAP))
        self.state_min = np.hstack((0, 0, 0))
        self.start_time = pygame.time.get_ticks()

        return np.array([0.1, 0.5, 0.5], dtype=np.float32)

    def draw(self):

        pygame.draw.rect(self.screen, (125, 125, 125), [750, 0, 100, 1500])
        pygame.draw.rect(self.screen, (125, 125, 125), [0, 500, 1500, 100])
        finish = pygame.image.load('../../finish.jpg')
        # finish = pygame.transform.rotate(finish, 90)
        finish = pygame.transform.scale(finish, (100, 40))
        self.screen.blit(finish, (750, 250))

        sprite = pygame.image.load('../../car.png')
        sprite = pygame.transform.rotate(sprite, -90)
        sprite = pygame.transform.scale(sprite, (80, 80))

        sprite1 = pygame.transform.rotate(sprite, self.back_car.angle)
        self.screen.blit(sprite1, (self.back_car.x, self.back_car.y))

        sprite2 = pygame.transform.rotate(sprite, self.front_car.angle)
        self.screen.blit(sprite2, (self.front_car.x, self.front_car.y))

        sprite3 = pygame.transform.rotate(sprite, self.ego_car.angle)
        self.screen.blit(sprite3, (self.ego_car.x, self.ego_car.y))
        

    def render(self):
        self.screen = pygame.display.set_mode((1500, 1000))
        self.screen.fill((255, 255, 255))
        self.draw()
        pygame.display.flip()

    def feature_scaling(self, state):
        return (state - self.state_min) / (self.state_max - self.state_min)

    def best_action(self, V):
        if V.x < 650:
            return [0.0, 0]
        if V.x >= 650 and V.angle < 89:
            return [0.0, 0.5]
        
        return [0.0, 0]
    
    def pid_controller(self):
        car = self.ego_car
        front_car = self.front_car

        if car.angle > front_car.angle:
            return 0
        
        steer = (front_car.angle-car.angle)/90

        return steer
