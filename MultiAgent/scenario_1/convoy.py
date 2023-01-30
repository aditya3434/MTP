from gym.spaces import Box
import numpy as np
import functools
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
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
        #self.angle += 10*action[1]*dt
        self.v += 5*self.acc*dt
        self.x += 5*self.v*np.cos(self.angle*np.pi/180)*dt
        self.y += 5*self.v*np.sin(self.angle*np.pi/180)*dt

def dist_euclid(v1, v2):
    return np.sqrt((v1.x-v2.x)**2+(v1.y-v2.y)**2)
        

def env(n_vehicles=3):
    env = raw_env(n_vehicles=n_vehicles)
    return env


class raw_env(AECEnv):

    metadata = {"render_modes": ["human"], "name": "rps_v2"}

    def __init__(self, n_vehicles=3):
        
        self.possible_agents = ["vehicle_" + str(r) for r in range(n_vehicles)]
        self.vehicles = [Vehicle(x_initial=r*GAP, y_initial=400) for r in range(n_vehicles)]
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )

        self._action_spaces = {agent: Box(0, 1.0, (1, 1)) for agent in self.possible_agents}
        self._observation_spaces = {
            agent: Box(-1000, 1000, (1, 3)) for agent in self.possible_agents
        }

        self.done = False

        pygame.init()
        self.clock = pygame.time.Clock()
        self.dt = self.clock.tick(FPS)/1000
        self.screen = pygame.Surface((800, 600))

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return Box(-1000, 1000, (1, 3))

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Box(0, 1.0, (1,))

    def draw(self):

        pygame.font.init()
        my_font = pygame.font.SysFont('Comic Sans MS', 30)

        pygame.draw.rect(self.screen, (125, 125, 125), [0, 400, 1000, 75])
        finish = pygame.image.load('../finish.jpg')
        finish = pygame.transform.rotate(finish, 90)
        finish = pygame.transform.scale(finish, (30, 75))
        self.screen.blit(finish, (950, 400))

        for agent in self.agents:
            vehicle = self.vehicles[self.agent_name_mapping[agent]]
            
            sprite = pygame.image.load('../car.png')
            sprite = pygame.transform.rotate(sprite, -90)
            sprite = pygame.transform.scale(sprite, (80, 80))

            sprite = pygame.transform.rotate(sprite, (-1)*vehicle.angle)
            self.screen.blit(sprite, (vehicle.x, vehicle.y))

        for index, agent in enumerate(self.agents):
            vehicle = self.vehicles[self.agent_name_mapping[agent]]
            text_surface = my_font.render(f'{agent} : [x: {round(vehicle.x, 3)}, y: {round(vehicle.y, 3)}, vel: {round(vehicle.v, 3)}, angle: {round(vehicle.angle, 3)}]', False, (0, 0, 0))
            self.screen.blit(text_surface, (0,index*30))

    def render(self, mode="human"):
        self.screen = pygame.display.set_mode((1000, 800))
        self.screen.fill((255, 255, 255))
        self.draw()
        pygame.display.flip()

    def observe(self, agent):
        return np.array(self.observations[agent])

    def close(self):
        pygame.event.pump()
        pygame.display.quit()

    def reset(self, seed=None, options=None):
        
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.state = {agent: None for agent in self.agents}
        self.observations = {agent: None for agent in self.agents}
        self.num_moves = 0
        """
        Our agent_selector utility allows easy cyclic stepping through the agents list.
        """
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

        self.vehicles = [Vehicle(x_initial=r*GAP, y_initial=400) for r in range(len(self.vehicles))]
        self.done = False

        return np.array([0])

    def step(self, action):
        
        # if self.dones[self.agent_selection]:
        #     # handles stepping an agent which is already done
        #     # accepts a None action for the one agent, and moves the agent_selection to
        #     # the next done agent,  or if there are no more done agents, to the next live agent
        #     return self._was_done_step(action)

        if all(x for x in self.dones.values()):
            self.done=True
            return

        agent = self.agent_selection
        vehicle = self.vehicles[self.agent_name_mapping[agent]]

        self._cumulative_rewards[agent] = 0

        self.state[self.agent_selection] = action

        action[0] = np.clip(action[0], 0, 1)
        vehicle.take_action(action, self.dt)

        if self._agent_selector.is_last():
            
            for agent in self.agents:

                if self.dones[agent]:
                    continue
                
                vehicle = self.vehicles[self.agent_name_mapping[agent]]
                self.rewards[agent] = 0
                if (vehicle.x >= 950):
                    self.dones[agent] = True
                    vehicle.color = (0, 0, 255)
                    self.rewards[agent] += 300

                index = self.agent_name_mapping[agent]

                if abs(vehicle.y-400) > OFFROAD_DIST:
                    self.rewards[agent] -= 100
                    self.done = True

                if index == len(self.agents)-1:
                    front = GAP
                    back = dist_euclid(vehicle, self.vehicles[index-1])
                elif index == 0:
                    front = dist_euclid(vehicle, self.vehicles[index+1])
                    back = GAP
                else:
                    front = dist_euclid(vehicle, self.vehicles[index+1])
                    back = dist_euclid(vehicle, self.vehicles[index-1])

                if (front < COLLISION_DIST and self.dones[self.agents[index+1]] != True) or (back < COLLISION_DIST and self.dones[self.agents[index-1]] != True):
                    self.rewards[agent] -= 100
                    self.done = True

                if(vehicle.v > 1):
                    self.rewards[agent] += 0.3
                
                if(front < 60 or back < 60):
                    self.rewards[agent] -= 0.3

                self.observations[agent] = [self.vehicles[self.agent_name_mapping[agent]].x/1000, front/(2*GAP), back/(2*GAP)]
        else:
            self._clear_rewards()

        self.agent_selection = self._agent_selector.next()
        self._accumulate_rewards()