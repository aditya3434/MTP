import convoy
import random

def policy(observation, agent):
    action = [0, 0.5]
    return action

env = convoy.env(n_vehicles=3)

env.reset()

total_episodes = 500

for episode in range(total_episodes):
    
    env.reset()

    for agent in env.agent_iter():
        observation, reward, done, info = env.last()

        action = [0.5, 0.0] if not done else None

        if agent == 'vehicle_1':
            action = policy(observation, agent) if not done else None
        
        env.step(action)
        env.render()

        if (env.done):
            break

    print(f'Episode {episode+1} : {env.rewards}')

env.close()