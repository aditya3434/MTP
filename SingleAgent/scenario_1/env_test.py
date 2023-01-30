import random
from my_env import myEnv

def get_action():
    a = float(random.randint(0, 10))/10
    b = float(random.randint(-10, 10))/10

    return [a, b]

episodes = 5
env = myEnv()

for episode in range(episodes):
    state = env.reset()
    done = False
    score = 0

    while not done:
        env.render()

        action = get_action()
        observation, reward, done, _ = env.step(action)

        score += reward

    print('Episode {} : Score {}'.format(episode+1, score))

env.close()

