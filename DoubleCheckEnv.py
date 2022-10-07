from Snakeenv import SnakeEnv

env = SnakeEnv()

episodes = 50

for episode in range(episodes):
    done = False
    obs = env.reset()
    while not done:
        random_action = env.action_space.sample()
        print("Action", random_action)
        obs, reward, done, info = env.step(random_action)
        print('reward', reward)