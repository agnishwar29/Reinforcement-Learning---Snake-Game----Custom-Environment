from Snakeenv import SnakeEnv
from stable_baselines3 import PPO
import gym

models_dir = "models/-1665123934"

env = SnakeEnv()

env.reset()

model_path = f"{models_dir}/10000.zip"
model = PPO.load(model_path, env=env)

episodes = 500

for ep in range(episodes):
    obs = env.reset()
    done= False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)

        print(rewards)