import gym
from stable_baselines3 import PPO
import os
import time
from Snakeenv import SnakeEnv


models_dir = f"models/-{int(time.time())}"
logdir = f"logs/-{int(time.time())}"


if not os.path.exists(models_dir):
    os.makedirs(models_dir)


if not os.path.exists(logdir):
    os.makedirs(logdir)


env = SnakeEnv()
env.reset()

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir)


TIMESTAMPS = 10000

for i in range(1,10000):
    model.learn(total_timesteps=TIMESTAMPS, reset_num_timesteps=False, tb_log_name="PPO")
    model.save(f"{models_dir}/{TIMESTAMPS}")


env.close()