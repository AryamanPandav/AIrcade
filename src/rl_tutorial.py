# Source: https://www.youtube.com/watch?v=Mut_u40Sqz4

# Import relevant modules

import os
import gym
from stable_baselines3 import PPO

# Stable baselines allows vectorization of environments, meaning that the RL
# agent can be trained on multiple environments at the same time. In this case
# we won't be vectorizing, so will be using DummyVecEnv as a wrapper.
from stable_baselines3.common.vec_env import DummyVecEnv

from stable_baselines3.common.evaluation import evaluate_policy

# Callback helpers
from stable_baselines3.common.callbacks import \
    EvalCallback, StopTrainingOnRewardThreshold

# Define paths

log_path = os.path.join('training', 'logs')
save_path = os.path.join('training', 'models')
model_path = os.path.join('training', 'models', 'PPO_MODEL_CARTPOLE')

# Load environment

ENV_NAME = 'CartPole-v0'
env = gym.make(ENV_NAME)
env = DummyVecEnv([lambda: env])

# Retrain model if desired

RETRAIN_MODEL = True

if RETRAIN_MODEL:

    # This callback will stop training once we pass a particular reward
    # threshold
    stop_callback = StopTrainingOnRewardThreshold(reward_threshold=200,
                                                  verbose=1)

    # This callback will be run at the end of each training run. eval_freq is
    # saying we want to run the eval callback every 10k timesteps.
    eval_callback = EvalCallback(env,
                                 callback_on_new_best=stop_callback,
                                 eval_freq=10000,
                                 best_model_save_path=save_path,
                                 verbose=1)

    POLICY = 'MlpPolicy'  # Multilayer perceptron without LSTM and CNNs
    model = PPO(POLICY, env, verbose=1, tensorboard_log=log_path)

    NUM_TIMESTEPS = 20000
    model.learn(NUM_TIMESTEPS, callback=eval_callback)

    # Save model
    model.save(model_path)

    # Delete model, just for the practice of reloading it again down below
    del model

# Load model

model = PPO.load(model_path, env=env)

# Evaluate model

NUM_EVAL_EPISODES = 10
evaluate_policy(model, env, n_eval_episodes=NUM_EVAL_EPISODES, render=True)

# Test model

# An episode is like one full "game" within an environment
NUM_EPISODES = 5
for episode in range(1, NUM_EPISODES+1):

    # Returns the initial set of observations for our environment. For the
    # CartPole-v0 environment, there are 4 values defining the environment.
    observation = env.reset()
    done = False
    score = 0

    while not done:

        # View the graphical representation of the environment
        env.render()

        # Generate a random action. The action space is the space of all
        # possible actions. This is a discrete action space with two actions,
        # moving the cart left or right.
        # action = env.action_space.sample()

        # Use the model to predict the next action. The next state is used in
        # recurrent policies, which we're not using in this example.
        action, _states = model.predict(observation)

        # Taking an action results in more observations, a reward, and whether
        # the episode is complete
        observation, reward, done, info = env.step(action)

        # Accumulating the reward
        score += reward

    print(f'Episode: {episode}, Score: {score}')

env.close()