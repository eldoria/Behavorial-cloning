import gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import os


def train_expert_model(name='PPO', steps=10000):
    os.makedirs('model/', exist_ok=True)

    model = PPO(policy='MlpPolicy', env=gym.make("CartPole-v1"), verbose=0).learn(steps)
    model.save(f'model/{name}')

    # mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
    # print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")


def load_expert_model(name):
    return PPO.load(f'model/{name}')


def return_dateset(name='weak_ppo', nb_steps=10000, render=False):
    loaded_model = load_expert_model(name)

    env = gym.make("CartPole-v1")

    X = []
    y = []

    obs = env.reset()

    for i in range(nb_steps):
        action, _states = loaded_model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        obs = obs.tolist()
        X.append(obs)
        y.append(float(action))

        if render:
            env.render()

        if done:
            obs = env.reset()

    env.close()
    return X, y


if __name__ == '__main__':
    train_expert_model('weak_ppo', 50)
    train_expert_model('medium_ppo', 500)
    train_expert_model('strong_ppo', 5000)



