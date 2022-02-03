import gym
from stable_baselines3 import PPO


def train_expert_model(env_name, algo_name='PPO', steps=10000):
    model = PPO(policy='MlpPolicy', env=gym.make(env_name), verbose=0).learn(steps)
    model.save(f'model/expert/{env_name}/{algo_name}')
    print(f'model {algo_name} on env {env_name} saved')

    # mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
    # print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")


def load_expert_model(env_name, algo_name):
    return PPO.load(f'model/expert/{env_name}/{algo_name}')


def return_dateset(env_name, algo_name='weak_ppo', nb_steps=10000, render=False):
    loaded_model = load_expert_model(env_name=env_name, algo_name=algo_name)

    env = gym.make(env_name)

    X = []
    y = []

    obs = env.reset()
    for i in range(nb_steps):
        action, _states = loaded_model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        obs = obs.tolist()
        X.append(obs)
        action = action.tolist()

        # y.append(float(action))
        if env_name == 'CartPole-v1':
            y.append(one_hot_encoding(action))
        else:
            y.append(action)

        if render:
            env.render()

        if done:
            obs = env.reset()

    env.close()
    return X, y


def one_hot_encoding(action):
    if action == 1:
        return [1, 0]
    else:
        return [0, 1]


if __name__ == '__main__':
    env_name = 'BipedalWalker-v3'
    train_expert_model(env_name, 'weak_ppo', 5000)
    train_expert_model(env_name, 'medium_ppo', 100000)
    # train_expert_model(env_name, 'strong_ppo', 2000000)



