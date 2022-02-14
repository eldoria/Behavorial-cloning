import tensorflow as tf
from tensorflow.keras import layers, callbacks
from tensorflow import keras

from sklearn.model_selection import train_test_split

import gym

import numpy as np

from expert import return_dateset


def return_model(env_name):
    if env_name == 'CartPole-v1':
        model = tf.keras.Sequential([
            layers.Dense(256, activation='relu', input_shape=(4,)),
            layers.Dense(126, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(2, activation='softmax')
        ])
        model.compile(optimizer='adam', loss=keras.losses.categorical_crossentropy, metrics=keras.metrics.categorical_accuracy)
    else:
        model = tf.keras.Sequential([
            layers.Dense(48, activation='relu', input_shape=(24,)),
            layers.Dense(48, activation='relu'),
            layers.Dense(48, activation='relu'),
            layers.Dense(48, activation='relu'),
            layers.Dense(48, activation='relu'),
            layers.Dense(48, activation='relu'),
            layers.Dense(48, activation='relu'),
            layers.Dense(4)
        ])
        model.compile(optimizer='adam', loss='mae')
    return model


def train_behavior_model(env_name, algo_name, nb_steps):
    X, y = return_dateset(env_name, algo_name, nb_steps)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = return_model(env_name)

    if env_name == 'CartPole-v1':
        model.fit(X_train, y_train,
                  validation_data=(X_test, y_test),
                  epochs=400,
                  batch_size=2048,
                  callbacks=[callbacks.EarlyStopping(monitor='val_loss', patience=30)])
    else:
        model.fit(X_train, y_train,
                  validation_data=(X_test, y_test),
                  epochs=1000,
                  batch_size=2048,
                  callbacks=[callbacks.EarlyStopping(monitor='val_loss', patience=70)])

    model.save_weights(f"model/apprentice/{env_name}/{algo_name.split('_')[0]}/behavior_model")

    return model


def test_behavior_model(env_name, algo_name, nb_steps=1000000, render=False):
    model = return_model(env_name)
    model.load_weights(f"model/apprentice/{env_name}/{algo_name.split('_')[0]}/behavior_model")

    env = gym.make(env_name)

    obs = env.reset()
    scores = []
    score = 0
    for i in range(nb_steps):

        if env_name == 'CartPole-v1':
            action = return_max(model.predict([obs.tolist()]))
        else:
            action = model.predict([obs.tolist()])[0]

        obs, reward, done, info = env.step(action)

        if render:
            env.render()
        score += reward
        if done:
            print(score)
            scores.append(score)
            score = 0
            obs = env.reset()
    env.close()
    print(f'score: {np.mean(scores)}')


def return_max(arr):
    arr = arr.tolist()[0]
    return arr.index(max(arr))


if __name__ == '__main__':
    env_name = 'BipedalWalker-v3'
    # train_behavior_model(env_name, 'weak_ppo', 100000)
    # train_behavior_model(env_name, 'medium_ppo', 100000)
    # train_behavior_model(env_name, 'strong_ppo', 100000)
    # test_behavior_model(env_name, 'weak_ppo', render=True)
    # test_behavior_model(env_name, 'medium_ppo', render=True)
    test_behavior_model(env_name, 'strong_ppo', render=True)


