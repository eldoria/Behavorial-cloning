import tensorflow as tf
from tensorflow.keras import layers, callbacks
from tensorflow import keras

from sklearn.model_selection import train_test_split

import gym

import numpy as np

from expert import return_dateset


model = tf.keras.Sequential([
    layers.Dense(4, activation='tanh', input_shape=(4,)),
    layers.Dense(4, activation='tanh'),
    layers.Dense(4, activation='tanh'),
    layers.Dense(2, activation='sigmoid')
])

model.compile(optimizer='adam',
              # loss=keras.losses.sparse_categorical_crossentropy, metrics=keras.metrics.sparse_categorical_accuracy)
              loss=keras.losses.categorical_crossentropy, metrics=keras.metrics.categorical_accuracy)


def train_behavior_model(name, nb_steps):
    X, y = return_dateset(name, nb_steps)

    # print(X)
    # print(y)

    model.fit(X, y,
              epochs=50,
              batch_size=1,
              callbacks=[callbacks.EarlyStopping(monitor='loss', patience=10)])

    return model


def test_beahvior_model(name, nb_steps=100000):
    model = train_behavior_model(name, nb_steps)

    env = gym.make("CartPole-v1")

    obs = env.reset()
    scores = []
    score = 0
    for i in range(1000):
        action = return_max(model.predict([obs.tolist()]))
        print(action)

        obs, reward, done, info = env.step(action)

        '''
        print(f'observation: {obs}')
        print(f'action: {action}')
        print('\n')
        '''

        env.render()
        score += 1
        if done:
            scores.append(score)
            score = 0
            obs = env.reset()
            print()
    env.close()
    print(f'score: {np.mean(scores)}')


def return_max(arr):
    arr = arr.tolist()[0]
    return arr.index(max(arr))


if __name__ == '__main__':
    test_beahvior_model('strong_ppo', 1000)


