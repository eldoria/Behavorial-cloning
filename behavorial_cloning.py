import tensorflow as tf
from tensorflow.keras import layers, callbacks
from tensorflow import keras

import gym

import numpy as np

from expert import return_dateset

'''
def OurModel(input_shape=4, action_space=2):
    X_input = Input(input_shape)

    # 'Dense' is the basic form of a neural network layer
    # Input Layer of state size(4) and Hidden Layer with 512 nodes
    X = Dense(512, input_shape=input_shape, activation="relu", kernel_initializer='he_uniform')(X_input)

    # Hidden layer with 256 nodes
    X = Dense(256, activation="relu", kernel_initializer='he_uniform')(X)

    # Hidden layer with 64 nodes
    X = Dense(64, activation="relu", kernel_initializer='he_uniform')(X)

    # Output Layer with # of actions: 2 nodes (left, right)
    X = Dense(action_space, activation="linear", kernel_initializer='he_uniform')(X)

    model = Model(inputs=X_input, outputs=X, name='CartPole DQN model')
    model.compile(loss="mse", optimizer=RMSprop(lr=0.00025, rho=0.95, epsilon=0.01), metrics=["accuracy"])

    model.summary()
    return model
'''


def return_model():
    model = tf.keras.Sequential([
        layers.Dense(256, activation='relu', input_shape=(4,)),
        layers.Dense(126, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam', loss=keras.losses.categorical_crossentropy, metrics=keras.metrics.categorical_accuracy)

    return model


def train_behavior_model(name, nb_steps):
    X, y = return_dateset(name, nb_steps)

    model = return_model()

    model.fit(X, y,
              epochs=300,
              batch_size=2048,
              callbacks=[callbacks.EarlyStopping(monitor='loss', patience=10)])

    model.save_weights(f"model/apprentice/{name.split('_')[0]}/behavior_model")

    return model


def test_beahvior_model(name, nb_steps=100000, render=False):
    # model = train_behavior_model(name, nb_steps)
    model = return_model()
    model.load_weights(f"model/apprentice/{name.split('_')[0]}/behavior_model")



    env = gym.make("CartPole-v1")

    obs = env.reset()
    scores = []
    score = 0
    for i in range(10000):
        action = return_max(model.predict([obs.tolist()]))

        obs, reward, done, info = env.step(action)

        if render:
            env.render()
        score += 1
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
    # test_beahvior_model('weak_ppo', render=True)
    # test_beahvior_model('medium_ppo', render=True)
    test_beahvior_model('strong_ppo', render=True)


