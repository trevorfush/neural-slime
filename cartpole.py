import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras

env = gym.make("CartPole-v1")
obs = env.reset()

n_inputs = 4

model = keras.models.Sequential([
    keras.layers.Dense(5, activation="elu", input_shape=[n_inputs]),
    keras.layers.Dense(1, activation="sigmoid"),
])

def basic_policy(obs):
    angle = obs[2]
    return 0 if angle < 0 else 1

def play_one_step(env, obs, model, loss_fn):
    with tf.GradientTape() as tape:

        # Using model to predict probability of choosing left
        left_proba = model(obs[np.newaxis])

        # Action is move left with probability defined above
        action = (tf.random.uniform([1,1]) > left_proba)

        # Target probability of moving left (if action=0, y_target=1 and vice versa)
        y_target = tf.constant([[1.]]) - tf.cast(action, tf.float32)

        # Compute loss based on left_prob and y_target
        loss = tf.reduce_mean(loss_fn(y_target, left_proba))

    grads = tape.gradient(loss, model.trainable_variables)
    obs, reward, done, info = env.step(int(action[0,0].numpy()))
    return obs, reward, done, grads

def play_multiple_episodes(env, n_episodes, n_max_steps, model, loss_fn):
    all_rewards = []
    all_grads = []
    for episode in range(n_episodes):
        current_rewards = []
        current_grads = []
        obs = env.reset()
        for step in range(n_max_steps):
            obs, reward, done, grads = play_one_step(env, obs, model, loss_fn)
            current_rewards.append(reward)
            current_grads.append(grads)
            if done:
                break
        all_rewards.append(current_rewards)
        all_grads.append(current_grads)
    return all_rewards, all_grads

def discount_rewards(rewards, discount_factor):
    discounted = np.array(rewards)
    for step in range(len(rewards)-2,-1,-1):
        discounted[step] += discounted[step+1]*discount_factor
    return discounted

def discount_and_normalize_rewards(all_rewards, discount_factor):
    all_discounted_rewards = [discount_rewards(rewards, discount_factor) for rewards in all_rewards]
    flat_rewards = np.concatenate(all_discounted_rewards)
    reward_mean = flat_rewards.mean()
    reward_std = flat_rewards.std()
    return [(discount_rewards - reward_mean)/reward_std for discount_rewards in all_discounted_rewards]

n_iterations = 150
n_episodes_per_update = 10
n_max_steps = 200
discount_factor = 0.95

optimizer = keras.optimizers.Adam(learning_rate=0.01)
loss_fn = keras.losses.binary_crossentropy


for iteration in range(n_iterations):
    all_rewards, all_grads = play_multiple_episodes(env, n_episodes_per_update, n_max_steps, model, loss_fn)
    all_final_rewards = discount_and_normalize_rewards(all_rewards, discount_factor)

    all_mean_grads = []
    for var_index in range(len(model.trainable_variables)):
        mean_grads = tf.reduce_mean(
            [final_reward * all_grads[episode_index][step][var_index] 
            for episode_index, final_rewards in enumerate(all_final_rewards)
                for step, final_reward in enumerate(final_rewards)], axis=0
        )
        all_mean_grads.append(mean_grads)
    print(len(all_mean_grads), model.trainable_variables)
    optimizer.apply_gradients(zip(all_mean_grads, model.trainable_variables))

# totals = []
# for episode in range(500):
#     episode_rewards = 0
#     obs = env.reset()
#     for step in range(200):
#         action = basic_policy(obs)
#         obs, reward, done, info = env.step(action)
#         episode_rewards += reward
#         if done:
#             break

#     totals.append(episode_rewards)

#print(np.mean(totals), np.std(totals), np.min(totals), np.max(totals))