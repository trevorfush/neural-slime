import tensorflow as tf
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from tensorflow import keras
from tqdm import tqdm
import argparse

from envclass import SlimeSpace

#------------------------------------------------------------------#
#                                                                  #
#   Implementing actor-only policy gradient method for training    #
#   neural networks. Continuous action space based on paper        #
#   cited in proposal.                                             #
#                                                                  #
#------------------------------------------------------------------#

# Adapted from https://github.com/woutervanheeswijk/example_continuous_control
def construct_network(num_hidden_layers, hidden_layer_arch, input_layers, output_activation_func):

    hidden_layer_list = []

    input = keras.layers.Input(shape=(input_layers,))

    for i in range(num_hidden_layers):
        if i == 0:
            hidden_layer_list.append(keras.layers.Dense(hidden_layer_arch[i])(input))
        else:
            hidden_layer_list.append(keras.layers.Dense(hidden_layer_arch[i])(hidden_layer_list[-1]))
    
    # hidden = keras.layers.Dense(10)(input)

    mu = keras.layers.Dense(1, activation=output_activation_func)(hidden_layer_list[-1])

    sigma = keras.layers.Dense(1, activation=output_activation_func)(hidden_layer_list[-1])

    model = keras.Model(inputs=input, outputs=[mu, sigma])

    return model

def play_one_step_multi(env, obs, model_list, loss_fn):
    loss_list = np.zeros(len(model_list))
    grads_list = []
    action_list = np.zeros(len(model_list))

    for p,model in enumerate(model_list):
        
        with tf.GradientTape() as tape:
            # print(model(np.atleast_2d(obs[p,:,:].flatten())).numpy().shape)
            
            mu, sigma = model(np.atleast_2d(obs[p,:,:].flatten()))

            mu = 2*np.pi*mu

            action = tf.random.normal([1], mean=mu, stddev=sigma)
            action_list[p] = action

            step_loss = loss_fn(model, obs[p,:,:], action, env.step_reward[p])
            # print(step_loss)
            loss_list[p] = step_loss

        to_be_appended_grad = tape.gradient(step_loss, model.trainable_variables)
        grads_list.append(to_be_appended_grad)

        # print(action.numpy()[0][0])
    # print(f"Action List : {action_list}")
    obs, reward = env._step(action_list)

    return obs, reward, grads_list

def play_one_step_single(env, obs, action, model, loss_fn):
    with tf.GradientTape() as tape:
        # print(model(np.atleast_2d(obs[p,:,:].flatten())).numpy().shape)
        
        mu, sigma = model(np.atleast_2d(obs[:,:].flatten()))

        mu = 2*np.pi*mu

        if mu < 0:
            mu = 2*np.pi + mu

        action[0] = tf.random.normal([1], mean=mu, stddev=sigma)

        step_loss = loss_fn(model, obs[:,:], action[0], env.step_reward)

    grads = tape.gradient(step_loss, model.trainable_variables)
    obs, reward = env._step(action)
    return obs, reward, grads


def play_multiple_episodes_multi(env, n_episodes, n_max_steps, model_list, loss_fn):
    all_rewards = []
    all_grads   = []

    for episode in tqdm(range(n_episodes), desc="Episode progress:"):
        current_rewards = []
        current_grads = []
        obs = env._reset()
        for step in range(n_max_steps):
            obs, reward, grads = play_one_step_multi(env, obs, model_list, loss_fn)
            current_rewards.append(reward)
            current_grads.append(grads)
            if env._episode_ended:
                break
        all_rewards.append(current_rewards)
        all_grads.append(current_grads)
    return all_rewards, all_grads

def play_multiple_episodes_single(env, init_action, n_episodes, n_max_steps, model, loss_fn):
    all_rewards = []
    all_grads = []
    for episode in tqdm(range(n_episodes), desc="Episode progress:"):
        current_rewards = []
        current_grads = []
        obs = env._reset()
        for step in range(n_max_steps):
            obs, reward, grads = play_one_step_single(env, obs, init_action, model, loss_fn)
            current_rewards.append(reward)
            current_grads.append(grads)
            if env._episode_ended:
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

def user_loss_fn(model, obs, action, reward):
    
    mu, sigma = model(np.atleast_2d(obs.flatten()))

    mu = 2*np.pi*mu

    if mu < 0:
        mu = 2*np.pi + mu

    # print(f"Mu : {mu}, Sigma : {sigma}")

    l = 1/ (sigma * tf.sqrt(2*np.pi)) * tf.exp(-0.5 * ((action - mu)/sigma)**2)

    # if l < 0:
    #     L = -tf.math.log(abs(l) + 1e-5) # 1e-5 is the floor value for prob
    # else:
    L = tf.math.log(l + 1e-5)

    # print(l, L)

    loss = -reward * L

    return loss



if __name__ == "__main__":
    """
    Currently implemented to train a model from one slime in a field with others:
    
    Other slimes have a random action space, but the one "chosen" slime has a brain
    and will learn that it is rewarded for following other slimes but penalized
    if it gets too close to too many.

    This model will then be saved and given to every slime and then they will each
    have brains. Once this works, scale up to train individual brains for each slime
    which is much more computationally expensive.
    """

    parser = argparse.ArgumentParser(description="Initialize workflow for slime agents")

    parser.add_argument(
        "--file", type=str, help="Input file for slime environment parameters"
    )

    parser.add_argument(
        "--train_iters", type=int, help="Number of training iterations"
    )

    parser.add_argument(
        "--n_episodes", type=int, help="Number of episodes to simulate per training iteration"
    )

    parser.add_argument(
        "--dprobe", type=int, help="Number of cells in single direction passed as observation space to neural network",
        default=10
    )

    parser.add_argument(
        "--activation", type=str, help="Type of activation function to use for output nodes",
        choices=["relu", "sigmoid", "softmax", "softplus", "softsign", "tanh", "selu", "elu", "exponential"],
        default="relu"
    )

    parser.add_argument(
        "--hidden_layers", type=int, help="Number of hidden layers in neural network architecture",
        default=1
    )

    parser.add_argument(
        "--hidden_arch", nargs="+", default=[10],
        help="List describing the number of neurons for each hidden layer. Creates dense layers with keras"
    )

    parser.add_argument(
        "-T", "--train_only", help="Do training only (debugging purposes)", action="store_true"
    )


    args = parser.parse_args()

    # Initialize slime environment
    Nslime = 600
    tmax   = 40
    dt     = 0.01

    nx     = 500
    ny     = 250

    xmax  = 2
    ymax  = 1
    decay = 0.001
    vel   = 0.08
    k     = 5e-5

    alpha = 10 # Reward weight times average value
    thresh = 40 * 3 / 81 # Need to play around with these

    n_iterations = args.train_iters
    n_episodes_per_update = args.n_episodes
    n_max_steps = 400
    discount_factor = 0.95

    env = SlimeSpace(seed=70, nx=nx, ny=ny, Nslime=Nslime, dprobe=args.dprobe,
                     xmax=xmax, ymax=ymax, decay=decay, vel=vel,
                     tmax=tmax, dt=dt, k=k, alpha=alpha, thresh=thresh,
                     single_slime=True)

    optimizer = keras.optimizers.Adam(learning_rate=0.01)
    loss_fn = user_loss_fn
    
    # SINGLE AGENT SETUP
    model = construct_network(args.hidden_layers, args.hidden_arch, (2*args.dprobe)**2, args.activation)

    # Initialize an action array for slimes without brain
    # action[0] denotes the slime with the brain being trained
    action = np.random.random(Nslime) * 2 * np.pi
    
    print(f"\n\n\n[TRAINING] Initializing training iterations\n")

    for iteration in range(n_iterations):
        print(f"[TRAINING] Iteration : {str(iteration).zfill(5)}")
        all_rewards, all_grads = play_multiple_episodes_single(env, action, n_episodes_per_update, n_max_steps, model, loss_fn)
        # all_final_rewards = discount_and_normalize_rewards(all_rewards, discount_factor)
        all_final_rewards = discount_and_normalize_rewards(all_rewards, discount_factor)

        all_mean_grads = []
        for var_index in range(len(model.trainable_variables)):
            mean_grads = tf.reduce_mean(
                [final_reward * all_grads[episode_index][step][var_index] 
                for episode_index, final_rewards in enumerate(all_final_rewards)
                    for step, final_reward in enumerate(final_rewards)], axis=0
            )

        optimizer.apply_gradients(zip(all_mean_grads, model.trainable_variables))

    print(f"\n\n[TRAINING] Finished training model")

    if args.train_only != True:
        print(f"\n[TESTING] Beginning analysis run to create visualizations...")

        # Run it with each slime having a brain NOT for training, but for visualization:
        env = SlimeSpace(seed=70, nx=nx, ny=ny, Nslime=Nslime, dprobe=args.dprobe,
                        xmax=xmax, ymax=ymax, decay=decay, vel=vel,
                        tmax=tmax, dt=dt, k=k, alpha=alpha, thresh=thresh,
                        single_slime=False)

        plot_iter = 0

        for iter in tqdm(range(int(tmax/dt)), desc="Slime Iteration:"):
            
            if iter > 50:
                for i in range(len(action)):
                    mu, sigma = model(np.atleast_2d(env.step_obs[i,:,:].flatten()))
                    mu = 2*np.pi*mu
                    action[i] = tf.random.normal([1], mean=mu, stddev=sigma)

            env._step(action)

            if iter % 50 == 0:
                env._Visualize(plot_iter)
                plot_iter += 1
    else:
        pass

    # Initializing the array of slime brains --> MULTI AGENT SETUP
    # model_list = []

    # print(2*dprobe, 2*dprobe)

    # for n in range(Nslime):
    #     model_list.append(construct_network(1, [10], (2*dprobe)**2, "sigmoid"))

    # optimizer = keras.optimizers.Adam(learning_rate=0.01)
    # loss_fn = user_loss_fn

    # for iteration in range(n_iterations):
    #     print(f"[STARTING] Iteration : {str(iteration).zfill(5)}")
    #     all_rewards, all_grads = play_multiple_episodes_multi(env, n_episodes_per_update, n_max_steps, model_list, loss_fn)
    #     all_final_rewards = discount_and_normalize_rewards(all_rewards, discount_factor)

    #     for p,model in enumerate(model_list):
    #         print(f"Grads: {len(all_grads)}")
    #         print(f"Trainable Variables: {model.trainable_variables}")
    #         optimizer.apply_gradients(zip(all_grads[p], model.trainable_variables))