import tensorflow as tf
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from tensorflow import keras
from tqdm import tqdm
import argparse
import yaml
import os
import sys

from envclass import SlimeSpace

#------------------------------------------------------------------#
#                                                                  #
#   Implementing actor-only policy gradient method for training    #
#   neural networks. Continuous action space based on paper        #
#   cited in proposal.                                             #
#                                                                  #
#------------------------------------------------------------------#


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# Adapted from https://github.com/woutervanheeswijk/example_continuous_control
def construct_network(input_layers, output_activation_func):

    input = keras.layers.Input(shape=(input_layers,))
    
    # arch1
    # hidden = keras.layers.Dense(10)(input)

    # arch2
    hidden1 = keras.layers.Dense(20)(input)
    hidden2 = keras.layers.Dense(15)(hidden1)
    hidden3 = keras.layers.Dense(10)(hidden2)
    hidden4 = keras.layers.Dense(5)(hidden3)

    mu = keras.layers.Dense(1, activation=output_activation_func)(hidden4)

    sigma = keras.layers.Dense(1, activation=output_activation_func)(hidden4)

    model = keras.Model(inputs=input, outputs=[mu, sigma])

    return model

def play_one_step_single(env, obs, action, model, loss_fn):
    with tf.GradientTape() as tape:
        
        mu, sigma = model(np.atleast_2d(obs[:,:].flatten()))

        mu = 2*np.pi*mu

        if mu < 0:
            mu = 2*np.pi + mu

        action[0] = tf.random.normal([1], mean=mu, stddev=sigma)

        step_loss = loss_fn(model, obs[:,:], action[0], env.step_reward)

    grads = tape.gradient(step_loss, model.trainable_variables)
    obs, reward = env._step(action)
    return obs, reward, grads

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

    l = 1/ (sigma * tf.sqrt(2*np.pi)) * tf.exp(-0.5 * ((action - mu)/sigma)**2)

    L = tf.math.log(l + 1e-5)

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
        "--input", "-I", type=str, help="Name of the simulation type from slimesetup.yml file"
    )

    parser.add_argument(
        "--output", "-O", type=str, help="Output directory for images and saved arrays if applicable"
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
        "--savefreq", type=int, help="Iteration frequency to save images or numpy arrays of the state",
        default=100
    )

    parser.add_argument(
        "-N", "--numpy", action="store_true", help="Save numpy arrays at provided frequency"
    )

    parser.add_argument(
        "-P", "--plot", action='store_true', help="Save plots of state at provided frequency"
    )

    parser.add_argument(
        "-T", "--train_only", help="Do training only (debugging purposes)", action="store_true"
    )


    args = parser.parse_args()

    # Initialize slime environment

    with open("slimesetup.yml", "r") as f:
        config = yaml.safe_load(f)
        f.close()

    if args.input != None:
        simparams = config[args.input]
    else:
        print(f"Please provide an input simulation with --input or -I !!!")
        sys.exit(0)
        
    Nslime = int(simparams["Nslime"])
    tmax   = float(simparams["tmax"])
    dt     = float(simparams["dt"])

    nx     = int(simparams["nx"])
    ny     = int(simparams["ny"])

    xmax  = float(simparams["xmax"])
    ymax  = float(simparams["ymax"])
    decay = float(simparams["decay"])
    vel   = float(simparams["vel"])
    k     = float(simparams["k"])

    alpha = float(simparams["alpha"]) # Reward weight times average value
    thresh = float(simparams["thresh"]) # Need to play around with these

    n_iterations = args.train_iters
    n_episodes_per_update = args.n_episodes
    n_max_steps = int(tmax/dt)-1
    discount_factor = float(simparams["discount_factor"])

    # print(n_max_steps)

    if args.output != None:
        if os.path.isdir(args.output):
            os.chdir(args.output)
        else:
            os.mkdir(args.output)
            os.chdir(args.output)

    # for key in simparams.keys():
    #     print(f"Key : {key}, Value : {simparams[key]}")

    env = SlimeSpace(seed=70, nx=nx, ny=ny, Nslime=Nslime, dprobe=args.dprobe,
                     xmax=xmax, ymax=ymax, decay=decay, vel=vel,
                     tmax=tmax, dt=dt, k=k, alpha=alpha, thresh=thresh,
                     single_slime=True)

    optimizer = keras.optimizers.Adam(learning_rate=0.01)
    loss_fn = user_loss_fn
    
    # SINGLE AGENT SETUP
    model = construct_network((2*args.dprobe)**2, args.activation)

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

        print(all_final_rewards)

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

            if iter % args.savefreq == 0:

                if args.numpy:
                    # print(env._state[env.ilo+1:env.ihi,env.jlo+1:env.jhi])
                    np.save(f"STATE{str(plot_iter).zfill(4)}", env._state[env.ilo+1:env.ihi,env.jlo+1:env.jhi])
                
                if args.plot:
                    env._Visualize(plot_iter)

                plot_iter += 1
    else:
        pass