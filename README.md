# neural-slime
Physarum (slime) mold transport networks using reinforcement learning with neural networks


https://user-images.githubusercontent.com/54543048/164559351-dbe9c85d-21ce-43bb-8283-13a16803aa14.mp4


<p align="center">
<embed src="https://user-images.githubusercontent.com/54543048/164559351-dbe9c85d-21ce-43bb-8283-13a16803aa14.mp4" width="500" />
</p>

## Running the code
1. To run the code, there are various settings that need to be included on the command line:

  | Flag | Purpose |
  |------|---------|
  | `-I`, `--input` | Name of the simulation type as outlined in the slimesetup.yml file (see **Implementing New Simulations** section below|
  | `-O`, `--output` | Name of directory where results from simulation should be saved to |
  | `--train_iters` | Number of training iterations |
  | `--n_episodes` | Number of episodes per training iteration (one episode is one complete simulation) |
  | `--dprobe` | Number of cells in a single direction passed as observation space to neural network (2 dprobe x 2 dprobe observation square) |
  | `--activation` | Activation function to use on the output neurons ("relu", "sigmoid", "softmax", "softplus", "softsign", "tanh", "selu", "elu", "exponential") |
  | `--savefreq` | Iteration frequency to save images and/or numpy arrays of state |
  | `-N`, `--numpy` | When set on command line, saves numpy arrays of state with frequency set by `--savefreq` |
  | `-P`, `--plot` | When set on command line, saves plots of state with frequency set by `--savefreq` |
  | `-T`, `--train_only` | Do training only (no testing simulation, for debugging purposes) |
  
2. Running `python main.py ...` with desired settings will run the simulation

## Implementing New Simulations
### New Neural Network Architectures

To implement new neural network architectures, the function `construct_network` in `main.py` can be changed to use any desired network architecture, as long as the keras model is being returned by the function.

### New Simulation Setups

To implement a new simulation setup, in the `slimesetup.yml` file, a new set of parameters can be implemented following the outline of the `standard_test` simulation provided for the CMSE 890 final project. All variables set in the `standard_test` setting need to be set in future simulation settings.

# Plotting

To change the visualization returned by the slime environment, edit the `_Visualize` method in the `envclass.py` script. The file `posterplots.ipynb` contains a routine for generating plots in my poster presentation.
 
  
