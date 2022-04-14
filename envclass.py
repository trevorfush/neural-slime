import tensorflow as tf
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from tensorflow import keras
from tqdm import tqdm


class SlimeSpace():

    def __init__(self, seed, nx, ny, Nslime, dprobe, xmax, ymax, decay, vel, tmax, dt, k, alpha, thresh, single_slime):

        self.single = single_slime # Used for training individual model

        self.nx = nx
        self.ny = ny
        self.xmax = xmax
        self.ymax = ymax 

        self.dprobe = dprobe
        self.Nslime = Nslime
        self.decay = decay
        self.k     = k
        self.cmap = "cividis"
        self.cmap_on = False

        self.v = vel

        self.tmax = tmax
        self.dt = dt
        self.t  = 0.0

        self.dx = xmax/nx
        self.dy = ymax/ny

        self._state = np.zeros((ny+2*self.dprobe, nx+2*self.dprobe), dtype=np.float64)

        # MULTI AGENT LEARNING SETUP
        if self.single == False:
            self.step_reward = np.zeros(self.Nslime)
            self.step_obs    = np.zeros((self.Nslime, 2*self.dprobe, 2*self.dprobe))

        # SINGLE AGENT LEARNING SETUP
        if self.single == True:
            self.step_reward = 0
            self.step_obs    = np.zeros((2*self.dprobe, 2*self.dprobe))

        self.ilo = self.dprobe
        self.ihi = self.ny + self.dprobe
        self.jlo = self.dprobe
        self.jhi = self.nx + self.dprobe

        self.alpha = alpha 
        self.thresh = thresh

        self.new_loc = np.zeros((Nslime,2),dtype=np.float64)

        self._episode_ended = False

        self.seed = seed
        np.random.seed(self.seed)

        self.new_array = np.zeros_like(self._state)

        for i in range(self.Nslime):
            
            x0 = np.random.uniform(self.dx, xmax)
            y0 = np.random.uniform(self.dy, ymax)

            (ix, iy) = self.getGridLoc(x0, y0)[0]
            
            if self._state[ix, iy] != 1.0:
                self._state[ix, iy] = 1.0

                self.new_loc[i][0] = x0
                self.new_loc[i][1] = y0
            else:
                i -= 1

    def getGridLoc(self, x, y):

        ix = int(x/self.dx)
        iy = int(y/self.dy)
        
        # Periodic boundary conditions, keep stuff on grid
        if (ix >= self.nx+1):
            ix = 1
            x = self.dx

        if (iy >= self.ny+1):
            iy = 1
            y = self.dx

        if (ix <= 0):
            ix = self.nx+1
            x = (self.nx) * self.dx
        if (iy <= 0):
            iy = self.ny+1
            y = (self.ny) * self.dy

        return (iy+self.ilo, ix+self.jlo), x, y

    def _reset(self):

        # Reset the necessary thingssss
        self._state = np.zeros((self.ny+2*self.dprobe, self.nx+2*self.dprobe), dtype=np.float64)
        self.new_array[:,:] = 0.0
        self.t = 0.0
        self._episode_ended = False

        # Set random seed
        np.random.seed(self.seed)
        
        # Initialize positions of all slimes
        for i in range(self.Nslime):
            
            x0 = np.random.uniform(self.dx, self.xmax)
            y0 = np.random.uniform(self.dy, self.ymax)

            (ix, iy) = self.getGridLoc(x0, y0)[0]
            
            self._state[ix, iy] = 1.0

            self.new_loc[i][0] = x0
            self.new_loc[i][1] = y0

        # Make initial observations to return
        for i in range(self.Nslime):
            self.getObsReward(i)

        return self.step_obs

    def getObsReward(self, i):
        
        # TO GET THIS BACK TO MULTI AGENT:
        # for step_obs and step_reward,
        # change to step_obs[i,:,:] and step_reward[i]


        # Find where the slime is on the grid
        index_loc = self.getGridLoc(self.new_loc[i][0], self.new_loc[i][1])[0]

        # Get the observation for given slime
        if self.single == True:
            self.step_obs[:,:] = self._state[(index_loc[0]-self.dprobe):(index_loc[0]+self.dprobe),
                                            (index_loc[1]-self.dprobe):(index_loc[1]+self.dprobe)]
            
            # Calculate average pheromone value within sight for reward
            avg_val = self.step_obs[:,:].mean()

            # If it is near others but not too close, give reward ignoring its own trail
            if avg_val < self.thresh and avg_val > (self.dprobe):
                self.step_reward = avg_val * self.alpha
            
            # If it is above threshold, give penalty
            elif avg_val > self.thresh:
                self.step_reward = -avg_val * self.alpha
            
            # If it's just in open space then carry on
            else:
                self.step_reward = 0.1

        else:
            self.step_obs[i,:,:] = self._state[(index_loc[0]-self.dprobe):(index_loc[0]+self.dprobe),
                                            (index_loc[1]-self.dprobe):(index_loc[1]+self.dprobe)]
            
            # Calculate average pheromone value within sight for reward
            avg_val = self.step_obs[i,:,:].mean()

            # If it is near others but not too close, give reward ignoring its own trail
            if avg_val < self.thresh and avg_val > (self.dprobe):
                self.step_reward[i] = avg_val * self.alpha
            
            # If it is above threshold, give penalty
            elif avg_val > self.thresh:
                self.step_reward[i] = -avg_val * self.alpha
            
            # If it's just in open space then carry on
            else:
                self.step_reward[i] = 0.1
        

    def _Move(self, ind, action):

        old_x, old_y = self.new_loc[ind][0], self.new_loc[ind][1]

        # Get new updated slime locations
        new_x = self.new_loc[ind][0] + self.v * self.dt * np.cos(action)
        new_y = self.new_loc[ind][1] + self.v * self.dt * np.sin(action)
        
        # Snap locations to the grid
        self.ind, new_x_bc, new_y_bc     = self.getGridLoc(new_x, new_y)
        self.old_ind, old_x_bc, old_y_bc = self.getGridLoc(old_x, old_y)

        # Make sure agents can't occupy the same state
        if self._state[self.ind] != 1.0:

            self.new_loc[ind][0] = new_x_bc
            self.new_loc[ind][1] = new_y_bc

            self._state[self.ind] = 1.0
            self._state[self.old_ind] = 1.0 - self.decay

        else:

            self.new_loc[ind][0] = old_x
            self.new_loc[ind][1] = old_y
        

    def _Decay(self):

        # Decrease values != 1 by decay constant
        decay_ind = np.where(self._state < 1.0)
        self._state[decay_ind] -= self.decay

        # Reset zero value to be 0
        below_zero = np.where(self._state < 0.0)
        self._state[below_zero] = 0.0

    def _Diffuse(self):
        
        a_ip1_j = self._state[2:,1:-1]
        
        a_i_jp1 = self._state[1:-1,2:]
        
        a_im1_j = self._state[:-2,1:-1]
        
        a_i_jm1 = self._state[1:-1,:-2]
        
        a_i_j = self._state[1:-1,1:-1]

        self.new_array[1:-1,1:-1] = self.k * self.dt/self.dx**2 * (a_ip1_j + a_i_jp1 + a_im1_j + a_i_jm1 - 4 * a_i_j)  + a_i_j

        return self.new_array[1:-1,1:-1]

    def _step(self, action):

        # Check if the episode has ended
        if self._episode_ended:
            return self._reset()
        
        # Boundaries
        self._state[:,:self.jlo] = self._state[:,(self.jhi-self.dprobe):self.jhi]
        self._state[:,self.jhi:] = self._state[:,self.jlo:(2*self.jlo)]
        self._state[:self.ilo,:] = self._state[(self.ihi-self.dprobe):self.ihi,:]
        self._state[self.ihi:,:] = self._state[self.ilo:(2*self.ilo),:]

        # Only need to loop over agents once :)
        for i in range(self.Nslime):

            # Move the slimes according to the action, create trails
            self._Move(i, action[i])

            # Get rewards and observations
            # Maybe make a separate function for this... and play around with the type of reward
            # i = 0 is designed to be the agent with the neural network in single agent learning
            if self.single == True:
                if i == 0:
                    self.getObsReward(i)
            else:
                self.getObsReward(i)

        # Decay the trails
        self._Decay()

        # Diffuse the trails
        self._state[1:-1,1:-1] = self._Diffuse()

        # Update time
        self.t += self.dt

        # Check for completion criteria, return necessary things
        if self.t >= self.tmax:
            self._episode_ended=True
            return self._state
        else:
            return self.step_obs, self.step_reward

    def _Visualize(self, iter):

        fig = plt.figure()
    
        plt.imshow(self._state[self.ilo+1:self.ihi,self.jlo+1:self.jhi], 
                   cmap=self.cmap, interpolation="none")
        if self.cmap_on == True:
            plt.colorbar()

        fig.tight_layout()

        plt.axis("off")
        # plt.show()
        plt.savefig(f"slime_test_{str(iter).zfill(4)}.png", dpi=300, bbox_inches="tight")
        # print(f"[UPDATE] : Saved plot to 'slime_test_{str(iter).zfill(4)}.png'")
        plt.close()

if __name__ == "__main__":

    Nslime = 10
    tmax = 20
    dt = 0.01

    alpha = 10
    thresh = 40 * 3 /81 # Need to play around with these

    penv = SlimeSpace(seed=70, nx=20, ny=20, Nslime=Nslime, dprobe=3, 
                      xmax=2, ymax=1, decay=.001, vel=0.08, 
                      tmax=tmax, dt=dt, k=5e-6, alpha=alpha, thresh=thresh)

    action = np.random.random(Nslime) * 2 * np.pi
    
    plot_iter = 0

    for iter in tqdm(range(int(tmax/dt)), desc="Slime Iteration"):
        penv._step(action)

        if iter % 200 == 0:
            penv._Visualize(plot_iter)
            plot_iter += 1
    