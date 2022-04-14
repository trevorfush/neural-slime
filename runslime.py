import numpy as np
from numba import njit
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import time

class Agent(object):
    
    def __init__(self, pos0, ang0, vel, dt, p_stray, gridargs):
        
        self.x           = pos0[0]
        self.y           = pos0[1]
        
        self.old_x       = self.x
        self.old_y       = self.y
        
        self.v           = vel
        self.dt          = dt
        
        self.theta       = ang0
        
        self.gridargs    = gridargs
        self.xmax        = gridargs["xmax"]
        self.xmin        = gridargs["xmin"]
        self.ymax        = gridargs["ymax"]
        self.ymin        = gridargs["ymin"]
        self.nx          = gridargs["nx"]
        self.ny          = gridargs["ny"]
        
        self.dx = (self.xmax - self.xmin)/self.nx
        self.dy = (self.ymax - self.ymin)/self.ny

        self.p  = p_stray
        
        self.ind = self.getGridLoc(self.x, self.y)[0]
        
        # Sensing parameters
        self.SA          = 22.5 * np.pi / 180 # Sensor angle
        self.SO          = self.dx * 7        # Sensor offset
        self.RA          = 45 * np.pi / 180   # Agent rotation angle
        
        
    def getGridLoc(self, x, y):
        
        ix = int(x/self.dx)
        iy = int(y/self.dy)
        
        # Periodic boundary conditions
        if (ix >= self.nx+2):
            ix = 0
            x = 0.0

        if (iy >= self.ny+2):
            iy = 0
            y = 0.0

        if (ix < 0):
            ix = self.nx+1
            x = self.nx * self.dx
        if (iy < 0):
            iy = self.ny+1
            y = self.ny * self.dy
            
        return (iy, ix), x, y
        
    def Move(self, grid):
        
        new_x = self.x + self.v * self.dt * np.cos(self.theta)
        new_y = self.y + self.v * self.dt * np.sin(self.theta)
        
        self.old_x = self.x
        self.old_y = self.y
        
        
        self.ind, new_x, new_y     = self.getGridLoc(new_x, new_y)
        self.old_ind, old_x, old_y = self.getGridLoc(self.old_x, self.old_y)
            
        self.x = new_x
        self.y = new_y
        
        grid[self.ind] = 1.0
            
        if np.random.random() < self.p:
            
            self.theta = np.random.random() * 2 * np.pi
            
    def Sense(self, grid):
        
        F_x = self.x + self.SO * np.cos(self.theta)
        F_y = self.y + self.SO * np.sin(self.theta)
        
        FL_x = self.x + self.SO * np.cos(self.theta - self.SA)
        FL_y = self.y + self.SO * np.sin(self.theta - self.SA)
        
        FR_x = self.x + self.SO * np.cos(self.theta + self.SA)
        FR_y = self.y + self.SO * np.sin(self.theta + self.SA)
        
        F_ind = self.getGridLoc(F_x, F_y)[0]
        FL_ind = self.getGridLoc(FL_x, FL_y)[0]
        FR_ind = self.getGridLoc(FR_x, FR_y)[0]
        
        F = grid[F_ind]
        FL = grid[FL_ind]
        FR = grid[FR_ind]
        C  = grid[self.getGridLoc(self.x, self.y)[0]]
        
        if (F > FL) and (F > FR):
            self.theta = self.theta
        elif (F < FL) and (F < FR):
            if np.random.random() < 0.5:
                self.theta -= self.RA
            else:
                self.theta += self.RA
        elif (FL < FR):
            self.theta += self.RA
        elif (FR < FL):
            self.theta -= self.RA
        else:
            self.theta = self.theta

def genGrid(gridargs, simargs):
    
    xmax = gridargs["xmax"]
    xmin = gridargs["xmin"]
    ymax = gridargs["ymax"]
    ymin = gridargs["ymin"]
    nx   = gridargs["nx"]
    ny   = gridargs["ny"]
    Nagent = gridargs["Nagent"]

    initstyle = gridargs["init"]
    rad       = gridargs["radius"]
    
    dt = simargs["dt"]
    
    dx = (xmax - xmin)/nx
    dy = (ymax - ymin)/ny
    
    grid = np.zeros((ny + 2, nx + 2), dtype=np.float64)
    diffgrid = np.ones((ny + 2, nx + 2), dtype=np.float64)
    
    grid[:,0] = 2
    grid[:,-1] = 2
    grid[0,:] = 2
    grid[-1,:] = 2
    
    diffgrid[:,0] = 0
    diffgrid[:,-1] = 0
    diffgrid[0,:] = 0
    diffgrid[-1,:] = 0
    
    Agents = []
    
    for i in range(Nagent):
        

        if initstyle == "circle":
            
            r = np.random.random() * rad
            theta = np.random.random() * 2 * np.pi

            x0 = r * np.cos(theta) + (xmax-xmin)/2
            y0 = r * np.sin(theta) + (ymax-ymin)/2

            # ang0 = np.pi/2 - np.arctan(y0/x0)
            ang0 = np.random.random() * 2 * np.pi

            pos0 = [x0,y0]

        else:

            x0 = np.random.uniform(xmin+dx, xmax)
            y0 = np.random.uniform(ymin+dy, ymax)

            # x0 = xmax / 2
            # y0 = ymax / 2
            ang0 = np.random.random() * 2 * np.pi
            
            pos0 = [x0, y0]
        
        # Choosing velocity of agents : could be same or distribution
        
        if simargs["vel_dist"] == "normal":
            vel = np.random.normal(simargs["v_mean"], simargs["sigma_v"])
        elif simargs["vel_dist"] == "constant":
            vel = simargs["v_mean"]
        
        Agents.append(Agent(pos0, ang0, vel = vel, dt = dt, p_stray=simargs["p_stray"], gridargs=gridargs))
        
        grid[Agents[i].ind] = 1.0
        
    return Agents, grid, diffgrid

@njit()
def diffuse(grid, diffgrid, k, dt, dx):
    
    new_array = np.zeros_like(grid)

    a_ip1_j = grid[2:,1:-1]
    b_ip1_j = diffgrid[2:,1:-1]
    
    a_i_jp1 = grid[1:-1,2:]
    b_i_jp1 = diffgrid[1:-1,2:]
    
    a_im1_j = grid[:-2,1:-1]
    b_im1_j = diffgrid[:-2,1:-1]
    
    a_i_jm1 = grid[1:-1,:-2]
    b_i_jm1 = diffgrid[1:-1,:-2]
    
    a_i_j = grid[1:-1,1:-1]
    b_i_j = diffgrid[1:-1,1:-1]

    new_array[1:-1,1:-1] = k * dt/dx**2 * (b_ip1_j * a_ip1_j + b_i_jp1 * a_i_jp1 + b_im1_j * a_im1_j + b_i_jm1 * a_i_jm1 - 4 * b_i_j * a_i_j)  + b_i_j * a_i_j
    
    return new_array[1:-1,1:-1]

@njit()
def decay(grid, decay_const):
    
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if grid[i,j] > decay_const:
                
                grid[i,j] -= decay_const
            
            else:
                
                grid[i,j] = 0.0
    
    return grid[1:-1,1:-1]

def plotGrid(grid, iters, cmap="viridis", cmap_on=True):
    
    fig = plt.figure()
    
    plt.imshow(grid, cmap=cmap, interpolation="spline36")
    if cmap_on == True:
        plt.colorbar()

    fig.tight_layout()

    plt.axis("off")
    # plt.show()
    plt.savefig(f"slime_test_{str(iters).zfill(4)}.png", dpi=300, bbox_inches="tight")
    print(f"[UPDATE] : Saved plot to 'slime_test_{str(iters).zfill(4)}.png'")
    plt.close()

def runSim(simargs, gridargs):
    
    tmax = simargs["tmax"]
    dt   = simargs["dt"]
    cmap = simargs["cmap"]
    cmap_on = simargs["cmap_on"]
    save_freq = simargs["save_freq"]
    
    decay_const = simargs["decay_const"]

    k = gridargs["diff_coeff"]
    xmax = gridargs["xmax"]
    xmin = gridargs["xmin"]
    nx = gridargs["nx"]


    agents, grid, diffgrid = genGrid(gridargs, simargs)
    
    initgrid = np.copy(grid)

    dx = (xmax-xmin)/nx
    
    t = 0.0
    iters = 0
    plotiters = 0
    
    while t < tmax:

        grid[0,:]  = grid[-2,:]
        grid[-1,:] = grid[1,:]
        grid[:,0]  = grid[:,-2]
        grid[:,-1] = grid[:,1]
        
        grid[1:-1,1:-1] = decay(grid, decay_const)
        grid[1:-1,1:-1] = diffuse(grid, diffgrid, k, dt, dx)
        
        for agent in agents:
            agent.Move(grid)
            agent.Sense(grid)
        
        if simargs["save_plot"] == True:
            if iters % save_freq == 0:
                plotGrid(grid, plotiters, cmap=cmap, cmap_on=cmap_on)
                plotiters += 1
                
        t += dt
        iters += 1
        
    return initgrid, grid

if __name__ == "__main__":

    gridargs = {
                "xmax"       : 2.0, 
                "xmin"       : 0.0, 
                "ymax"       : 1.0, 
                "ymin"       : 0.0, 
                "nx"         : 800, 
                "ny"         : 400, 
                "Nagent"     : 10000, 
                "diff_coeff" : 5e-6,
                "init"       : "circle",
                "radius"     : 0.003 * 20
                }

    simargs  = {
                "tmax"        : 40.0, 
                "dt"          : 0.01, 
                "cmap"        : "cividis", 
                "cmap_on"     : False, 
                "decay_const" : .001, 
                "vel_dist"    : "normal", 
                "v_mean"      : 0.08, 
                "sigma_v"     : 0.02, 
                "save_plot"   : True, 
                "save_freq"   : 200,
                "p_stray"     : 1e-4
                }

    start = time.time()

    print(f"[STARTING] Running simulation\n")
    initgrid, sol = runSim(simargs, gridargs)

    end = time.time()
    print(f"\n[COMPLETE] Simulation complete")
    print(f"[RUNTIME] {end-start} sec. ({(end-start)/60} min.)")

