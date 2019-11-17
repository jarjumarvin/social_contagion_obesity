import numpy as np
import random
import matplotlib.pyplot as plt


if __name__ == "__main__":
        
    T = 200          # simulation time period
    p = 0.53         # occupation probability
    
    lattice = np.zeros((T+1,int(1.5*T)))    
    
    active = []    
    lattice[0,len(lattice[0,:])//2] = 1
    active.append(len(lattice[0,:])//2)    
    
    for t in range(T):
        
        active_new = []

        # iterate over all active sites from previous time step
        for i in range(len(active)):
            
            # activate lower left neighbor
            if random.random() < p:
                lattice[t+1,active[i]-1] = 1
                active_new.append(active[i]-1)
             
            # activate lower right neighbor
            if random.random() < p:
                lattice[t+1,active[i]+1] = 1
                active_new.append(active[i]+1)
        
        if len(active_new) == 0:
            break
        
        active = active_new

    plt.figure()
    plt.imshow(lattice, interpolation='None', cmap=plt.get_cmap('binary'))
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.show()