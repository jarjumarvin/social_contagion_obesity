import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt

def kmcInfection(G, u, infectedList, susceptibleList, adjSusceptibleList):
    """Infection process.

    Args:
        G:                  Interaction graph.
        u:                  Node that recovers.
        infectedList:       List of infected nodes.
        susceptibleList:    List of suscpetible nodes.
        adjSusceptibleList: List of susceptible nodes adjacent to infected ones.
    
    Returns: 
        infectedList:       List of infected nodes.
        susceptibleList:    List of suscpetible nodes.
        adjSusceptibleList: List of susceptible nodes adjacent to infected ones.
    """
    
    # remove node "u" from "susceptibleList" and add it to "infectedList"
    susceptibleList.remove(u)
    infectedList.append(u)
     
    # set state of node "u" to infected
    G.node[u]['infected'] = 1
        
    # update "adjSusceptibleList"
    for v in G.neighbors(u):
        if G.node[v]['infected'] == 1:
            adjSusceptibleList.remove(u) 
        if G.node[v]['infected'] == 0:
            adjSusceptibleList.append(v)
            
    return infectedList, susceptibleList, adjSusceptibleList
 
def kmcRecovery(G, u, infectedList, susceptibleList, adjSusceptibleList):
    """Recovery process.

    Args:
        G:                  Interaction graph.
        u:                  Node that recovers.
        infectedList:       List of infected nodes.
        susceptibleList:    List of suscpetible nodes.
        adjSusceptibleList: List of susceptible nodes adjacent to infected ones.
    
    Returns: 
        infectedList:       List of infected nodes.
        susceptibleList:    List of suscpetible nodes.
        adjSusceptibleList: List of susceptible nodes adjacent to infected ones.
    
    """
     
    # remove node "u" from "infectedList" and add it to "susceptibleList"
    infectedList.remove(u)
    susceptibleList.append(u)
    
    # set state of node "u" to susceptible
    G.node[u]['infected'] = 0 
    
    # update "adjSusceptibleList"
    for v in G.neighbors(u):
        if G.node[v]['infected'] == 0:
            adjSusceptibleList.remove(v)   
        if G.node[v]['infected'] == 1:
            adjSusceptibleList.append(u)

    return infectedList, susceptibleList, adjSusceptibleList


def kmcInitialize(G, rho):
    """Initialize kinetic Monte Carlo lists.

    Args:
        G:   Interaction graph.
        rho: Initial fraction of infected.

    Returns: 
        infectedList:       List of infected nodes.
        susceptibleList:    List of suscpetible nodes.
        adjSusceptibleList: List of susceptible nodes adjacent to infected ones.
        
    """
    # lists of infected and susceptible nodes
    infectedList = []
    susceptibleList = []
    
    # adjacent susceptible list
    adjSusceptibleList = []
    
    for u in G.nodes():
        if np.random.rand() < rho:
            G.node[u]['infected'] = 1
            infectedList.append(u)
        else:
            susceptibleList.append(u)
    
    for u in G.nodes():
        if G.node[u]['infected'] == 1:
            for v in G.neighbors(u):
                if G.node[v]['infected'] == 0:
                    adjSusceptibleList.append(v)
     
    return infectedList, susceptibleList, adjSusceptibleList
    

def kmcSIS(G, T, lamb, rho):
    """SIS kinetic Monte Carlo.

    Args:
        G:     Interaction graph.
        T:     Simulation time period.
        lamb:  Effective spreading rate.
        rho:   Initial fraction of infected.
        
    Returns:
        t:     Time array.
        S:     Susceptible array.
        I:     Infected array.

    """
    
    # initialize SIS dynamics
    infectedList, susceptibleList, adjSusceptibleList = kmcInitialize(G, rho)
    
    t_tot = 0

    # t, S, and I arrays
    t = []
    S = []
    I = []
    
    t.append(t_tot)
    S.append(len(susceptibleList))
    I.append(len(infectedList))

    #pos = nx.circular_layout(G) #dict( (n, n) for n in G.nodes() )

    while(t_tot < T):

        #plt.figure()     
        #nx.draw(G, pos, edge_color="Grey", cmap=plt.get_cmap('RdYlGn_r'), node_color=[G.node[x]['infected'] for x in G.nodes()], vmin=0, vmax=1.0)
        # adjust the plot limits
        #cut = 1.14
        #xmax= cut*max(xx for xx,yy in pos.values())
        #ymax= cut*max(yy for xx,yy in pos.values())   
        #plt.xlim([-xmax, xmax])
        #plt.ylim([-ymax, ymax])
        #plt.savefig('imgs/%04d.png'%len(t), dpi = 300)    
        #plt.close('all')
        
        # total infection rate    
        Q1 = lamb*len(adjSusceptibleList)
        
        # total recovery rate
        Q2 = len(infectedList)
   
        R = Q1 + Q2
                
        if R == 0:
            break
        
        # infection
        if np.random.random() < Q1/(Q1+Q2):
            rand_state = random.randint(0, len(adjSusceptibleList)-1)
            infectedList, susceptibleList, adjSusceptibleList = kmcInfection(G, adjSusceptibleList[rand_state], infectedList, susceptibleList, adjSusceptibleList)
            
        # recovery
        else:
            rand_state = random.randint(0, len(infectedList)-1)
            infectedList, susceptibleList, adjSusceptibleList = kmcRecovery(G, infectedList[rand_state], infectedList, susceptibleList, adjSusceptibleList)

        delta_t = -np.log(1-random.random())/R
        t_tot += delta_t
        
        # update lists
        t.append(t_tot)
        S.append(len(susceptibleList))
        I.append(len(infectedList))
                        
    return np.asarray(t), np.asarray(S), np.asarray(I)

if __name__ == "__main__":
        
    T = 100         # simulation time period
    n = 1000         # number of nodes
    
    rho = 0.05      # initial infected fraction
    lamb = 0.5      # effective spreading rate
    
    # interaction network
    G = nx.watts_strogatz_graph(n, 4, 0.5)#nx.grid_2d_graph(int(n**0.5), int(n**0.5), periodic = True)
    nx.set_node_attributes(G, 0, 'infected')    
    
    # simulate SIS dynamics with kinetic Monte Carlo
    t, S, I = kmcSIS(G, T, lamb, rho)
    
    plt.figure()
    plt.plot(t, S/len(G), linewidth = 1.5, label = r'$S$')
    plt.plot(t, I/len(G), linewidth = 1.5, label = r'$I$')
    plt.xlabel(r'Time')
    plt.ylabel(r'Fraction')
    plt.legend(frameon = False)
    plt.tight_layout()
    plt.show()