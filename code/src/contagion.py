import os
import networkx as nx
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from copy import deepcopy

from networkx.algorithms.community import LFR_benchmark_graph

dirname = os.path.dirname(__file__)
class Agent:
    """ 
    Class for individual agents in the network
    
    Attributes:
        - ID    : unique number for every agent
        - age   : age of the agent (from 15 to 99)
        - sex   : 'm' for male, 'f' for female
        - obese : True for obese, False for not obese
    """

    def __init__(self, ID, age, sex, obese):
        self.ID = ID
        self.age = age
        self.sex = sex
        self.obese = obese
        self.next_state = obese
    
    def __str__(self):
        return '%d\t%d\t%s\t%s' % (self.ID, self.age, self.sex, self.obese)

    def interact(self, G, rate_transmission, rate_recovery, rate_spontaneous):
        # loop through neighbours, set own obesity probabilty
        deg = G.degree(self.ID)
        eps = np.random.rand()
        if(self.obese):
            if(eps < rate_recovery):
                self.next_state = False
        else:
            p = rate_spontaneous
            for agent in G.neighbors(self.ID):
                neighbor = G.nodes[agent]['data']
                if neighbor.obese:
                    p += (rate_transmission / deg)

            if(eps < p):
                self.next_state = True
    
    def update(self):
        self.obese = self.next_state
"""
    Returns a uniformly distributed age within a given age group
"""
def ageFromGroup(group):
    if group == 0:
        return np.random.randint(15, 24)
    elif group == 1:
        return np.random.randint(25, 54)
    elif group == 2:
        return np.random.randint(55, 64)
    elif group == 3:
        return np.random.randint(64, 99)

def createSwissAgents(n):
    """
        Returns a list of n Agent instances whose attributes are distributed according to the Swiss age distribution
        and the Swiss obesity rates according to age and gender:

        Age Distribution:
        [15-24] : 14.67%
        [25-54] : 47.00%
        [55-65] : 16.39%
        [65+]   : 21.94%
        
        Obesity Rates:
        Age         Male        Female
        [15-24]     5.1%        3%
        [25-34]     9.2%        5.7%
        [35-44]     11%         9.1%
        [45-54]     14.4%       12.1%
        [55-64]     17.4%       15.3%
        [65-74]     17.7%       14.1%
        [75+]       11.7%       12.5%
    """
    # obesityRateMale = { '15-24': 0.051, '25-34': 0.092, '35-44': 0.11, '45-54': 0.144, '55-64': 0.174, '65-74': 0.177, '75+': 0.117 }
    # obesityRateFemale = { '15-24': 0.03, '25-34': 0.057, '35-44': 0.091, '45-54': 0.121, '55-64': 0.153, '65-74': 0.141, '75+': 0.125 }

    obesityRateMale = { '15-24': 0.011, '25-34': 0.038, '35-44': 0.053, '45-54': 0.088, '55-64': 0.107, '65-74': 0.094, '75+': 0.064 }
    obesityRateFemale = { '15-24': 0.07, '25-34': 0.024, '35-44': 0.043, '45-54': 0.054, '55-64': 0.087, '65-74': 0.083, '75+': 0.071 }

    # Create discrete age probability distribution
    pk = (14.67, 47., 16.39, 21.94)
    pk_norm = tuple(p / sum(pk) for p in pk)
    ageDistribution = stats.rv_discrete(values=(np.arange(4), pk_norm))

    agents = []

    # create n agents
    for i in range(0, n):
        
        # age sampled from the age distribution
        age = ageFromGroup(ageDistribution.rvs()) 
        # 50% male / 50% female
        sex = 'm' if i % 2 == 0 else 'f' 
        
        #set obesity according to obesity rates
        rates = obesityRateFemale if sex == 'f' else obesityRateMale
        eps = np.random.rand()
        state = False

        if age < 25 and eps < rates['15-24']: state = True
        elif age in range(25, 35) and eps < rates['25-34']: state = True
        elif age in range(35, 45) and eps < rates['35-44']: state = True
        elif age in range(45, 55) and eps < rates['45-54']: state = True
        elif age in range(55, 65) and eps < rates['55-64']: state = True
        elif age in range(65, 75) and eps < rates['65-74']: state = True
        elif age > 74 and eps < rates['75+']: state = True

        agents.append(Agent(i, age, sex, state))
    return agents

def exportNetwork(G, name):
    """
        Generates a .gexf file. This graph contains all nodes, edges and a special column indicating the state of obesity.
    """
    G_ = deepcopy(G)
    agents = nx.get_node_attributes(G_, 'data')

    for agent in agents:
        agents[agent] = 1 if agents[agent].obese else 0
    
    nx.set_node_attributes(G_, agents, 'data')
    nx.write_gexf(G_, os.path.join(dirname, "graph/" ,name + ".gexf"))

def obesityRateNetwork(G):
    """
        Returns the obesity rate of a given network. Element of [0, 1]
    """
    n = len(G.nodes)
    count = 0
    for agent in G.nodes:
        if G.nodes[agent]['data'].obese:
            count += 1
    return count / n


def createNetwork(agents):
    """
        Creates a network using the LFR generator
    """
    n = len(agents)
    tau1 = 3
    tau2 = 1.5
    mu = 0.1

    G = nx.Graph()

    for agent in agents:
        G.add_node(agent.ID, data=agent)
    
    G_ = LFR_benchmark_graph(n, tau1, tau2, mu, average_degree=5,min_community=20)    
    
    G.add_edges_from(G_.edges)

    G.remove_edges_from(nx.selfloop_edges(G))
    return G

def kmcInfection(G, u, infectedList, susceptibleList, adjSusceptibleList):
    # update lists
    susceptibleList.remove(u)
    infectedList.append(u)
     
    # set state of u to "obese"
    G.nodes[u]['data'].obese = True

    # update "adjSusceptibleList"
    for v in G.neighbors(u):
        if G.node[v]['data'].obese:
            adjSusceptibleList.remove(u) 
        else:
            adjSusceptibleList.append(v)
    
    # return lists
    return infectedList, susceptibleList, adjSusceptibleList

def kmcRecovery(G, u, infectedList, susceptibleList, adjSusceptibleList):
    # update lists
    infectedList.remove(u)
    susceptibleList.append(u)
    
    # set state of node "u" to susceptible
    G.node[u]['data'].obese = False
    
    # update "adjSusceptibleList"
    for v in G.neighbors(u):
        if G.node[v]['data'].obese:
            adjSusceptibleList.append(u)   
        else:
            adjSusceptibleList.remove(v)

    return infectedList, susceptibleList, adjSusceptibleList

def kmcInitialise(G):
    """
        takes graph G and returns lists
    """
    infectedList = []
    susceptibleList = []

    adjSusceptibleList = []
    
    for u in G.nodes():
        if G.node[u]['data'].obese:
            infectedList.append(u)
            for v in G.neighbors(u):
                if not G.node[v]['data'].obese:
                    adjSusceptibleList.append(v)
        else:
            susceptibleList.append(u)

    return infectedList, susceptibleList, adjSusceptibleList

def kmcSISa(G, T, beta, gamma, alpha):
    infectedList, susceptibleList, adjSusceptibleList = kmcInitialise(G)
    t_tot = 0

    t = []
    S = []
    I = []

    t.append(t_tot)
    S.append(len(susceptibleList))
    I.append(len(infectedList))

    while(t_tot < T):
        # total transmissionary infection rate
        Q1 = beta * len(adjSusceptibleList)
        # total recovery rate
        Q2 = gamma * len(infectedList)
        # total spontaneous infection rate
        Q3 = alpha * len(susceptibleList)
        # overall rates
        Q = Q1 + Q2 + Q3 

        if Q == 0: break

        eps = np.random.rand() * Q

        a = len(susceptibleList)
        b = len(infectedList)
        c = len(adjSusceptibleList)

        if 0 <= eps < Q1 and len(adjSusceptibleList) > 0: # transmissionary infection
            n = np.random.randint(0, len(adjSusceptibleList))
            infectedList, susceptibleList, adjSusceptibleList = kmcInfection(G, adjSusceptibleList[n], infectedList, susceptibleList, adjSusceptibleList)
        elif Q1 <= eps < Q1 + Q2 and len(susceptibleList) > 0: # spontaneous infection
            n = np.random.randint(0, len(susceptibleList))
            infectedList, susceptibleList, adjSusceptibleList = kmcInfection(G, susceptibleList[n], infectedList, susceptibleList, adjSusceptibleList)
        elif Q1 + Q2 <= eps < Q and len(infectedList) > 0: # recovery
            n = np.random.randint(0, len(infectedList))
            infectedList, susceptibleList, adjSusceptibleList = kmcRecovery(G, infectedList[n], infectedList, susceptibleList, adjSusceptibleList)

        dt = -np.log(1 - np.random.random()) / Q
        t_tot += dt

        t.append(t_tot)
        S.append(len(susceptibleList))
        I.append(len(infectedList))
    return np.asarray(t), np.asarray(S), np.asarray(I)

def main():
    G = nx.read_graphml(os.path.join(dirname, 'graph/contagion.graphml'), node_type=int)
    n = 1005
    agents = createSwissAgents(n)
    for node in G.nodes:
        G.nodes[node]['data'] = agents[node]

    t, S, I = kmcSISa(G, 1000, 0.05, 0.04, 0.2)

    plt.figure()
    plt.plot(t, S/len(G), linewidth = 1.5, label = r'$S$')
    plt.plot(t, I/len(G), linewidth = 1.5, label = r'$I$')
    plt.xlabel(r'Time')
    plt.ylabel(r'Fraction')
    plt.legend(frameon = False)
    plt.tight_layout()
    plt.show()
    # simulate()
if __name__== "__main__":
    main()