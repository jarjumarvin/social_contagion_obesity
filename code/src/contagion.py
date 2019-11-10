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

"""
    Generates a .gexf file. This graph contains all nodes, edges and a special column indicating the state of obesity.
"""
def exportNetwork(G, name):
    G_ = deepcopy(G)
    agents = nx.get_node_attributes(G_, 'data')

    for agent in agents:
        agents[agent] = 1 if agents[agent].obese else 0
    
    nx.set_node_attributes(G_, agents, 'data')
    nx.write_gexf(G_, os.path.join(dirname, "graph/" ,name + ".gexf"))

def obesityRateNetwork(G):
    n = len(G.nodes)
    count = 0
    for agent in G.nodes:
        if G.nodes[agent]['data'].obese:
            count += 1
    return count / n
"""
    Creates a network using the LFR
"""
def createNetwork(agents):
    aveDeg = 15
    maxDeg = 40
    gamma = 3
    beta = 2
    mu = 0.25

    G = nx.Graph()

    for agent in agents:
        G.add_node(agent.ID, data=agent)
    
    G_ = LFR_benchmark_graph(len(agents), gamma, beta, mu, max_degree=maxDeg, average_degree=aveDeg, max_iters=1000)
    G.add_edges_from(G_.edges)

    G.remove_edges_from(nx.selfloop_edges(G))
    return G

def step(G, rate_transmission, rate_recovery, rate_spontaneous):
    for agent in G.nodes:
        G.nodes[agent]['data'].interact(G, rate_transmission, rate_recovery, rate_spontaneous)

    for agent in G.nodes:
        G.nodes[agent]['data'].update()

def simulate(n = 300, num_timesteps = 25, rate_transmission=0.005, rate_recovery=0.049, rate_spontaneous=0.01):
    agents = createSwissAgents(n)
    G = createNetwork(agents)

    exportNetwork(G, "start")
    print('initial obesity rate: %f' % obesityRateNetwork(G))
    for i in range(num_timesteps):
        step(G, rate_transmission, rate_recovery, rate_spontaneous)

    exportNetwork(G, "end")
    print('obesity rate after %d timesteps (years): %f' % (num_timesteps, obesityRateNetwork(G)))

def main():
    simulate()

if __name__== "__main__":
    main()