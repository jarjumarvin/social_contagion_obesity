import os
import networkx as nx
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from networkx.algorithms.community import LFR_benchmark_graph

dirname = os.path.dirname(__file__)

obesityRateFemale = { "15-24": 0.03, "25-34": 0.057, "35-44": 0.091, "45-54": 0.121, "55-64": 0.153, "65-74": 0.141, "75+": 0.125 }
obesityRateMale = { "15-24": 0.051, "25-34": 0.092, "35-44": 0.11, "45-54": 0.144, "55-64": 0.174, "65-74": 0.177, "75+": 0.117 }

xk = np.arange(4) # ages 15-99 inclusive.
pk = (14.67, 47., 16.39, 21.94) # 15-24, 25-54, 55-65, 65+
pk_norm = tuple(p/sum(pk) for p in pk)
ageDistribution = stats.rv_discrete(name="swissAge", values=(xk, pk_norm))

class Agent:
    def __init__(self, ID, age, sex, obese):
        self.ID = ID
        self.age = age
        self.sex = sex # 0 for male, 1 for female
        self.obese = obese
        self.tempState = obese
    
    def __str__(self):
        return "%d\t%d\t%d\t%s" % (self.ID, self.age, self.sex, self.obese)

    def act(self, G):
        print(G[self.ID])

def ageFromGroup(group):
    if group == 0:
        return np.random.randint(15, 24)
    elif group == 1:
        return np.random.randint(25, 54)
    elif group == 2:
        return np.random.randint(55, 64)
    elif group == 3:
        return np.random.randint(64, 99)


def createAgents(n):
    agents = []
    mid = int((n + 1) / 2)
    for i in range(0, n): # create n agents
        ageGroup = ageDistribution.rvs() # random age group according to the probability distribution
        age = ageFromGroup(ageGroup) # random age from within that age group

        sex = 0 if i % 2 == 0 else 1 # 50% male / female

        # set obesity state (y or n)
        rates = obesityRateFemale if sex == 0 else obesityRateMale
        eps = np.random.rand()
        state = False
        if age < 25 and eps < rates["15-24"]: state = True
        elif age in range(25, 35) and eps < rates["25-34"]: state = True
        elif age in range(35, 45) and eps < rates["35-44"]: state = True
        elif age in range(45, 55) and eps < rates["45-54"]: state = True
        elif age in range(55, 65) and eps < rates["55-64"]: state = True
        elif age in range(65, 75) and eps < rates["65-74"]: state = True
        elif age > 74 and eps < rates["75+"]: state = True

        agents.append(Agent(i, age, sex, state))
    return agents

def createNetwork(n):
    agents = createAgents(n)
    maxDeg = 30
    aveDeg = 10
    gamma = 2
    beta = 1.1
    mu = 0.25
    G = LFR_benchmark_graph(n, gamma, beta, mu, average_degree=aveDeg, max_degree=maxDeg)
    G.remove_edges_from(nx.selfloop_edges(G))
    
    nx.write_edgelist(G, os.path.join(dirname, 'graph.csv'), data=False)

n = 300
createNetwork(n)
# N = 1000 # number of agents
# agents = createPopulation(N)
# G_ = nx.barabasi_albert_graph(N, 2, 0)
# G = nx.Graph()

# for agent in agents:
#     G.add_node(agent.ID, data=agent)

# G.add_edges_from(G_.edges)

# for neighbour in G[3]:
#     print(G.nodes[neighbour]['data'])

# nx.write_edgelist(G, "./Assignments/Sem3/ABM/project/src/test.csv", data=False)