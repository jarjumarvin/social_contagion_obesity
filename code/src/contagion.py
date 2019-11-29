import os
import networkx as nx
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from copy import deepcopy
from timeit import default_timer as timer
from datetime import datetime

from networkx.algorithms.community import LFR_benchmark_graph

dirname = os.path.dirname(__file__)

np.random.seed(12) # 7 - 0.052, 12 - 0.056

class Agent:
    """
    Class for individual agents in the network. Each agent represents one person.
    
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
        """
            Takes the graph G and the rates of transmission, recovery and spontaneous. 

            If the agent is not obese, he recovers with probability `rate_recovery`.

            Otherwise, the agent observes the states of all his neighbours, and increases the probability of becoming obese
            `rate_spontaneous` by `rate_transmission` divided by the amount of neighbours.
            
            Both of these cases lead to next_state being set to either true or false.
        """
        deg = G.degree(self.ID)

        eps = np.random.rand()
        mu = np.random.normal(1, 0.25)
    
        if(self.obese):
            if(eps < mu * rate_recovery):
                self.next_state = False
        else:
            p = mu * rate_spontaneous

            for agent in G.neighbors(self.ID):
                neighbor = G.nodes[agent]['data']
                if neighbor.obese:
                    delta = np.random.normal(1, 0.25)
                    p += delta * (rate_transmission / deg)

            if(eps < p):
                self.next_state = True
    
    def update(self):
        """
            simply updates the current state to the state determined in interact()
        """
        self.obese = self.next_state

def ageFromGroup(group):
    """
        Returns a uniformly distributed age within a given age group.

        Age groups:
        0: 15-24
        1: 25-54
        2: 55-65
        3: 65+
    """
    if group == 0:
        return np.random.randint(15, 24)
    elif group == 1:
        return np.random.randint(25, 54)
    elif group == 2:
        return np.random.randint(55, 64)
    elif group == 3:
        return np.random.randint(64, 99)

def createSwissAgents1992(n):
    obesityRateMale1992 = { '15-24': 0.011, '25-34': 0.038, '35-44': 0.053, '45-54': 0.088, '55-64': 0.107, '65-74': 0.094, '75+': 0.064 }
    obesityRateFemale1992 = { '15-24': 0.07, '25-34': 0.024, '35-44': 0.043, '45-54': 0.054, '55-64': 0.087, '65-74': 0.083, '75+': 0.071 }

    return createSwissAgents(n, [obesityRateMale1992, obesityRateFemale1992])

def createSwissAgents2017(n):
    obesityRateMale2017 = { '15-24': 0.051, '25-34': 0.092, '35-44': 0.11, '45-54': 0.144, '55-64': 0.174, '65-74': 0.177, '75+': 0.117 }
    obesityRateFemale2017 = { '15-24': 0.03, '25-34': 0.057, '35-44': 0.091, '45-54': 0.121, '55-64': 0.153, '65-74': 0.141, '75+': 0.125 }

    return createSwissAgents(n, [obesityRateMale2017, obesityRateFemale2017])

def createSwissAgents(n, obesity_rates):
    """
        Returns a list of n Agent instances whose attributes are distributed according to the Swiss age distribution
        and the Swiss obesity rates according to age and sex:

        Age Distribution (2017):
        [15-24] : 14.67%
        [25-54] : 47.00%
        [55-65] : 16.39%
        [65+]   : 21.94%
        
        Obesity Rates (1992 / 2017):
        Age         Male        Female
        [15-24]     5.1%        3%
        [25-34]     9.2%        5.7%
        [35-44]     11%         9.1%
        [45-54]     14.4%       12.1%
        [55-64]     17.4%       15.3%
        [65-74]     17.7%       14.1%
        [75+]       11.7%       12.5%

        Obesity Rates (1992):
        Age         Male        Female
        [15-24]     1.1%        7%
        [25-34]     3.8%        2.4%
        [35-44]     5.3%        4.3%
        [45-54]     8.8%        5.4%
        [55-64]     10.7%       8.7%
        [65-74]     9.4%        8.3%
        [75+]       6.4%        7.1%
    """

    # Create discrete age probability distribution
    pk = (14.67, 47., 16.39, 21.94)
    pk_norm = tuple(p / sum(pk) for p in pk)
    ageDistribution = stats.rv_discrete(values=(np.arange(4), pk_norm))

    # list of agents
    agents = []

    # create n agents
    for i in range(0, n):
        # age sampled from the age distribution
        age = ageFromGroup(ageDistribution.rvs()) 
        # 50% male / 50% female
        sex = 'm' if i % 2 == 0 else 'f' 
        #pick obesity rates depening on sex
        rates = obesity_rates[0] if sex == 'm' else obesity_rates[1]
        ## rates = obesityRateFemale2017 if sex == 'f' else obesityRateMale2017

        # set obesity of agent according to distribution
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

def createNetwork(agents):
    """
        Creates a network using the LFR Benchmark Algorithm (to be improved?)
    """
    aveDeg = 10
    minCom = 20
    gamma = 3
    beta = 1.5
    mu = 0.2

    G = nx.Graph()

    for agent in agents:
        G.add_node(agent.ID, data=agent)
    
    G_ = LFR_benchmark_graph(len(agents), gamma, beta, mu, min_community=minCom, average_degree=aveDeg, max_iters=1000, seed=2)

    G.add_edges_from(G_.edges)

    G.remove_edges_from(nx.selfloop_edges(G))
    return G

def obesityRateNetwork(G):
    """
        Returns the rate of obesity in the network.
    """
    n = len(G.nodes)
    count = 0
    for agent in G.nodes:
        if G.nodes[agent]['data'].obese:
            count += 1
    return count / n

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

def step(G, rate_transmission, rate_recovery, rate_spontaneous):
    """
        Takes a graph G, the transmission, recovery an spontaneous infection rates and does one timestep
    """
    for agent in G.nodes:
        G.nodes[agent]['data'].interact(G, rate_transmission, rate_recovery, rate_spontaneous)
    for agent in G.nodes:
        G.nodes[agent]['data'].update()

def simulate(G, num_timesteps = 25, rate_transmission=0.005, rate_recovery=0.04, rate_spontaneous=0.02):
    """
        takes a graph G, and performs `num_timesteps` timesteps on it given the rates
        for each iteration, save the rate of obesity
    """
    rates = []
    rates.append(obesityRateNetwork(G))
    for i in range(num_timesteps):
        step(G, rate_transmission, rate_recovery, rate_spontaneous)
        rates.append(obesityRateNetwork(G))
    
    return rates

def plotParameterDependenceAndDoRegression(size=12, n=500, recovery=(0.001, 0.06), spontaneous=(0.001, 0.04), timesteps=25, iterations=5):
    """
        produces a parameter dependence plot (rate of recovery, rate of spontaneous infection)
        does (`size` * `size`) * `iterations` simulations of `timesteps` timesteps
        after each `iterations` iterations, it normalises the resulting final rate of obesity

        then performs a 'linear regression' on these two parameters given the rates above
        finds a recovery rate and spontaneous rate such that the sum of the square differences between
        simulated and actual rates in 1992, 1997, 2002, 2007, 2012, 2017 is minimal.

        plots the parameter dependence plot and the time evolution using the found best fitting rates

        returns G (initial Graph) and the best fitting recovery and spontaneous rate
    """

    print("starting parameter dependence and regression plots")
    start = timer()

    rate_recovery_range = np.linspace(recovery[0], recovery[1], size)
    rate_spontaneous_range = np.linspace(spontaneous[0], spontaneous[1], size)

    # generate population
    agents = createSwissAgents1992(n)
    G = createNetwork(agents)
    init = obesityRateNetwork(G)

    finalRate = np.zeros((size, size))
    ratesByTimeStep = np.zeros((size, size, timesteps + 1))
    for i in range(size):
        print(i)
        for j in range(size):
            rate_spontaneous = rate_spontaneous_range[i]
            rate_recovery = rate_recovery_range[j]
            sum = 0

            ratesInd = np.zeros(timesteps + 1) # rates for every timestep, normalised over 5 iterations
            for k in range(iterations):
                agents_copy = deepcopy(agents)
                G_copy = deepcopy(G)
                res = simulate(G_copy, timesteps, rate_transmission=0.005, rate_recovery=rate_recovery, rate_spontaneous=rate_spontaneous)

                ratesInd += np.array(res)
                sum += obesityRateNetwork(G_copy)

            ratesInd /= iterations # rates for this tuple of parameters normalised
            ratesByTimeStep[i][j] = ratesInd
            finalRate[i][j] = sum / iterations

    # rates for 1997, 2002, 2007, 2012, 2017
    # iterations 5, 10, 15, 20 and 25
    # find i, j such that sum over above iterations of |observed rate - actual rates| is minimal
    norms = np.zeros((size, size))
    years = [1992, 1997, 2002, 2007, 2012, 2017]
    yearsFull = list(range(1992, 2018))
    ratesYear = [0.054, 0.068, 0.077, 0.086, 0.112, 0.123]

    # generate square norms
    for i in range(size):
        for j in range(size):
            for k in range(1, 6): # 1,2,3,4,5
                residual = ratesByTimeStep[i][j][k * 5] - ratesYear[k]
                residual *= residual
                norms[i][j] = residual
    
    bestFitRecovery = 1 # best fitting recovery rate
    indexRecovery = 0 # index in range_recovery_rate
    bestFitSpontaneous = 1 # best fitting spontaneous rate
    indexSpontaneous = 0 # index in range_spontaneous_rate

    currentMin = 1000000 # current minimal norm

    for i in range(size): # find minimal rates an indexes
        for j in range(size):
            if norms[i][j] < currentMin:
                currentMin = norms[i][j]
                bestFitSpontaneous = rate_spontaneous_range[i]
                bestFitRecovery = rate_recovery_range[j]
                indexSpontaneous = i
                indexRecovery = j

    # norme rates for 1992, 1997, 2002, 2007, 2012, 2017
    condensedTimeStepRates = []
    for i in range(6):
        condensedTimeStepRates.append(ratesByTimeStep[indexSpontaneous][indexRecovery][i * 5])

    end = timer()
    print('time elapsed:', end - start, 'seconds')

    # plots
    ## parameter depenence
    plt.figure(figsize = (16, 6))
    plt.subplot(1, 2, 1)
    levels = np.linspace(0,1, 20)
    contour = plt.contour(rate_recovery_range, rate_spontaneous_range, finalRate, levels, colors='k')
    plt.clabel(contour, colors = 'k', fmt = '%1.3f', fontsize=16)
    contour_filled = plt.contourf(rate_recovery_range, rate_spontaneous_range, finalRate, levels)
    plt.colorbar(contour_filled)
    plt.title('obesity after %d years, initial rate %1.3f' % (timesteps, init), fontsize = 'xx-large')
    plt.xlabel('recovery', fontsize = 'xx-large')
    plt.ylabel('spontaneous infection', fontsize = 'xx-large')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    ## time evolution using least squares
    plt.subplot(1, 2, 2)
    plt.title('time evolution using least squares solution\nrecovery rate %1.6f and spontaneous rate %1.6f' % (bestFitRecovery, bestFitSpontaneous), fontsize = 'xx-large')
    plt.plot(years, ratesYear, '-o', label='real data')
    plt.plot(yearsFull, ratesByTimeStep[indexSpontaneous][indexRecovery], '-^', label='simulation')
    plt.ylabel('rate of obesity', fontsize = 'xx-large')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend()

    now = datetime.now()
    date_time = now.strftime("%Y%m%d-%H:%M:%S")
    plt.savefig(os.path.join(dirname, 'plots/','%s-n-%d-size-%d-init-%1.3f-steps-%d-iter-%d.png' % (date_time, n, size, init, timesteps, iterations)))

    plt.show()

    return G, bestFitRecovery, bestFitSpontaneous

def produceClosestGraph(G, timesteps, recovery, spontaneous, k = 15):
    """
        saves the given graph G in graph/normedBegin

        does `k` 25-timestep iterations on G using the given rates

        saves the graph thats closest to the average in final obesity rate to the average in graph/optnormedStopimalStop
    """
    graphs = []
    obesities = []

    avg = 0
    for i in range(k):
        G_copy = deepcopy(G)
        simulate(G_copy, timesteps, rate_transmission=0.005, rate_recovery=recovery, rate_spontaneous=spontaneous)
        graphs.append(G_copy)
        obesities.append(obesityRateNetwork(G_copy))
        avg += obesityRateNetwork(G_copy)

    avg /= k

    closest = 0
    for i in range(k):
        r = obesities[i] - avg
        c = obesities[closest] - avg
        if (r * r) < (c * c):
            closest = i

    exportNetwork(G, "normedBegin")
    exportNetwork(graphs[closest], "normedStop")


def main():
    G, bestFitRecovery, bestFitSpontaneous = plotParameterDependenceAndDoRegression(size=25)
    produceClosestGraph(G, 25, bestFitRecovery, bestFitSpontaneous, 15)

if __name__== "__main__":
    main()