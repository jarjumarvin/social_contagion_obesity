# Agent-Based Modeling and Social System Simulation 2019


> * Group Name: Hufflepuff
> * Group participants names: Chio Ge, Marvin Jarju, Giulia Zheng
> * Project Title: Spread of Obesity in Social Networks

## General Introduction

In the last 25 years, the prevalence of obesity has surged across the globe. As of 2017, 40% of the Swiss population is overweight and another 10% is considered obese, whereby in 1992, the obesity rate was at 5%, which means that within 25 years, Switzerland has experienced a full 100% increase in the fraction of obese people in the population. Rather than viewing obesity as a purely personal or genetic issue, we want to consider obesity as an social contagion, i.e an pseudo-epidemic transmittable via social networks, similiar to an infection or virus that spawns further infections via interpersonal contact.

## The Model

The model we are employing is a modification upon the SIS-model, which is typically used for mathematical modelling of epidemiology in networks. This model is a fitting abstraction for the problem we are studying, since our agents can be in one of two states, "susceptible" and "infectious" similiar to a regular infection. The agents in our model cannot develop immunity and enter a "recovered" state which is why we are not choosing alternative models such as the SIR-model. The SIS-model traditionally takes two variables, the so-called "transmission" and "recovery" rates. We need an additional rate, the "spontaneous" rate, since our agents have the ability to contract obesity by themselves, independent of the states of the agents in their social circle. We will be using the spontaneous and recovery rates as parameters in our simulations to interpolate between the obesity rates of 1992 and 2017 in Switzerland. We will fix the transmission rate at 0.005 as found in similiar studies.

## Fundamental Questions

Is it possible to model the spread of obesity with this simplified model and consider it as an "epidemic"?
If want to model the exact development of obesity in Switzerland from 1992 - 2017, which spontanoeus and recovery rate do we obtain?
What does the distribution of obesity within our social network look like?
With the rates found in our interpolation, what obesity rate would we obtain if we ran the simulation for another 25 years? I.e. what is our predicted obesity rate in 2042?
How can our simple model be expanded upon further in order to obtain an even more accurate result?

## Expected Results

We expect that after runnning our simulations, we will find a formation of clusters in our social network, i.e. that obese agents are concentrated within social groups. We also expect to obtain recovery and spontaneous infection rates that are in the same order of magnitude as those found in similiar studies pertaining the US. Since the US has a larger obesity rate in general, we expect our rates to be slightly lower, though. Further, using the rates we find in our populations, we expect that if we run the simulation for another 25 years, our obesity rate should double again in 2042, i.e. we would obtain a rate of around 20%.

## Reproducibility

See /code

## References 

[1] CIA. "Age Structure of Switzerland 2018". url:https://www.cia.gov/library/publications/the-world-factbook/geos/sz.html.

[2] Bundesamt für Statistik. "Body Mass Index (BMI) nach Geschlecht, Alter, Bildungsniveau, Sprachgebiet - 1992, 1997, 2002, 2007, 2012, 2017: Tabelle". url: https://www.bfs.admin.ch/bfs/de/home/statistiken/gesundheit/determinanten/uebergewicht.assetdetail.6466017.html.

[3] Christakis, Nicholas A. and Fowler, James. "The Spread of Obesity in a Large Social Network over 32 Years". In: New England Journal of Medicine 357.4 (2007). doi: 10.1056/NEJMsa066082.

[4] Hill, Alison L. AND Rand, David G. AND Nowak, Martin A. AND Christakis, Nicholas A. "Infectious Disease Modeling of Social Contagion in Networks". In: PLOS Computational Biology (2010). doi: 10.1371/journal.pcbi.1000968.

[5] Lancichinetti Andrea. "Benchmark graphs for testing community detection algorithms". In: Physical Review E 78.4 (2008). doi: 10.1103/physreve.78.046110.
