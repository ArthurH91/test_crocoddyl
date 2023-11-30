import matplotlib.pyplot as plt
import numpy as np

def plot_costs(rd):    
    
    ### Construction of the dictionnary
    costs_dict = {}
    for name in rd[0].differential.costs.costs.todict():
        costs_dict[name] = []
    for name in costs_dict:
        for data in rd:
            costs_dict[name].append(data.differential.costs.costs[name].cost)

    ### Plotting
    
    for name_cost in costs_dict:
        if "col" in name_cost:
            plt.plot(costs_dict[name_cost], "-o" ,label = name_cost, markersize= 3)
        else:
            plt.plot(costs_dict[name_cost], "o" ,label = name_cost, markersize=3 )
    plt.xlabel("Nodes")
    plt.ylabel("Cost (log)")
    plt.legend()
    plt.yscale("log")
    plt.show()
