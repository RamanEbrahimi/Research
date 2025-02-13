import matplotlib.pyplot as plt
import networkx as nx
import random
from MultiplexMaker import *
from MultiplexDiffusion import *

# Step 1: Create a multiplex network (100 nodes, 3 layers)
multiplex = MultiplexNetwork(num_nodes=100, num_layers=3, graph_methods=["ER", "BA", "WS"])

# Compute degree centrality for layer 1
centrality = nx.degree_centrality(multiplex.multiplex["layer_1"])
sorted_nodes_L1 = sorted(centrality, key=centrality.get, reverse=True)
# Compute degree centrality for layer 2
centrality = nx.degree_centrality(multiplex.multiplex["layer_2"])
sorted_nodes_L2 = sorted(centrality, key=centrality.get, reverse=True)
# Compute degree centrality for layer 3
centrality = nx.degree_centrality(multiplex.multiplex["layer_3"])
sorted_nodes_L3 = sorted(centrality, key=centrality.get, reverse=True)

def run_diffusion_and_plot(seed_sets, title, diffusion_type="LT"):
    """
    Runs diffusion for multiple seed sets and plots the results.

    :param seed_sets: Dictionary with labels as keys and seed node lists as values.
    :param title: Title for the plot.
    :param diffusion_type: "LT" or "IC" for diffusion models.
    """
    plt.figure(figsize=(8, 6))

    for label, seeds in seed_sets.items():
        diffusion = MultiplexDiffusion(multiplex, seeds, diffusion_type=diffusion_type)
        diffusion.run_diffusion()
        plt.plot(diffusion.time_steps, diffusion.informed_fraction, marker="o", label=label)

    plt.xlabel("Time Steps")
    plt.ylabel("Fraction of Informed Nodes")
    plt.title(title)
    plt.legend()
    plt.show()



# Comparing 5 vs. 10 random seeds vs. 5 central seeds**
comparison_seed_sets = {
    "5 Random Seeds": random.sample(range(100), 5),
    "10 Random Seeds": random.sample(range(100), 10),
    "5 Central Seeds - Layer 1": sorted_nodes_L1[:5],
    "5 Central Seeds - Layer 2": sorted_nodes_L2[:5],
    "5 Central Seeds - Layer 3": sorted_nodes_L3[:5],
}
run_diffusion_and_plot(comparison_seed_sets, "Comparison of 5 vs. 10 Random Seeds vs. 5 Central Seeds (IC)", diffusion_type="IC")
