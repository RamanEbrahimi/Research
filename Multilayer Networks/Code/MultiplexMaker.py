import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

class MultiplexNetwork:
    def __init__(self, num_nodes, num_layers, graph_methods=None):
        """
        Creates a multiplex network with given number of nodes and layers.

        :param num_nodes: Number of nodes in each layer.
        :param num_layers: Number of layers in the multiplex network.
        :param graph_methods: A list of graph generation methods (one per layer). 
                              Supported: "ER", "BA", "WS".
        """
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        self.graph_methods = graph_methods or ["ER"] * num_layers  # Default: ER graphs
        self.multiplex = self.create_multiplex_network()

    def create_multiplex_network(self):
        """Creates a multiplex network with the specified graph generation methods."""
        layers = {}
        for i, method in enumerate(self.graph_methods):
            if method == "ER":
                G = nx.erdos_renyi_graph(self.num_nodes, 0.1)
            elif method == "BA":
                G = nx.barabasi_albert_graph(self.num_nodes, 3)
            elif method == "WS":
                G = nx.watts_strogatz_graph(self.num_nodes, 4, 0.1)
            else:
                raise ValueError(f"Unsupported graph generation method: {method}")

            # Assign random edge weights between 0.1 and 1.0
            for u, v in G.edges():
                G[u][v]["weight"] = np.random.uniform(0.1, 1.0)

            layers[f"layer_{i+1}"] = G
        
        return layers

    def plot_layer(self, layer_index):
        """
        Plots a single layer of the multiplex network.
        
        :param layer_index: Index of the layer (1-based).
        """
        layer_name = f"layer_{layer_index}"
        if layer_name not in self.multiplex:
            raise ValueError(f"Layer {layer_index} does not exist.")

        G = self.multiplex[layer_name]
        pos = nx.spring_layout(G)
        plt.figure(figsize=(6, 6))
        nx.draw(G, pos, with_labels=True, node_size=300, node_color="lightblue", edge_color="gray")
        plt.title(f"Multiplex Network - {layer_name}")
        plt.show()

    def plot_all_layers(self):
        """Plots all layers in the multiplex network."""
        num_cols = min(3, self.num_layers)
        num_rows = (self.num_layers + num_cols - 1) // num_cols  # Arrange in rows
        
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows))
        axes = axes.flatten() if self.num_layers > 1 else [axes]

        for i, (layer_name, G) in enumerate(self.multiplex.items()):
            ax = axes[i]
            pos = nx.spring_layout(G)
            nx.draw(G, pos, ax=ax, with_labels=True, node_size=300, node_color="lightblue", edge_color="gray")
            ax.set_title(layer_name)

        for j in range(i+1, len(axes)):  # Hide unused subplots
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()

# # Create a multiplex network with 100 nodes and 3 layers (ER, BA, WS)
# multiplex = MultiplexNetwork(num_nodes=100, num_layers=3, graph_methods=["ER", "BA", "WS"])

# # Plot a single layer (e.g., layer 1)
# multiplex.plot_layer(layer_index=1)

# # Plot all layers
# multiplex.plot_all_layers()
