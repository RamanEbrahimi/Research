import networkx as nx
import random
import matplotlib.pyplot as plt
import numpy as np

class MultiplexDiffusion:
    def __init__(self, num_nodes, num_layers):
        """
        Initializes the multiplex diffusion object.
        
        Parameters:
            num_nodes (int): Number of nodes in each layer.
            num_layers (int): Number of layers in the multiplex network.
        """
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        self.layers = []  # List of layers (each a dict with keys: "graph", "model", "param", etc.)
        self.diffusion_sequence = None  # Stores the most recent diffusion sequence.
    
    def create_network(self, layer_models):
        """
        Creates each layer as an Erdős–Rényi (ER) graph with p in [0.5, 1] and assigns a diffusion model.
        
        Parameters:
            layer_models (list): A list (length=num_layers) where each element is a tuple:
                - For IC: ("IC", p_share) where p_share is the sharing probability.
                - For LT: ("LT", None) (thresholds are assigned randomly).
        """
        if len(layer_models) != self.num_layers:
            raise ValueError("The number of layer model specifications must equal num_layers.")
            
        self.layers = []  # Reset layers if needed
        
        for i in range(self.num_layers):
            # Random edge probability for the ER graph (p ∈ [0.5, 1])
            p_edge = random.uniform(0.5, 1)
            G = nx.erdos_renyi_graph(self.num_nodes, p_edge)
            
            model, param = layer_models[i]
            layer_info = {"graph": G, "model": model, "param": param}
            
            if model == "LT":
                # For LT model, assign each node a random threshold in [0, 1]
                thresholds = {node: random.random() for node in G.nodes()}
                layer_info["thresholds"] = thresholds
            
            self.layers.append(layer_info)
    
    def run_diffusion(self, k):
        """
        Runs the diffusion process over the multiplex network.
        
        The process:
          - Starts with a random seed set of k nodes.
          - At each time step, only nodes activated in the previous step attempt to activate neighbors.
          - For IC: each attempt is successful with probability p_share.
          - For LT: an inactive node is activated if the fraction of its neighbors (activated in the previous step)
            meets or exceeds its threshold.
          - Continues until no new activations occur.
        
        Parameters:
            k (int): Size of the initial seed set.
            
        Returns:
            List[set]: A list where each element is the set of nodes newly activated at that time step.
        """
        if k > self.num_nodes:
            raise ValueError("k cannot be larger than the total number of nodes.")
        initial_active = set(random.sample(range(self.num_nodes), k))
        diffusion_sequence = [initial_active]
        activated_global = set(initial_active)
        new_active = initial_active
        
        while new_active:
            step_new_active = set()
            for layer in self.layers:
                G = layer["graph"]
                if layer["model"] == "IC":
                    p_share = layer["param"]
                    for node in new_active:
                        for neighbor in G.neighbors(node):
                            if neighbor not in activated_global:
                                if random.random() < p_share:
                                    step_new_active.add(neighbor)
                elif layer["model"] == "LT":
                    thresholds = layer["thresholds"]
                    for node in G.nodes():
                        if node not in activated_global:
                            new_neighbors = set(G.neighbors(node)).intersection(new_active)
                            if new_neighbors:
                                deg = G.degree(node)
                                if deg > 0:
                                    influence = len(new_neighbors) / deg
                                    if influence >= thresholds[node]:
                                        step_new_active.add(node)
                else:
                    raise ValueError(f"Unknown diffusion model: {layer['model']}")
            
            new_active = step_new_active - activated_global
            if new_active:
                diffusion_sequence.append(new_active)
                activated_global.update(new_active)
        
        self.diffusion_sequence = diffusion_sequence
        return diffusion_sequence

    def plot_layers(self, time_step, diffusion_sequence=None):
        """
        Plots each layer at a given diffusion time step.
        Activated nodes (up to the specified time step) are shown in red; inactive nodes in blue.
        
        Parameters:
            time_step (int): Time step to visualize (union of activated nodes from step 0 to time_step).
            diffusion_sequence (list of set, optional): If not provided, uses the stored diffusion_sequence.
        """
        if diffusion_sequence is None:
            if self.diffusion_sequence is None:
                raise ValueError("No diffusion sequence available. Run run_diffusion() first or provide one.")
            diffusion_sequence = self.diffusion_sequence
        
        activated_union = set()
        steps = min(time_step + 1, len(diffusion_sequence))
        for i in range(steps):
            activated_union.update(diffusion_sequence[i])
        
        ncols = min(self.num_layers, 3)
        nrows = (self.num_layers + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
        
        if self.num_layers == 1:
            axes = [axes]
        elif nrows == 1:
            axes = list(axes)
        else:
            axes = axes.flatten()
        
        for i, layer in enumerate(self.layers):
            G = layer["graph"]
            ax = axes[i]
            pos = nx.spring_layout(G)
            node_colors = ['red' if node in activated_union else 'blue' for node in G.nodes()]
            nx.draw_networkx(G, pos=pos, ax=ax, node_color=node_colors, with_labels=True)
            ax.set_title(f"Layer {i} ({layer['model']})")
        
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')
        
        plt.tight_layout()
        plt.show()

    def get_adjacency_matrices(self):
        """
        Extracts the adjacency matrices for each layer.
        
        Returns:
            List[np.array]: A list of adjacency matrices (as NumPy arrays) corresponding to each layer.
        """
        matrices = []
        for layer in self.layers:
            A = nx.to_numpy_array(layer["graph"])
            matrices.append(A)
        return matrices

# # Example usage:
# if __name__ == "__main__":
#     num_nodes = 20
#     num_layers = 2
#     # Define models for each layer.
#     # For example:
#     # Layer 0: IC with sharing probability 0.3,
#     # Layer 1: LT (with random thresholds),
#     # Layer 2: IC with sharing probability 0.5.
#     layer_models = [("IC", 0.1), ("LT", None)]
    
#     md = MultiplexDiffusion(num_nodes, num_layers)
#     md.create_network(layer_models)
    
#     # Run diffusion starting with a random set of 5 nodes.
#     diffusion_steps = md.run_diffusion(2)
#     for t, new_nodes in enumerate(diffusion_steps):
#         print(f"Step {t}: Activated nodes = {new_nodes}")
    
#     # Plot the state of each layer at time step 2 (change as needed)
#     md.plot_layers(time_step=0)
#     md.plot_layers(time_step=1)
    
