import numpy as np
import matplotlib.pyplot as plt

class MultiplexDiffusion:
    def __init__(self, multiplex_network, seed_nodes, diffusion_type="LT", thresholds=None, activation_probs=None):
        """
        Initializes the diffusion process.

        :param multiplex_network: An instance of the MultiplexNetwork class.
        :param seed_nodes: List of initial seed nodes.
        :param diffusion_type: "LT" for Linear Threshold, "IC" for Independent Cascade.
        :param thresholds: Dictionary specifying node thresholds per layer for LT model.
                           If None, assigns random thresholds.
        :param activation_probs: Dictionary specifying activation probabilities per layer for IC model.
                                 If None, assigns random probabilities.
        """
        self.multiplex = multiplex_network.multiplex  # Dictionary of layers
        self.num_nodes = multiplex_network.num_nodes
        self.num_layers = multiplex_network.num_layers
        self.diffusion_type = diffusion_type
        self.seed_nodes = set(seed_nodes)

        # Active nodes (activated at any point) & Newly activated nodes (current step)
        self.active_nodes = set(seed_nodes)  # Nodes activated so far
        self.newly_activated = set(seed_nodes)  # Nodes activated in the last step

        # Assign thresholds for LT model
        self.thresholds = thresholds or self.generate_random_thresholds() if diffusion_type == "LT" else None

        # Assign activation probabilities for IC model
        self.activation_probs = activation_probs or self.generate_random_activation_probs() if diffusion_type == "IC" else None

        # Stores diffusion results
        self.time_steps = []
        self.informed_fraction = []

    def generate_random_thresholds(self):
        """Generates random thresholds for each node in each layer (for LT model)."""
        return {
            layer: {node: np.random.uniform(0.1, 0.9) for node in range(self.num_nodes)}
            for layer in self.multiplex
        }

    def generate_random_activation_probs(self):
        """Generates random activation probabilities per layer (for IC model)."""
        return {layer: np.random.uniform(0.01, 0.5) for layer in self.multiplex}

    def step_LT(self):
        """Performs one step of the Linear Threshold (LT) diffusion process."""
        newly_activated = set()

        for layer, G in self.multiplex.items():
            for node in G.nodes():
                if node not in self.active_nodes:
                    influence_sum = sum(
                        G[neighbor][node]["weight"]
                        for neighbor in G.neighbors(node)
                        if neighbor in self.newly_activated  # Only influence from last step
                    )
                    if influence_sum >= self.thresholds[layer][node]:
                        newly_activated.add(node)

        # A node activated in any layer is activated in all layers
        self.active_nodes.update(newly_activated)
        return newly_activated

    def step_IC(self):
        """Performs one step of the Independent Cascade (IC) diffusion process."""
        newly_activated = set()

        for layer, G in self.multiplex.items():
            p = self.activation_probs[layer]  # Probability of activation in this layer
            for node in list(self.newly_activated):  # Only nodes activated in the last step
                for neighbor in G.neighbors(node):
                    if neighbor not in self.active_nodes and np.random.rand() < p:
                        newly_activated.add(neighbor)

        # A node activated in any layer is activated in all layers
        self.active_nodes.update(newly_activated)
        return newly_activated

    def run_diffusion(self):
        """Runs the diffusion process step-by-step until no more nodes are activated."""
        while self.newly_activated:  # Continue while new nodes are being activated
            self.time_steps.append(len(self.time_steps))
            self.informed_fraction.append(len(self.active_nodes) / self.num_nodes)

            # Perform diffusion step based on model
            if self.diffusion_type == "LT":
                self.newly_activated = self.step_LT()
            elif self.diffusion_type == "IC":
                self.newly_activated = self.step_IC()
            else:
                raise ValueError("Invalid diffusion type. Choose 'LT' or 'IC'.")

    def plot_diffusion(self):
        """Plots the fraction of informed nodes over time."""
        plt.plot(self.time_steps, self.informed_fraction, marker="o", label=f"{self.diffusion_type} Model")
        plt.xlabel("Time Steps")
        plt.ylabel("Fraction of Informed Nodes")
        plt.title(f"Diffusion Process ({self.diffusion_type} Model)")
        plt.legend()
        plt.show()
