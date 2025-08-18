import networkx as nx
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from gtda.homology import VietorisRipsPersistence
import warnings
import os

# Suppress warnings from giotto-tda for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

class MultiplexNetwork:
    """
    A class to represent and manage multiplex networks with multiple layers.
    """
    
    def __init__(self, num_nodes, num_layers):
        """
        Initialize a multiplex network.
        
        Args:
            num_nodes (int): Number of nodes in the network
            num_layers (int): Number of layers in the multiplex network
        """
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        self.layers = {}
        self.spreading_coefficients = {}
        
    def add_layer(self, layer_id, graph, spreading_coefficient=0.1):
        """
        Add a layer to the multiplex network.
        
        Args:
            layer_id (int): Unique identifier for the layer
            graph (networkx.Graph): NetworkX graph for this layer
            spreading_coefficient (float): Layer-specific spreading coefficient
        """
        self.layers[layer_id] = graph
        self.spreading_coefficients[layer_id] = spreading_coefficient
        
    def get_layer(self, layer_id):
        """Get a specific layer."""
        return self.layers.get(layer_id)
    
    def get_all_nodes(self):
        """Get all unique nodes across all layers."""
        all_nodes = set()
        for layer in self.layers.values():
            all_nodes.update(layer.nodes())
        return sorted(list(all_nodes))

def create_multiplex_network(num_nodes=50, num_layers=3):
    """
    Creates a synthetic multiplex network with different topologies for each layer.
    
    Args:
        num_nodes (int): Number of nodes in the network
        num_layers (int): Number of layers in the multiplex network
        
    Returns:
        MultiplexNetwork: A multiplex network object
    """
    multiplex = MultiplexNetwork(num_nodes, num_layers)
    
    # Create different network topologies for each layer
    network_generators = [
        ("Erdős-Rényi", lambda: nx.erdos_renyi_graph(num_nodes, 0.3, seed=42)),
        ("Barabási-Albert", lambda: nx.barabasi_albert_graph(num_nodes, 3, seed=42)),
        ("Watts-Strogatz", lambda: nx.watts_strogatz_graph(num_nodes, 6, 0.3, seed=42)),
        ("Community", lambda: create_community_graph(num_nodes)),
        ("Scale-Free", lambda: nx.powerlaw_cluster_graph(num_nodes, 3, 0.1, seed=42))
    ]
    
    # Different spreading coefficients for each layer
    spreading_coefficients = [0.15, 0.25, 0.35]  # Different influence strengths per layer
    
    for layer_id in range(num_layers):
        # Select network generator (cycle through them)
        generator_name, generator_func = network_generators[layer_id % len(network_generators)]
        
        # Create the graph
        G = generator_func()
        
        # Ensure all nodes are present (some generators might not include all nodes)
        for i in range(num_nodes):
            if i not in G.nodes():
                G.add_node(i)
        
        # Add random weights to edges
        for (u, v) in G.edges():
            G.edges[u, v]['weight'] = np.random.uniform(0.01, 0.2)
        
        # Add the layer to multiplex network
        spreading_coeff = spreading_coefficients[layer_id % len(spreading_coefficients)]
        multiplex.add_layer(layer_id, G, spreading_coeff)
        
        print(f"Layer {layer_id}: {generator_name} network with {G.number_of_edges()} edges, "
              f"spreading coefficient = {spreading_coeff:.2f}")
    
    return multiplex

def create_community_graph(num_nodes):
    """Helper function to create a community graph."""
    sizes = [num_nodes // 3] * 3
    sizes[-1] += num_nodes % 3
    G = nx.random_partition_graph(sizes, 0.3, 0.05, seed=42)
    
    # Ensure all nodes are present
    for i in range(num_nodes):
        if i not in G.nodes():
            G.add_node(i)
    
    return G

def run_multiplex_ic_model(multiplex, seed_nodes, mc_simulations=100):
    """
    Simulates the Multiplex Independent Cascade (IC) model.
    
    In this model:
    - Each layer has its own spreading coefficient
    - A node activated in any layer is considered active in all layers
    - Influence spreads independently in each layer but activation is global
    
    Args:
        multiplex (MultiplexNetwork): The multiplex network
        seed_nodes (list): Initial set of activated nodes
        mc_simulations (int): Number of Monte Carlo simulations
        
    Returns:
        float: Average fraction of nodes influenced across all simulations
    """
    total_influenced = 0
    all_nodes = multiplex.get_all_nodes()
    num_nodes = len(all_nodes)
    
    for _ in range(mc_simulations):
        # Global activation state (active in any layer = active globally)
        globally_activated = set(seed_nodes)
        
        # Track newly activated nodes per layer
        newly_activated_per_layer = {layer_id: list(seed_nodes) for layer_id in multiplex.layers.keys()}
        
        # Continue until no new activations in any layer
        while any(newly_activated_per_layer.values()):
            for layer_id, layer_graph in multiplex.layers.items():
                if not newly_activated_per_layer[layer_id]:
                    continue
                
                current_node = newly_activated_per_layer[layer_id].pop(0)
                
                # Try to activate neighbors in this layer
                for neighbor in layer_graph.neighbors(current_node):
                    if neighbor not in globally_activated:
                        # Use layer-specific spreading coefficient
                        layer_spreading_coeff = multiplex.spreading_coefficients[layer_id]
                        influence_prob = layer_graph.edges[current_node, neighbor]['weight'] * layer_spreading_coeff
                        
                        if np.random.rand() < influence_prob:
                            globally_activated.add(neighbor)
                            # Add to newly activated for all layers
                            for other_layer_id in multiplex.layers.keys():
                                if neighbor in multiplex.layers[other_layer_id].nodes():
                                    newly_activated_per_layer[other_layer_id].append(neighbor)
        
        total_influenced += len(globally_activated) / num_nodes  # Return fraction instead of count
    
    return total_influenced / mc_simulations

def run_multiplex_ic_model_with_tracking(multiplex, seed_nodes, mc_simulations=100):
    """
    Simulates the Multiplex IC model and tracks the spreading process.
    
    Args:
        multiplex (MultiplexNetwork): The multiplex network
        seed_nodes (list): Initial set of activated nodes
        mc_simulations (int): Number of Monte Carlo simulations
        
    Returns:
        tuple: (average_influence_fraction, spreading_curves) where spreading_curves is a list of 
               lists containing the fraction of activated nodes at each step for each simulation.
    """
    spreading_curves = []
    total_influenced = 0
    all_nodes = multiplex.get_all_nodes()
    num_nodes = len(all_nodes)
    
    for _ in range(mc_simulations):
        globally_activated = set(seed_nodes)
        newly_activated_per_layer = {layer_id: list(seed_nodes) for layer_id in multiplex.layers.keys()}
        curve = [len(globally_activated) / num_nodes]  # Start with initial seeds (fraction)
        
        while any(newly_activated_per_layer.values()):
            for layer_id, layer_graph in multiplex.layers.items():
                if not newly_activated_per_layer[layer_id]:
                    continue
                
                current_node = newly_activated_per_layer[layer_id].pop(0)
                
                for neighbor in layer_graph.neighbors(current_node):
                    if neighbor not in globally_activated:
                        layer_spreading_coeff = multiplex.spreading_coefficients[layer_id]
                        influence_prob = layer_graph.edges[current_node, neighbor]['weight'] * layer_spreading_coeff
                        
                        if np.random.rand() < influence_prob:
                            globally_activated.add(neighbor)
                            for other_layer_id in multiplex.layers.keys():
                                if neighbor in multiplex.layers[other_layer_id].nodes():
                                    newly_activated_per_layer[other_layer_id].append(neighbor)
            
            curve.append(len(globally_activated) / num_nodes)  # Store fraction
        
        spreading_curves.append(curve)
        total_influenced += len(globally_activated) / num_nodes  # Return fraction
    
    return total_influenced / mc_simulations, spreading_curves

def multiplex_greedy_selection(multiplex, k, mc_simulations=100):
    """
    Greedy algorithm for multiplex network influence maximization.
    
    Args:
        multiplex (MultiplexNetwork): The multiplex network
        k (int): Number of seeds to select
        mc_simulations (int): Number of Monte Carlo simulations
        
    Returns:
        tuple: (selected_seeds, final_influence, selection_time)
    """
    start_time = time.time()
    
    seed_set = []
    all_nodes = multiplex.get_all_nodes()
    
    for _ in range(k):
        best_marginal_gain = -1
        best_node = -1
        
        for node in all_nodes:
            if node not in seed_set:
                current_influence = run_multiplex_ic_model(multiplex, seed_set, mc_simulations)
                new_influence = run_multiplex_ic_model(multiplex, seed_set + [node], mc_simulations)
                marginal_gain = new_influence - current_influence
                
                if marginal_gain > best_marginal_gain:
                    best_marginal_gain = marginal_gain
                    best_node = node
        
        seed_set.append(best_node)
    
    final_influence = run_multiplex_ic_model(multiplex, seed_set, mc_simulations)
    end_time = time.time()
    
    return seed_set, final_influence, end_time - start_time

def multiplex_tda_selection(multiplex, k):
    """
    TDA-based seed selection for multiplex networks.
    
    This approach:
    1. Computes persistence diagrams for each layer
    2. Combines topological information across layers
    3. Scores nodes based on their topological importance across all layers
    
    Args:
        multiplex (MultiplexNetwork): The multiplex network
        k (int): Number of seeds to select
        
    Returns:
        tuple: (selected_seeds, selection_time)
    """
    start_time = time.time()
    
    all_nodes = multiplex.get_all_nodes()
    num_nodes = len(all_nodes)
    node_scores = np.zeros(num_nodes)
    
    # Process each layer
    for layer_id, layer_graph in multiplex.layers.items():
        print(f"Processing TDA for layer {layer_id}...")
        
        # Create distance matrix for this layer
        nodes = list(layer_graph.nodes())
        num_layer_nodes = len(nodes)
        dist_matrix = np.full((num_layer_nodes, num_layer_nodes), np.inf)
        np.fill_diagonal(dist_matrix, 0)
        
        node_map = {node: i for i, node in enumerate(nodes)}
        
        # Add distance attributes for Floyd-Warshall
        for u, v, data in layer_graph.edges(data=True):
            dist = 1.0 - data['weight']
            i, j = node_map[u], node_map[v]
            dist_matrix[i, j] = dist_matrix[j, i] = dist
            layer_graph.edges[u, v]['distance'] = dist
        
        # Use Floyd-Warshall to get all-pairs shortest paths
        dist_matrix = nx.floyd_warshall_numpy(layer_graph, nodelist=nodes, weight='distance')
        
        # Compute persistent homology
        vr_persistence = VietorisRipsPersistence(
            homology_dimensions=(0, 1),
            metric="precomputed"
        )
        diagrams = vr_persistence.fit_transform([dist_matrix])
        
        # Extract H0 and H1 diagrams
        h0_diagram = diagrams[0, diagrams[0, :, 2] == 0]
        h1_diagram = diagrams[0, diagrams[0, :, 2] == 1]
        
        # Score nodes based on persistence features
        layer_node_scores = np.zeros(num_nodes)
        
        # H0 features (connected components)
        for feature in h0_diagram:
            birth, death, _ = feature
            if np.isinf(death): continue
            persistence = death - birth
            rows, cols = np.where(np.isclose(dist_matrix, death))
            for i, j in zip(rows, cols):
                if i < j:
                    node_i = nodes[i]
                    node_j = nodes[j]
                    if node_i < num_nodes:
                        layer_node_scores[node_i] += persistence
                    if node_j < num_nodes:
                        layer_node_scores[node_j] += persistence
        
        # H1 features (loops)
        for feature in h1_diagram:
            birth, death, _ = feature
            if np.isinf(death): continue
            persistence = death - birth
            rows, cols = np.where(np.isclose(dist_matrix, birth))
            for i, j in zip(rows, cols):
                if i < j:
                    node_i = nodes[i]
                    node_j = nodes[j]
                    if node_i < num_nodes:
                        layer_node_scores[node_i] += persistence
                    if node_j < num_nodes:
                        layer_node_scores[node_j] += persistence
        
        # Weight layer scores by spreading coefficient
        layer_weight = multiplex.spreading_coefficients[layer_id]
        node_scores += layer_node_scores * layer_weight
    
    # Select top k nodes
    top_k_indices = np.argsort(node_scores)[-k:]
    seed_set = [all_nodes[i] for i in top_k_indices]
    
    end_time = time.time()
    return seed_set, end_time - start_time

def multiplex_degree_centrality_selection(multiplex, k):
    """
    Degree centrality-based selection for multiplex networks.
    
    Args:
        multiplex (MultiplexNetwork): The multiplex network
        k (int): Number of seeds to select
        
    Returns:
        tuple: (selected_seeds, selection_time)
    """
    start_time = time.time()
    
    all_nodes = multiplex.get_all_nodes()
    node_scores = defaultdict(float)
    
    # Aggregate degree centrality across all layers
    for layer_id, layer_graph in multiplex.layers.items():
        degree_centrality = nx.degree_centrality(layer_graph)
        layer_weight = multiplex.spreading_coefficients[layer_id]
        
        for node, centrality in degree_centrality.items():
            node_scores[node] += centrality * layer_weight
    
    # Select top k nodes
    sorted_nodes = sorted(node_scores.items(), key=lambda x: x[1], reverse=True)
    seed_set = [node for node, _ in sorted_nodes[:k]]
    
    end_time = time.time()
    return seed_set, end_time - start_time

def plot_multiplex_spreading_comparison(spreading_data, network_name):
    """
    Plots the spreading process comparison for multiplex networks.
    
    Args:
        spreading_data (dict): Dictionary with method names as keys and spreading curves as values.
        network_name (str): Name of the network for the plot title.
    """
    plt.figure(figsize=(12, 8))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    methods = list(spreading_data.keys())
    
    for i, method in enumerate(methods):
        curves = spreading_data[method]
        
        # Calculate average curve
        max_length = max(len(curve) for curve in curves)
        avg_curve = np.zeros(max_length)
        count_curve = np.zeros(max_length)
        
        for curve in curves:
            for j, value in enumerate(curve):
                avg_curve[j] += value
                count_curve[j] += 1
        
        avg_curve = avg_curve / count_curve
        
        # Plot average curve with confidence interval
        steps = range(len(avg_curve))
        plt.plot(steps, avg_curve, label=method, color=colors[i % len(colors)], linewidth=2)
        
        # Add confidence interval
        std_curve = np.zeros(max_length)
        for curve in curves:
            for j, value in enumerate(curve):
                std_curve[j] += (value - avg_curve[j]) ** 2
        
        std_curve = np.sqrt(std_curve / count_curve)
        plt.fill_between(steps, avg_curve - std_curve, avg_curve + std_curve, 
                        alpha=0.2, color=colors[i % len(colors)])
    
    plt.xlabel('Spreading Steps')
    plt.ylabel('Fraction of Activated Nodes')
    plt.title(f'Multiplex Influence Spreading Process - {network_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Create Figures directory if it doesn't exist
    os.makedirs('Figures', exist_ok=True)
    plt.savefig(f'Figures/multiplex_spreading_comparison_{network_name.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.show()

def visualize_multiplex_network(multiplex, seed_nodes=None):
    """
    Visualizes the multiplex network structure.
    
    Args:
        multiplex (MultiplexNetwork): The multiplex network to visualize
        seed_nodes (list): Optional list of seed nodes to highlight
    """
    fig, axes = plt.subplots(1, len(multiplex.layers), figsize=(5*len(multiplex.layers), 4))
    if len(multiplex.layers) == 1:
        axes = [axes]
    
    for i, (layer_id, layer_graph) in enumerate(multiplex.layers.items()):
        ax = axes[i]
        
        # Create layout
        pos = nx.spring_layout(layer_graph, seed=42)
        
        # Draw nodes
        node_colors = ['lightblue'] * len(layer_graph.nodes())
        if seed_nodes:
            for j, node in enumerate(layer_graph.nodes()):
                if node in seed_nodes:
                    node_colors[j] = 'red'
        
        nx.draw_networkx_nodes(layer_graph, pos, node_color=node_colors, 
                              node_size=300, ax=ax)
        nx.draw_networkx_edges(layer_graph, pos, alpha=0.6, ax=ax)
        nx.draw_networkx_labels(layer_graph, pos, font_size=8, ax=ax)
        
        ax.set_title(f'Layer {layer_id}\nSpreading Coeff: {multiplex.spreading_coefficients[layer_id]:.2f}')
        ax.axis('off')
    
    plt.tight_layout()
    
    # Create Figures directory if it doesn't exist
    os.makedirs('Figures', exist_ok=True)
    plt.savefig('Figures/multiplex_network_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Parameters
    NUM_NODES = 100
    NUM_LAYERS = 3
    NUM_SEEDS = 5
    MC_SIMULATIONS = 50
    
    print("=== Multiplex Network TDA Influence Maximization ===")
    print(f"Parameters: N={NUM_NODES}, Layers={NUM_LAYERS}, k={NUM_SEEDS}, MC Sims={MC_SIMULATIONS}\n")
    
    # Create multiplex network
    print("Creating multiplex network...")
    multiplex = create_multiplex_network(NUM_NODES, NUM_LAYERS)
    
    # Visualize the network
    print("\nVisualizing multiplex network structure...")
    visualize_multiplex_network(multiplex)
    
    # Define seed selection methods
    seed_selection_methods = {
        "Multiplex Greedy": lambda: multiplex_greedy_selection(multiplex, NUM_SEEDS, MC_SIMULATIONS),
        "Multiplex TDA": lambda: multiplex_tda_selection(multiplex, NUM_SEEDS),
        "Multiplex Degree Centrality": lambda: multiplex_degree_centrality_selection(multiplex, NUM_SEEDS)
    }
    
    # Results storage
    results = {
        'seeds': {},
        'influence': {},
        'time': {},
        'spreading_curves': {}
    }
    
    # Test each method
    for method_name, method_func in seed_selection_methods.items():
        print(f"\n--- Running {method_name} ---")
        
        try:
            if method_name == "Multiplex Greedy":
                seeds, influence, selection_time = method_func()
                results['seeds'][method_name] = seeds
                results['influence'][method_name] = influence
                results['time'][method_name] = selection_time
            else:
                seeds, selection_time = method_func()
                influence = run_multiplex_ic_model(multiplex, seeds, MC_SIMULATIONS)
                results['seeds'][method_name] = seeds
                results['influence'][method_name] = influence
                results['time'][method_name] = selection_time
            
            # Get spreading curves
            _, spreading_curves = run_multiplex_ic_model_with_tracking(multiplex, seeds, MC_SIMULATIONS)
            results['spreading_curves'][method_name] = spreading_curves
            
            print(f"{method_name} Seeds: {seeds}")
            print(f"{method_name} Influence Spread: {influence:.2f} nodes")
            print(f"{method_name} Selection Time: {selection_time:.4f} seconds")
            
        except Exception as e:
            print(f"Error running {method_name}: {str(e)}")
            continue
    
    # Print comparison summary
    print(f"\n--- Multiplex Network Comparison Summary ---")
    print("-" * 80)
    print(f"| {'Method':<25} | {'Influence Fraction':<15} | {'Selection Time (s)':<15} |")
    print("-" * 80)
    
    for method_name in results['influence'].keys():
        influence = results['influence'][method_name]
        selection_time = results['time'][method_name]
        print(f"| {method_name:<25} | {influence:<15.3f} | {selection_time:<15.4f} |")
    print("-" * 80)
    
    # Create spreading comparison plot
    print(f"\nGenerating spreading comparison plot...")
    plot_multiplex_spreading_comparison(results['spreading_curves'], "Multiplex Network")
    
    # Visualize results with seed nodes
    print(f"\nVisualizing results with selected seed nodes...")
    best_method = max(results['influence'].items(), key=lambda x: x[1])
    best_seeds = results['seeds'][best_method[0]]
    visualize_multiplex_network(multiplex, best_seeds)
    
    print(f"\nAnalysis complete!")
    print(f"Best performing method: {best_method[0]} (influence: {best_method[1]:.2f})")
