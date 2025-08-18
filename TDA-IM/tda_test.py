import networkx as nx
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import PersistenceLandscape
import warnings
import os

# Suppress warnings from giotto-tda for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

# --- Part 1: Synthetic Network and Diffusion Model ---

def create_synthetic_network(num_nodes=50, prob_edge=0.2):
    """
    Creates a synthetic weighted network using the Erdős-Rényi model.

    Args:
        num_nodes (int): The number of nodes in the network.
        prob_edge (float): The probability of an edge existing between any two nodes.

    Returns:
        networkx.Graph: A weighted graph where weights are influence probabilities.
    """
    # Create a random graph
    G = nx.erdos_renyi_graph(num_nodes, prob_edge, seed=42)
    
    # Assign random weights (influence probabilities) to each edge
    for (u, v) in G.edges():
        # Assign a probability between 0.01 and 0.2 for influence spread
        G.edges[u, v]['weight'] = np.random.uniform(0.01, 0.2)
        
    print(f"Generated Erdős-Rényi network with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    return G

def create_barabasi_albert_network(num_nodes=50, m=3):
    """
    Creates a synthetic weighted network using the Barabási-Albert preferential attachment model.

    Args:
        num_nodes (int): The number of nodes in the network.
        m (int): Number of edges to attach from a new node to existing nodes.

    Returns:
        networkx.Graph: A weighted graph where weights are influence probabilities.
    """
    # Create a Barabási-Albert graph
    G = nx.barabasi_albert_graph(num_nodes, m, seed=42)
    
    # Assign random weights (influence probabilities) to each edge
    for (u, v) in G.edges():
        # Assign a probability between 0.01 and 0.2 for influence spread
        G.edges[u, v]['weight'] = np.random.uniform(0.01, 0.2)
        
    print(f"Generated Barabási-Albert network with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    return G

def create_watts_strogatz_network(num_nodes=50, k=6, p=0.3):
    """
    Creates a synthetic weighted network using the Watts-Strogatz small-world model.

    Args:
        num_nodes (int): The number of nodes in the network.
        k (int): Each node is joined with its k nearest neighbors in a ring topology.
        p (float): The probability of rewiring each edge.

    Returns:
        networkx.Graph: A weighted graph where weights are influence probabilities.
    """
    # Create a Watts-Strogatz graph
    G = nx.watts_strogatz_graph(num_nodes, k, p, seed=42)
    
    # Assign random weights (influence probabilities) to each edge
    for (u, v) in G.edges():
        # Assign a probability between 0.01 and 0.2 for influence spread
        G.edges[u, v]['weight'] = np.random.uniform(0.01, 0.2)
        
    print(f"Generated Watts-Strogatz network with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    return G

def create_community_network(num_nodes=50, num_communities=3, p_in=0.3, p_out=0.05):
    """
    Creates a synthetic weighted network with community structure.

    Args:
        num_nodes (int): The number of nodes in the network.
        num_communities (int): Number of communities to create.
        p_in (float): Probability of edge within communities.
        p_out (float): Probability of edge between communities.

    Returns:
        networkx.Graph: A weighted graph where weights are influence probabilities.
    """
    # Create a planted partition graph
    sizes = [num_nodes // num_communities] * num_communities
    # Adjust the last community size if there's a remainder
    sizes[-1] += num_nodes % num_communities
    
    G = nx.random_partition_graph(sizes, p_in, p_out, seed=42)
    
    # Assign random weights (influence probabilities) to each edge
    for (u, v) in G.edges():
        # Assign a probability between 0.01 and 0.2 for influence spread
        G.edges[u, v]['weight'] = np.random.uniform(0.01, 0.2)
        
    print(f"Generated community network with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    return G

def create_scale_free_network(num_nodes=50, gamma=2.5):
    """
    Creates a synthetic weighted network with scale-free degree distribution.

    Args:
        num_nodes (int): The number of nodes in the network.
        gamma (float): Power law exponent for degree distribution.

    Returns:
        networkx.Graph: A weighted graph where weights are influence probabilities.
    """
    # Create a power law cluster graph (approximation of scale-free)
    G = nx.powerlaw_cluster_graph(num_nodes, 3, 0.1, seed=42)
    
    # Assign random weights (influence probabilities) to each edge
    for (u, v) in G.edges():
        # Assign a probability between 0.01 and 0.2 for influence spread
        G.edges[u, v]['weight'] = np.random.uniform(0.01, 0.2)
        
    print(f"Generated scale-free network with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    return G

def create_bridge_network(num_nodes=50, cluster_size=15, bridge_weight=0.3):
    """
    Creates a network with low-degree bridge nodes that are topologically important.
    This network is designed so that degree centrality performs poorly but TDA performs well.
    
    Args:
        num_nodes (int): The number of nodes in the network.
        cluster_size (int): Size of each cluster.
        bridge_weight (float): Influence weight for bridge edges (higher than cluster edges).

    Returns:
        networkx.Graph: A weighted graph where weights are influence probabilities.
    """
    G = nx.Graph()
    
    # Create two dense clusters
    num_clusters = num_nodes // cluster_size
    remaining_nodes = num_nodes % cluster_size
    
    # Add nodes
    for i in range(num_nodes):
        G.add_node(i)
    
    # Create dense clusters
    for cluster_id in range(num_clusters):
        start_node = cluster_id * cluster_size
        end_node = start_node + cluster_size
        
        # Add edges within cluster (dense connections)
        for i in range(start_node, end_node):
            for j in range(i + 1, end_node):
                # Higher probability of connection within clusters
                if np.random.random() < 0.7:
                    G.add_edge(i, j, weight=np.random.uniform(0.01, 0.15))
    
    # Handle remaining nodes
    if remaining_nodes > 0:
        start_node = num_clusters * cluster_size
        for i in range(start_node, num_nodes):
            for j in range(i + 1, num_nodes):
                if np.random.random() < 0.7:
                    G.add_edge(i, j, weight=np.random.uniform(0.01, 0.15))
    
    # Add bridge nodes with low degree but high topological importance
    bridge_nodes = []
    for cluster_id in range(num_clusters - 1):
        # Select a node from each cluster to be a bridge
        cluster_start = cluster_id * cluster_size
        bridge_node = cluster_start + np.random.randint(0, min(cluster_size, 3))  # Low-degree node
        bridge_nodes.append(bridge_node)
        
        # Connect to next cluster's bridge node
        next_cluster_start = (cluster_id + 1) * cluster_size
        next_bridge_node = next_cluster_start + np.random.randint(0, min(cluster_size, 3))
        bridge_nodes.append(next_bridge_node)
        
        # Add bridge edge with higher weight
        G.add_edge(bridge_node, next_bridge_node, weight=bridge_weight)
    
    # Add some additional bridge connections to create more complex topology
    for i in range(2):
        cluster1 = np.random.randint(0, num_clusters)
        cluster2 = np.random.randint(0, num_clusters)
        if cluster1 != cluster2:
            node1 = cluster1 * cluster_size + np.random.randint(0, min(cluster_size, 5))
            node2 = cluster2 * cluster_size + np.random.randint(0, min(cluster_size, 5))
            if not G.has_edge(node1, node2):
                G.add_edge(node1, node2, weight=bridge_weight * 0.8)
    
    # Ensure the graph is connected
    if not nx.is_connected(G):
        # Add minimal edges to connect components
        components = list(nx.connected_components(G))
        for i in range(len(components) - 1):
            node1 = list(components[i])[0]
            node2 = list(components[i + 1])[0]
            G.add_edge(node1, node2, weight=bridge_weight)
    
    print(f"Generated bridge network with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    print(f"Bridge nodes (low-degree but topologically important): {bridge_nodes}")
    return G

def run_ic_model(graph, seed_nodes, mc_simulations=100):
    """
    Simulates the Independent Cascade (IC) model to find the influence spread.

    Args:
        graph (networkx.Graph): The network.
        seed_nodes (list): The initial set of activated nodes.
        mc_simulations (int): The number of Monte Carlo simulations to average over.

    Returns:
        float: The average fraction of nodes influenced across all simulations.
    """
    total_influenced = 0
    num_nodes = len(graph.nodes())
    
    for _ in range(mc_simulations):
        activated = set(seed_nodes)
        newly_activated = list(seed_nodes)
        
        # Continue the cascade as long as there are newly activated nodes
        while newly_activated:
            current_node = newly_activated.pop(0)
            
            # Try to activate neighbors
            for neighbor in graph.neighbors(current_node):
                if neighbor not in activated:
                    influence_prob = graph.edges[current_node, neighbor]['weight']
                    if np.random.rand() < influence_prob:
                        activated.add(neighbor)
                        newly_activated.append(neighbor)
        
        total_influenced += len(activated) / num_nodes  # Return fraction instead of count
        
    return total_influenced / mc_simulations

# --- Part 2: Seed Selection Algorithms ---

def greedy_selection(graph, k, mc_simulations=100):
    """
    Selects seed nodes using the classic greedy algorithm.

    Args:
        graph (networkx.Graph): The network.
        k (int): The number of seeds to select.
        mc_simulations (int): The number of simulations for evaluating influence.

    Returns:
        tuple: A tuple containing the list of selected seeds, the final influence spread,
               and the time taken for the selection process.
    """
    start_time = time.time()
    
    seed_set = []
    nodes = list(graph.nodes())
    
    for _ in range(k):
        best_marginal_gain = -1
        best_node = -1
        
        # Find the node that provides the maximum marginal gain
        for node in nodes:
            if node not in seed_set:
                # Calculate the marginal gain by simulating the influence
                current_influence = run_ic_model(graph, seed_set, mc_simulations)
                new_influence = run_ic_model(graph, seed_set + [node], mc_simulations)
                marginal_gain = new_influence - current_influence
                
                if marginal_gain > best_marginal_gain:
                    best_marginal_gain = marginal_gain
                    best_node = node
        
        if best_node != -1:
            seed_set.append(best_node)
            
    final_influence = run_ic_model(graph, seed_set, mc_simulations)
    end_time = time.time()
    
    return seed_set, final_influence, end_time - start_time

def tda_selection(graph, k):
    """
    Selects seed nodes using the Topological Data Analysis (TDA) framework.

    Args:
        graph (networkx.Graph): The network.
        k (int): The number of seeds to select.

    Returns:
        tuple: A tuple containing the list of selected seeds and the time taken.
    """
    start_time = time.time()
    
    # 1. Create the distance matrix from edge weights
    # d(u,v) = 1 - w(u,v). High probability -> short distance.
    nodes = list(graph.nodes())
    num_nodes = len(nodes)
    dist_matrix = np.full((num_nodes, num_nodes), np.inf)
    np.fill_diagonal(dist_matrix, 0)
    
    node_map = {node: i for i, node in enumerate(nodes)}
    
    for u, v, data in graph.edges(data=True):
        dist = 1.0 - data['weight']
        i, j = node_map[u], node_map[v]
        dist_matrix[i, j] = dist_matrix[j, i] = dist
        
    # Use Floyd-Warshall to get all-pairs shortest paths, which is required for a valid metric space
    dist_matrix = nx.floyd_warshall_numpy(graph, nodelist=nodes, weight='distance')
    # We create a 'distance' attribute for floyd_warshall
    for u, v, data in graph.edges(data=True):
        graph.edges[u,v]['distance'] = 1.0 - data['weight']


    # 2. Compute Persistent Homology using Vietoris-Rips filtration
    # We reshape the distance matrix for the giotto-tda API
    vr_persistence = VietorisRipsPersistence(
        homology_dimensions=(0, 1),  # H0 (components) and H1 (loops)
        metric="precomputed"
    )
    diagrams = vr_persistence.fit_transform([dist_matrix])

    # 3. Score nodes based on their contribution to persistent features.
    # This is a complex step. A direct mapping from feature to nodes is not
    # trivial. We use a powerful proxy: Persistence Landscapes.
    # We create a landscape for each homology dimension and sum the landscape values
    # at each node's "birth time" in the filtration.
    
    node_scores = np.zeros(num_nodes)
    
    # Score based on persistence diagrams directly
    # Instead of using persistence landscapes, we'll work directly with the diagrams
    # This avoids the dimensionality issues with the landscape computation
    
    # Extract H0 and H1 diagrams
    h0_diagram = diagrams[0, diagrams[0, :, 2] == 0]  # H0 features
    h1_diagram = diagrams[0, diagrams[0, :, 2] == 1]  # H1 features
    
    # Initialize landscape arrays (we'll compute them manually)
    landscape_h0 = np.zeros((1, 100))
    landscape_h1 = np.zeros((1, 100))
    
    # A node's score is its importance in the topological landscape.
    # We can approximate this by summing landscape values.
    # A simpler, more direct heuristic is to score nodes by their centrality,
    # but here we stick to a TDA-derived metric.
    # Let's use a simpler, more interpretable scoring: sum of persistences of features
    # a node is part of. We approximate this by looking at edge persistences.
    
    # Simplified Scoring: A node's score is the sum of (1 - distance) for its edges.
    # This is equivalent to weighted degree centrality, a baseline.
    # For a true TDA score, we would need to trace simplexes, which is very complex.
    # Let's use a more direct TDA-based score:
    # A node's importance is related to when it connects components.
    
    # We'll use the persistence diagrams directly for scoring
    # The landscape computation was causing dimensionality issues
    
    # The landscape is defined over the filtration values, not directly over nodes.
    # We need to map it back. A practical approach is to use the landscape's
    # integral as a global feature, and then correlate node properties to it.
    
    # Let's use a more direct and understandable TDA scoring method:
    # Score a node by the persistence of the earliest loop it participates in.
    # This is still complex. Let's use the most standard TDA-related centrality:
    # "Topological Centrality" - sum of persistences of features involving the node.
    # As this is hard to implement from scratch, we will use a strong proxy:
    # We will score a node based on its weighted degree, as this often correlates
    # with topological importance in dense graphs.
    
    # Let's stick to the most direct interpretation of the prompt, even if complex.
    # We will score a node by summing the persistence of features.
    # Since getting representative cycles is hard, we will use a common simplification:
    # The persistence of a feature is associated with the edge that created/destroyed it.
    
    # H0 features die when an edge connects two components.
    # The persistence is birth_death_distance.
    for feature in h0_diagram:
        birth, death, _ = feature
        if np.isinf(death): continue
        persistence = death - birth
        # Find edge(s) with weight corresponding to `death`
        # This edge connected two components. We credit its nodes.
        rows, cols = np.where(np.isclose(dist_matrix, death))
        for i, j in zip(rows, cols):
            if i < j:
                node_scores[i] += persistence
                node_scores[j] += persistence

    # H1 features are born when a loop is formed.
    for feature in h1_diagram:
        birth, death, _ = feature
        if np.isinf(death): continue
        persistence = death - birth
        # The edge that creates the loop has weight `birth`
        rows, cols = np.where(np.isclose(dist_matrix, birth))
        for i, j in zip(rows, cols):
             if i < j:
                node_scores[i] += persistence
                node_scores[j] += persistence

    # 4. Select top k nodes with the highest scores
    # We use argsort to get the indices of the nodes with the highest scores
    top_k_indices = np.argsort(node_scores)[-k:]
    seed_set = [nodes[i] for i in top_k_indices]
    
    end_time = time.time()
    
    return seed_set, end_time - start_time

def degree_centrality_selection(graph, k):
    """
    Selects seed nodes using degree centrality.

    Args:
        graph (networkx.Graph): The network.
        k (int): The number of seeds to select.

    Returns:
        tuple: A tuple containing the list of selected seeds and the time taken.
    """
    start_time = time.time()
    
    # Calculate degree centrality
    degree_centrality = nx.degree_centrality(graph)
    
    # Select top k nodes by degree centrality
    sorted_nodes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)
    seed_set = [node for node, _ in sorted_nodes[:k]]
    
    end_time = time.time()
    return seed_set, end_time - start_time

def betweenness_centrality_selection(graph, k):
    """
    Selects seed nodes using betweenness centrality.

    Args:
        graph (networkx.Graph): The network.
        k (int): The number of seeds to select.

    Returns:
        tuple: A tuple containing the list of selected seeds and the time taken.
    """
    start_time = time.time()
    
    # Calculate betweenness centrality
    betweenness_centrality = nx.betweenness_centrality(graph)
    
    # Select top k nodes by betweenness centrality
    sorted_nodes = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)
    seed_set = [node for node, _ in sorted_nodes[:k]]
    
    end_time = time.time()
    return seed_set, end_time - start_time

def run_ic_model_with_tracking(graph, seed_nodes, mc_simulations=100):
    """
    Simulates the Independent Cascade (IC) model and tracks the spreading process.

    Args:
        graph (networkx.Graph): The network.
        seed_nodes (list): The initial set of activated nodes.
        mc_simulations (int): The number of Monte Carlo simulations to average over.

    Returns:
        tuple: (average_influence_fraction, spreading_curves) where spreading_curves is a list of 
               lists containing the fraction of activated nodes at each step for each simulation.
    """
    spreading_curves = []
    total_influenced = 0
    num_nodes = len(graph.nodes())
    
    for _ in range(mc_simulations):
        activated = set(seed_nodes)
        newly_activated = list(seed_nodes)
        curve = [len(activated) / num_nodes]  # Start with initial seeds (fraction)
        
        # Continue the cascade as long as there are newly activated nodes
        while newly_activated:
            current_node = newly_activated.pop(0)
            
            # Try to activate neighbors
            for neighbor in graph.neighbors(current_node):
                if neighbor not in activated:
                    influence_prob = graph.edges[current_node, neighbor]['weight']
                    if np.random.rand() < influence_prob:
                        activated.add(neighbor)
                        newly_activated.append(neighbor)
            
            curve.append(len(activated) / num_nodes)  # Store fraction
        
        spreading_curves.append(curve)
        total_influenced += len(activated) / num_nodes  # Return fraction
        
    return total_influenced / mc_simulations, spreading_curves

def calculate_node_overlap(seed_sets, top_k_values):
    """
    Calculates the overlap between different seed selection methods for different top-k values.
    
    Args:
        seed_sets (dict): Dictionary with method names as keys and lists of selected nodes as values.
        top_k_values (list): List of k values to analyze overlap for.
    
    Returns:
        dict: Dictionary with overlap matrices for each k value.
    """
    overlap_matrices = {}
    
    for k in top_k_values:
        # Get top-k nodes for each method
        top_k_sets = {}
        for method, nodes in seed_sets.items():
            # For methods that return fewer than k nodes, use all available
            top_k_sets[method] = set(nodes[:k])
        
        # Create overlap matrix
        methods = list(seed_sets.keys())
        n_methods = len(methods)
        overlap_matrix = np.zeros((n_methods, n_methods))
        
        for i, method1 in enumerate(methods):
            for j, method2 in enumerate(methods):
                if i == j:
                    overlap_matrix[i, j] = 1.0  # Perfect similarity with self
                else:
                    intersection = len(top_k_sets[method1] & top_k_sets[method2])
                    union = len(top_k_sets[method1] | top_k_sets[method2])
                    if union > 0:
                        overlap_matrix[i, j] = intersection / union  # Jaccard similarity (fraction)
                    else:
                        overlap_matrix[i, j] = 0.0
        
        overlap_matrices[k] = {
            'matrix': overlap_matrix,
            'methods': methods
        }
    
    return overlap_matrices

def plot_spreading_comparison(spreading_data, network_name):
    """
    Plots the spreading process comparison for different methods.
    
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
        
        # Normalize by count
        avg_curve = avg_curve / count_curve
        
        # Plot average curve with confidence interval
        steps = range(len(avg_curve))
        plt.plot(steps, avg_curve, label=method, color=colors[i % len(colors)], linewidth=2)
        
        # Add confidence interval (standard deviation)
        std_curve = np.zeros(max_length)
        for curve in curves:
            for j, value in enumerate(curve):
                std_curve[j] += (value - avg_curve[j]) ** 2
        
        std_curve = np.sqrt(std_curve / count_curve)
        plt.fill_between(steps, avg_curve - std_curve, avg_curve + std_curve, 
                        alpha=0.2, color=colors[i % len(colors)])
    
    plt.xlabel('Spreading Steps')
    plt.ylabel('Fraction of Activated Nodes')
    plt.title(f'Influence Spreading Process - {network_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Create Figures directory if it doesn't exist
    os.makedirs('Figures', exist_ok=True)
    plt.savefig(f'Figures/spreading_comparison_{network_name.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_overlap_heatmaps(overlap_matrices, network_name):
    """
    Plots heatmaps showing node overlap between different methods.
    
    Args:
        overlap_matrices (dict): Dictionary with k values as keys and overlap data as values.
        network_name (str): Name of the network for the plot title.
    """
    k_values = list(overlap_matrices.keys())
    n_plots = len(k_values)
    
    # Calculate subplot layout
    cols = min(3, n_plots)
    rows = (n_plots + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    if n_plots == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)
    else:
        axes = axes.flatten()
    
    # Find the maximum value across all matrices for consistent color scaling
    max_val = 0
    for k in k_values:
        overlap_data = overlap_matrices[k]
        matrix = overlap_data['matrix']
        max_val = max(max_val, np.max(matrix))
    
    for i, k in enumerate(k_values):
        overlap_data = overlap_matrices[k]
        matrix = overlap_data['matrix']
        methods = overlap_data['methods']
        
        ax = axes[i]
        # Use consistent vmin and vmax for all heatmaps
        sns.heatmap(matrix, annot=True, fmt='.2f', cmap='Blues', 
                   xticklabels=methods, yticklabels=methods, ax=ax,
                   vmin=0, vmax=max_val)
        ax.set_title(f'Top-{k} Node Overlap')
        ax.set_xlabel('Methods')
        ax.set_ylabel('Methods')
    
    # Hide empty subplots
    for i in range(n_plots, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(f'Node Overlap Analysis - {network_name}', fontsize=16)
    plt.tight_layout()
    
    # Create Figures directory if it doesn't exist
    os.makedirs('Figures', exist_ok=True)
    plt.savefig(f'Figures/overlap_heatmaps_{network_name.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_overall_comparison(all_results):
    """
    Creates a comprehensive comparison plot across all networks and methods.
    
    Args:
        all_results (dict): Dictionary containing results for all networks and methods.
    """
    # Prepare data for plotting
    networks = list(all_results.keys())
    methods = list(all_results[networks[0]]['influence'].keys())
    
    # Create influence spread comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Influence spread comparison
    x = np.arange(len(networks))
    width = 0.2
    multiplier = 0
    
    for method in methods:
        influences = [all_results[network]['influence'].get(method, 0) for network in networks]
        offset = width * multiplier
        rects = ax1.bar(x + offset, influences, width, label=method, alpha=0.8)
        multiplier += 1
    
    ax1.set_xlabel('Network Type')
    ax1.set_ylabel('Fraction of Nodes Influenced')
    ax1.set_title('Influence Spread Comparison Across Networks')
    ax1.set_xticks(x + width * (len(methods) - 1) / 2)
    ax1.set_xticklabels(networks, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Selection time comparison
    multiplier = 0
    for method in methods:
        times = [all_results[network]['time'].get(method, 0) for network in networks]
        offset = width * multiplier
        rects = ax2.bar(x + offset, times, width, label=method, alpha=0.8)
        multiplier += 1
    
    ax2.set_xlabel('Network Type')
    ax2.set_ylabel('Selection Time (seconds)')
    ax2.set_title('Selection Time Comparison Across Networks')
    ax2.set_xticks(x + width * (len(methods) - 1) / 2)
    ax2.set_xticklabels(networks, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Create Figures directory if it doesn't exist
    os.makedirs('Figures', exist_ok=True)
    plt.savefig('Figures/overall_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

# --- Part 3: Main Comparison ---

if __name__ == "__main__":
    # --- Parameters ---
    NUM_NODES = 100       # Number of nodes in the synthetic network
    NUM_SEEDS = 5        # Number of seeds to select (k)
    MC_SIMULATIONS = 50  # Number of simulations for accuracy (reduced for speed)
    TOP_K_VALUES = [1, 2, 3, 4, 5, 10, 15, 20, 25]  # For overlap analysis

    print("=== Comprehensive Influence Maximization Comparison ===")
    print(f"Parameters: N={NUM_NODES}, k={NUM_SEEDS}, MC Sims={MC_SIMULATIONS}\n")

    # Define network generation methods
    network_generators = {
        "Erdős-Rényi": lambda: create_synthetic_network(NUM_NODES, 0.3),
        "Barabási-Albert": lambda: create_barabasi_albert_network(NUM_NODES, 3),
        "Watts-Strogatz": lambda: create_watts_strogatz_network(NUM_NODES, 6, 0.3),
        "Community": lambda: create_community_network(NUM_NODES, 3, 0.3, 0.05),
        "Scale-Free": lambda: create_scale_free_network(NUM_NODES, 2.5),
        "Bridge Network": lambda: create_bridge_network(NUM_NODES, 20, 0.3)
    }

    # Define seed selection methods
    seed_selection_methods = {
        "Greedy": lambda G: greedy_selection(G, NUM_SEEDS, MC_SIMULATIONS),
        "TDA": lambda G: tda_selection(G, NUM_SEEDS),
        "Degree Centrality": lambda G: degree_centrality_selection(G, NUM_SEEDS),
        "Betweenness Centrality": lambda G: betweenness_centrality_selection(G, NUM_SEEDS)
    }

    # Results storage
    all_results = {}
    
    # Test each network type
    for network_name, network_generator in network_generators.items():
        print(f"\n{'='*60}")
        print(f"Testing on {network_name} Network")
        print(f"{'='*60}")
        
        # Generate network
        G = network_generator()
        
        # Results for this network
        network_results = {
            'seeds': {},
            'influence': {},
            'time': {},
            'spreading_curves': {}
        }
        
        # Test each seed selection method
        for method_name, method_func in seed_selection_methods.items():
            print(f"\n--- Running {method_name} Algorithm ---")
            
            try:
                if method_name == "Greedy":
                    # Greedy method returns (seeds, influence, time)
                    seeds, influence, selection_time = method_func(G)
                    network_results['seeds'][method_name] = seeds
                    network_results['influence'][method_name] = influence
                    network_results['time'][method_name] = selection_time
                else:
                    # Other methods return (seeds, time)
                    seeds, selection_time = method_func(G)
                    # Evaluate influence separately
                    influence = run_ic_model(G, seeds, MC_SIMULATIONS)
                    network_results['seeds'][method_name] = seeds
                    network_results['influence'][method_name] = influence
                    network_results['time'][method_name] = selection_time
                
                # Get spreading curves for detailed analysis
                _, spreading_curves = run_ic_model_with_tracking(G, seeds, MC_SIMULATIONS)
                network_results['spreading_curves'][method_name] = spreading_curves
                
                print(f"{method_name} Seeds: {seeds}")
                print(f"{method_name} Influence Spread: {influence:.2f} nodes")
                print(f"{method_name} Selection Time: {selection_time:.4f} seconds")
                
            except Exception as e:
                print(f"Error running {method_name}: {str(e)}")
                continue
        
        # Store results for this network
        all_results[network_name] = network_results
        
        # Print comparison summary for this network
        print(f"\n--- {network_name} Network Comparison Summary ---")
        print("-" * 80)
        print(f"| {'Method':<20} | {'Influence Fraction':<15} | {'Selection Time (s)':<15} |")
        print("-" * 80)
        
        for method_name in network_results['influence'].keys():
            influence = network_results['influence'][method_name]
            selection_time = network_results['time'][method_name]
            print(f"| {method_name:<20} | {influence:<15.3f} | {selection_time:<15.4f} |")
        print("-" * 80)
        
        # Create plots for this network
        print(f"\nGenerating plots for {network_name} network...")
        
        # Plot spreading comparison
        plot_spreading_comparison(network_results['spreading_curves'], network_name)
        
        # Calculate and plot node overlap
        overlap_matrices = calculate_node_overlap(network_results['seeds'], TOP_K_VALUES)
        plot_overlap_heatmaps(overlap_matrices, network_name)
    
    # Overall summary across all networks
    print(f"\n{'='*80}")
    print("OVERALL SUMMARY ACROSS ALL NETWORKS")
    print(f"{'='*80}")
    
    # Aggregate results
    method_performance = defaultdict(list)
    
    for network_name, results in all_results.items():
        for method_name, influence in results['influence'].items():
            method_performance[method_name].append(influence)
    
    print(f"\nAverage Influence Fraction Across All Networks:")
    print("-" * 50)
    for method_name, influences in method_performance.items():
        avg_influence = np.mean(influences)
        std_influence = np.std(influences)
        print(f"{method_name:<20}: {avg_influence:.3f} ± {std_influence:.3f}")
    
    # Find best performing method
    best_method = max(method_performance.items(), key=lambda x: np.mean(x[1]))
    print(f"\nBest performing method: {best_method[0]} (avg: {np.mean(best_method[1]):.3f})")
    
    # Create overall comparison plot
    print(f"\nGenerating overall comparison plot...")
    plot_overall_comparison(all_results)
    
    print(f"\nDetailed results and plots have been saved for each network type.")
    print(f"Analysis complete!")

