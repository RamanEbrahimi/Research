import networkx as nx
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from gtda.homology import VietorisRipsPersistence
import warnings
import os

# Import functions from the main TDA script
from tda_test import (
    create_synthetic_network, create_barabasi_albert_network, 
    create_watts_strogatz_network, create_community_network,
    create_scale_free_network, create_bridge_network,
    greedy_selection, tda_selection, degree_centrality_selection,
    betweenness_centrality_selection, run_ic_model
)

# Suppress warnings from giotto-tda for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

def select_seeds_for_all_methods(graph, max_seeds=25):
    """
    Select seeds for all methods up to max_seeds.
    
    Args:
        graph (networkx.Graph): The network
        max_seeds (int): Maximum number of seeds to select
        
    Returns:
        dict: Dictionary with method names as keys and lists of selected seeds as values
    """
    print(f"Selecting seeds for all methods (up to {max_seeds} seeds)...")
    
    seed_sets = {}
    
    # Greedy selection (returns influence and time, but we only need seeds)
    print("Running Greedy selection...")
    greedy_seeds, _, _ = greedy_selection(graph, max_seeds, mc_simulations=50)
    seed_sets["Greedy"] = greedy_seeds
    
    # TDA selection
    print("Running TDA selection...")
    tda_seeds, _ = tda_selection(graph, max_seeds)
    seed_sets["TDA"] = tda_seeds
    
    # Degree centrality selection
    print("Running Degree Centrality selection...")
    degree_seeds, _ = degree_centrality_selection(graph, max_seeds)
    seed_sets["Degree Centrality"] = degree_seeds
    
    # Betweenness centrality selection
    print("Running Betweenness Centrality selection...")
    betweenness_seeds, _ = betweenness_centrality_selection(graph, max_seeds)
    seed_sets["Betweenness Centrality"] = betweenness_seeds
    
    return seed_sets

def evaluate_seed_performance(graph, seed_sets, seed_counts, mc_simulations=100):
    """
    Evaluate the performance of each method for different numbers of seeds.
    
    Args:
        graph (networkx.Graph): The network
        seed_sets (dict): Dictionary with method names and their selected seeds
        seed_counts (list): List of seed counts to evaluate
        mc_simulations (int): Number of Monte Carlo simulations
        
    Returns:
        dict: Dictionary with results for each method and seed count
    """
    print(f"Evaluating performance for seed counts: {seed_counts}")
    print(f"Running {mc_simulations} Monte Carlo simulations for each configuration...")
    
    results = {}
    
    for method_name, seeds in seed_sets.items():
        results[method_name] = {}
        
        for k in seed_counts:
            if k <= len(seeds):
                # Take top k seeds
                top_k_seeds = seeds[:k]
                
                # Run multiple simulations
                influences = []
                for _ in range(mc_simulations):
                    influence = run_ic_model(graph, top_k_seeds, mc_simulations=1)
                    influences.append(influence)
                
                # Calculate statistics
                avg_influence = np.mean(influences)
                std_influence = np.std(influences)
                
                results[method_name][k] = {
                    'avg_influence': avg_influence,
                    'std_influence': std_influence,
                    'seeds': top_k_seeds,
                    'all_influences': influences
                }
                
                print(f"  {method_name} with {k} seeds: {avg_influence:.3f} ± {std_influence:.3f}")
    
    return results

def plot_seed_performance_comparison(results, seed_counts, network_name):
    """
    Plot the performance comparison for different seed counts.
    
    Args:
        results (dict): Results from evaluate_seed_performance
        seed_counts (list): List of seed counts
        network_name (str): Name of the network
    """
    methods = list(results.keys())
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Average influence vs number of seeds
    for i, method in enumerate(methods):
        avg_influences = []
        std_influences = []
        
        for k in seed_counts:
            if k in results[method]:
                avg_influences.append(results[method][k]['avg_influence'])
                std_influences.append(results[method][k]['std_influence'])
            else:
                avg_influences.append(np.nan)
                std_influences.append(np.nan)
        
        # Plot with error bars
        ax1.errorbar(seed_counts, avg_influences, yerr=std_influences, 
                    label=method, color=colors[i % len(colors)], 
                    marker='o', linewidth=2, capsize=5, capthick=2)
    
    ax1.set_xlabel('Number of Seeds (k)')
    ax1.set_ylabel('Average Influence Fraction')
    ax1.set_title(f'Influence Fraction vs Number of Seeds - {network_name}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')  # Log scale for better visualization
    
    # Plot 2: Influence gain per additional seed
    for i, method in enumerate(methods):
        marginal_gains = []
        
        for j, k in enumerate(seed_counts):
            if k in results[method]:
                if j == 0:
                    marginal_gains.append(results[method][k]['avg_influence'])
                else:
                    prev_k = seed_counts[j-1]
                    if prev_k in results[method]:
                        gain = results[method][k]['avg_influence'] - results[method][prev_k]['avg_influence']
                        marginal_gains.append(gain)
                    else:
                        marginal_gains.append(np.nan)
            else:
                marginal_gains.append(np.nan)
        
        ax2.plot(seed_counts, marginal_gains, label=method, 
                color=colors[i % len(colors)], marker='s', linewidth=2)
    
    ax2.set_xlabel('Number of Seeds (k)')
    ax2.set_ylabel('Marginal Influence Gain')
    ax2.set_title(f'Marginal Gain per Additional Seed - {network_name}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    
    plt.tight_layout()
    
    # Create Figures directory if it doesn't exist
    os.makedirs('Figures', exist_ok=True)
    plt.savefig(f'Figures/seed_performance_comparison_{network_name.lower().replace(" ", "_")}.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def plot_seed_overlap_analysis(seed_sets, seed_counts, network_name):
    """
    Analyze and plot the overlap between different methods for different seed counts.
    
    Args:
        seed_sets (dict): Dictionary with method names and their selected seeds
        seed_counts (list): List of seed counts to analyze
        network_name (str): Name of the network
    """
    methods = list(seed_sets.keys())
    
    # Create overlap matrices for each seed count
    overlap_data = {}
    
    for k in seed_counts:
        # Get top-k seeds for each method
        top_k_sets = {}
        for method, seeds in seed_sets.items():
            if len(seeds) >= k:
                top_k_sets[method] = set(seeds[:k])
        
        if len(top_k_sets) >= 2:  # Need at least 2 methods to compare
            # Create overlap matrix
            n_methods = len(top_k_sets)
            overlap_matrix = np.zeros((n_methods, n_methods))
            method_names = list(top_k_sets.keys())
            
            for i, method1 in enumerate(method_names):
                for j, method2 in enumerate(method_names):
                    if i == j:
                        overlap_matrix[i, j] = len(top_k_sets[method1])
                    else:
                        intersection = len(top_k_sets[method1] & top_k_sets[method2])
                        union = len(top_k_sets[method1] | top_k_sets[method2])
                        if union > 0:
                            overlap_matrix[i, j] = intersection / union  # Jaccard similarity
                        else:
                            overlap_matrix[i, j] = 0.0
            
            overlap_data[k] = {
                'matrix': overlap_matrix,
                'methods': method_names
            }
    
    # Plot overlap heatmaps
    k_values = list(overlap_data.keys())
    n_plots = len(k_values)
    
    if n_plots > 0:
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
        
        # Find maximum value for consistent color scaling
        max_val = 0
        for k in k_values:
            max_val = max(max_val, np.max(overlap_data[k]['matrix']))
        
        for i, k in enumerate(k_values):
            overlap_info = overlap_data[k]
            matrix = overlap_info['matrix']
            methods = overlap_info['methods']
            
            ax = axes[i]
            sns.heatmap(matrix, annot=True, fmt='.2f', cmap='Blues', 
                       xticklabels=methods, yticklabels=methods, ax=ax,
                       vmin=0, vmax=max_val)
            ax.set_title(f'Top-{k} Seed Overlap')
            ax.set_xlabel('Methods')
            ax.set_ylabel('Methods')
        
        # Hide empty subplots
        for i in range(n_plots, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(f'Seed Selection Overlap Analysis - {network_name}', fontsize=16)
        plt.tight_layout()
        
        # Create Figures directory if it doesn't exist
        os.makedirs('Figures', exist_ok=True)
        plt.savefig(f'Figures/seed_overlap_analysis_{network_name.lower().replace(" ", "_")}.png', 
                    dpi=300, bbox_inches='tight')
        plt.show()

def print_detailed_results(results, seed_counts):
    """
    Print detailed results in a formatted table.
    
    Args:
        results (dict): Results from evaluate_seed_performance
        seed_counts (list): List of seed counts
    """
    methods = list(results.keys())
    
    print("\n" + "="*100)
    print("DETAILED PERFORMANCE RESULTS")
    print("="*100)
    
    # Print header
    header = f"{'Method':<20} |"
    for k in seed_counts:
        header += f" k={k:<8} |"
    print(header)
    print("-" * len(header))
    
    # Print results for each method
    for method in methods:
        row = f"{method:<20} |"
        for k in seed_counts:
            if k in results[method]:
                avg_inf = results[method][k]['avg_influence']
                std_inf = results[method][k]['std_influence']
                row += f" {avg_inf:6.3f}±{std_inf:4.3f} |"
            else:
                row += f" {'N/A':<8} |"
        print(row)
    
    print("-" * len(header))
    
    # Print best method for each seed count
    print("\nBEST PERFORMING METHOD FOR EACH SEED COUNT:")
    print("-" * 50)
    for k in seed_counts:
        best_method = None
        best_influence = -1
        
        for method in methods:
            if k in results[method]:
                influence = results[method][k]['avg_influence']
                if influence > best_influence:
                    best_influence = influence
                    best_method = method
        
        if best_method:
            print(f"k={k:2d}: {best_method:<20} ({best_influence:.3f} fraction influenced)")

def analyze_network_performance(graph, network_name, seed_counts=None):
    """
    Complete analysis of a network for different seed counts.
    
    Args:
        graph (networkx.Graph): The network to analyze
        network_name (str): Name of the network
        seed_counts (list): List of seed counts to analyze
    """
    if seed_counts is None:
        seed_counts = [1, 2, 3, 4, 5, 10, 15, 20, 25]
    
    print(f"\n{'='*80}")
    print(f"ANALYZING {network_name.upper()} NETWORK")
    print(f"{'='*80}")
    
    # Step 1: Select seeds for all methods
    max_seeds = max(seed_counts)
    seed_sets = select_seeds_for_all_methods(graph, max_seeds)
    
    # Step 2: Evaluate performance for different seed counts
    results = evaluate_seed_performance(graph, seed_sets, seed_counts, mc_simulations=100)
    
    # Step 3: Print detailed results
    print_detailed_results(results, seed_counts)
    
    # Step 4: Create visualizations
    print(f"\nGenerating performance comparison plots...")
    plot_seed_performance_comparison(results, seed_counts, network_name)
    
    print(f"Generating seed overlap analysis...")
    plot_seed_overlap_analysis(seed_sets, seed_counts, network_name)
    
    return results, seed_sets

if __name__ == "__main__":
    # Parameters
    NUM_NODES = 100
    SEED_COUNTS = [1, 2, 3, 4, 5, 10, 15, 20, 25]
    MC_SIMULATIONS = 100
    
    print("=== Seed Count Performance Analysis ===")
    print(f"Parameters: N={NUM_NODES}, Seed Counts={SEED_COUNTS}, MC Sims={MC_SIMULATIONS}\n")
    
    # Define network generation methods
    network_generators = {
        "Erdős-Rényi": lambda: create_synthetic_network(NUM_NODES, 0.3),
        "Barabási-Albert": lambda: create_barabasi_albert_network(NUM_NODES, 3),
        "Watts-Strogatz": lambda: create_watts_strogatz_network(NUM_NODES, 6, 0.3),
        "Community": lambda: create_community_network(NUM_NODES, 3, 0.3, 0.05),
        "Scale-Free": lambda: create_scale_free_network(NUM_NODES, 2.5),
        "Bridge Network": lambda: create_bridge_network(NUM_NODES, 20, 0.3)
    }
    
    # Store all results
    all_results = {}
    
    # Analyze each network type
    for network_name, network_generator in network_generators.items():
        print(f"\n{'='*60}")
        print(f"Testing on {network_name} Network")
        print(f"{'='*60}")
        
        # Generate network
        G = network_generator()
        
        # Analyze performance
        results, seed_sets = analyze_network_performance(G, network_name, SEED_COUNTS)
        
        # Store results
        all_results[network_name] = {
            'results': results,
            'seed_sets': seed_sets
        }
    
    # Overall summary
    print(f"\n{'='*80}")
    print("OVERALL SUMMARY ACROSS ALL NETWORKS")
    print(f"{'='*80}")
    
    # Aggregate results across networks
    method_performance = defaultdict(list)
    
    for network_name, network_data in all_results.items():
        results = network_data['results']
        
        for method_name in results.keys():
            # Use k=5 as a representative seed count
            if 5 in results[method_name]:
                influence = results[method_name][5]['avg_influence']
                method_performance[method_name].append(influence)
    
    print(f"\nAverage Influence Fraction (k=5) Across All Networks:")
    print("-" * 60)
    for method_name, influences in method_performance.items():
        avg_influence = np.mean(influences)
        std_influence = np.std(influences)
        print(f"{method_name:<20}: {avg_influence:.3f} ± {std_influence:.3f}")
    
    # Find best performing method
    best_method = max(method_performance.items(), key=lambda x: np.mean(x[1]))
    print(f"\nBest performing method: {best_method[0]} (avg: {np.mean(best_method[1]):.3f})")
    
    print(f"\nAnalysis complete! Check the generated plots for detailed visualizations.")
