# TDA Influence Maximization

A comprehensive comparison of influence maximization algorithms using Topological Data Analysis (TDA) and traditional methods across multiple network topologies.

## 📋 Overview

This project implements and compares different influence maximization algorithms:

- **Greedy Algorithm**: Optimal but computationally expensive
- **TDA (Topological Data Analysis)**: Novel approach using persistent homology
- **Degree Centrality**: Fast heuristic based on node degrees
- **Betweenness Centrality**: Fast heuristic based on node betweenness

The algorithms are tested on various synthetic network topologies to understand their performance characteristics.

## 🏗️ Network Types

The project supports multiple network generation models:

1. **Erdős-Rényi**: Random networks with uniform edge probability
2. **Barabási-Albert**: Scale-free networks with preferential attachment
3. **Watts-Strogatz**: Small-world networks with rewiring
4. **Community Networks**: Networks with clear community structure
5. **Scale-Free**: Networks with power-law degree distribution
6. **Bridge Network**: Networks with low-degree bridge nodes that are topologically important

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone or download the project files**

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv myenv
   source myenv/bin/activate  # On Windows: myenv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 📦 Dependencies

The main dependencies include:

- **NetworkX**: Network analysis and generation
- **NumPy**: Numerical computations
- **Matplotlib & Seaborn**: Plotting and visualization
- **giotto-tda**: Topological data analysis library
- **scikit-learn**: Machine learning utilities

See `requirements.txt` for the complete list with specific versions.

## 🎯 Usage

### Basic Usage

Run the main comparison script:

```bash
python tda_test.py
```

This will:
- Generate 5 different types of synthetic networks
- Run 4 different influence maximization algorithms on each network
- Generate comprehensive comparison plots
- Provide detailed performance analysis

### Parameters

You can modify the following parameters in the script:

```python
NUM_NODES = 100       # Number of nodes in networks
NUM_SEEDS = 5         # Number of seed nodes to select
MC_SIMULATIONS = 50   # Monte Carlo simulations for accuracy
TOP_K_VALUES = [1, 2, 3, 4, 5, 10, 15, 20, 25]  # For overlap analysis
```

### Individual Functions

You can also use individual functions for custom analysis:

```python
# Generate a specific network type
G = create_barabasi_albert_network(100, m=3)

# Run a specific algorithm
seeds, influence, time = greedy_selection(G, k=5, mc_simulations=100)

# Evaluate influence spread
influence = run_ic_model(G, seeds, mc_simulations=100)
```

## 📊 Output and Results

### Console Output

The script provides detailed console output including:
- Network generation statistics
- Algorithm performance metrics
- Comparison summaries for each network type
- Overall performance analysis

### Generated Plots

For each network type, the following plots are generated:

1. **Spreading Comparison** (`spreading_comparison_[network].png`)
   - Shows how influence spreads over time for each method
   - Includes confidence intervals
   - Helps identify which methods achieve faster/better spreading

2. **Node Overlap Heatmaps** (`overlap_heatmaps_[network].png`)
   - Shows similarity between method selections for different top-k values
   - Uses Jaccard similarity metric
   - Helps understand method agreement

3. **Overall Comparison** (`overall_comparison.png`)
   - Bar charts comparing performance across all networks
   - Shows both influence spread and selection time

### Performance Metrics

The analysis provides:
- **Influence Spread**: Average number of nodes influenced
- **Selection Time**: Time taken to select seed nodes
- **Node Overlap**: Similarity between different method selections
- **Statistical Summary**: Mean and standard deviation across networks

## 🔬 Algorithm Details

### Greedy Algorithm
- **Approach**: Iteratively selects nodes with maximum marginal gain
- **Guarantee**: (1-1/e) approximation to optimal solution
- **Complexity**: O(kn²) where k is number of seeds, n is number of nodes
- **Best for**: Small to medium networks where optimality is crucial

### TDA Algorithm
- **Approach**: Uses persistent homology to identify topologically important nodes
- **Method**: Computes Vietoris-Rips persistence diagrams and scores nodes based on feature persistence
- **Complexity**: O(n³) for persistence computation
- **Best for**: Networks with clear topological structure

### Degree Centrality
- **Approach**: Selects nodes with highest degrees
- **Complexity**: O(n)
- **Best for**: Fast approximation in large networks

### Betweenness Centrality
- **Approach**: Selects nodes with highest betweenness centrality
- **Complexity**: O(n³) for exact computation
- **Best for**: Networks where bridge nodes are important

## 📈 Expected Results

### Performance Characteristics

- **Greedy**: Highest influence spread, slowest execution
- **TDA**: Competitive influence spread, moderate speed
- **Degree Centrality**: Fast execution, moderate influence spread
- **Betweenness Centrality**: Moderate speed and influence spread

### Network-Specific Performance

- **Scale-free networks**: Degree centrality often performs well
- **Community networks**: TDA may identify bridge nodes effectively
- **Random networks**: All methods perform similarly
- **Small-world networks**: Betweenness centrality may be effective
- **Bridge networks**: TDA should outperform degree centrality by identifying low-degree but topologically important bridge nodes

## 🛠️ Customization

### Adding New Network Types

```python
def create_custom_network(num_nodes=50, **params):
    """Create your custom network type"""
    G = nx.your_network_generator(num_nodes, **params)
    
    # Add weights
    for (u, v) in G.edges():
        G.edges[u, v]['weight'] = np.random.uniform(0.01, 0.2)
    
    return G
```

### Adding New Algorithms

```python
def custom_selection(graph, k):
    """Your custom seed selection algorithm"""
    start_time = time.time()
    
    # Your algorithm implementation
    seed_set = your_algorithm(graph, k)
    
    end_time = time.time()
    return seed_set, end_time - start_time
```

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed correctly
2. **Memory Issues**: Reduce `NUM_NODES` or `MC_SIMULATIONS` for large networks
3. **Slow Execution**: TDA computation can be slow for large networks
4. **Plot Display**: Ensure matplotlib backend is configured correctly

### Performance Tips

- Use smaller networks for quick testing
- Reduce MC simulations for faster execution
- Consider using parallel processing for large-scale experiments
- Monitor memory usage for very large networks

## 📚 References

- Kempe, D., Kleinberg, J., & Tardos, É. (2003). Maximizing the spread of influence through a social network.
- Edelsbrunner, H., & Harer, J. (2010). Computational topology: an introduction.
- NetworkX Documentation: https://networkx.org/
- giotto-tda Documentation: https://giotto-ai.github.io/gtda-docs/

## 📄 License

This project is provided as-is for educational and research purposes.

## 🤝 Contributing

Feel free to:
- Report bugs or issues
- Suggest improvements
- Add new network types or algorithms
- Improve documentation

## 📞 Contact

For questions or feedback, please open an issue in the project repository.
