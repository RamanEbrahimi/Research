# Multiplex Network TDA Influence Maximization

An extension of the TDA Influence Maximization project to handle **multiplex networks** - networks with multiple layers where each layer represents different types of relationships or interactions.

## 🌐 What are Multiplex Networks?

Multiplex networks consist of multiple layers, where each layer represents a different type of relationship between the same set of nodes. For example:
- **Social Networks**: Facebook friends (layer 1), Twitter followers (layer 2), LinkedIn connections (layer 3)
- **Transportation**: Road network (layer 1), subway network (layer 2), bus routes (layer 3)
- **Biological Networks**: Protein-protein interactions (layer 1), gene co-expression (layer 2), metabolic pathways (layer 3)

## 🚀 Key Features

### **Multiplex Independent Cascade (IC) Model**
- **Layer-specific spreading coefficients**: Each layer has its own influence strength
- **Cross-layer activation**: A node activated in any layer is considered active in all layers
- **Independent spreading**: Influence spreads independently in each layer but activation is global

### **TDA for Multiplex Networks**
- **Multi-layer persistence analysis**: Computes persistent homology for each layer
- **Cross-layer topological scoring**: Combines topological information across all layers
- **Layer-weighted importance**: Weights node importance by layer-specific spreading coefficients

### **Network Visualization**
- **Multi-layer visualization**: Shows all layers side-by-side
- **Seed node highlighting**: Highlights selected seed nodes across all layers
- **Layer-specific metrics**: Displays spreading coefficients and edge counts per layer

## 📊 Multiplex Network Structure

### **Layer Types**
Each layer can have different network topologies:
1. **Erdős-Rényi**: Random connections
2. **Barabási-Albert**: Scale-free with preferential attachment
3. **Watts-Strogatz**: Small-world with rewiring
4. **Community**: Clear community structure
5. **Scale-Free**: Power-law degree distribution

### **Spreading Coefficients**
- **Layer 0**: 0.15 (low influence strength)
- **Layer 1**: 0.25 (medium influence strength)
- **Layer 2**: 0.35 (high influence strength)

## 🎯 Usage

### **Basic Usage**

```bash
python multiplex_tda.py
```

This will:
- Create a 3-layer multiplex network with 50 nodes
- Run 3 different influence maximization algorithms
- Generate comprehensive visualizations
- Provide detailed performance analysis

### **Parameters**

You can modify these parameters in the script:

```python
NUM_NODES = 50        # Number of nodes in the network
NUM_LAYERS = 3        # Number of layers in the multiplex network
NUM_SEEDS = 5         # Number of seed nodes to select
MC_SIMULATIONS = 50   # Monte Carlo simulations for accuracy
```

### **Custom Multiplex Network**

```python
# Create a custom multiplex network
multiplex = MultiplexNetwork(num_nodes=100, num_layers=4)

# Add layers with different topologies
layer0 = create_erdos_renyi_network(100, 0.2)
multiplex.add_layer(0, layer0, spreading_coefficient=0.1)

layer1 = create_barabasi_albert_network(100, m=2)
multiplex.add_layer(1, layer1, spreading_coefficient=0.3)

# Run TDA selection
seeds, time = multiplex_tda_selection(multiplex, k=10)
```

## 🔬 Algorithm Details

### **Multiplex Greedy Algorithm**
- **Approach**: Iteratively selects nodes with maximum marginal gain across all layers
- **Complexity**: O(kn²L) where L is number of layers
- **Guarantee**: (1-1/e) approximation to optimal solution
- **Best for**: Small to medium multiplex networks

### **Multiplex TDA Algorithm**
- **Approach**: 
  1. Computes persistence diagrams for each layer
  2. Scores nodes based on topological importance in each layer
  3. Combines scores weighted by layer spreading coefficients
- **Complexity**: O(n³L) for persistence computation across all layers
- **Best for**: Multiplex networks with clear topological structure

### **Multiplex Degree Centrality**
- **Approach**: Aggregates degree centrality across all layers
- **Method**: Weighted sum of degree centrality by layer spreading coefficients
- **Complexity**: O(nL)
- **Best for**: Fast approximation in large multiplex networks

## 📈 Expected Results

### **Performance Characteristics**

- **Multiplex Greedy**: Highest influence spread, slowest execution
- **Multiplex TDA**: Competitive influence spread, moderate speed
- **Multiplex Degree Centrality**: Fast execution, moderate influence spread

### **Layer-Specific Insights**

- **High spreading coefficient layers**: More important for influence spread
- **Dense layers**: Provide more opportunities for local spreading
- **Sparse layers**: May contain critical bridge connections
- **Heterogeneous layers**: TDA can identify cross-layer topological importance

## 🎨 Generated Visualizations

### **1. Multiplex Network Structure**
- Shows all layers side-by-side
- Displays layer-specific spreading coefficients
- Highlights selected seed nodes in red

### **2. Influence Spreading Process**
- Tracks activation across all layers
- Shows confidence intervals
- Compares different selection methods

### **3. Performance Comparison**
- Influence spread comparison
- Selection time comparison
- Method ranking

## 🔧 Advanced Features

### **Custom Layer Creation**

```python
def create_custom_layer(num_nodes, topology_type, **params):
    """Create a custom layer with specific topology"""
    if topology_type == "custom":
        G = nx.Graph()
        # Your custom network generation logic
        return G
    # ... other topologies
```

### **Custom Spreading Model**

```python
def custom_multiplex_ic_model(multiplex, seed_nodes, **params):
    """Custom multiplex IC model with additional parameters"""
    # Your custom spreading logic
    pass
```

### **Layer-Specific Analysis**

```python
# Analyze individual layers
for layer_id, layer_graph in multiplex.layers.items():
    layer_centrality = nx.betweenness_centrality(layer_graph)
    print(f"Layer {layer_id} betweenness centrality: {layer_centrality}")
```

## 🐛 Troubleshooting

### **Common Issues**

1. **Memory Issues**: Reduce `NUM_NODES` or `NUM_LAYERS` for large networks
2. **Slow Execution**: TDA computation scales with O(n³L)
3. **Layer Connectivity**: Ensure all layers are connected
4. **Node Alignment**: All layers should have the same set of nodes

### **Performance Tips**

- Use fewer layers for quick testing
- Reduce MC simulations for faster execution
- Consider layer importance when setting spreading coefficients
- Monitor memory usage for very large multiplex networks

## 📚 Research Applications

### **Social Networks**
- Multi-platform influence maximization
- Cross-platform viral marketing
- Social media campaign optimization

### **Biological Networks**
- Multi-omics data integration
- Disease propagation modeling
- Drug target identification

### **Transportation Networks**
- Multi-modal transportation planning
- Infrastructure optimization
- Emergency response planning

### **Financial Networks**
- Multi-market risk analysis
- Systemic risk assessment
- Portfolio optimization

## 🔗 Integration with Main Project

The multiplex extension can be used alongside the main TDA influence maximization project:

```python
# Compare single-layer vs multiplex performance
single_layer_results = run_single_layer_analysis()
multiplex_results = run_multiplex_analysis()

# Compare results
compare_performance(single_layer_results, multiplex_results)
```

## 📄 Files

- `multiplex_tda.py`: Main multiplex network analysis script
- `MULTIPLEX_README.md`: This documentation file
- Generated plots: Various visualization files

## 🤝 Contributing

Feel free to:
- Add new layer generation methods
- Implement additional multiplex algorithms
- Improve visualization capabilities
- Add real-world network datasets

## 📞 Contact

For questions about the multiplex extension, please refer to the main project documentation or open an issue in the repository.
