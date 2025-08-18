# Seed Count Performance Analysis

A comprehensive analysis tool that compares the performance of different influence maximization methods based on the number of seed nodes selected.

## 🎯 Purpose

This analysis answers critical questions about influence maximization:
- **How does performance scale with the number of seeds?**
- **Which method performs best for different seed counts?**
- **How much marginal gain do additional seeds provide?**
- **How similar are the seed selections between different methods?**

## 📊 Analysis Overview

### **Seed Counts Analyzed**
- **k = 1, 2, 3, 4, 5**: Small seed sets (typical for viral marketing)
- **k = 10, 15, 20, 25**: Larger seed sets (campaign optimization)

### **Methods Compared**
1. **Greedy Algorithm**: Optimal but computationally expensive
2. **TDA (Topological Data Analysis)**: Novel topological approach
3. **Degree Centrality**: Fast heuristic based on node degrees
4. **Betweenness Centrality**: Fast heuristic based on node betweenness

### **Networks Tested**
- **Erdős-Rényi**: Random networks
- **Barabási-Albert**: Scale-free networks
- **Watts-Strogatz**: Small-world networks
- **Community**: Networks with community structure
- **Scale-Free**: Power-law degree distribution
- **Bridge Network**: Networks with low-degree bridge nodes

## 🚀 Usage

### **Basic Usage**

```bash
python seed_analysis.py
```

This will:
1. Generate 6 different network types
2. Select seeds for all methods (up to 25 seeds)
3. Run 100 Monte Carlo simulations for each seed count
4. Generate comprehensive visualizations
5. Provide detailed performance analysis

### **Parameters**

You can modify these parameters in the script:

```python
NUM_NODES = 100                    # Number of nodes in networks
SEED_COUNTS = [1, 2, 3, 4, 5, 10, 15, 20, 25]  # Seed counts to analyze
MC_SIMULATIONS = 100               # Monte Carlo simulations per configuration
```

## 📈 Analysis Process

### **Step 1: Seed Selection**
- Run each method to select up to 25 seeds
- Store the complete ordered list of selected seeds
- This ensures fair comparison across different seed counts

### **Step 2: Performance Evaluation**
- For each seed count k, take the top k seeds from each method
- Run 100 Monte Carlo simulations for each configuration
- Calculate average influence spread and standard deviation

### **Step 3: Analysis & Visualization**
- Generate performance comparison plots
- Analyze seed selection overlap between methods
- Provide detailed statistical summaries

## 📊 Generated Outputs

### **1. Performance Comparison Plots**

#### **Influence Spread vs Number of Seeds**
- Shows how influence spread increases with more seeds
- Includes error bars showing standard deviation
- Uses log scale for better visualization of trends
- Compares all methods on the same plot

#### **Marginal Gain per Additional Seed**
- Shows the incremental benefit of each additional seed
- Helps identify diminishing returns
- Reveals which methods provide better marginal gains

### **2. Seed Overlap Analysis**

#### **Overlap Heatmaps**
- Shows similarity between method selections for each seed count
- Uses Jaccard similarity (intersection/union)
- Diagonal shows number of seeds selected
- Off-diagonal shows overlap between different methods

### **3. Detailed Results Table**

```
Method                | k=1     | k=2     | k=3     | k=4     | k=5     | ...
Greedy                | 15.2±2.1| 28.5±3.2| 41.3±4.1| 52.8±4.8| 62.1±5.2| ...
TDA                   | 14.8±2.0| 27.9±3.1| 40.1±3.9| 51.2±4.5| 60.8±4.9| ...
Degree Centrality     | 13.5±1.9| 25.8±2.8| 37.2±3.6| 47.9±4.2| 57.1±4.6| ...
Betweenness Centrality| 14.1±2.0| 26.9±3.0| 38.7±3.7| 49.1±4.3| 58.3±4.7| ...
```

### **4. Best Method Summary**

```
BEST PERFORMING METHOD FOR EACH SEED COUNT:
k= 1: Greedy                (15.2 nodes influenced)
k= 2: Greedy                (28.5 nodes influenced)
k= 3: Greedy                (41.3 nodes influenced)
k= 4: Greedy                (52.8 nodes influenced)
k= 5: Greedy                (62.1 nodes influenced)
...
```

## 🔍 Key Insights

### **Performance Scaling**
- **Small seed sets (k ≤ 5)**: All methods perform similarly
- **Medium seed sets (k = 10-15)**: TDA often catches up to Greedy
- **Large seed sets (k ≥ 20)**: Diminishing returns become apparent

### **Method Characteristics**

#### **Greedy Algorithm**
- **Consistent performance**: Best or near-best across all seed counts
- **Diminishing returns**: Marginal gains decrease with more seeds
- **Computational cost**: Expensive for large seed sets

#### **TDA Algorithm**
- **Competitive performance**: Often matches Greedy for larger seed sets
- **Topological insights**: May identify different types of important nodes
- **Scalability**: Moderate computational cost

#### **Centrality Methods**
- **Fast execution**: Quick to compute for any seed count
- **Consistent selection**: Similar performance across seed counts
- **Limited insight**: May miss complex topological features

### **Network-Specific Patterns**

#### **Scale-Free Networks**
- **Degree centrality**: Often performs well due to hub structure
- **TDA**: May identify structural bridges between communities

#### **Community Networks**
- **TDA**: Excels at identifying bridge nodes between communities
- **Betweenness centrality**: Also good at finding bridges

#### **Bridge Networks**
- **TDA**: Designed to outperform degree centrality
- **Degree centrality**: Fails to identify low-degree bridge nodes

## 📊 Expected Results

### **Typical Performance Rankings**

1. **Greedy**: Highest influence spread across all seed counts
2. **TDA**: Competitive performance, especially for larger seed sets
3. **Betweenness Centrality**: Moderate performance
4. **Degree Centrality**: Good for scale-free networks, poor for bridge networks

### **Marginal Gain Patterns**

- **k = 1 to 5**: High marginal gains, steep learning curve
- **k = 5 to 10**: Moderate gains, good value for additional seeds
- **k = 10 to 15**: Declining gains, diminishing returns
- **k = 15 to 25**: Low gains, may not be cost-effective

## 🎨 Visualization Features

### **Interactive Elements**
- **Error bars**: Show uncertainty in performance estimates
- **Log scale**: Better visualization of trends across seed counts
- **Color coding**: Consistent colors for each method
- **Grid lines**: Easy reading of values

### **Statistical Information**
- **Mean and standard deviation**: Robust performance estimates
- **Confidence intervals**: Uncertainty quantification
- **Marginal analysis**: Incremental benefit analysis

## 🔧 Customization

### **Adding New Methods**

```python
def custom_selection_method(graph, k):
    """Your custom seed selection method"""
    # Your implementation
    return selected_seeds

# Add to seed_sets in select_seeds_for_all_methods()
seed_sets["Custom Method"] = custom_selection_method(graph, max_seeds)
```

### **Modifying Seed Counts**

```python
# Analyze different seed count ranges
SEED_COUNTS = [1, 2, 5, 10, 20, 50, 100]  # For larger networks
SEED_COUNTS = [1, 2, 3, 4, 5]             # For quick testing
```

### **Custom Network Types**

```python
def create_custom_network(num_nodes):
    """Your custom network generator"""
    # Your implementation
    return G

# Add to network_generators
network_generators["Custom Network"] = lambda: create_custom_network(NUM_NODES)
```

## 📈 Research Applications

### **Viral Marketing**
- **Optimal campaign size**: How many influencers to recruit
- **Budget allocation**: Cost-benefit analysis of seed set size
- **Platform selection**: Which social networks to target

### **Disease Control**
- **Vaccination strategies**: How many people to vaccinate
- **Contact tracing**: Optimal number of initial cases to trace
- **Resource allocation**: Limited resources for intervention

### **Information Diffusion**
- **News spreading**: Optimal number of initial sources
- **Rumor control**: How many fact-checkers to deploy
- **Emergency communication**: Optimal number of emergency contacts

## 🐛 Troubleshooting

### **Common Issues**

1. **Memory Issues**: Reduce `NUM_NODES` or `MC_SIMULATIONS`
2. **Slow Execution**: Reduce `MC_SIMULATIONS` for faster results
3. **Plot Display**: Ensure matplotlib backend is configured correctly
4. **Import Errors**: Ensure all dependencies are installed

### **Performance Tips**

- Start with smaller networks for testing
- Use fewer Monte Carlo simulations for quick results
- Focus on specific seed count ranges of interest
- Monitor memory usage for large networks

## 📄 Files

- `seed_analysis.py`: Main analysis script
- `SEED_ANALYSIS_README.md`: This documentation
- Generated plots: Various visualization files

## 🤝 Contributing

Feel free to:
- Add new seed selection methods
- Implement additional performance metrics
- Improve visualization capabilities
- Add statistical significance testing

## 📞 Contact

For questions about the seed analysis, please refer to the main project documentation or open an issue in the repository.
