import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np
import seaborn as sns
import os

# Make sure we have an images folder at all times
if not os.path.exists('images'):
    os.makedirs('images')

def visualize_frequent_itemsets(itemsets, top_n=10):
    """Shows top patterns as a bar chart"""
    if not itemsets:
        print("No patterns to visualize")
        return
    
    # Take top N itemsets
    top_itemsets = itemsets[:top_n]
    
    # Format for display
    labels = [', '.join(sorted(list(itemset['itemset']))) for itemset in top_itemsets]
    supports = [itemset['support'] for itemset in top_itemsets]
    
    plt.figure(figsize=(12, 6))
    bars = plt.barh(range(len(labels)), supports, align='center')
    plt.yticks(range(len(labels)), labels)
    plt.xlabel('Support')
    plt.title(f'Top {top_n} Frequent Patterns')
    
    # Add values as text
    for i, bar in enumerate(bars):
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{supports[i]:.3f}', va='center')
    
    plt.savefig('images/frequent_itemsets.png')
    plt.close()

def visualize_association_rules(rules, top_n=10):
    """Shows rules as a network graph"""
    if not rules:
        print("No rules to visualize")
        return
    
    # Get top rules
    top_rules = rules[:top_n]
    
    # Create graph
    G = nx.DiGraph()
    
    # Add connections for each rule
    for rule in top_rules:
        antecedent = ', '.join(sorted(rule['antecedent']))
        consequent = ', '.join(sorted(rule['consequent']))
        
        G.add_node(antecedent, type='antecedent')
        G.add_node(consequent, type='consequent')
        G.add_edge(antecedent, consequent, 
                  weight=rule['confidence'],
                  support=rule['support'],
                  label=f"conf={rule['confidence']:.2f}\nsupp={rule['support']:.2f}")
    
    plt.figure(figsize=(12, 10))
    
    # Position nodes
    pos = nx.spring_layout(G, seed=42)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, 
                           node_color=['lightblue' if G.nodes[n]['type'] == 'antecedent' else 'lightgreen' 
                                    for n in G.nodes],
                           node_size=2000, alpha=0.8)
    
    # Draw connections
    edge_widths = [G[u][v]['weight'] *.3 for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.7, 
                          edge_color='gray', arrows=True, arrowsize=20)
    
    # Add labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    
    # Add edge labels
    edge_labels = {(u, v): G[u][v]['label'] for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    
    plt.title(f'Top {top_n} Association Rules')
    plt.axis('off')
    plt.savefig('images/association_rules.png')
    plt.close()

def visualize_similarity_network(similar_pairs, profile_data, threshold=0.7):
    """Shows patient similarities as a network"""
    if not similar_pairs:
        print("No similar patients to visualize")
        return
    
    # Filter by similarity
    filtered_pairs = [(i, j, sim) for i, j, sim in similar_pairs if sim >= threshold]
    
    if not filtered_pairs:
        print(f"No patient pairs with similarity above {threshold}")
        return
    
    # Create graph
    G = nx.Graph()
    
    # Add patient nodes
    unique_profiles = set()
    for i, j, _ in filtered_pairs:
        unique_profiles.add(i)
        unique_profiles.add(j)
    
    for profile_idx in unique_profiles:
        G.add_node(profile_idx)
    
    # Add similarity connections
    for i, j, similarity in filtered_pairs:
        G.add_edge(i, j, weight=similarity, label=f"{similarity:.2f}")
    
    plt.figure(figsize=(14, 12))
    
    # Position nodes
    pos = nx.spring_layout(G, seed=42)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                          node_size=500, alpha=0.8)
    
    # Draw edges with varying thickness
    edge_colors = [G[u][v]['weight'] for u, v in G.edges()]
    edge_widths = [G[u][v]['weight'] * 3 for u, v in G.edges()]
    
    edges = nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.7,
                                edge_color=edge_colors, edge_cmap=plt.get_cmap('Blues'))
    
    # Add node labels
    nx.draw_networkx_labels(G, pos, font_size=10)
    
    # Add edge labels
    edge_labels = {(u, v): f"{G[u][v]['weight']:.2f}" for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    
    plt.colorbar(edges, label='Similarity')
    plt.title(f'Patient Similarity Network (threshold={threshold})')
    plt.axis('off')
    plt.savefig('images/similarity_network.png')
    plt.close()

def compare_algorithm_performances(baseline_time, enhanced_time, baseline_results, enhanced_results):
    """Compares algorithm versions"""
    # Time comparison
    plt.figure(figsize=(10, 6))
    
    algorithms = ['Baseline', 'Enhanced']
    times = [baseline_time, enhanced_time]
    
    plt.bar(algorithms, times, color=['#1f77b4', '#ff7f0e'])
    plt.ylabel('Time (seconds)')
    plt.title('Execution Time Comparison')
    
    # Add time values
    for i, time in enumerate(times):
        plt.text(i, time + 0.1, f'{time:.2f}s', ha='center')
    
    plt.savefig('images/time_comparison.png')
    plt.close()
    
    # Results comparison
    if isinstance(baseline_results, dict) and isinstance(enhanced_results, dict):
        if 'n_itemsets' in baseline_results and 'n_itemsets' in enhanced_results:
            plt.figure(figsize=(10, 6))
            
            metrics = ['Frequent Itemsets', 'Association Rules']
            baseline_values = [baseline_results['n_itemsets'], baseline_results.get('n_rules', 0)]
            enhanced_values = [enhanced_results['n_itemsets'], enhanced_results.get('n_rules', 0)]
            
            x = np.arange(len(metrics))
            width = 0.35
            
            plt.bar(x - width/2, baseline_values, width, label='Baseline')
            plt.bar(x + width/2, enhanced_values, width, label='Enhanced')
            
            plt.xlabel('Metric')
            plt.ylabel('Count')
            plt.title('Algorithm Results Comparison')
            plt.xticks(x, metrics)
            plt.legend()
            
            plt.savefig('images/results_comparison.png')
            plt.close()