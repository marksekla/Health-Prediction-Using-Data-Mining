import os
import pandas as pd
import numpy as np
import time

from src.preprocessing import load_data, preprocess_heart_data, create_transaction_data
from src.apriori import Apriori
from src.minhash import MinHash
from src.visualization import (
    visualize_frequent_itemsets, visualize_association_rules,
    visualize_similarity_network, compare_algorithm_performances
)

def main():
    print("\n")
    print("="*25)
    print("Health Prediction Project")
    print("="*25)
    
    # Load and preprocess data
    print("\nLoading data...")
    
    if not os.path.exists('data'):
        os.makedirs('data')
        print("Created data folder - please add your dataset here!")
        return
    
    data_path = 'data/heart_disease.csv'
    if not os.path.exists(data_path):
        print(f"Couldn't find the dataset! Make sure it's in {data_path}")
        return
    
    data = load_data(data_path)
    if data is None:
        return
    
    # Preprocess data
    processed_data = preprocess_heart_data(data)
    transactions = create_transaction_data(processed_data)
    
    # Apply Apriori
    print("\nRunning Apriori to find patterns...")
    print()
    
    # Basic Apriori
    print("Standard Apriori:")
    baseline_apriori = Apriori(min_support=0.1, min_confidence=0.5)
    baseline_frequent_itemsets = baseline_apriori.fit(transactions)
    baseline_rules = baseline_apriori.rules
    baseline_apriori_time = baseline_apriori.execution_time
    
    print(f"Found {len(baseline_frequent_itemsets)} patterns and {len(baseline_rules)} rules")
    print()
    
    # Enhanced Apriori
    print("Enhanced Apriori:")
    enhanced_apriori = Apriori(min_support=0.05, min_confidence=0.6)
    enhanced_frequent_itemsets = enhanced_apriori.fit(transactions)
    enhanced_rules = enhanced_apriori.rules
    enhanced_apriori_time = enhanced_apriori.execution_time
    
    print(f"Found {len(enhanced_frequent_itemsets)} patterns and {len(enhanced_rules)} rules")
    
    # MinHash
    print("\nRunning MinHash to find similar patients...")
    print()
    
    # Create profile sets
    profile_sets = []
    for _, row in processed_data.iterrows():
        profile_features = {f"{col}={val}" for col, val in row.items() if pd.notna(val)}
        profile_sets.append(profile_features)
    
    # Basic MinHash
    print("Standard MinHash:")
    baseline_minhash = MinHash(n_hash_functions=100)
    baseline_minhash.fit_transform(profile_sets)
    baseline_similar_pairs = baseline_minhash.get_similar_profiles(threshold=0.5)
    baseline_minhash_time = baseline_minhash.execution_time
    print()
    
    # Enhanced MinHash 
    print("Enhanced MinHash with LSH:")
    enhanced_minhash = MinHash(n_hash_functions=100)
    enhanced_similar_pairs = enhanced_minhash.enhanced_minhash(profile_sets, bands=20)
    enhanced_minhash_time = enhanced_minhash.execution_time
    
    # Visualize Results
    print("\nCreating visualizations...")
    
    all_visualizations_created = True
    
    try:
        visualize_frequent_itemsets(baseline_frequent_itemsets, top_n=10)
        visualize_association_rules(baseline_rules, top_n=10)
        visualize_similarity_network(baseline_similar_pairs, processed_data, threshold=0.6)
        
        # Compare performance
        apriori_baseline_results = {
            'n_itemsets': len(baseline_frequent_itemsets),
            'n_rules': len(baseline_rules)
        }
        
        apriori_enhanced_results = {
            'n_itemsets': len(enhanced_frequent_itemsets),
            'n_rules': len(enhanced_rules)
        }
        
        compare_algorithm_performances(
            baseline_apriori_time, enhanced_apriori_time,
            apriori_baseline_results, apriori_enhanced_results
        )
        
        # Compare MinHash results
        minhash_baseline_results = {
            'n_similar_pairs': len(baseline_similar_pairs)
        }
        
        minhash_enhanced_results = {
            'n_similar_pairs': len(enhanced_similar_pairs)
        }
        
        compare_algorithm_performances(
            baseline_minhash_time, enhanced_minhash_time,
            minhash_baseline_results, minhash_enhanced_results
        )
    except Exception as e:
        print(f"Error creating visualizations: {e}")
        all_visualizations_created = False
    
    if all_visualizations_created:
        print("All visualizations saved to images folder")
    
    print("\nKey findings:")
    print("Top patterns found:")
    for i, itemset in enumerate(baseline_frequent_itemsets[:3]):
        items_str = ', '.join(itemset['itemset'])
        print(f"{i+1}. {items_str} (support: {itemset['support']:.3f})")
    
    print("\nTop rules:")
    for i, rule in enumerate(baseline_rules[:3]):
        antecedent = ', '.join(rule['antecedent'])
        consequent = ', '.join(rule['consequent'])
        print(f"{i+1}. {antecedent} -> {consequent} (conf: {rule['confidence']:.3f})")
    print()
    print("Done! Check the images folder for visualizations.")

if __name__ == "__main__":
    main()