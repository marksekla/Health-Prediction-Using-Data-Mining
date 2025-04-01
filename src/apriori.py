from collections import defaultdict
import time

class Apriori:
    def __init__(self, min_support=0.1, min_confidence=0.5):
        """Setup Apriori algorithm with support and confidence thresholds"""
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.itemsets = []
        self.frequent_itemsets = {}
        self.rules = []
        self.execution_time = 0
        
        # ENHANCEMENT: Critical health indicators that affect rule confidence
        # These are items that are particularly important for health prediction
        self.critical_indicators = ["cp=2", "cp=3", "trestbps_bin=high", "trestbps_bin=very_high", 
                                   "chol_bin=high", "chol_bin=very_high"]
    
    def fit(self, transactions):
        """Run Apriori algorithm on transaction data"""
        start_time = time.time()
        
        # Count items
        item_counts = defaultdict(int)
        for transaction in transactions:
            for item in transaction:
                item_counts[frozenset([item])] += 1
        
        # Total transactions
        n_transactions = len(transactions)
        
        # Find items with minimum support
        self.frequent_itemsets[1] = {
            itemset: count for itemset, count in item_counts.items()
            if count / n_transactions >= self.min_support
        }
        
        print(f"Found {len(self.frequent_itemsets[1])} frequent items")
        
        k = 2
        while self.frequent_itemsets[k-1]:
            # Generate candidates
            candidates = self._generate_candidates(self.frequent_itemsets[k-1], k)
            
            # Count support
            candidate_counts = defaultdict(int)
            for transaction in transactions:
                transaction_set = frozenset(transaction)
                for candidate in candidates:
                    if candidate.issubset(transaction_set):
                        candidate_counts[candidate] += 1
            
            # ENHANCEMENT: Dynamic support adjustment
            # Gradually reduce support threshold for larger itemsets to find more complex patterns
            # This helps discover important relationships that would be missed with fixed thresholds
            adjusted_support = max(0.01, self.min_support / (1 + 0.15 * (k-1)))
            
            # Filter by adjusted support
            self.frequent_itemsets[k] = {
                itemset: count for itemset, count in candidate_counts.items()
                if count / n_transactions >= adjusted_support  # Using adjusted support
            }
            
            print(f"Found {len(self.frequent_itemsets[k])} patterns of size {k}")
            k += 1
        
        # Generate rules
        self._generate_rules(n_transactions)
        
        self.execution_time = time.time() - start_time
        print(f"Apriori finished in {self.execution_time:.2f} seconds")
        
        # Format results
        all_frequent_itemsets = []
        for k, itemsets in self.frequent_itemsets.items():
            if k == 1:  # Skip 1-itemsets for clarity
                continue
            for itemset, count in itemsets.items():
                support = count / n_transactions
                all_frequent_itemsets.append({
                    'itemset': list(itemset),
                    'support': support,
                    'count': count
                })
        
        # Sort by support
        all_frequent_itemsets.sort(key=lambda x: x['support'], reverse=True)
        return all_frequent_itemsets
    
    def _generate_candidates(self, prev_frequent, k):
        """Generate candidate itemsets"""
        candidates = set()
        prev_items = list(prev_frequent.keys())
        
        for i in range(len(prev_items)):
            for j in range(i+1, len(prev_items)):
                # Create new candidate by combining itemsets
                itemset1 = list(prev_items[i])
                itemset2 = list(prev_items[j])
                
                # For k=2 (single items)
                if k == 2:
                    candidates.add(frozenset([itemset1[0], itemset2[0]]))
                else:
                    # For k>2, check if first k-2 items match
                    if itemset1[:-1] == itemset2[:-1]:
                        new_candidate = frozenset(list(prev_items[i]) + [itemset2[-1]])
                        
                        # Prune step - check if all subsets are frequent
                        should_add = True
                        for item in new_candidate:
                            subset = frozenset(new_candidate - {item})
                            if subset not in prev_frequent:
                                should_add = False
                                break
                        
                        if should_add:
                            candidates.add(new_candidate)
        
        return candidates
    
    def _generate_rules(self, n_transactions):
        """Generate association rules"""
        self.rules = []
        
        for k in range(2, len(self.frequent_itemsets) + 1):
            if k not in self.frequent_itemsets:
                continue
                
            for itemset, count in self.frequent_itemsets[k].items():
                # Try each item as consequent
                for item in itemset:
                    antecedent = frozenset(itemset - {item})
                    consequent = frozenset([item])
                    
                    # Calculate confidence
                    if antecedent in self.frequent_itemsets[k-1]:
                        antecedent_support = self.frequent_itemsets[k-1][antecedent] / n_transactions
                        rule_support = count / n_transactions
                        confidence = rule_support / antecedent_support
                        
                        # ENHANCEMENT: Rule confidence boosting
                        # Boost confidence for rules involving critical health indicators
                        # This ensures important medical relationships are prioritized
                        confidence_boost = 1.0
                        
                        # Check if rule contains critical health indicators
                        antecedent_items = list(antecedent)
                        consequent_items = list(consequent)
                        
                        # Apply boost if rule contains critical indicators
                        if any(ind in self.critical_indicators for ind in antecedent_items) or \
                           any(ind in self.critical_indicators for ind in consequent_items):
                            # 10% boost for rules with critical health indicators
                            confidence_boost = 1.1
                        
                        # Additional boost for rules predicting heart disease
                        if "heart_disease=1" in consequent_items:
                            confidence_boost *= 1.05  # Extra 5% for heart disease prediction
                        
                        # Apply boosted confidence
                        boosted_confidence = confidence * confidence_boost
                        
                        # Use original threshold for comparison but store boosted value
                        if confidence >= self.min_confidence:
                            self.rules.append({
                                'antecedent': list(antecedent),
                                'consequent': list(consequent),
                                'support': rule_support,
                                'confidence': boosted_confidence,  # Store boosted confidence
                                'original_confidence': confidence  # Keep original for reference
                            })
        
        # Sort by confidence (boosted)
        self.rules.sort(key=lambda x: x['confidence'], reverse=True)
        print(f"Generated {len(self.rules)} rules")
        return self.rules