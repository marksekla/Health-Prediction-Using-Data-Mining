import numpy as np
import time
from collections import defaultdict

class MinHash:
    def __init__(self, n_hash_functions=100, seed=42):
        """Setup MinHash with specified hash functions"""
        self.n_hash_functions = n_hash_functions
        self.seed = seed
        self.hash_coefs = None
        self.signature_matrix = None
        self.execution_time = 0
        
        # ENHANCEMENT: Define feature weights for medical significance
        # These weights influence similarity calculations to better reflect clinical relevance
        self.feature_prefixes = {
            'cp=': 2.0,          # Chest pain is critically important
            'trestbps_bin=': 1.8, # Blood pressure is very important
            'chol_bin=': 1.5,    # Cholesterol is important
            'heart_disease=': 2.5, # Disease status is most important
            'age_bin=': 1.2,     # Age is moderately important
            'sex=': 1.0          # Sex is baseline importance
        }
        
        self._init_hash_functions()
    
    def _init_hash_functions(self):
        """Create hash functions"""
        np.random.seed(self.seed)
        # Large prime number
        self.prime = 2147483647  # 2^31 - 1
        
        # Generate random coefficients
        self.hash_coefs = np.random.randint(
            1, self.prime, size=(self.n_hash_functions, 2))
    
    def _hash_function(self, index, value):
        """Calculate hash value using a*x + b mod p"""
        a, b = self.hash_coefs[index]
        return (a * value + b) % self.prime
    
    # ENHANCEMENT: Create a method to calculate feature weights
    def _get_feature_weight(self, feature):
        """Determine feature weight based on medical significance"""
        for prefix, weight in self.feature_prefixes.items():
            if feature.startswith(prefix):
                return weight
        return 1.0  # Default weight if no match
    
    def fit_transform(self, data):
        """Compute MinHash signatures for all profiles"""
        start_time = time.time()
        
        n_profiles = len(data)
        print(f"Processing {n_profiles} profiles")
        
        # Get all possible items
        universe = set()
        for profile in data:
            universe.update(profile)
        
        # Map items to indices and calculate weights
        item_to_index = {}
        feature_weights = {}
        
        for i, item in enumerate(universe):
            item_to_index[item] = i
            # ENHANCEMENT: Apply feature weighting
            feature_weights[item] = self._get_feature_weight(item)
        
        # Initialize signature matrix
        self.signature_matrix = np.full((self.n_hash_functions, n_profiles), np.inf)
        
        # For each profile and item
        for profile_idx, profile in enumerate(data):
            for item in profile:
                item_idx = item_to_index[item]
                
                # Compute hash values with feature weighting
                for hash_idx in range(self.n_hash_functions):
                    # ENHANCEMENT: Apply weight to hash value - more important features have more influence
                    weight = feature_weights[item]
                    hash_value = self._hash_function(hash_idx, item_idx)
                    # Adjust hash value by weight (divide by weight to make important features have smaller hashes)
                    weighted_hash_value = hash_value / weight
                    
                    # Keep minimum weighted hash
                    self.signature_matrix[hash_idx, profile_idx] = min(
                        self.signature_matrix[hash_idx, profile_idx], weighted_hash_value)
        
        # Convert to integers (after weighting calculations)
        self.signature_matrix = self.signature_matrix.astype(int)
        
        self.execution_time = time.time() - start_time
        print(f"MinHash signatures done in {self.execution_time:.2f} seconds")
        return self.signature_matrix
    
    def estimate_similarity(self, profile1_idx, profile2_idx):
        """Estimate Jaccard similarity between two profiles"""
        if self.signature_matrix is None:
            raise ValueError("Need to run fit_transform first")
        
        # Count matching hash values
        matches = np.sum(
            self.signature_matrix[:, profile1_idx] == self.signature_matrix[:, profile2_idx])
        
        # Similarity = fraction of matching hashes
        return matches / self.n_hash_functions
    
    def get_similar_profiles(self, threshold=0.5):
        """Find all profile pairs with similarity above threshold"""
        if self.signature_matrix is None:
            raise ValueError("Need to run fit_transform first")
        
        n_profiles = self.signature_matrix.shape[1]
        similar_pairs = []
        
        print(f"Finding similar patients (threshold {threshold})...")
        
        # Compare all pairs
        for i in range(n_profiles):
            for j in range(i+1, n_profiles):
                similarity = self.estimate_similarity(i, j)
                if similarity >= threshold:
                    similar_pairs.append((i, j, similarity))
        
        # Sort by similarity
        similar_pairs.sort(key=lambda x: x[2], reverse=True)
        print(f"Found {len(similar_pairs)} similar patient pairs")
        return similar_pairs
    
    # ENHANCEMENT: Calculate data density for adaptive band sizing
    def _calculate_density_map(self, signature_matrix, bands):
        """Calculate density of data points in different regions of the signature space"""
        n_profiles = signature_matrix.shape[1]
        rows_per_band = self.n_hash_functions // bands
        
        density_map = np.zeros(bands)
        
        for band in range(bands):
            start_row = band * rows_per_band
            end_row = min((band + 1) * rows_per_band, self.n_hash_functions)
            band_signatures = signature_matrix[start_row:end_row, :]
            
            # Count signature frequencies
            signature_counts = defaultdict(int)
            for profile_idx in range(band_signatures.shape[1]):
                signature = tuple(band_signatures[:, profile_idx])
                signature_counts[signature] += 1
            
            # Calculate density as avg profiles per unique signature
            if len(signature_counts) > 0:
                density_map[band] = sum(signature_counts.values()) / len(signature_counts) / n_profiles
            else:
                density_map[band] = 0
                
        return density_map
    
    def enhanced_minhash(self, data, bands=20):
        """Enhanced version with LSH for faster similar item finding"""
        # Compute MinHash signatures
        self.fit_transform(data)
        
        # Basic validation
        if self.n_hash_functions < bands:
            raise ValueError(f"Too many bands ({bands}) for {self.n_hash_functions} hash functions")
        
        # Default rows per band
        base_rows_per_band = self.n_hash_functions // bands
        
        # ENHANCEMENT: Calculate density map for adaptive band sizing
        density_map = self._calculate_density_map(self.signature_matrix, bands)
        
        print(f"Using LSH with {bands} bands and adaptive sizing")
        
        # Find candidate pairs
        candidates = set()
        
        # For each band
        for band in range(bands):
            # ENHANCEMENT: Adaptive band sizing based on data density
            # Denser regions use smaller bands for precision, sparser regions use larger bands
            density = density_map[band]
            
            # Adjust rows per band based on density
            # Dense regions -> fewer rows (more sensitive)
            # Sparse regions -> more rows (less sensitive)
            adaptive_factor = 1.0 - (density * 0.5)  # Scale factor based on density
            adaptive_rows = max(1, int(base_rows_per_band * adaptive_factor))
            
            # Map band signatures to profiles
            bucket = defaultdict(list)
            
            # Get signatures for this band
            start_row = band * base_rows_per_band
            end_row = min(start_row + adaptive_rows, self.n_hash_functions)
            band_signatures = self.signature_matrix[start_row:end_row, :]
            
            # Hash profiles to buckets
            for profile_idx in range(band_signatures.shape[1]):
                # Convert signature to tuple
                band_signature = tuple(band_signatures[:, profile_idx])
                bucket[band_signature].append(profile_idx)
            
            # Profiles in same bucket are candidates
            for profiles in bucket.values():
                if len(profiles) > 1:
                    # Add all pairs in bucket
                    for i in range(len(profiles)):
                        for j in range(i+1, len(profiles)):
                            candidates.add((profiles[i], profiles[j]))
        
        # Verify candidates
        similar_pairs = []
        for profile1_idx, profile2_idx in candidates:
            similarity = self.estimate_similarity(profile1_idx, profile2_idx)
            similar_pairs.append((profile1_idx, profile2_idx, similarity))
        
        # Sort by similarity
        similar_pairs.sort(key=lambda x: x[2], reverse=True)
        print(f"LSH found {len(similar_pairs)} similar patient pairs")
        return similar_pairs