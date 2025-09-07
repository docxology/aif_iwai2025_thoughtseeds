"""
Network computation functionality for the meditation simulation framework.

This module contains specialized network computation classes that can be used
independently or as mixins with the main learner classes. All methods are
preserved exactly from the original ActInfLearner implementation.
"""

import numpy as np
from typing import Dict, Any


class NetworkComputation:
    """
    Specialized class for network activation computations.
    
    Contains all network-related computation methods extracted from the
    original ActInfLearner class. These methods can be used independently
    or mixed into learner classes.
    """
    
    def __init__(self):
        """Initialize network computation parameters."""
        pass
    
    @staticmethod
    def compute_baseline_activations(networks: list) -> Dict[str, float]:
        """Initialize baseline network activations."""
        return {net: 0.2 for net in networks}
    
    @staticmethod
    def get_experience_weights(experience_level: str) -> tuple:
        """Get bottom-up and top-down weights based on experience level."""
        if experience_level == 'expert':
            return 0.5, 0.5  # bottom_up, top_down
        else:
            return 0.6, 0.4  # More bottom-up for novices
    
    @staticmethod
    def apply_network_bounds(network_acts: Dict[str, float], 
                           noise_level: float) -> Dict[str, float]:
        """Apply biological bounds and noise to network activations."""
        for net in network_acts:
            network_acts[net] = np.clip(network_acts[net], 0.05, 1.0)
            network_acts[net] += np.random.normal(0, noise_level)
            network_acts[net] = np.clip(network_acts[net], 0.05, 1.0)
        
        # VAN physiological limits
        if 'VAN' in network_acts and network_acts['VAN'] > 0.85:
            network_acts['VAN'] = 0.85
            
        return network_acts
