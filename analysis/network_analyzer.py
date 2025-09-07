"""
Network-specific analysis functionality for meditation simulation data.

This module provides specialized analysis tools for neural network dynamics
including connectivity analysis, network efficiency metrics, and temporal
network evolution patterns.
"""

import numpy as np
from typing import Dict, List, Any

try:
    from scipy.stats import pearsonr
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    def pearsonr(x, y):
        return np.corrcoef(x, y)[0, 1], 0.05  # correlation, p-value


class NetworkAnalyzer:
    """
    Specialized analyzer for neural network dynamics in meditation.
    
    Provides analysis tools specifically designed for the four-network
    model (DMN, VAN, DAN, FPN) used in meditation research.
    """
    
    def __init__(self):
        """Initialize network analyzer."""
        self.has_scipy = SCIPY_AVAILABLE
    
    def calculate_network_correlations(self, learner: Any) -> Dict[str, Any]:
        """Calculate network correlation matrix."""
        if not hasattr(learner, 'network_activations_history'):
            return {}
        
        connectivity = self._analyze_connectivity(learner)
        correlation_matrix = connectivity.get('correlation_matrix', [])
        
        # Convert matrix to more accessible dict format
        if correlation_matrix and hasattr(learner, 'networks'):
            network_correlations = {}
            networks = learner.networks
            
            for i, net1 in enumerate(networks):
                network_correlations[net1] = {}
                for j, net2 in enumerate(networks):
                    if i < len(correlation_matrix) and j < len(correlation_matrix[i]):
                        network_correlations[net1][net2] = correlation_matrix[i][j]
            
            return {
                'correlation_matrix': correlation_matrix,
                'network_correlations': network_correlations,
                'strongest_connections': connectivity.get('strongest_connections', [])
            }
        
        return connectivity
    
    def analyze_dmn_dan_anticorrelation(self, learner: Any) -> Dict[str, Any]:
        """Analyze DMN-DAN anticorrelation patterns."""
        return self._analyze_anticorrelations(learner)
    
    def calculate_network_dominance(self, learner: Any) -> Dict[str, float]:
        """Calculate network dominance measures."""
        if not hasattr(learner, 'network_activations_history'):
            return {}
        
        stats = self._calculate_network_statistics(learner)
        dominance = {}
        
        for net, net_stats in stats.items():
            dominance[net] = net_stats['mean']  # Simple dominance = average activation
        
        return dominance
    
    def analyze_state_network_patterns(self, learner: Any) -> Dict[str, Any]:
        """Analyze network patterns by meditation state."""
        return self._analyze_state_patterns(learner)
    
    def analyze_network_dynamics(self, learner: Any) -> Dict[str, Any]:
        """Comprehensive analysis of network dynamics."""
        if not hasattr(learner, 'network_activations_history'):
            return {'error': 'No network data available'}
        
        return {
            'network_statistics': self._calculate_network_statistics(learner),
            'connectivity_analysis': self._analyze_connectivity(learner),
            'state_dependent_patterns': self._analyze_state_patterns(learner),
            'temporal_evolution': self._analyze_temporal_evolution(learner),
            'anticorrelation_analysis': self._analyze_anticorrelations(learner)
        }
    
    def _calculate_network_statistics(self, learner: Any) -> Dict[str, Any]:
        """Calculate basic network statistics."""
        network_array = np.array([
            [step[net] for net in learner.networks]
            for step in learner.network_activations_history
        ])
        
        stats = {}
        for i, net in enumerate(learner.networks):
            net_data = network_array[:, i]
            stats[net] = {
                'mean': float(np.mean(net_data)),
                'std': float(np.std(net_data)),
                'min': float(np.min(net_data)),
                'max': float(np.max(net_data)),
                'dynamic_range': float(np.max(net_data) - np.min(net_data))
            }
        
        return stats
    
    def _analyze_connectivity(self, learner: Any) -> Dict[str, Any]:
        """Analyze network connectivity patterns."""
        network_array = np.array([
            [step[net] for net in learner.networks]
            for step in learner.network_activations_history
        ])
        
        correlation_matrix = np.corrcoef(network_array.T)
        
        return {
            'correlation_matrix': correlation_matrix.tolist(),
            'strongest_connections': self._find_strongest_connections(correlation_matrix, learner.networks),
            'anticorrelations': self._find_anticorrelations(correlation_matrix, learner.networks)
        }
    
    def _analyze_state_patterns(self, learner: Any) -> Dict[str, Any]:
        """Analyze network patterns by meditation state."""
        patterns = {}
        
        for state in learner.states:
            state_indices = [i for i, s in enumerate(learner.state_history) if s == state]
            if state_indices:
                state_networks = {}
                for net in learner.networks:
                    net_values = [learner.network_activations_history[i][net] for i in state_indices]
                    state_networks[net] = {
                        'mean': float(np.mean(net_values)),
                        'std': float(np.std(net_values))
                    }
                patterns[state] = state_networks
        
        return patterns
    
    def _analyze_temporal_evolution(self, learner: Any) -> Dict[str, Any]:
        """Analyze how networks evolve over time."""
        network_array = np.array([
            [step[net] for net in learner.networks]
            for step in learner.network_activations_history
        ])
        
        trends = {}
        for i, net in enumerate(learner.networks):
            net_data = network_array[:, i]
            time_steps = np.arange(len(net_data))
            
            slope, intercept = np.polyfit(time_steps, net_data, 1)
            trends[net] = {
                'slope': float(slope),
                'direction': 'increasing' if slope > 0.001 else ('decreasing' if slope < -0.001 else 'stable')
            }
        
        return trends
    
    def _analyze_anticorrelations(self, learner: Any) -> Dict[str, Any]:
        """Analyze anticorrelation patterns (especially DMN-DAN)."""
        network_array = np.array([
            [step[net] for net in learner.networks]
            for step in learner.network_activations_history
        ])
        
        dmn_idx = learner.networks.index('DMN')
        dan_idx = learner.networks.index('DAN')
        
        dmn_dan_corr = np.corrcoef(network_array[:, dmn_idx], network_array[:, dan_idx])[0, 1]
        
        return {
            'correlation': float(dmn_dan_corr),
            'dmn_dan_correlation': float(dmn_dan_corr),
            'anticorrelation_strength': float(-dmn_dan_corr) if dmn_dan_corr < 0 else 0.0,
            'is_anticorrelated': dmn_dan_corr < -0.1
        }
    
    def _find_strongest_connections(self, corr_matrix: np.ndarray, networks: List[str]) -> List[Dict[str, Any]]:
        """Find strongest positive connections."""
        connections = []
        
        for i in range(len(networks)):
            for j in range(i + 1, len(networks)):
                connections.append({
                    'networks': (networks[i], networks[j]),
                    'correlation': float(corr_matrix[i, j])
                })
        
        return sorted(connections, key=lambda x: abs(x['correlation']), reverse=True)[:3]
    
    def _find_anticorrelations(self, corr_matrix: np.ndarray, networks: List[str]) -> List[Dict[str, Any]]:
        """Find strongest negative correlations."""
        anticorrelations = []
        
        for i in range(len(networks)):
            for j in range(i + 1, len(networks)):
                corr = corr_matrix[i, j]
                if corr < -0.1:  # Only significant anticorrelations
                    anticorrelations.append({
                        'networks': (networks[i], networks[j]),
                        'correlation': float(corr)
                    })
        
        return sorted(anticorrelations, key=lambda x: x['correlation'])
