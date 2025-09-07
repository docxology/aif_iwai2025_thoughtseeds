"""
CSV export functionality for spreadsheet-compatible data analysis.

This module provides CSV export strategies optimized for different analysis workflows:
- Standard CSV: Time series data in tabular format
- Network CSV: Network-specific analysis format
- Summary CSV: Statistical summaries and comparisons
"""

import csv
from typing import Dict, Any, Optional, List
from pathlib import Path
import numpy as np
from .base_exporter import BaseExporter


class CSVExporter(BaseExporter):
    """
    Standard CSV exporter for time series data in tabular format.
    
    Exports simulation data in a format suitable for spreadsheet analysis,
    statistical software (R, SPSS), and data visualization tools.
    """
    
    def __init__(self, output_dir: str = "./exports",
                 include_headers: bool = True,
                 delimiter: str = ',',
                 **kwargs):
        """
        Initialize CSV exporter.
        
        Args:
            output_dir: Directory for exported files
            include_headers: Whether to include column headers
            delimiter: CSV field delimiter
            **kwargs: Additional arguments for base exporter
        """
        super().__init__(output_dir, **kwargs)
        self.include_headers = include_headers
        self.delimiter = delimiter
    
    def export(self, learner: Any, filename: Optional[str] = None) -> Dict[str, str]:
        """Export complete learner data to CSV format."""
        base_name = filename or "meditation_data"
        
        # Export main time series data
        main_filepath = self._export_time_series(learner, base_name)
        
        # Export summary statistics
        summary_filepath = self._export_summary(learner, f"{base_name}_summary")
        
        result = {
            'time_series': main_filepath,
            'summary': summary_filepath
        }
        
        # Export network data if available
        if hasattr(learner, 'network_activations_history'):
            network_filepath = self._export_networks(learner, f"{base_name}_networks")
            result['networks'] = network_filepath
        
        return result
    
    def _export_time_series(self, learner: Any, base_name: str) -> str:
        """Export main time series data."""
        filepath = self._generate_filename(base_name, "csv", learner.experience_level)
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f, delimiter=self.delimiter)
            
            # Prepare headers
            headers = ['timestep', 'state', 'meta_awareness', 'dominant_thoughtseed']
            headers.extend([f'ts_{ts}' for ts in learner.thoughtseeds])
            
            if hasattr(learner, 'network_activations_history'):
                headers.extend([f'net_{net}' for net in learner.networks])
                headers.append('free_energy')
            
            if self.include_headers:
                writer.writerow(headers)
            
            # Write data rows
            for i in range(len(learner.activations_history)):
                row = [
                    i,  # timestep
                    learner.state_history[i],
                    learner.meta_awareness_history[i],
                    learner.dominant_ts_history[i]
                ]
                
                # Add thoughtseed activations
                row.extend(learner.activations_history[i])
                
                # Add network activations if available
                if hasattr(learner, 'network_activations_history'):
                    network_vals = [learner.network_activations_history[i][net] 
                                  for net in learner.networks]
                    row.extend(network_vals)
                    row.append(learner.free_energy_history[i])
                
                writer.writerow(row)
        
        return filepath
    
    def _export_summary(self, learner: Any, base_name: str) -> str:
        """Export summary statistics."""
        filepath = self._generate_filename(base_name, "csv", learner.experience_level)
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f, delimiter=self.delimiter)
            
            if self.include_headers:
                writer.writerow(['metric', 'category', 'item', 'value'])
            
            # Export thoughtseed statistics
            activations_array = np.array(learner.activations_history)
            
            for i, ts in enumerate(learner.thoughtseeds):
                ts_data = activations_array[:, i]
                
                writer.writerow(['mean_activation', 'thoughtseed', ts, np.mean(ts_data)])
                writer.writerow(['std_activation', 'thoughtseed', ts, np.std(ts_data)])
                writer.writerow(['max_activation', 'thoughtseed', ts, np.max(ts_data)])
                writer.writerow(['min_activation', 'thoughtseed', ts, np.min(ts_data)])
                
                # Dominance frequency
                dominance_freq = np.mean([
                    learner.dominant_ts_history[t] == ts 
                    for t in range(len(learner.dominant_ts_history))
                ])
                writer.writerow(['dominance_frequency', 'thoughtseed', ts, dominance_freq])
            
            # Export state statistics
            for state in learner.states:
                frequency = learner.state_history.count(state) / len(learner.state_history)
                writer.writerow(['frequency', 'state', state, frequency])
            
            # Export meta-awareness statistics
            writer.writerow(['mean', 'meta_awareness', 'overall', np.mean(learner.meta_awareness_history)])
            writer.writerow(['std', 'meta_awareness', 'overall', np.std(learner.meta_awareness_history)])
            
            # Export network statistics if available
            if hasattr(learner, 'network_activations_history'):
                network_array = np.array([
                    [step[net] for net in learner.networks]
                    for step in learner.network_activations_history
                ])
                
                for i, net in enumerate(learner.networks):
                    net_data = network_array[:, i]
                    writer.writerow(['mean_activation', 'network', net, np.mean(net_data)])
                    writer.writerow(['std_activation', 'network', net, np.std(net_data)])
                
                # Free energy statistics
                writer.writerow(['mean', 'free_energy', 'overall', np.mean(learner.free_energy_history)])
                writer.writerow(['std', 'free_energy', 'overall', np.std(learner.free_energy_history)])
                writer.writerow(['final', 'free_energy', 'overall', learner.free_energy_history[-1]])
            
            # Export simulation parameters
            if hasattr(learner, 'precision_weight'):
                writer.writerow(['precision_weight', 'parameter', 'simulation', learner.precision_weight])
            if hasattr(learner, 'complexity_penalty'):
                writer.writerow(['complexity_penalty', 'parameter', 'simulation', learner.complexity_penalty])
            if hasattr(learner, 'learning_rate'):
                writer.writerow(['learning_rate', 'parameter', 'simulation', learner.learning_rate])
        
        return filepath
    
    def _export_networks(self, learner: Any, base_name: str) -> str:
        """Export network-specific data."""
        filepath = self._generate_filename(base_name, "csv", learner.experience_level)
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f, delimiter=self.delimiter)
            
            # Headers
            headers = ['timestep', 'state']
            headers.extend(learner.networks)
            headers.extend(['free_energy', 'prediction_error', 'precision'])
            
            if self.include_headers:
                writer.writerow(headers)
            
            # Data rows
            for i in range(len(learner.network_activations_history)):
                row = [i, learner.state_history[i]]
                
                # Network activations
                row.extend([learner.network_activations_history[i][net] for net in learner.networks])
                
                # Free energy components
                row.extend([
                    learner.free_energy_history[i],
                    learner.prediction_error_history[i],
                    learner.precision_history[i]
                ])
                
                writer.writerow(row)
        
        return filepath


class NetworkCSVExporter(BaseExporter):
    """
    Specialized CSV exporter for network analysis.
    
    Exports data in format optimized for network analysis, correlations,
    and comparative studies between different networks.
    """
    
    def export(self, learner: Any, filename: Optional[str] = None) -> Dict[str, str]:
        """Export network analysis data."""
        if not hasattr(learner, 'network_activations_history'):
            raise ValueError("Learner must have network data for network export")
        
        base_name = filename or "network_analysis"
        
        # Export correlation matrix
        corr_filepath = self._export_correlations(learner, f"{base_name}_correlations")
        
        # Export state-based network profiles
        profile_filepath = self._export_state_profiles(learner, f"{base_name}_state_profiles")
        
        return {
            'correlations': corr_filepath,
            'state_profiles': profile_filepath
        }
    
    def _export_correlations(self, learner: Any, base_name: str) -> str:
        """Export network correlation matrix."""
        filepath = self._generate_filename(base_name, "csv", learner.experience_level)
        
        # Calculate correlation matrix
        network_array = np.array([
            [step[net] for net in learner.networks]
            for step in learner.network_activations_history
        ])
        
        corr_matrix = np.corrcoef(network_array.T)
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f, delimiter=',')
            
            # Header row
            headers = ['network'] + list(learner.networks)
            writer.writerow(headers)
            
            # Correlation matrix rows
            for i, net in enumerate(learner.networks):
                row = [net] + list(corr_matrix[i, :])
                writer.writerow(row)
        
        return filepath
    
    def _export_state_profiles(self, learner: Any, base_name: str) -> str:
        """Export network activation profiles by meditation state."""
        filepath = self._generate_filename(base_name, "csv", learner.experience_level)
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f, delimiter=',')
            
            # Headers
            headers = ['state'] + list(learner.networks) + ['count', 'free_energy_mean']
            writer.writerow(headers)
            
            # Calculate state-based profiles
            for state in learner.states:
                # Find indices for this state
                state_indices = [i for i, s in enumerate(learner.state_history) if s == state]
                
                if state_indices:
                    # Calculate mean network activations for this state
                    state_networks = []
                    for net in learner.networks:
                        net_values = [learner.network_activations_history[i][net] for i in state_indices]
                        state_networks.append(np.mean(net_values))
                    
                    # Calculate mean free energy for this state
                    state_fe = [learner.free_energy_history[i] for i in state_indices]
                    mean_fe = np.mean(state_fe)
                    
                    # Write row
                    row = [state] + state_networks + [len(state_indices), mean_fe]
                    writer.writerow(row)
        
        return filepath
