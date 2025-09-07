"""
JSON export functionality with enhanced structure and flexibility.

This module provides multiple JSON export strategies:
- Standard JSON: Complete data export in structured format
- Time Series JSON: Optimized for temporal analysis
- Summary JSON: Condensed statistics and metadata
"""

import json
from typing import Dict, Any, Optional, List
from pathlib import Path
import numpy as np
from .base_exporter import BaseExporter


class JSONExporter(BaseExporter):
    """
    Standard JSON exporter with complete data structure.
    
    Exports all available data in a hierarchical JSON format suitable
    for web applications, APIs, and general data interchange.
    """
    
    def __init__(self, output_dir: str = "./exports", 
                 indent: int = 2,
                 compress: bool = False,
                 **kwargs):
        """
        Initialize JSON exporter.
        
        Args:
            output_dir: Directory for exported files
            indent: JSON indentation (None for compact)
            compress: Whether to compress large arrays
            **kwargs: Additional arguments for base exporter
        """
        super().__init__(output_dir, **kwargs)
        self.indent = indent
        self.compress = compress
    
    def export(self, learner: Any, filename: Optional[str] = None) -> Dict[str, str]:
        """Export complete learner data to JSON format."""
        data = self._prepare_data(learner)
        self._validate_data(data)
        
        # Apply compression if requested
        if self.compress:
            data = self._compress_arrays(data)
        
        base_name = filename or "meditation_simulation"
        filepath = self._generate_filename(
            base_name, "json", learner.experience_level
        )
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=self.indent, ensure_ascii=False)
        
        return {'complete_data': filepath}
    
    def _compress_arrays(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Compress large arrays for more efficient storage."""
        compressed = data.copy()
        
        # Compress time series data
        if 'time_series' in compressed:
            ts_data = compressed['time_series']
            
            # Sample large arrays if they exceed threshold
            threshold = 1000
            
            for key, value in ts_data.items():
                if isinstance(value, list) and len(value) > threshold:
                    # Keep every nth element for compression
                    step = max(1, len(value) // threshold)
                    ts_data[f"{key}_compressed"] = value[::step]
                    ts_data[f"{key}_compression_factor"] = step
                    
                    # Keep original metadata
                    ts_data[f"{key}_original_length"] = len(value)
        
        return compressed


class TimeSeriesJSONExporter(BaseExporter):
    """
    Specialized JSON exporter optimized for time series analysis.
    
    Exports data in a format optimized for temporal analysis tools
    and visualization libraries like D3.js, Plotly, etc.
    """
    
    def __init__(self, output_dir: str = "./exports",
                 sample_rate: Optional[int] = None,
                 **kwargs):
        """
        Initialize time series JSON exporter.
        
        Args:
            output_dir: Directory for exported files  
            sample_rate: Optional downsampling rate for large datasets
            **kwargs: Additional arguments for base exporter
        """
        super().__init__(output_dir, **kwargs)
        self.sample_rate = sample_rate
    
    def export(self, learner: Any, filename: Optional[str] = None) -> Dict[str, str]:
        """Export learner data optimized for time series analysis."""
        data = self._prepare_time_series_data(learner)
        
        base_name = filename or "time_series"
        filepath = self._generate_filename(
            base_name, "json", learner.experience_level
        )
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        return {'time_series': filepath}
    
    def _prepare_time_series_data(self, learner: Any) -> Dict[str, Any]:
        """Prepare data in time series optimized format."""
        timesteps = range(len(learner.activations_history))
        
        # Apply sampling if specified
        if self.sample_rate and len(timesteps) > self.sample_rate:
            step = len(timesteps) // self.sample_rate
            indices = list(range(0, len(timesteps), step))
        else:
            indices = list(timesteps)
        
        # Create time series records
        time_series = []
        
        for i in indices:
            record = {
                'timestep': i,
                'state': learner.state_history[i],
                'meta_awareness': learner.meta_awareness_history[i],
                'dominant_thoughtseed': learner.dominant_ts_history[i],
                'thoughtseed_activations': {
                    ts: float(learner.activations_history[i][j])
                    for j, ts in enumerate(learner.thoughtseeds)
                }
            }
            
            # Add network data if available
            if hasattr(learner, 'network_activations_history'):
                record['network_activations'] = {
                    net: float(learner.network_activations_history[i][net])
                    for net in learner.networks
                }
                record['free_energy'] = float(learner.free_energy_history[i])
            
            time_series.append(record)
        
        return {
            'metadata': {
                **self.metadata,
                'experience_level': learner.experience_level,
                'total_timesteps': len(learner.activations_history),
                'sampled_timesteps': len(time_series),
                'thoughtseeds': learner.thoughtseeds,
                'states': learner.states,
                'sample_rate': self.sample_rate
            },
            'time_series': time_series
        }


class SummaryJSONExporter(BaseExporter):
    """
    Condensed JSON exporter for summary statistics and metadata.
    
    Exports only key statistics, parameters, and summary information
    without full time series data. Ideal for dashboards and reports.
    """
    
    def export(self, learner: Any, filename: Optional[str] = None) -> Dict[str, str]:
        """Export summary statistics and metadata."""
        data = self._prepare_summary_data(learner)
        
        base_name = filename or "summary"
        filepath = self._generate_filename(
            base_name, "json", learner.experience_level
        )
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        return {'summary': filepath}
    
    def _prepare_summary_data(self, learner: Any) -> Dict[str, Any]:
        """Prepare condensed summary data."""
        # Calculate summary statistics
        activations_array = np.array(learner.activations_history)
        
        summary_stats = {
            'thoughtseed_statistics': {
                ts: {
                    'mean_activation': float(np.mean(activations_array[:, i])),
                    'std_activation': float(np.std(activations_array[:, i])),
                    'max_activation': float(np.max(activations_array[:, i])),
                    'min_activation': float(np.min(activations_array[:, i])),
                    'dominance_frequency': float(np.mean([
                        learner.dominant_ts_history[t] == ts 
                        for t in range(len(learner.dominant_ts_history))
                    ]))
                }
                for i, ts in enumerate(learner.thoughtseeds)
            },
            
            'state_statistics': {
                state: {
                    'frequency': float(learner.state_history.count(state) / len(learner.state_history)),
                    'average_duration': self._calculate_average_duration(learner.state_history, state)
                }
                for state in learner.states
            },
            
            'meta_awareness_statistics': {
                'mean': float(np.mean(learner.meta_awareness_history)),
                'std': float(np.std(learner.meta_awareness_history)),
                'trend': self._calculate_trend(learner.meta_awareness_history)
            }
        }
        
        # Add network statistics if available
        if hasattr(learner, 'network_activations_history'):
            network_array = np.array([
                [step[net] for net in learner.networks]
                for step in learner.network_activations_history
            ])
            
            summary_stats['network_statistics'] = {
                net: {
                    'mean_activation': float(np.mean(network_array[:, i])),
                    'std_activation': float(np.std(network_array[:, i])),
                    'correlation_with_meta_awareness': float(np.corrcoef(
                        network_array[:, i], learner.meta_awareness_history
                    )[0, 1])
                }
                for i, net in enumerate(learner.networks)
            }
            
            summary_stats['free_energy_statistics'] = {
                'mean': float(np.mean(learner.free_energy_history)),
                'std': float(np.std(learner.free_energy_history)),
                'trend': self._calculate_trend(learner.free_energy_history),
                'final_value': float(learner.free_energy_history[-1])
            }
        
        return {
            'metadata': {
                **self.metadata,
                'experience_level': learner.experience_level,
                'simulation_duration': len(learner.activations_history),
                'natural_transitions': getattr(learner, 'natural_transition_count', 0),
                'forced_transitions': getattr(learner, 'forced_transition_count', 0)
            },
            'summary_statistics': summary_stats,
            'parameters': self._prepare_data(learner)['parameters']
        }
    
    def _calculate_average_duration(self, state_history: List[str], target_state: str) -> float:
        """Calculate average duration spent in a specific state."""
        durations = []
        current_duration = 0
        
        for state in state_history:
            if state == target_state:
                current_duration += 1
            else:
                if current_duration > 0:
                    durations.append(current_duration)
                    current_duration = 0
        
        # Add final duration if simulation ended in target state
        if current_duration > 0:
            durations.append(current_duration)
        
        return float(np.mean(durations)) if durations else 0.0
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction for time series values."""
        if len(values) < 2:
            return 'insufficient_data'
        
        # Simple linear trend
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope > 0.001:
            return 'increasing'
        elif slope < -0.001:
            return 'decreasing'
        else:
            return 'stable'
