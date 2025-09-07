"""
HDF5 export functionality for high-performance scientific computing.

This module provides HDF5 export capabilities optimized for large datasets,
scientific computing workflows, and integration with tools like Python
scientific stack, MATLAB, and other HDF5-compatible applications.

Note: Requires h5py package. Falls back gracefully if not available.
"""

from typing import Dict, Any, Optional
import numpy as np
from .base_exporter import BaseExporter

try:
    import h5py
    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False


class HDF5Exporter(BaseExporter):
    """
    HDF5 exporter for high-performance scientific data storage.
    
    Provides hierarchical data organization, compression, and metadata
    support ideal for large-scale analysis and scientific computing.
    Requires h5py package.
    """
    
    def __init__(self, output_dir: str = "./exports",
                 compression: str = 'gzip',
                 compression_opts: int = 9,
                 shuffle: bool = True,
                 **kwargs):
        """
        Initialize HDF5 exporter.
        
        Args:
            output_dir: Directory for exported files
            compression: HDF5 compression algorithm ('gzip', 'lzf', 'szip')
            compression_opts: Compression level (0-9 for gzip)
            shuffle: Enable shuffle filter for better compression
            **kwargs: Additional arguments for base exporter
        """
        super().__init__(output_dir, **kwargs)
        
        if not HDF5_AVAILABLE:
            raise ImportError(
                "h5py is required for HDF5 export. Install with: pip install h5py"
            )
        
        self.compression = compression
        self.compression_opts = compression_opts
        self.shuffle = shuffle
    
    def export(self, learner: Any, filename: Optional[str] = None) -> Dict[str, str]:
        """Export complete learner data to HDF5 format."""
        base_name = filename or "meditation_simulation"
        filepath = self._generate_filename(base_name, "h5", learner.experience_level)
        
        with h5py.File(filepath, 'w') as f:
            # Create hierarchical structure
            self._create_metadata_group(f, learner)
            self._create_time_series_group(f, learner)
            self._create_parameters_group(f, learner)
            self._create_statistics_group(f, learner)
            
            # Add network data if available
            if hasattr(learner, 'network_activations_history'):
                self._create_networks_group(f, learner)
        
        return {'hdf5_data': filepath}
    
    def _create_metadata_group(self, f, learner: Any) -> None:
        """Create metadata group with simulation information."""
        meta_group = f.create_group('metadata')
        
        # Scalar metadata
        meta_group.attrs['experience_level'] = learner.experience_level
        meta_group.attrs['timesteps'] = learner.timesteps
        meta_group.attrs['natural_transitions'] = getattr(learner, 'natural_transition_count', 0)
        meta_group.attrs['forced_transitions'] = getattr(learner, 'forced_transition_count', 0)
        meta_group.attrs['export_timestamp'] = self.metadata['export_timestamp']
        meta_group.attrs['format_version'] = self.metadata['format_version']
        
        # String arrays for thoughtseeds and states
        dt = h5py.string_dtype(encoding='utf-8')
        thoughtseeds_ds = meta_group.create_dataset(
            'thoughtseeds', data=learner.thoughtseeds, dtype=dt
        )
        states_ds = meta_group.create_dataset(
            'states', data=learner.states, dtype=dt
        )
        
        # Add descriptions
        thoughtseeds_ds.attrs['description'] = 'List of thoughtseed entities in the simulation'
        states_ds.attrs['description'] = 'List of meditation states in the simulation'
    
    def _create_time_series_group(self, f, learner: Any) -> None:
        """Create time series data group."""
        ts_group = f.create_group('time_series')
        
        # Thoughtseed activations (2D array: timestep × thoughtseed)
        activations_array = np.array(learner.activations_history)
        activations_ds = ts_group.create_dataset(
            'thoughtseed_activations',
            data=activations_array,
            compression=self.compression,
            compression_opts=self.compression_opts,
            shuffle=self.shuffle
        )
        activations_ds.attrs['description'] = 'Thoughtseed activation levels over time'
        activations_ds.attrs['shape'] = f'{activations_array.shape[0]} timesteps × {activations_array.shape[1]} thoughtseeds'
        activations_ds.attrs['units'] = 'activation_level (0-1)'
        
        # State history (1D string array)
        dt = h5py.string_dtype(encoding='utf-8')
        state_ds = ts_group.create_dataset(
            'state_history', data=learner.state_history, dtype=dt
        )
        state_ds.attrs['description'] = 'Meditation state at each timestep'
        
        # Meta-awareness history (1D float array)
        meta_awareness_ds = ts_group.create_dataset(
            'meta_awareness_history',
            data=learner.meta_awareness_history,
            compression=self.compression,
            compression_opts=self.compression_opts,
            shuffle=self.shuffle
        )
        meta_awareness_ds.attrs['description'] = 'Meta-awareness level over time'
        meta_awareness_ds.attrs['units'] = 'awareness_level (0-1)'
        
        # Dominant thoughtseed history (1D string array)
        dominant_ts_ds = ts_group.create_dataset(
            'dominant_thoughtseed_history', data=learner.dominant_ts_history, dtype=dt
        )
        dominant_ts_ds.attrs['description'] = 'Most active thoughtseed at each timestep'
    
    def _create_parameters_group(self, f, learner: Any) -> None:
        """Create simulation parameters group."""
        params_group = f.create_group('parameters')
        
        # Active inference parameters
        if hasattr(learner, 'precision_weight'):
            params_group.attrs['precision_weight'] = learner.precision_weight
        if hasattr(learner, 'complexity_penalty'):
            params_group.attrs['complexity_penalty'] = learner.complexity_penalty
        if hasattr(learner, 'learning_rate'):
            params_group.attrs['learning_rate'] = learner.learning_rate
        if hasattr(learner, 'noise_level'):
            params_group.attrs['noise_level'] = learner.noise_level
        if hasattr(learner, 'memory_factor'):
            params_group.attrs['memory_factor'] = learner.memory_factor
        
        params_group.attrs['description'] = 'Active inference simulation parameters'
    
    def _create_statistics_group(self, f, learner: Any) -> None:
        """Create statistics and analysis group."""
        stats_group = f.create_group('statistics')
        
        # Thoughtseed statistics
        ts_stats_group = stats_group.create_group('thoughtseeds')
        activations_array = np.array(learner.activations_history)
        
        for i, ts in enumerate(learner.thoughtseeds):
            ts_data = activations_array[:, i]
            ts_subgroup = ts_stats_group.create_group(ts)
            
            ts_subgroup.attrs['mean_activation'] = np.mean(ts_data)
            ts_subgroup.attrs['std_activation'] = np.std(ts_data)
            ts_subgroup.attrs['max_activation'] = np.max(ts_data)
            ts_subgroup.attrs['min_activation'] = np.min(ts_data)
            
            # Dominance frequency
            dominance_freq = np.mean([
                learner.dominant_ts_history[t] == ts 
                for t in range(len(learner.dominant_ts_history))
            ])
            ts_subgroup.attrs['dominance_frequency'] = dominance_freq
        
        # State statistics
        state_stats_group = stats_group.create_group('states')
        
        for state in learner.states:
            state_subgroup = state_stats_group.create_group(state)
            frequency = learner.state_history.count(state) / len(learner.state_history)
            state_subgroup.attrs['frequency'] = frequency
            
            # State durations
            durations = self._calculate_state_durations(learner.state_history, state)
            if durations:
                durations_ds = state_subgroup.create_dataset(
                    'durations', data=durations,
                    compression=self.compression,
                    compression_opts=self.compression_opts
                )
                durations_ds.attrs['description'] = f'Duration of each {state} episode'
                durations_ds.attrs['units'] = 'timesteps'
        
        # Meta-awareness statistics
        meta_stats_group = stats_group.create_group('meta_awareness')
        meta_stats_group.attrs['mean'] = np.mean(learner.meta_awareness_history)
        meta_stats_group.attrs['std'] = np.std(learner.meta_awareness_history)
        meta_stats_group.attrs['min'] = np.min(learner.meta_awareness_history)
        meta_stats_group.attrs['max'] = np.max(learner.meta_awareness_history)
    
    def _create_networks_group(self, f, learner: Any) -> None:
        """Create network data group for ActInfLearner."""
        net_group = f.create_group('networks')
        
        # Network activations (2D array: timestep × network)
        network_data = []
        for step in learner.network_activations_history:
            network_data.append([step[net] for net in learner.networks])
        
        network_array = np.array(network_data)
        network_ds = net_group.create_dataset(
            'activations',
            data=network_array,
            compression=self.compression,
            compression_opts=self.compression_opts,
            shuffle=self.shuffle
        )
        network_ds.attrs['description'] = 'Network activation levels over time'
        network_ds.attrs['networks'] = [net.encode('utf-8') for net in learner.networks]
        network_ds.attrs['shape'] = f'{network_array.shape[0]} timesteps × {network_array.shape[1]} networks'
        
        # Free energy history
        fe_ds = net_group.create_dataset(
            'free_energy_history',
            data=learner.free_energy_history,
            compression=self.compression,
            compression_opts=self.compression_opts,
            shuffle=self.shuffle
        )
        fe_ds.attrs['description'] = 'Variational free energy over time'
        fe_ds.attrs['units'] = 'free_energy_units'
        
        # Prediction error history
        pe_ds = net_group.create_dataset(
            'prediction_error_history',
            data=learner.prediction_error_history,
            compression=self.compression,
            compression_opts=self.compression_opts,
            shuffle=self.shuffle
        )
        pe_ds.attrs['description'] = 'Prediction error over time'
        
        # Precision history
        precision_ds = net_group.create_dataset(
            'precision_history',
            data=learner.precision_history,
            compression=self.compression,
            compression_opts=self.compression_opts,
            shuffle=self.shuffle
        )
        precision_ds.attrs['description'] = 'Precision weighting over time'
        
        # Network statistics
        net_stats_group = net_group.create_group('statistics')
        
        for i, net in enumerate(learner.networks):
            net_data = network_array[:, i]
            net_subgroup = net_stats_group.create_group(net)
            
            net_subgroup.attrs['mean_activation'] = np.mean(net_data)
            net_subgroup.attrs['std_activation'] = np.std(net_data)
            net_subgroup.attrs['correlation_with_meta_awareness'] = np.corrcoef(
                net_data, learner.meta_awareness_history
            )[0, 1]
        
        # Free energy statistics
        fe_stats_group = net_stats_group.create_group('free_energy')
        fe_stats_group.attrs['mean'] = np.mean(learner.free_energy_history)
        fe_stats_group.attrs['std'] = np.std(learner.free_energy_history)
        fe_stats_group.attrs['final_value'] = learner.free_energy_history[-1]
        fe_stats_group.attrs['trend'] = self._calculate_trend(learner.free_energy_history)
    
    def _calculate_state_durations(self, state_history: list, target_state: str) -> list:
        """Calculate durations for each occurrence of a specific state."""
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
        
        return durations
    
    def _calculate_trend(self, values: list) -> str:
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
