"""
Pickle export functionality for complete Python object serialization.

This module provides pickle export capabilities for complete object preservation,
allowing full reconstruction of learner objects for continued analysis or
sharing between Python environments.
"""

import pickle
from typing import Dict, Any, Optional
from .base_exporter import BaseExporter


class PickleExporter(BaseExporter):
    """
    Pickle exporter for complete Python object serialization.
    
    Provides complete object preservation allowing full reconstruction
    of learner instances. Ideal for checkpointing, resuming simulations,
    or sharing complete analysis objects.
    """
    
    def __init__(self, output_dir: str = "./exports",
                 protocol: int = pickle.HIGHEST_PROTOCOL,
                 compress: bool = True,
                 **kwargs):
        """
        Initialize pickle exporter.
        
        Args:
            output_dir: Directory for exported files
            protocol: Pickle protocol version
            compress: Whether to compress pickle files
            **kwargs: Additional arguments for base exporter
        """
        super().__init__(output_dir, **kwargs)
        self.protocol = protocol
        self.compress = compress
        
        if compress:
            try:
                import gzip
                self.gzip_available = True
            except ImportError:
                self.gzip_available = False
                self.compress = False
    
    def export(self, learner: Any, filename: Optional[str] = None) -> Dict[str, str]:
        """Export complete learner object using pickle."""
        base_name = filename or "meditation_learner"
        
        # Export complete learner object
        learner_filepath = self._export_learner(learner, base_name)
        
        # Export data-only version (lighter weight)
        data_filepath = self._export_data_only(learner, f"{base_name}_data")
        
        return {
            'complete_learner': learner_filepath,
            'data_only': data_filepath
        }
    
    def _export_learner(self, learner: Any, base_name: str) -> str:
        """Export complete learner object."""
        extension = "pkl.gz" if self.compress else "pkl"
        filepath = self._generate_filename(base_name, extension, learner.experience_level)
        
        if self.compress and self.gzip_available:
            import gzip
            with gzip.open(filepath, 'wb') as f:
                pickle.dump(learner, f, protocol=self.protocol)
        else:
            with open(filepath, 'wb') as f:
                pickle.dump(learner, f, protocol=self.protocol)
        
        return filepath
    
    def _export_data_only(self, learner: Any, base_name: str) -> str:
        """Export data-only version (without methods)."""
        # Create data-only representation
        data_object = {
            'metadata': {
                'experience_level': learner.experience_level,
                'timesteps': learner.timesteps,
                'thoughtseeds': learner.thoughtseeds,
                'states': learner.states,
                'export_metadata': self.metadata
            },
            
            'time_series_data': {
                'activations_history': learner.activations_history,
                'state_history': learner.state_history,
                'meta_awareness_history': learner.meta_awareness_history,
                'dominant_ts_history': learner.dominant_ts_history,
                'state_history_over_time': learner.state_history_over_time
            },
            
            'simulation_parameters': {
                'noise_level': getattr(learner, 'noise_level', None),
                'precision_weight': getattr(learner, 'precision_weight', None),
                'complexity_penalty': getattr(learner, 'complexity_penalty', None),
                'learning_rate': getattr(learner, 'learning_rate', None),
                'memory_factor': getattr(learner, 'memory_factor', None),
                'fpn_enhancement': getattr(learner, 'fpn_enhancement', None)
            },
            
            'transition_data': {
                'transition_counts': getattr(learner, 'transition_counts', {}),
                'natural_transition_count': getattr(learner, 'natural_transition_count', 0),
                'forced_transition_count': getattr(learner, 'forced_transition_count', 0),
                'transition_activations': getattr(learner, 'transition_activations', {}),
                'distraction_buildup_rates': getattr(learner, 'distraction_buildup_rates', [])
            }
        }
        
        # Add network data if available
        if hasattr(learner, 'network_activations_history'):
            data_object['network_data'] = {
                'networks': learner.networks,
                'network_activations_history': learner.network_activations_history,
                'free_energy_history': learner.free_energy_history,
                'prediction_error_history': learner.prediction_error_history,
                'precision_history': learner.precision_history,
                'learned_network_profiles': learner.learned_network_profiles,
                'transition_thresholds': getattr(learner, 'transition_thresholds', {}),
                'network_modulation': getattr(learner, 'network_modulation', {}),
                'non_linear_dynamics': getattr(learner, 'non_linear_dynamics', {})
            }
        
        extension = "pkl.gz" if self.compress else "pkl"
        filepath = self._generate_filename(base_name, extension, learner.experience_level)
        
        if self.compress and self.gzip_available:
            import gzip
            with gzip.open(filepath, 'wb') as f:
                pickle.dump(data_object, f, protocol=self.protocol)
        else:
            with open(filepath, 'wb') as f:
                pickle.dump(data_object, f, protocol=self.protocol)
        
        return filepath


class PickleLoader:
    """
    Utility class for loading pickled meditation data.
    
    Provides safe loading and validation of pickled learner objects
    and data-only representations.
    """
    
    @staticmethod
    def load_learner(filepath: str, validate: bool = True) -> Any:
        """
        Load complete learner object from pickle file.
        
        Args:
            filepath: Path to pickle file
            validate: Whether to validate loaded object
            
        Returns:
            Loaded learner object
        """
        try:
            if filepath.endswith('.gz'):
                import gzip
                with gzip.open(filepath, 'rb') as f:
                    learner = pickle.load(f)
            else:
                with open(filepath, 'rb') as f:
                    learner = pickle.load(f)
            
            if validate:
                PickleLoader._validate_learner(learner)
            
            return learner
            
        except Exception as e:
            raise ValueError(f"Failed to load learner from {filepath}: {e}")
    
    @staticmethod
    def load_data(filepath: str, validate: bool = True) -> Dict[str, Any]:
        """
        Load data-only representation from pickle file.
        
        Args:
            filepath: Path to pickle file
            validate: Whether to validate loaded data
            
        Returns:
            Loaded data dictionary
        """
        try:
            if filepath.endswith('.gz'):
                import gzip
                with gzip.open(filepath, 'rb') as f:
                    data = pickle.load(f)
            else:
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
            
            if validate:
                PickleLoader._validate_data(data)
            
            return data
            
        except Exception as e:
            raise ValueError(f"Failed to load data from {filepath}: {e}")
    
    @staticmethod
    def _validate_learner(learner: Any) -> None:
        """Validate loaded learner object."""
        required_attrs = [
            'experience_level', 'timesteps', 'thoughtseeds', 'states',
            'activations_history', 'state_history', 'meta_awareness_history'
        ]
        
        for attr in required_attrs:
            if not hasattr(learner, attr):
                raise ValueError(f"Loaded learner missing required attribute: {attr}")
    
    @staticmethod
    def _validate_data(data: Dict[str, Any]) -> None:
        """Validate loaded data dictionary."""
        required_sections = ['metadata', 'time_series_data', 'simulation_parameters']
        
        for section in required_sections:
            if section not in data:
                raise ValueError(f"Loaded data missing required section: {section}")
        
        # Validate time series consistency
        ts_data = data['time_series_data']
        if 'activations_history' in ts_data and 'state_history' in ts_data:
            act_len = len(ts_data['activations_history'])
            state_len = len(ts_data['state_history'])
            
            if act_len != state_len:
                raise ValueError(f"Time series length mismatch: activations={act_len}, states={state_len}")
