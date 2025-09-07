"""
Base exporter class providing common interface for all data export formats.

This module defines the abstract base class that all specialized exporters
inherit from, ensuring consistent APIs and behavior across different formats.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import numpy as np
from datetime import datetime


class BaseExporter(ABC):
    """
    Abstract base class for all data exporters.
    
    Provides common interface and utility methods for exporting meditation
    simulation data in various formats. All concrete exporters must implement
    the export() method.
    """
    
    def __init__(self, output_dir: str = "./exports", 
                 timestamp: bool = True,
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize base exporter.
        
        Args:
            output_dir: Directory for exported files
            timestamp: Whether to add timestamps to filenames
            metadata: Additional metadata to include in exports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = timestamp
        self.metadata = metadata or {}
        
        # Add system metadata
        self.metadata.update({
            'export_timestamp': datetime.now().isoformat(),
            'exporter_type': self.__class__.__name__,
            'format_version': '1.0'
        })
    
    @abstractmethod
    def export(self, learner: Any, filename: Optional[str] = None) -> Dict[str, str]:
        """
        Export learner data in the specific format.
        
        Args:
            learner: Trained learner instance with data to export
            filename: Optional custom filename (without extension)
            
        Returns:
            Dictionary mapping export types to created file paths
        """
        pass
    
    def _generate_filename(self, base_name: str, extension: str, 
                          experience_level: Optional[str] = None) -> str:
        """Generate standardized filename with optional timestamp."""
        parts = [base_name]
        
        if experience_level:
            parts.append(experience_level)
            
        if self.timestamp:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            parts.append(timestamp_str)
            
        filename = "_".join(parts) + f".{extension}"
        return str(self.output_dir / filename)
    
    def _prepare_data(self, learner: Any) -> Dict[str, Any]:
        """
        Prepare and standardize data from learner for export.
        
        Args:
            learner: Trained learner instance
            
        Returns:
            Standardized data dictionary
        """
        # Convert numpy arrays to lists for JSON compatibility
        def convert_arrays(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_arrays(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_arrays(item) for item in obj]
            else:
                return obj
        
        data = {
            'metadata': {
                **self.metadata,
                'experience_level': learner.experience_level,
                'timesteps': learner.timesteps,
                'thoughtseeds': learner.thoughtseeds,
                'states': learner.states,
                'natural_transitions': getattr(learner, 'natural_transition_count', 0),
                'forced_transitions': getattr(learner, 'forced_transition_count', 0)
            },
            
            'time_series': {
                'activations_history': convert_arrays(learner.activations_history),
                'state_history': learner.state_history,
                'meta_awareness_history': learner.meta_awareness_history,
                'dominant_ts_history': learner.dominant_ts_history,
            },
            
            'parameters': {
                'precision_weight': getattr(learner, 'precision_weight', None),
                'complexity_penalty': getattr(learner, 'complexity_penalty', None),
                'learning_rate': getattr(learner, 'learning_rate', None),
                'noise_level': getattr(learner, 'noise_level', None),
                'memory_factor': getattr(learner, 'memory_factor', None),
            },
            
            'statistics': {
                'transition_counts': convert_arrays(getattr(learner, 'transition_counts', {})),
                'distraction_buildup_rates': convert_arrays(getattr(learner, 'distraction_buildup_rates', [])),
            }
        }
        
        # Add network-specific data if available (ActInfLearner)
        if hasattr(learner, 'network_activations_history'):
            data['networks'] = {
                'activations_history': convert_arrays(learner.network_activations_history),
                'free_energy_history': learner.free_energy_history,
                'prediction_error_history': learner.prediction_error_history,
                'precision_history': learner.precision_history,
                'learned_profiles': convert_arrays(learner.learned_network_profiles)
            }
        
        return data
    
    def _validate_data(self, data: Dict[str, Any]) -> bool:
        """
        Validate prepared data for export.
        
        Args:
            data: Prepared data dictionary
            
        Returns:
            True if data is valid for export
        """
        required_keys = ['metadata', 'time_series', 'parameters', 'statistics']
        
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Missing required data section: {key}")
        
        # Validate time series consistency
        time_series = data['time_series']
        if 'activations_history' in time_series and 'state_history' in time_series:
            act_len = len(time_series['activations_history'])
            state_len = len(time_series['state_history'])
            
            if act_len != state_len:
                raise ValueError(f"Time series length mismatch: activations={act_len}, states={state_len}")
        
        return True
