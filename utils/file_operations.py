"""
File operations utilities for the meditation simulation framework.

This module contains additional file operation utilities that complement
the core data management functions. These are extensions that provide
enhanced file handling capabilities while maintaining compatibility.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, List


def create_output_structure(base_path: str = './results_act_inf') -> Dict[str, str]:
    """
    Create complete output directory structure with validation.
    
    Args:
        base_path: Base directory for simulation outputs
        
    Returns:
        Dictionary mapping structure names to created paths
    """
    structure = {
        'base': base_path,
        'data': f'{base_path}/data',
        'plots': f'{base_path}/plots',
        'logs': f'{base_path}/logs',
        'checkpoints': f'{base_path}/checkpoints'
    }
    
    for name, path in structure.items():
        os.makedirs(path, exist_ok=True)
    
    return structure


def validate_data_integrity(data_dir: str) -> Dict[str, bool]:
    """
    Validate integrity of generated data files.
    
    Args:
        data_dir: Directory containing data files to validate
        
    Returns:
        Dictionary mapping file types to validation status
    """
    validation_results = {}
    expected_files = [
        'thoughtseed_params_novice.json',
        'thoughtseed_params_expert.json', 
        'active_inference_params_novice.json',
        'active_inference_params_expert.json',
        'transition_stats_novice.json',
        'transition_stats_expert.json'
    ]
    
    for filename in expected_files:
        filepath = os.path.join(data_dir, filename)
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    # Basic validation - check if data is not empty
                    validation_results[filename] = len(data) > 0
            else:
                validation_results[filename] = False
        except (json.JSONDecodeError, IOError):
            validation_results[filename] = False
    
    return validation_results


def get_latest_results(results_dir: str = './results_act_inf') -> Dict[str, Any]:
    """
    Get metadata about the latest simulation results.
    
    Args:
        results_dir: Directory containing simulation results
        
    Returns:
        Dictionary with metadata about latest results
    """
    metadata = {
        'timestamp': None,
        'data_files': [],
        'plot_files': [],
        'total_size': 0
    }
    
    if not os.path.exists(results_dir):
        return metadata
    
    # Get data files
    data_dir = os.path.join(results_dir, 'data')
    if os.path.exists(data_dir):
        metadata['data_files'] = [f for f in os.listdir(data_dir) if f.endswith('.json')]
    
    # Get plot files  
    plots_dir = os.path.join(results_dir, 'plots')
    if os.path.exists(plots_dir):
        metadata['plot_files'] = [f for f in os.listdir(plots_dir) if f.endswith('.png')]
    
    # Calculate total size
    for root, dirs, files in os.walk(results_dir):
        for file in files:
            filepath = os.path.join(root, file)
            if os.path.exists(filepath):
                metadata['total_size'] += os.path.getsize(filepath)
    
    return metadata
