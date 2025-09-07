"""
Utility module for the meditation simulation framework.

This module contains utility functions for data management, file operations,
and output processing. All utility methods are preserved exactly from the
original implementation to maintain complete compatibility.
"""

from .data_management import ensure_directories, _save_json_outputs, convert_numpy_to_lists
from .file_operations import create_output_structure, validate_data_integrity, get_latest_results
from .free_energy_tracer import FreeEnergyTracer, FreeEnergyTrace, FreeEnergySnapshot
from .exporters import (
    ExportManager, ExportConfig, ExportPipeline,
    JSONExporter, CSVExporter, HDF5Exporter, PickleExporter,
    MATLABExporter, RExporter
)

__all__ = [
    # Legacy data management
    'ensure_directories',
    '_save_json_outputs',
    'convert_numpy_to_lists',
    'create_output_structure',
    'validate_data_integrity',
    'get_latest_results',

    # Free energy tracing
    'FreeEnergyTracer',
    'FreeEnergyTrace', 
    'FreeEnergySnapshot',

    # Modern export system
    'ExportManager',
    'ExportConfig',
    'ExportPipeline',
    'JSONExporter',
    'CSVExporter',
    'HDF5Exporter',
    'PickleExporter',
    'MATLABExporter',
    'RExporter'
]
