"""
Visualization module for the meditation simulation framework.

This module contains all plotting and visualization functionality for analyzing
meditation simulation data. All plotting methods are preserved exactly from the
original implementation to maintain complete compatibility and functionality.
"""

from .plotting import (
    generate_all_plots, plot_network_radar, plot_free_energy_comparison,
    plot_hierarchy, plot_time_series, load_json_data, set_plot_style,
    STATE_COLORS, NETWORK_COLORS, THOUGHTSEED_COLORS
)
from .free_energy_visualizer import FreeEnergyVisualizer
from .dynamics_visualizer import DynamicsVisualizer
from .advanced_free_energy_visualizer import AdvancedFreeEnergyVisualizer
from .statistical_dashboard import StatisticalDashboard
from .enhanced_visualization_suite import EnhancedVisualizationSuite, generate_enhanced_plots

__all__ = [
    # Core plotting functions
    'generate_all_plots',
    'plot_network_radar',
    'plot_free_energy_comparison',
    'plot_hierarchy',
    'plot_time_series',
    'load_json_data',
    'set_plot_style',
    
    # Color schemes
    'STATE_COLORS',
    'NETWORK_COLORS',
    'THOUGHTSEED_COLORS',
    
    # Visualization classes
    'FreeEnergyVisualizer',
    'DynamicsVisualizer',
    'AdvancedFreeEnergyVisualizer',
    'StatisticalDashboard',
    'EnhancedVisualizationSuite',
    
    # Enhanced plotting functions
    'generate_enhanced_plots'
]
