"""
Dynamic Attentional Agents in Focused Attention Meditation

A hierarchical computational modeling framework for expert-novice differences in meditation
using Active Inference principles. This package implements a three-level thoughtseeds framework
modeling attentional network dynamics, thoughtseed competition, and metacognitive regulation.

Key Components:
    - Active Inference learner implementation
    - Rules-based foundation classes  
    - Network dynamics modeling
    - Thoughtseed competition simulation
    - Visualization and analysis tools
    - Professional configuration management

Author: Research Team
Version: 1.0.0
License: MIT
"""

from .core import ActInfLearner, RuleBasedLearner
from .config import ActiveInferenceConfig, THOUGHTSEEDS, STATES
from .visualization import generate_all_plots, FreeEnergyVisualizer
from .utils import (
    ensure_directories, ExportManager, ExportConfig, 
    FreeEnergyTracer, FreeEnergyTrace, FreeEnergySnapshot
)
from .analysis import (
    StatisticalAnalyzer, ComparisonAnalyzer, MetricsCalculator,
    NetworkAnalyzer, TimeSeriesAnalyzer
)

# Package metadata
__version__ = "1.0.0"
__author__ = "Research Team"
__license__ = "MIT"

# Public API
__all__ = [
    # Core classes
    'ActInfLearner',
    'RuleBasedLearner',
    
    # Configuration
    'ActiveInferenceConfig',
    'THOUGHTSEEDS', 
    'STATES',
    
    # Visualization
    'generate_all_plots',
    'FreeEnergyVisualizer',
    
    # Utilities
    'ensure_directories',
    
    # Free energy tracing
    'FreeEnergyTracer',
    'FreeEnergyTrace',
    'FreeEnergySnapshot',
    
    # Modern export system
    'ExportManager',
    'ExportConfig',
    
    # Analysis system
    'StatisticalAnalyzer',
    'ComparisonAnalyzer',
    'MetricsCalculator',
    'NetworkAnalyzer',
    'TimeSeriesAnalyzer'
]
