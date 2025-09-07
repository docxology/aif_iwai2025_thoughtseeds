"""
Configuration module for the meditation simulation framework.

This module contains all configuration classes, parameters, and data structures
used throughout the meditation simulation system. All configuration classes and
values are preserved exactly from the original implementation to maintain
complete compatibility.
"""

from .meditation_config import (
    # Core constants
    THOUGHTSEEDS, STATES, STATE_DWELL_TIMES, THOUGHTSEED_AGENTS, NETWORK_PROFILES,
    
    # Configuration classes  
    EffectMagnitude, ContextualEffect, DwellTimeConfig, ThoughtseedAgent,
    NetworkProfile, NonLinearDynamicsConfig, NetworkModulationConfig,
    TransitionThresholds, StateTargetActivations, ActiveInferenceParameters,
    ThoughtseedParams, MetacognitionParams,
    
    # Access interface
    ActiveInferenceConfig
)

__all__ = [
    # Core constants
    'THOUGHTSEEDS', 'STATES', 'STATE_DWELL_TIMES', 'THOUGHTSEED_AGENTS', 'NETWORK_PROFILES',
    
    # Configuration classes
    'EffectMagnitude', 'ContextualEffect', 'DwellTimeConfig', 'ThoughtseedAgent',
    'NetworkProfile', 'NonLinearDynamicsConfig', 'NetworkModulationConfig', 
    'TransitionThresholds', 'StateTargetActivations', 'ActiveInferenceParameters',
    'ThoughtseedParams', 'MetacognitionParams',
    
    # Access interface
    'ActiveInferenceConfig',
]
