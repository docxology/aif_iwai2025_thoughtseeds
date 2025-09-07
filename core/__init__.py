"""
Core module for the meditation simulation framework.

This module contains the main learner classes and core functionality:
    - ActInfLearner: Full Active Inference implementation
    - RuleBasedLearner: Foundation class with core methods
    - BaseNetwork: Network computation functionality
    - StateMachine: State transition management

All core methods are preserved exactly as implemented in the original codebase.
"""

from .learners import ActInfLearner, RuleBasedLearner
from .networks import NetworkComputation
from .transitions import StateTransitionManager

__all__ = [
    'ActInfLearner',
    'RuleBasedLearner', 
    'NetworkComputation',
    'StateTransitionManager',
]
