"""
State transition management for the meditation simulation framework.

This module contains specialized state transition functionality that can be used
independently or as part of the main learner classes. All transition logic is
preserved exactly from the original implementation.
"""

import numpy as np
from typing import Dict, List, Tuple, Any
from collections import defaultdict


class StateTransitionManager:
    """
    Specialized class for managing state transitions in meditation simulation.
    
    Contains all state transition logic extracted from the original ActInfLearner
    class. These methods handle natural vs forced transitions, free energy-based
    transition probabilities, and biological transition smoothing.
    """
    
    def __init__(self, states: List[str], experience_level: str):
        """Initialize state transition manager."""
        self.states = states
        self.experience_level = experience_level
        self.transition_counts = defaultdict(lambda: defaultdict(int))
        self.natural_transition_count = 0
        self.forced_transition_count = 0
    
    def calculate_transition_probability(self, t: int, timesteps: int, 
                                       free_energy: float) -> float:
        """
        Calculate natural transition probability based on time and free energy.
        
        Implements concepts from Equation 4 where higher free energy increases
        transition probability.
        """
        # Higher base probability for more natural transitions
        natural_prob = 0.8 + min(0.15, t / timesteps * 0.2)
        
        # Higher free energy increases transition probability
        precision_factor = 1.5 if self.experience_level == 'expert' else 0.8
        fe_factor = min(0.3, free_energy * 0.3 * precision_factor)
        natural_prob = min(0.95, natural_prob + fe_factor)
        
        return natural_prob
    
    def check_natural_transition(self, current_state: str, 
                                thoughtseed_activations: np.ndarray,
                                network_activations: Dict[str, float],
                                thoughtseeds: List[str],
                                transition_thresholds: Dict[str, float]) -> Tuple[bool, str]:
        """
        Check if natural transition should occur based on current activations.
        
        Returns tuple of (transition_occurred, next_state).
        Preserves exact logic from original implementation.
        """
        natural_transition = False
        next_state = None
        
        # FOCUSED STATES TO MIND WANDERING
        if current_state in ["breath_control", "redirect_breath"]:
            # Calculate combined distraction level
            pain_idx = thoughtseeds.index("pain_discomfort")
            tasks_idx = thoughtseeds.index("pending_tasks")
            distraction_level = (thoughtseed_activations[pain_idx] + 
                               thoughtseed_activations[tasks_idx])
            
            # Consider DMN/DAN ratio
            dmn_dan_ratio = network_activations['DMN'] / (network_activations['DAN'] + 0.1)
            
            # Either high distraction OR high DMN/DAN ratio can trigger transition
            if (distraction_level > transition_thresholds['mind_wandering'] or 
                dmn_dan_ratio > transition_thresholds['dmn_dan_ratio']):
                next_state = "mind_wandering"
                natural_transition = True
        
        # MIND WANDERING TO META-AWARENESS
        elif current_state == "mind_wandering":
            # Self-reflection is the key factor
            reflection_idx = thoughtseeds.index("self_reflection")
            self_reflection = thoughtseed_activations[reflection_idx]
            
            # Consider VAN activation as secondary factor
            van_activation = network_activations['VAN']
            
            # Experience-dependent thresholds
            awareness_threshold = 0.35 if self.experience_level == 'expert' else 0.45
            
            if (self_reflection > awareness_threshold or 
                (van_activation > 0.4 and self_reflection > 0.3)):
                next_state = "meta_awareness"
                natural_transition = True
        
        # META-AWARENESS TO FOCUSED STATES
        elif current_state == "meta_awareness":
            # Base transition values (from activations)
            bf_idx = thoughtseeds.index("breath_focus")
            eq_idx = thoughtseeds.index("equanimity")
            bf_value = thoughtseed_activations[bf_idx]
            eq_value = thoughtseed_activations[eq_idx]
            
            # Network influences on transitions
            bf_value += network_activations['DAN'] * 0.2  # DAN enhances breath focus
            eq_value += network_activations['FPN'] * 0.2  # FPN enhances equanimity
            
            # Lower threshold for more reliable transitions
            threshold = transition_thresholds['return_focus']
            
            if bf_value > threshold and eq_value > threshold:
                # If both high, experts favor equanimity/redirect_breath
                if self.experience_level == 'expert' and eq_value > bf_value:
                    next_state = "redirect_breath"
                else:
                    next_state = "breath_control"
                natural_transition = True
            elif bf_value > threshold + 0.1:  # Higher certainty for single condition
                next_state = "breath_control"
                natural_transition = True
            elif eq_value > threshold + 0.1:  # Higher certainty for single condition
                next_state = "redirect_breath"
                natural_transition = True
        
        return natural_transition, next_state
    
    def create_transition_target(self, current_activations: np.ndarray,
                               new_target: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Create smooth transition between activation states with biological variability.
        
        Returns tuple of (blended_activations, transition_counter).
        """
        # Add variability to target activations
        varied_target = new_target.copy()
        for i in range(len(varied_target)):
            variation = 1.0 + np.random.uniform(-0.05, 0.1)  # 5-10% variation
            varied_target[i] *= variation
            varied_target[i] = max(0.06, varied_target[i])  # Minimum threshold
        
        # Conservative blending with variability
        blend_factor = 0.4 * (1.0 + np.random.uniform(-0.1, 0.1))  # 36-44% blend
        blended = (1 - blend_factor) * current_activations + blend_factor * varied_target
        
        # Variable transition time
        transition_counter = 3 + np.random.randint(0, 2)
        
        return blended, transition_counter
    
    def record_transition(self, current_state: str, next_state: str, 
                         natural: bool) -> None:
        """Record transition statistics."""
        self.transition_counts[current_state][next_state] += 1
        if natural:
            self.natural_transition_count += 1
        else:
            self.forced_transition_count += 1
