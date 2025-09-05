"""
rules_based_learner.py

This module implements a rule-based meditation model that serves as the foundation
for the active inference implementation. It provides core methods for thoughtseed
dynamics and meta-awareness calculation.
"""

import numpy as np
from collections import defaultdict
from meditation_config import (
    THOUGHTSEEDS, STATES, STATE_DWELL_TIMES, 
    ActiveInferenceConfig, ThoughtseedParams, MetacognitionParams
)

class RuleBasedLearner:
    """
    Foundation class implementing rule-based thoughtseed dynamics.
    
    This class provides core methods for thoughtseed behavior, meta-awareness,
    and state transitions, serving as a foundation for the active inference implementation.
    It models meditation dynamics using rule-based interactions between thoughtseeds.
    """
    
    def __init__(self, experience_level='novice', timesteps_per_cycle=200):

        # Core parameters
        self.experience_level = experience_level
        self.timesteps = timesteps_per_cycle
        self.thoughtseeds = THOUGHTSEEDS
        self.states = STATES
        self.num_thoughtseeds = len(self.thoughtseeds)
        
        # State tracking
        self.state_indices = {state: i for i, state in enumerate(self.states)}
        self.transition_counts = defaultdict(lambda: defaultdict(int))
        self.natural_transition_count = 0
        self.forced_transition_count = 0
        
        # History tracking
        self.activations_history = []
        self.state_history = []
        self.meta_awareness_history = []
        self.dominant_ts_history = []
        self.state_history_over_time = []
        
        # Get noise level from config
        aif_params = ActiveInferenceConfig.get_params(experience_level)
        self.noise_level = aif_params['noise_level']
        
        # Track activation patterns at transition points
        self.transition_activations = {state: [] for state in self.states}
        
        # Track distraction buildup patterns
        self.distraction_buildup_rates = []

    def get_target_activations(self, state, meta_awareness):
        """
        Generate target activations for each thoughtseed based on state and meta-awareness.        
        This determines the ideal activation pattern for each state,
        which is then modulated by neural network dynamics.
        
        """
        # Get target activations from the parameter class
        targets_dict = ThoughtseedParams.get_target_activations(
            state, meta_awareness, self.experience_level)
        
        # Convert dictionary to numpy array in the correct order
        target_activations = np.zeros(self.num_thoughtseeds)
        for i, ts in enumerate(self.thoughtseeds):
            target_activations[i] = targets_dict[ts]
        
        # Add noise for biological plausibility
        target_activations += np.random.normal(0, self.noise_level, size=self.num_thoughtseeds)
        
        # Ensure values are in proper range
        return np.clip(target_activations, 0.05, 1.0)

    def get_dwell_time(self, state):
        """
        Generate a random dwell time for the given state, based on experience level.

        """
        # Get the configured range from STATE_DWELL_TIMES
        config_min, config_max = STATE_DWELL_TIMES[self.experience_level][state]
        
        # Ensure minimal biological plausibility while respecting configured values
        if state in ['meta_awareness', 'redirect_breath']:
            # For brief states: at least 1 timestep, respect configured max
            min_biological = 1
            max_biological = config_max
        else:
            # For longer states: at least 3 timesteps, respect configured max
            min_biological = 3
            max_biological = config_max
        
        # Generate dwell time with proper constraints
        return max(min_biological, min(max_biological, np.random.randint(config_min, config_max + 1)))

    def get_meta_awareness(self, current_state, activations):
        """
        Calculate meta-awareness based on state and thoughtseed activations.
        
        Meta-awareness is higher in experts, higher during meta_awareness state,
        and influenced by self-reflection and equanimity thoughtseeds.
        """
        # Convert activations array to dictionary for parameter class
        activations_dict = {}
        for i, ts in enumerate(self.thoughtseeds):
            activations_dict[ts] = activations[i]
        
        # Get meta-awareness from parameter class
        meta_awareness = MetacognitionParams.calculate_meta_awareness(
            current_state, activations_dict, self.experience_level)
        
        # Add small random noise for variability
        meta_awareness += np.random.normal(0, 0.05)
        
        return np.clip(meta_awareness, 0.2, 0.85 if self.experience_level == 'novice' else 0.9)