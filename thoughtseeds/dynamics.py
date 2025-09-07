"""
Thoughtseed dynamics for the meditation simulation framework.

This module contains specialized thoughtseed dynamic functionality that can be used
independently or as part of the main learner classes. All thoughtseed logic is
preserved exactly from the original implementation.
"""

import numpy as np
from typing import List, Dict, Tuple, Any
from config import ThoughtseedParams


class ThoughtseedDynamics:
    """
    Specialized class for thoughtseed dynamic computations.
    
    Contains all thoughtseed-related computation methods extracted from the
    original implementation. These methods handle target activation generation,
    biological noise application, and experience-level adjustments.
    """
    
    def __init__(self, thoughtseeds: List[str], experience_level: str):
        """Initialize thoughtseed dynamics."""
        self.thoughtseeds = thoughtseeds
        self.experience_level = experience_level
        self.num_thoughtseeds = len(thoughtseeds)
    
    def get_target_activations(self, state: str, meta_awareness: float, 
                             noise_level: float) -> np.ndarray:
        """
        Generate target activations for each thoughtseed based on state and meta-awareness.
        
        Exact preservation of the original method from RuleBasedLearner.
        """
        # Get target activations from the parameter class
        targets_dict = ThoughtseedParams.get_target_activations(
            state, meta_awareness, self.experience_level)
        
        # Convert dictionary to numpy array in the correct order
        target_activations = np.zeros(self.num_thoughtseeds)
        for i, ts in enumerate(self.thoughtseeds):
            target_activations[i] = targets_dict[ts]
        
        # Add noise for biological plausibility
        target_activations += np.random.normal(0, noise_level, size=self.num_thoughtseeds)
        
        # Ensure values are in proper range
        return np.clip(target_activations, 0.05, 1.0)
    
    def apply_state_specific_modulation(self, activations: np.ndarray, 
                                      current_state: str, meta_awareness: float,
                                      current_dwell: int) -> np.ndarray:
        """
        Apply state-specific modulation to thoughtseed activations.
        
        Extracted from the main training loop in ActInfLearner.
        """
        modulated = activations.copy()
        
        # Mind wandering state modulations
        if current_state == "mind_wandering" and meta_awareness < 0.6:
            for ts in self.thoughtseeds:
                i = self.thoughtseeds.index(ts)
                if ts == "breath_focus":
                    # Much more biological variability for suppressed attention
                    base_level = max(0.05, modulated[i] * 0.3)
                    # Larger noise component based on time in state
                    noise_amplitude = 0.03 + 0.01 * min(1.0, current_dwell/10)
                    modulated[i] = base_level + np.random.normal(0, noise_amplitude)
                elif ts in ["pain_discomfort", "pending_tasks"]:
                    # Add fluctuations to dominant thoughtseeds
                    growth_factor = 1.2 * (1.0 + np.random.uniform(-0.1, 0.1))
                    modulated[i] *= growth_factor
                else:
                    # More variable suppression for other thoughtseeds
                    suppress_factor = 0.5 * (1.0 + np.random.uniform(-0.15, 0.15))
                    modulated[i] *= suppress_factor
        
        # Meta-awareness state modulations  
        elif current_state == "meta_awareness" and meta_awareness >= 0.8:
            for ts in self.thoughtseeds:
                i = self.thoughtseeds.index(ts)
                if ts == "self_reflection":
                    modulated[i] *= 1.5
                else:
                    modulated[i] *= 0.2
        
        # Redirect breath state modulations
        elif current_state == "redirect_breath" and meta_awareness >= 0.8:
            for ts in self.thoughtseeds:
                i = self.thoughtseeds.index(ts)
                if ts == "equanimity":
                    modulated[i] *= 1.5
                elif ts == "breath_focus":
                    modulated[i] *= 1.1
                else:
                    modulated[i] *= 0.3
        
        return np.clip(modulated, 0.05, 1.0)
    
    def apply_experience_specific_effects(self, activations: np.ndarray,
                                        current_state: str) -> np.ndarray:
        """
        Apply experience-level specific effects to thoughtseed activations.
        
        Extracted from expert-specific adjustments in the original implementation.
        """
        if self.experience_level != 'expert':
            return activations
        
        modulated = activations.copy()
        
        # Expert synergies in redirect_breath and meta_awareness
        if current_state in ["redirect_breath", "meta_awareness"]:
            bf_idx = self.thoughtseeds.index("breath_focus")
            eq_idx = self.thoughtseeds.index("equanimity")
            
            if modulated[bf_idx] > 0.3 and modulated[eq_idx] > 0.3:
                boost = 0.03 * min(modulated[bf_idx], modulated[eq_idx])
                modulated[bf_idx] += boost
                modulated[eq_idx] += boost
            
            if modulated[eq_idx] > 0.4:
                pd_idx = self.thoughtseeds.index("pain_discomfort")
                modulated[pd_idx] = max(0.05, modulated[pd_idx] - 0.02 * modulated[eq_idx])
        
        # Expert adjustments in focused states
        elif current_state in ["breath_control", "redirect_breath"]:
            bf_idx = self.thoughtseeds.index("breath_focus")
            eq_idx = self.thoughtseeds.index("equanimity")
            
            if current_state == "breath_control" and current_dwell < 10:  # Simplified dwell check
                # Lower initial equanimity in breath_control
                modulated[eq_idx] *= 0.85
            elif modulated[bf_idx] > 0.4:
                # Breath focus facilitates equanimity
                facilitation = 0.08 * modulated[bf_idx]
                modulated[eq_idx] += facilitation * (1.0 + np.random.uniform(-0.2, 0.2))
                modulated[eq_idx] = min(1.0, modulated[eq_idx])
        
        return np.clip(modulated, 0.05, 1.0)
    
    def apply_physiological_noise(self, activations: np.ndarray, 
                                current_state: str) -> np.ndarray:
        """
        Apply physiological noise to all activations for biological plausibility.
        
        Extracted from the noise application in the original training loop.
        """
        noisy_activations = activations.copy()
        
        for i, ts in enumerate(self.thoughtseeds):
            # Base noise level
            noise_level = 0.005
            
            # More noise during mind_wandering to avoid fixed values
            if current_state == "mind_wandering":
                noise_level = 0.015
            
            # Different thoughtseeds have different noise characteristics
            if ts in ["breath_focus", "equanimity"]:
                # More stable attentional focus has less noise
                noise_level *= 0.8
            elif ts in ["pain_discomfort"]:
                # Pain fluctuates more
                noise_level *= 1.5
            
            # Apply the noise
            noisy_activations[i] += np.random.normal(0, noise_level)
        
        return np.clip(noisy_activations, 0.05, 1.0)
