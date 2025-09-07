"""
Thoughtseed agent classes for the meditation simulation framework.

This module contains individual thoughtseed agent implementations that can be used
for more detailed modeling of individual thoughtseed behaviors. These extend the
basic thoughtseed dynamics with agent-specific functionality.
"""

import numpy as np
from typing import Dict, Any
from config import THOUGHTSEED_AGENTS


class ThoughtseedAgent:
    """
    Individual thoughtseed agent implementation.
    
    This class represents a single thoughtseed as an autonomous agent with its own
    dynamics, decay/recovery rates, and interaction patterns. This allows for more
    detailed modeling of individual thoughtseed behaviors.
    """
    
    def __init__(self, thoughtseed_name: str, experience_level: str):
        """Initialize individual thoughtseed agent."""
        self.name = thoughtseed_name
        self.experience_level = experience_level
        
        # Load configuration for this thoughtseed
        if thoughtseed_name in THOUGHTSEED_AGENTS:
            config = THOUGHTSEED_AGENTS[thoughtseed_name]
            self.agent_id = config['id']
            self.category = config['category']
            self.decay_rate = config['decay_rate']
            self.recovery_rate = config['recovery_rate']
            self.intentional_weight = config['intentional_weights'][experience_level]
        else:
            raise ValueError(f"Unknown thoughtseed: {thoughtseed_name}")
        
        # Agent state
        self.current_activation = 0.1
        self.target_activation = 0.1
        self.history = []
    
    def update_activation(self, target: float, dt: float = 1.0) -> float:
        """
        Update agent activation towards target with decay/recovery dynamics.
        
        Args:
            target: Target activation level
            dt: Time step size
            
        Returns:
            Updated activation level
        """
        self.target_activation = target
        
        # Apply decay or recovery based on target
        if target > self.current_activation:
            # Recovery towards target
            change = self.recovery_rate * (target - self.current_activation) * dt
        else:
            # Decay towards target  
            change = -self.decay_rate * (self.current_activation - target) * dt
        
        # Update activation
        self.current_activation += change
        
        # Apply intentional weight constraint
        self.current_activation = min(self.current_activation, self.intentional_weight)
        
        # Biological bounds
        self.current_activation = np.clip(self.current_activation, 0.05, 1.0)
        
        # Record history
        self.history.append(self.current_activation)
        
        return self.current_activation
    
    def get_dominance_score(self, other_activations: Dict[str, float]) -> float:
        """Calculate how dominant this thoughtseed is relative to others."""
        total_activation = sum(other_activations.values()) + self.current_activation
        if total_activation > 0:
            return self.current_activation / total_activation
        return 0.0
    
    def get_competition_pressure(self, other_activations: Dict[str, float]) -> float:
        """Calculate competitive pressure from other thoughtseeds."""
        # Thoughtseeds in same category compete more strongly
        same_category_pressure = 0.0
        other_category_pressure = 0.0
        
        for name, activation in other_activations.items():
            if name in THOUGHTSEED_AGENTS:
                other_category = THOUGHTSEED_AGENTS[name]['category']
                if other_category == self.category:
                    same_category_pressure += activation
                else:
                    other_category_pressure += activation * 0.5  # Less competition
        
        return same_category_pressure + other_category_pressure
    
    def reset(self):
        """Reset agent to initial state."""
        self.current_activation = 0.1
        self.target_activation = 0.1
        self.history = []
