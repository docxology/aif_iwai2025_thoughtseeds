"""
Resource competition mechanisms for thoughtseeds.

This module contains competition and resource allocation mechanisms for thoughtseeds,
modeling how different thoughtseeds compete for limited attentional resources during
meditation practice.
"""

import numpy as np
from typing import List, Dict, Tuple
from .agents import ThoughtseedAgent


class ResourceCompetition:
    """
    Resource competition manager for thoughtseed interactions.
    
    This class manages competition between thoughtseeds for limited attentional
    resources, implementing winner-take-more dynamics and resource constraints
    that affect thoughtseed activation patterns.
    """
    
    def __init__(self, thoughtseeds: List[str], experience_level: str):
        """Initialize resource competition manager."""
        self.thoughtseeds = thoughtseeds
        self.experience_level = experience_level
        
        # Create individual agents
        self.agents = {
            ts: ThoughtseedAgent(ts, experience_level) 
            for ts in thoughtseeds
        }
        
        # Competition parameters
        self.total_resources = 1.0  # Total attentional resources
        self.competition_strength = 0.7 if experience_level == 'novice' else 0.5
        
    def allocate_resources(self, target_activations: np.ndarray) -> np.ndarray:
        """
        Allocate limited attentional resources among competing thoughtseeds.
        
        Args:
            target_activations: Desired activation levels for each thoughtseed
            
        Returns:
            Resource-constrained activation levels
        """
        # Convert to dict for easier handling
        targets = {ts: target_activations[i] for i, ts in enumerate(self.thoughtseeds)}
        
        # Calculate total demand
        total_demand = sum(targets.values())
        
        # If total demand exceeds resources, apply competition
        if total_demand > self.total_resources:
            allocated = self._apply_competition(targets)
        else:
            allocated = targets
        
        # Convert back to array
        result = np.array([allocated[ts] for ts in self.thoughtseeds])
        return np.clip(result, 0.05, 1.0)
    
    def _apply_competition(self, targets: Dict[str, float]) -> Dict[str, float]:
        """Apply competitive resource allocation."""
        allocated = {}
        
        # Sort by priority (category-dependent)
        priority_order = self._get_priority_order(targets)
        
        remaining_resources = self.total_resources
        
        for ts in priority_order:
            # Calculate competitive pressure from other thoughtseeds
            pressure = self._calculate_pressure(ts, targets)
            
            # Reduce target based on competition
            competitive_target = targets[ts] * (1 - pressure * self.competition_strength)
            
            # Allocate resources (limited by remaining)
            allocation = min(competitive_target, remaining_resources)
            allocated[ts] = allocation
            remaining_resources -= allocation
            
            if remaining_resources <= 0:
                break
        
        # Ensure all thoughtseeds get some minimal allocation
        for ts in self.thoughtseeds:
            if ts not in allocated:
                allocated[ts] = 0.05
        
        return allocated
    
    def _get_priority_order(self, targets: Dict[str, float]) -> List[str]:
        """Determine priority order for resource allocation."""
        # Sort by activation level (higher gets priority)
        sorted_by_activation = sorted(targets.items(), key=lambda x: x[1], reverse=True)
        
        # Apply category-based adjustments
        priority_categories = ['focus', 'regulation', 'metacognition', 'distraction']
        
        final_order = []
        for category in priority_categories:
            category_thoughtseeds = [
                ts for ts, _ in sorted_by_activation 
                if ts in self.agents and self.agents[ts].category == category
            ]
            final_order.extend(category_thoughtseeds)
        
        # Add any remaining thoughtseeds
        for ts, _ in sorted_by_activation:
            if ts not in final_order:
                final_order.append(ts)
        
        return final_order
    
    def _calculate_pressure(self, target_ts: str, targets: Dict[str, float]) -> float:
        """Calculate competitive pressure on a target thoughtseed."""
        if target_ts not in self.agents:
            return 0.0
        
        target_category = self.agents[target_ts].category
        pressure = 0.0
        
        for ts, activation in targets.items():
            if ts != target_ts and ts in self.agents:
                competitor_category = self.agents[ts].category
                
                # Same category creates more pressure
                if competitor_category == target_category:
                    pressure += activation * 0.8
                else:
                    pressure += activation * 0.3
        
        return min(pressure, 0.9)  # Cap pressure
    
    def get_dominant_thoughtseed(self, activations: np.ndarray) -> str:
        """Identify the currently dominant thoughtseed."""
        max_idx = np.argmax(activations)
        return self.thoughtseeds[max_idx]
    
    def get_competition_state(self, activations: np.ndarray) -> Dict[str, float]:
        """Get current competition state metrics."""
        total = np.sum(activations)
        
        return {
            'total_activation': total,
            'resource_utilization': min(total / self.total_resources, 1.0),
            'dominant_share': np.max(activations) / total if total > 0 else 0.0,
            'competition_intensity': np.std(activations),
            'active_thoughtseeds': np.sum(activations > 0.1)
        }
