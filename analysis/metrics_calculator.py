"""
Meditation-specific metrics calculator.

This module provides calculation of standard meditation research metrics
including attention stability, mind-wandering episodes, meta-cognitive
awareness indices, and other domain-specific measures.
"""

import numpy as np
from typing import Dict, List, Any
from dataclasses import dataclass


@dataclass
class MeditationMetrics:
    """Container for meditation-specific metrics."""
    attention_stability: float
    mind_wandering_frequency: float
    metacognitive_efficiency: float
    distraction_resistance: float
    transition_smoothness: float
    network_connectivity: float
    overall_quality_score: float


class MetricsCalculator:
    """
    Calculator for meditation-specific performance metrics.
    
    Provides standardized metrics commonly used in meditation research
    including attention stability, distraction resistance, and
    metacognitive efficiency measures.
    """
    
    def calculate_all_metrics(self, learner: Any) -> MeditationMetrics:
        """Calculate all meditation metrics for a learner."""
        return MeditationMetrics(
            attention_stability=self._calculate_attention_stability(learner),
            mind_wandering_frequency=self._calculate_mind_wandering_frequency(learner),
            metacognitive_efficiency=self._calculate_meta_awareness_efficiency(learner),
            distraction_resistance=self._calculate_distraction_resistance(learner),
            transition_smoothness=self._calculate_transition_smoothness(learner),
            network_connectivity=self.calculate_network_connectivity(learner)['overall_connectivity'],
            overall_quality_score=self._calculate_overall_quality(learner)
        )
    
    def calculate_attention_stability(self, learner: Any) -> float:
        """Calculate attention stability index."""
        return self._calculate_attention_stability(learner)
    
    def calculate_distraction_resistance(self, learner: Any) -> float:
        """Calculate resistance to distracting thoughtseeds."""
        return self._calculate_distraction_resistance(learner)
    
    def calculate_metacognitive_efficiency(self, learner: Any) -> float:
        """Calculate metacognitive awareness efficiency."""
        return self._calculate_meta_awareness_efficiency(learner)
    
    def calculate_network_connectivity(self, learner: Any) -> Dict[str, float]:
        """Calculate network connectivity measures."""
        if not hasattr(learner, 'network_activations_history'):
            return {'overall_connectivity': 0.0, 'dmn_dan_anticorr': 0.0, 'stability_index': 0.0}
        
        network_array = np.array([
            [step.get(net, 0) for net in ['DMN', 'VAN', 'DAN', 'FPN']]
            for step in learner.network_activations_history
        ])
        
        # Overall connectivity as inverse of activation variance
        connectivity = 1 / (np.mean(np.var(network_array, axis=0)) + 1e-6)
        overall_connectivity = float(min(1.0, connectivity / 10))  # Normalize to 0-1
        
        # DMN-DAN anticorrelation if data available
        dmn_dan_anticorr = 0.0
        if network_array.shape[0] > 1 and network_array.shape[1] >= 3:
            dmn_values = network_array[:, 0]  # DMN is first
            dan_values = network_array[:, 2]  # DAN is third
            corr_matrix = np.corrcoef(dmn_values, dan_values)
            if not np.isnan(corr_matrix[0, 1]):
                dmn_dan_anticorr = -corr_matrix[0, 1]  # Negative correlation is good
        
        return {
            'overall_connectivity': overall_connectivity,
            'dmn_dan_anticorr': float(dmn_dan_anticorr),
            'stability_index': float(1.0 - np.mean(np.var(network_array, axis=0)))
        }
    
    def _calculate_attention_stability(self, learner: Any) -> float:
        """Calculate attention stability index."""
        activations = np.array(learner.activations_history)
        breath_idx = learner.thoughtseeds.index('breath_focus')
        breath_stability = 1 / (np.std(activations[:, breath_idx]) + 1e-6)
        return float(min(1.0, breath_stability / 10))  # Normalize to 0-1
    
    def _calculate_mind_wandering_frequency(self, learner: Any) -> float:
        """Calculate frequency of mind wandering episodes."""
        mw_count = learner.state_history.count('mind_wandering')
        return float(mw_count / len(learner.state_history))
    
    def _calculate_meta_awareness_efficiency(self, learner: Any) -> float:
        """Calculate metacognitive awareness efficiency."""
        ma_mean = np.mean(learner.meta_awareness_history)
        return float(min(1.0, ma_mean))
    
    def _calculate_distraction_resistance(self, learner: Any) -> float:
        """Calculate resistance to distracting thoughtseeds."""
        activations = np.array(learner.activations_history)
        distraction_thoughtseeds = ['pain_discomfort', 'pending_tasks']
        
        distraction_level = 0
        count = 0
        for ts in distraction_thoughtseeds:
            if ts in learner.thoughtseeds:
                idx = learner.thoughtseeds.index(ts)
                distraction_level += np.mean(activations[:, idx])
                count += 1
        
        if count == 0:
            return 1.0
        
        avg_distraction = distraction_level / count
        return float(max(0, 1 - avg_distraction))
    
    def _calculate_transition_smoothness(self, learner: Any) -> float:
        """Calculate smoothness of state transitions."""
        natural_transitions = getattr(learner, 'natural_transition_count', 0)
        total_transitions = natural_transitions + getattr(learner, 'forced_transition_count', 0)
        
        if total_transitions == 0:
            return 0.0
        
        return float(natural_transitions / total_transitions)
    
    def _calculate_overall_quality(self, learner: Any) -> float:
        """Calculate overall meditation quality score."""
        metrics = [
            self._calculate_attention_stability(learner),
            1 - self._calculate_mind_wandering_frequency(learner),  # Invert (less MW is better)
            self._calculate_meta_awareness_efficiency(learner),
            self._calculate_distraction_resistance(learner),
            self._calculate_transition_smoothness(learner)
        ]
        
        return float(np.mean(metrics))
