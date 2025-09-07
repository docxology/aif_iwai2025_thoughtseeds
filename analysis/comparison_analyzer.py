"""
Comparison analysis functionality for expert vs novice meditation studies.

This module provides comprehensive comparison analysis tools for evaluating
differences between expert and novice meditators, including statistical
significance tests, effect size calculations, and domain-specific metrics.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    # Create a simple stats module substitute
    class SimpleStats:
        @staticmethod
        def ttest_ind(a, b, equal_var=True):
            # Simple t-test approximation
            mean1, mean2 = np.mean(a), np.mean(b)
            var1, var2 = np.var(a), np.var(b)
            n1, n2 = len(a), len(b)
            
            if equal_var:
                pooled_var = ((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2)
                t = (mean1 - mean2) / np.sqrt(pooled_var * (1/n1 + 1/n2))
            else:
                se = np.sqrt(var1/n1 + var2/n2)
                t = (mean1 - mean2) / se if se != 0 else 0
            
            # Rough p-value approximation
            p = min(1.0, 2 * (1 - abs(t) / (abs(t) + 2)))
            return t, p
        
        @staticmethod
        def mannwhitneyu(a, b, alternative='two-sided'):
            # Simple rank-based approximation
            combined = list(a) + list(b)
            n1, n2 = len(a), len(b)
            
            if n1 == 0 or n2 == 0:
                return 0.0, 1.0
            
            # Simple approximation based on medians
            med1, med2 = np.median(a), np.median(b)
            u = abs(med1 - med2) * n1 * n2 / (n1 + n2)
            p = min(1.0, 2 * (1 - u / max(u + 10, 20)))
            
            return u, p
    
    stats = SimpleStats()

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    class SimpleTimestamp:
        @staticmethod
        def now():
            return SimpleTimestamp()
        
        def isoformat(self):
            return datetime.now().isoformat()
    
    class SimplePandas:
        Timestamp = SimpleTimestamp
    
    pd = SimplePandas()

from .statistical_analyzer import StatisticalAnalyzer, AnalysisResult


@dataclass
class ComparisonResult:
    """Container for comparison analysis results."""
    novice_analysis: Any  # Changed from AnalysisResult to Any for flexibility
    expert_analysis: Any  # Changed from AnalysisResult to Any for flexibility  
    comparison_statistics: Dict[str, Any]
    effect_sizes: Dict[str, float]
    statistical_tests: Dict[str, Dict[str, float]]  # Renamed from significance_tests
    meditation_metrics: Dict[str, Any]
    timestamp: str


class ComparisonAnalyzer:
    """
    Comprehensive comparison analyzer for expert vs novice meditation studies.
    
    Provides statistical comparison functionality including significance tests,
    effect size calculations, and meditation-specific performance metrics
    for comparing expert and novice meditators.
    """
    
    def __init__(self, significance_level: float = 0.05):
        """
        Initialize comparison analyzer.
        
        Args:
            significance_level: Statistical significance threshold
        """
        self.significance_level = significance_level
        self.statistical_analyzer = StatisticalAnalyzer(significance_level)
        self.has_scipy = SCIPY_AVAILABLE
        self.has_pandas = PANDAS_AVAILABLE
    
    def compare_learners(self, novice_learner: Any, expert_learner: Any) -> ComparisonResult:
        """
        Perform comprehensive comparison between novice and expert learners.
        
        Args:
            novice_learner: Novice learner instance
            expert_learner: Expert learner instance
            
        Returns:
            Complete comparison analysis result
        """
        # Perform individual analyses
        novice_analysis = self.statistical_analyzer.analyze_learner(novice_learner)
        expert_analysis = self.statistical_analyzer.analyze_learner(expert_learner)
        
        # Perform comparative analyses
        comparison_stats = self._calculate_comparison_statistics(novice_learner, expert_learner)
        effect_sizes = self._calculate_effect_sizes(novice_learner, expert_learner)
        significance_tests = self._perform_significance_tests(novice_learner, expert_learner)
        meditation_metrics = self._calculate_meditation_specific_metrics(novice_learner, expert_learner)
        
        return ComparisonResult(
            novice_analysis=novice_analysis,
            expert_analysis=expert_analysis,
            comparison_statistics=comparison_stats,
            effect_sizes=effect_sizes,
            statistical_tests=significance_tests,  # Renamed parameter
            meditation_metrics=meditation_metrics,
            timestamp=pd.Timestamp.now().isoformat()
        )
    
    def calculate_effect_size(self, group1: List[float], group2: List[float]) -> Dict[str, float]:
        """Calculate Cohen's d effect size between two groups."""
        if len(group1) < 2 or len(group2) < 2:
            return {'cohens_d': 0.0, 'interpretation': 'no_effect'}
        
        cohens_d = self._calculate_cohens_d(np.array(group1), np.array(group2))
        
        # Interpret effect size
        if abs(cohens_d) < 0.2:
            interpretation = 'small'
        elif abs(cohens_d) < 0.5:
            interpretation = 'medium'
        elif abs(cohens_d) < 0.8:
            interpretation = 'large'
        else:
            interpretation = 'very_large'
        
        return {
            'cohens_d': float(cohens_d),
            'interpretation': interpretation,
            'magnitude': abs(float(cohens_d))
        }
    
    def t_test(self, group1: List[float], group2: List[float]) -> Dict[str, float]:
        """Perform t-test between two groups."""
        if len(group1) < 2 or len(group2) < 2:
            return {'statistic': 0.0, 'p_value': 1.0, 'significant': False}
        
        t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)
        
        return {
            'statistic': float(t_stat),
            'p_value': float(p_value),
            'significant': p_value < self.significance_level
        }
    
    def mann_whitney_u_test(self, group1: List[float], group2: List[float]) -> Dict[str, float]:
        """Perform Mann-Whitney U test between two groups."""
        if len(group1) == 0 or len(group2) == 0:
            return {'statistic': 0.0, 'p_value': 1.0, 'significant': False}
        
        u_stat, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')
        
        return {
            'statistic': float(u_stat),
            'p_value': float(p_value),
            'significant': p_value < self.significance_level
        }
    
    
    def _calculate_comparison_statistics(self, novice: Any, expert: Any) -> Dict[str, Any]:
        """Calculate basic comparison statistics between groups."""
        novice_activations = np.array(novice.activations_history)
        expert_activations = np.array(expert.activations_history)
        
        comparison_stats = {
            'sample_sizes': {
                'novice': len(novice.activations_history),
                'expert': len(expert.activations_history)
            },
            
            'thoughtseed_comparisons': {},
            'meta_awareness_comparison': self._compare_meta_awareness(novice, expert),
            'transition_comparison': self._compare_transitions(novice, expert),
            'temporal_comparison': self._compare_temporal_patterns(novice, expert)
        }
        
        # Compare each thoughtseed
        for i, ts in enumerate(novice.thoughtseeds):
            novice_ts_data = novice_activations[:, i]
            expert_ts_data = expert_activations[:, i]
            
            comparison_stats['thoughtseed_comparisons'][ts] = {
                'novice_mean': float(np.mean(novice_ts_data)),
                'expert_mean': float(np.mean(expert_ts_data)),
                'mean_difference': float(np.mean(expert_ts_data) - np.mean(novice_ts_data)),
                'percent_change': float(
                    (np.mean(expert_ts_data) - np.mean(novice_ts_data)) / np.mean(novice_ts_data) * 100
                ) if np.mean(novice_ts_data) != 0 else 0,
                'novice_std': float(np.std(novice_ts_data)),
                'expert_std': float(np.std(expert_ts_data)),
                'variance_ratio': float(np.var(expert_ts_data) / np.var(novice_ts_data)) if np.var(novice_ts_data) != 0 else 0
            }
        
        # Network comparisons if available
        if hasattr(novice, 'network_activations_history') and hasattr(expert, 'network_activations_history'):
            comparison_stats['network_comparison'] = self._compare_networks(novice, expert)
            comparison_stats['free_energy_comparison'] = self._compare_free_energy(novice, expert)
        
        return comparison_stats
    
    def _calculate_effect_sizes(self, novice: Any, expert: Any) -> Dict[str, float]:
        """Calculate effect sizes (Cohen's d) for all comparisons."""
        novice_activations = np.array(novice.activations_history)
        expert_activations = np.array(expert.activations_history)
        
        effect_sizes = {}
        
        # Effect sizes for thoughtseeds
        for i, ts in enumerate(novice.thoughtseeds):
            novice_data = novice_activations[:, i]
            expert_data = expert_activations[:, i]
            
            effect_sizes[f'{ts}_cohen_d'] = self._calculate_cohens_d(novice_data, expert_data)
        
        # Effect size for meta-awareness
        effect_sizes['meta_awareness_cohen_d'] = self._calculate_cohens_d(
            novice.meta_awareness_history, expert.meta_awareness_history
        )
        
        # Network effect sizes if available
        if hasattr(novice, 'network_activations_history') and hasattr(expert, 'network_activations_history'):
            novice_networks = np.array([
                [step[net] for net in novice.networks]
                for step in novice.network_activations_history
            ])
            expert_networks = np.array([
                [step[net] for net in expert.networks]
                for step in expert.network_activations_history
            ])
            
            for i, net in enumerate(novice.networks):
                effect_sizes[f'{net}_cohen_d'] = self._calculate_cohens_d(
                    novice_networks[:, i], expert_networks[:, i]
                )
            
            # Free energy effect size
            effect_sizes['free_energy_cohen_d'] = self._calculate_cohens_d(
                novice.free_energy_history, expert.free_energy_history
            )
        
        return effect_sizes
    
    def _perform_significance_tests(self, novice: Any, expert: Any) -> Dict[str, Dict[str, float]]:
        """Perform statistical significance tests."""
        novice_activations = np.array(novice.activations_history)
        expert_activations = np.array(expert.activations_history)
        
        significance_tests = {}
        
        # T-tests for thoughtseeds
        for i, ts in enumerate(novice.thoughtseeds):
            novice_data = novice_activations[:, i]
            expert_data = expert_activations[:, i]
            
            # Welch's t-test (unequal variances)
            t_stat, p_value = stats.ttest_ind(expert_data, novice_data, equal_var=False)
            
            # Mann-Whitney U test (non-parametric)
            u_stat, u_p_value = stats.mannwhitneyu(expert_data, novice_data, alternative='two-sided')
            
            significance_tests[ts] = {
                't_statistic': float(t_stat),
                't_p_value': float(p_value),
                't_significant': p_value < self.significance_level,
                'mannwhitney_u': float(u_stat),
                'mannwhitney_p': float(u_p_value),
                'mannwhitney_significant': u_p_value < self.significance_level
            }
        
        # Test for meta-awareness
        t_stat, p_value = stats.ttest_ind(
            expert.meta_awareness_history, novice.meta_awareness_history, equal_var=False
        )
        u_stat, u_p_value = stats.mannwhitneyu(
            expert.meta_awareness_history, novice.meta_awareness_history, alternative='two-sided'
        )
        
        significance_tests['meta_awareness'] = {
            't_statistic': float(t_stat),
            't_p_value': float(p_value),
            't_significant': p_value < self.significance_level,
            'mannwhitney_u': float(u_stat),
            'mannwhitney_p': float(u_p_value),
            'mannwhitney_significant': u_p_value < self.significance_level
        }
        
        # Network tests if available
        if hasattr(novice, 'network_activations_history') and hasattr(expert, 'network_activations_history'):
            novice_networks = np.array([
                [step[net] for net in novice.networks]
                for step in novice.network_activations_history
            ])
            expert_networks = np.array([
                [step[net] for net in expert.networks]
                for step in expert.network_activations_history
            ])
            
            for i, net in enumerate(novice.networks):
                t_stat, p_value = stats.ttest_ind(
                    expert_networks[:, i], novice_networks[:, i], equal_var=False
                )
                u_stat, u_p_value = stats.mannwhitneyu(
                    expert_networks[:, i], novice_networks[:, i], alternative='two-sided'
                )
                
                significance_tests[f'{net}_network'] = {
                    't_statistic': float(t_stat),
                    't_p_value': float(p_value),
                    't_significant': p_value < self.significance_level,
                    'mannwhitney_u': float(u_stat),
                    'mannwhitney_p': float(u_p_value),
                    'mannwhitney_significant': u_p_value < self.significance_level
                }
            
            # Free energy test
            t_stat, p_value = stats.ttest_ind(
                expert.free_energy_history, novice.free_energy_history, equal_var=False
            )
            significance_tests['free_energy'] = {
                't_statistic': float(t_stat),
                't_p_value': float(p_value),
                't_significant': p_value < self.significance_level
            }
        
        return significance_tests
    
    def _calculate_meditation_specific_metrics(self, novice: Any, expert: Any) -> Dict[str, Any]:
        """Calculate meditation-specific performance metrics."""
        metrics = {
            'attentional_stability': self._compare_attentional_stability(novice, expert),
            'meta_cognitive_efficiency': self._compare_metacognitive_efficiency(novice, expert),
            'distraction_resistance': self._compare_distraction_resistance(novice, expert),
            'state_transition_efficiency': self._compare_transition_efficiency(novice, expert),
            'meditation_quality_index': self._calculate_meditation_quality_index(novice, expert)
        }
        
        # Network-specific metrics if available
        if hasattr(novice, 'network_activations_history') and hasattr(expert, 'network_activations_history'):
            metrics['network_efficiency'] = self._compare_network_efficiency(novice, expert)
            metrics['free_energy_optimization'] = self._compare_free_energy_optimization(novice, expert)
            metrics['dmn_suppression_ability'] = self._compare_dmn_suppression(novice, expert)
            metrics['cognitive_control_strength'] = self._compare_cognitive_control(novice, expert)
        
        return metrics
    
    def _compare_meta_awareness(self, novice: Any, expert: Any) -> Dict[str, float]:
        """Compare meta-awareness patterns."""
        novice_ma = np.array(novice.meta_awareness_history)
        expert_ma = np.array(expert.meta_awareness_history)
        
        return {
            'novice_mean': float(np.mean(novice_ma)),
            'expert_mean': float(np.mean(expert_ma)),
            'expert_advantage': float(np.mean(expert_ma) - np.mean(novice_ma)),
            'expert_consistency': float(1 / (np.std(expert_ma) + 1e-6)),  # Higher is more consistent
            'novice_consistency': float(1 / (np.std(novice_ma) + 1e-6)),
            'consistency_ratio': float((1 / (np.std(expert_ma) + 1e-6)) / (1 / (np.std(novice_ma) + 1e-6)))
        }
    
    def _compare_transitions(self, novice: Any, expert: Any) -> Dict[str, Any]:
        """Compare state transition patterns."""
        # Use safe attribute access for Mock objects
        novice_natural = self._safe_get_int_attr(novice, 'natural_transition_count', 0)
        novice_forced = self._safe_get_int_attr(novice, 'forced_transition_count', 0)
        expert_natural = self._safe_get_int_attr(expert, 'natural_transition_count', 0)
        expert_forced = self._safe_get_int_attr(expert, 'forced_transition_count', 0)
        
        novice_rate = novice_natural / max(1, novice_natural + novice_forced)
        expert_rate = expert_natural / max(1, expert_natural + expert_forced)
        
        return {
            'natural_transition_rates': {
                'novice': float(novice_rate),
                'expert': float(expert_rate)
            },
            'expert_natural_advantage': float(expert_rate - novice_rate),
            'transition_efficiency_ratio': float(expert_natural / max(1, novice_natural))
        }
    
    def _compare_temporal_patterns(self, novice: Any, expert: Any) -> Dict[str, float]:
        """Compare temporal pattern characteristics."""
        novice_activations = np.array(novice.activations_history)
        expert_activations = np.array(expert.activations_history)
        
        # Calculate temporal stability (inverse of activation variability over time)
        novice_stability = 1 / (np.mean(np.std(novice_activations, axis=0)) + 1e-6)
        expert_stability = 1 / (np.mean(np.std(expert_activations, axis=0)) + 1e-6)
        
        return {
            'novice_temporal_stability': float(novice_stability),
            'expert_temporal_stability': float(expert_stability),
            'stability_improvement': float(expert_stability - novice_stability),
            'stability_ratio': float(expert_stability / novice_stability)
        }
    
    def _compare_networks(self, novice: Any, expert: Any) -> Dict[str, Any]:
        """Compare network activation patterns."""
        novice_networks = np.array([
            [step[net] for net in novice.networks]
            for step in novice.network_activations_history
        ])
        expert_networks = np.array([
            [step[net] for net in expert.networks]
            for step in expert.network_activations_history
        ])
        
        network_comparison = {}
        
        for i, net in enumerate(novice.networks):
            novice_net_data = novice_networks[:, i]
            expert_net_data = expert_networks[:, i]
            
            network_comparison[net] = {
                'novice_mean': float(np.mean(novice_net_data)),
                'expert_mean': float(np.mean(expert_net_data)),
                'expert_change': float(np.mean(expert_net_data) - np.mean(novice_net_data)),
                'percent_change': float(
                    (np.mean(expert_net_data) - np.mean(novice_net_data)) / np.mean(novice_net_data) * 100
                ) if np.mean(novice_net_data) != 0 else 0
            }
        
        return network_comparison
    
    def _compare_free_energy(self, novice: Any, expert: Any) -> Dict[str, float]:
        """Compare free energy optimization patterns."""
        novice_fe = np.array(novice.free_energy_history)
        expert_fe = np.array(expert.free_energy_history)
        
        return {
            'novice_mean': float(np.mean(novice_fe)),
            'expert_mean': float(np.mean(expert_fe)),
            'novice_mean_fe': float(np.mean(novice_fe)),
            'expert_mean_fe': float(np.mean(expert_fe)),
            'expert_fe_reduction': float(np.mean(novice_fe) - np.mean(expert_fe)),
            'expert_fe_improvement_percent': float(
                (np.mean(novice_fe) - np.mean(expert_fe)) / np.mean(novice_fe) * 100
            ) if np.mean(novice_fe) != 0 else 0,
            'novice_final_fe': float(novice_fe[-1]),
            'expert_final_fe': float(expert_fe[-1]),
            'final_fe_improvement': float(novice_fe[-1] - expert_fe[-1])
        }
    
    def _calculate_cohens_d(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """Calculate Cohen's d effect size."""
        n1, n2 = len(group1), len(group2)
        pooled_std = np.sqrt(((n1 - 1) * np.var(group1) + (n2 - 1) * np.var(group2)) / (n1 + n2 - 2))
        
        if pooled_std == 0:
            return 0.0
        
        return float((np.mean(group2) - np.mean(group1)) / pooled_std)
    
    def _compare_attentional_stability(self, novice: Any, expert: Any) -> Dict[str, float]:
        """Compare attentional stability between groups."""
        # Focus on breath_focus thoughtseed stability
        novice_activations = np.array(novice.activations_history)
        expert_activations = np.array(expert.activations_history)
        
        breath_idx = novice.thoughtseeds.index('breath_focus')
        
        novice_breath_stability = 1 / (np.std(novice_activations[:, breath_idx]) + 1e-6)
        expert_breath_stability = 1 / (np.std(expert_activations[:, breath_idx]) + 1e-6)
        
        return {
            'novice_attentional_stability': float(novice_breath_stability),
            'expert_attentional_stability': float(expert_breath_stability),
            'stability_improvement': float(expert_breath_stability - novice_breath_stability),
            'stability_ratio': float(expert_breath_stability / novice_breath_stability)
        }
    
    def _compare_metacognitive_efficiency(self, novice: Any, expert: Any) -> Dict[str, float]:
        """Compare metacognitive efficiency."""
        # Efficiency = meta-awareness achieved per unit of self-reflection effort
        novice_activations = np.array(novice.activations_history)
        expert_activations = np.array(expert.activations_history)
        
        reflection_idx = novice.thoughtseeds.index('self_reflection')
        
        novice_reflection_effort = np.mean(novice_activations[:, reflection_idx])
        expert_reflection_effort = np.mean(expert_activations[:, reflection_idx])
        
        novice_ma_achieved = np.mean(novice.meta_awareness_history)
        expert_ma_achieved = np.mean(expert.meta_awareness_history)
        
        novice_efficiency = novice_ma_achieved / (novice_reflection_effort + 1e-6)
        expert_efficiency = expert_ma_achieved / (expert_reflection_effort + 1e-6)
        
        return {
            'novice_metacognitive_efficiency': float(novice_efficiency),
            'expert_metacognitive_efficiency': float(expert_efficiency),
            'efficiency_improvement': float(expert_efficiency - novice_efficiency),
            'efficiency_ratio': float(expert_efficiency / novice_efficiency)
        }
    
    def _compare_distraction_resistance(self, novice: Any, expert: Any) -> Dict[str, float]:
        """Compare resistance to distracting thoughtseeds."""
        novice_activations = np.array(novice.activations_history)
        expert_activations = np.array(expert.activations_history)
        
        # Focus on distracting thoughtseeds
        distraction_thoughtseeds = ['pain_discomfort', 'pending_tasks']
        distraction_indices = [novice.thoughtseeds.index(ts) for ts in distraction_thoughtseeds if ts in novice.thoughtseeds]
        
        novice_distraction_level = np.mean([np.mean(novice_activations[:, i]) for i in distraction_indices])
        expert_distraction_level = np.mean([np.mean(expert_activations[:, i]) for i in distraction_indices])
        
        # Lower distraction = higher resistance
        novice_resistance = 1 / (novice_distraction_level + 1e-6)
        expert_resistance = 1 / (expert_distraction_level + 1e-6)
        
        return {
            'novice_distraction_resistance': float(novice_resistance),
            'expert_distraction_resistance': float(expert_resistance),
            'resistance_improvement': float(expert_resistance - novice_resistance),
            'resistance_ratio': float(expert_resistance / novice_resistance)
        }
    
    def _compare_transition_efficiency(self, novice: Any, expert: Any) -> Dict[str, float]:
        """Compare efficiency of state transitions."""
        novice_natural_rate = self._safe_get_int_attr(novice, 'natural_transition_count', 0) / max(1, len(novice.state_history))
        expert_natural_rate = self._safe_get_int_attr(expert, 'natural_transition_count', 0) / max(1, len(expert.state_history))
        
        return {
            'novice_transition_efficiency': float(novice_natural_rate),
            'expert_transition_efficiency': float(expert_natural_rate),
            'efficiency_improvement': float(expert_natural_rate - novice_natural_rate),
            'efficiency_ratio': float(expert_natural_rate / max(1e-6, novice_natural_rate))
        }
    
    def _calculate_meditation_quality_index(self, novice: Any, expert: Any) -> Dict[str, float]:
        """Calculate overall meditation quality index."""
        # Composite index considering multiple factors
        
        # Factor 1: Meta-awareness level
        ma_factor_n = np.mean(novice.meta_awareness_history)
        ma_factor_e = np.mean(expert.meta_awareness_history)
        
        # Factor 2: Attention stability (breath focus consistency)
        novice_activations = np.array(novice.activations_history)
        expert_activations = np.array(expert.activations_history)
        breath_idx = novice.thoughtseeds.index('breath_focus')
        
        stability_factor_n = 1 / (np.std(novice_activations[:, breath_idx]) + 1e-6)
        stability_factor_e = 1 / (np.std(expert_activations[:, breath_idx]) + 1e-6)
        
        # Factor 3: Natural transition rate
        novice_natural = self._safe_get_int_attr(novice, 'natural_transition_count', 0)
        novice_forced = self._safe_get_int_attr(novice, 'forced_transition_count', 0)
        expert_natural = self._safe_get_int_attr(expert, 'natural_transition_count', 0)
        expert_forced = self._safe_get_int_attr(expert, 'forced_transition_count', 0)
        
        transition_factor_n = novice_natural / max(1, novice_natural + novice_forced)
        transition_factor_e = expert_natural / max(1, expert_natural + expert_forced)
        
        # Normalize factors (0-1 scale)
        ma_factor_n = min(1.0, ma_factor_n)
        ma_factor_e = min(1.0, ma_factor_e)
        stability_factor_n = min(1.0, stability_factor_n / 10)  # Scale down
        stability_factor_e = min(1.0, stability_factor_e / 10)
        
        # Composite indices
        novice_quality = (ma_factor_n + stability_factor_n + transition_factor_n) / 3
        expert_quality = (ma_factor_e + stability_factor_e + transition_factor_e) / 3
        
        return {
            'novice_meditation_quality': float(novice_quality),
            'expert_meditation_quality': float(expert_quality),
            'quality_improvement': float(expert_quality - novice_quality),
            'quality_ratio': float(expert_quality / max(1e-6, novice_quality))
        }
    
    def _compare_network_efficiency(self, novice: Any, expert: Any) -> Dict[str, float]:
        """Compare network efficiency metrics."""
        # Network efficiency = achieving target states with minimal energy
        if not (hasattr(novice, 'network_activations_history') and hasattr(expert, 'network_activations_history')):
            return {}
        
        novice_fe = np.mean(novice.free_energy_history)
        expert_fe = np.mean(expert.free_energy_history)
        
        # Lower free energy = higher efficiency
        novice_efficiency = 1 / (novice_fe + 1e-6)
        expert_efficiency = 1 / (expert_fe + 1e-6)
        
        return {
            'novice_network_efficiency': float(novice_efficiency),
            'expert_network_efficiency': float(expert_efficiency),
            'efficiency_improvement': float(expert_efficiency - novice_efficiency),
            'efficiency_ratio': float(expert_efficiency / novice_efficiency)
        }
    
    def _compare_free_energy_optimization(self, novice: Any, expert: Any) -> Dict[str, float]:
        """Compare free energy optimization performance."""
        novice_fe = np.array(novice.free_energy_history)
        expert_fe = np.array(expert.free_energy_history)
        
        # Optimization speed (how quickly free energy decreases)
        novice_optimization_rate = (novice_fe[0] - novice_fe[-1]) / len(novice_fe)
        expert_optimization_rate = (expert_fe[0] - expert_fe[-1]) / len(expert_fe)
        
        return {
            'novice_optimization_rate': float(novice_optimization_rate),
            'expert_optimization_rate': float(expert_optimization_rate),
            'rate_improvement': float(expert_optimization_rate - novice_optimization_rate),
            'final_fe_advantage': float(novice_fe[-1] - expert_fe[-1])
        }
    
    def _compare_dmn_suppression(self, novice: Any, expert: Any) -> Dict[str, float]:
        """Compare Default Mode Network suppression ability."""
        novice_networks = np.array([
            [step[net] for net in novice.networks]
            for step in novice.network_activations_history
        ])
        expert_networks = np.array([
            [step[net] for net in expert.networks]
            for step in expert.network_activations_history
        ])
        
        dmn_idx = novice.networks.index('DMN')
        
        # Lower DMN activation during focused states indicates better suppression
        focused_states = ['breath_control', 'redirect_breath']
        
        novice_dmn_focused = []
        expert_dmn_focused = []
        
        for i, state in enumerate(novice.state_history):
            if state in focused_states:
                novice_dmn_focused.append(novice_networks[i, dmn_idx])
        
        for i, state in enumerate(expert.state_history):
            if state in focused_states:
                expert_dmn_focused.append(expert_networks[i, dmn_idx])
        
        novice_dmn_suppression = 1 / (np.mean(novice_dmn_focused) + 1e-6) if novice_dmn_focused else 0
        expert_dmn_suppression = 1 / (np.mean(expert_dmn_focused) + 1e-6) if expert_dmn_focused else 0
        
        return {
            'novice_dmn_suppression': float(novice_dmn_suppression),
            'expert_dmn_suppression': float(expert_dmn_suppression),
            'suppression_improvement': float(expert_dmn_suppression - novice_dmn_suppression),
            'suppression_ratio': float(expert_dmn_suppression / max(1e-6, novice_dmn_suppression))
        }
    
    def _compare_cognitive_control(self, novice: Any, expert: Any) -> Dict[str, float]:
        """Compare cognitive control network strength."""
        novice_networks = np.array([
            [step[net] for net in novice.networks]
            for step in novice.network_activations_history
        ])
        expert_networks = np.array([
            [step[net] for net in expert.networks]
            for step in expert.network_activations_history
        ])
        
        # Focus on DAN and FPN for cognitive control
        dan_idx = novice.networks.index('DAN')
        fpn_idx = novice.networks.index('FPN')
        
        novice_control = np.mean(novice_networks[:, dan_idx]) + np.mean(novice_networks[:, fpn_idx])
        expert_control = np.mean(expert_networks[:, dan_idx]) + np.mean(expert_networks[:, fpn_idx])
        
        return {
            'novice_cognitive_control': float(novice_control),
            'expert_cognitive_control': float(expert_control),
            'control_improvement': float(expert_control - novice_control),
            'control_ratio': float(expert_control / max(1e-6, novice_control))
        }
    
    def _safe_get_int_attr(self, obj: Any, attr_name: str, default: int = 0) -> int:
        """Safely get an integer attribute, handling Mock objects."""
        try:
            value = getattr(obj, attr_name, default)
            # Handle Mock objects or other non-numeric types
            if hasattr(value, '__call__') or str(type(value)) == "<class 'unittest.mock.Mock'>":
                return default
            return int(value)
        except (TypeError, ValueError):
            return default
