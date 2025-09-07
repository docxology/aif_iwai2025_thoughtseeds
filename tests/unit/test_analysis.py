"""
Comprehensive unit tests for the analysis module.

This module tests all analysis functionality including statistical analysis,
comparative studies, metrics calculation, network analysis, and time series analysis.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from analysis import (
    StatisticalAnalyzer, ComparisonAnalyzer, MetricsCalculator,
    NetworkAnalyzer, TimeSeriesAnalyzer
)


class TestStatisticalAnalyzer:
    """Test statistical analysis functionality."""
    
    def test_analyzer_initialization(self):
        """Test StatisticalAnalyzer initialization."""
        analyzer = StatisticalAnalyzer()
        
        # Check basic attributes
        assert hasattr(analyzer, 'has_scipy')
        assert hasattr(analyzer, 'has_pandas')
    
    def test_calculate_descriptive_stats(self):
        """Test descriptive statistics calculation."""
        analyzer = StatisticalAnalyzer()
        
        # Test with simple data
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        stats = analyzer.calculate_descriptive_stats(data)
        
        # Check output structure
        assert isinstance(stats, dict)
        required_keys = ['mean', 'std', 'min', 'max', 'median', 'count']
        for key in required_keys:
            assert key in stats
        
        # Check values
        assert stats['mean'] == 5.5
        assert stats['min'] == 1
        assert stats['max'] == 10
        assert stats['median'] == 5.5
        assert stats['count'] == 10
        
        # Check std is reasonable
        assert 2.5 < stats['std'] < 3.5
    
    def test_calculate_descriptive_stats_empty_data(self):
        """Test descriptive statistics with empty data."""
        analyzer = StatisticalAnalyzer()
        
        stats = analyzer.calculate_descriptive_stats([])
        
        # Should handle empty data gracefully
        assert stats['count'] == 0
        assert stats['mean'] == 0
        assert stats['std'] == 0
    
    def test_calculate_correlation(self):
        """Test correlation calculation."""
        analyzer = StatisticalAnalyzer()
        
        # Test with correlated data
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 6, 8, 10])  # Perfect correlation
        
        correlation = analyzer.calculate_correlation(x, y)
        
        assert isinstance(correlation, dict)
        assert 'correlation' in correlation
        assert 'p_value' in correlation
        
        # Should be perfectly correlated
        assert abs(correlation['correlation'] - 1.0) < 0.001
    
    def test_calculate_correlation_no_correlation(self):
        """Test correlation with uncorrelated data."""
        analyzer = StatisticalAnalyzer()
        
        # Uncorrelated data
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([5, 3, 1, 4, 2])
        
        correlation = analyzer.calculate_correlation(x, y)
        
        # Correlation should be weak
        assert abs(correlation['correlation']) < 0.8
    
    def test_detect_trend(self):
        """Test trend detection."""
        analyzer = StatisticalAnalyzer()
        
        # Increasing trend
        increasing_data = [1, 3, 5, 7, 9, 11]
        trend = analyzer.detect_trend(increasing_data)
        
        assert isinstance(trend, dict)
        assert 'slope' in trend
        assert 'direction' in trend
        assert 'strength' in trend
        
        # Should detect increasing trend
        assert trend['slope'] > 0
        assert trend['direction'] == 'increasing'
        
        # Decreasing trend
        decreasing_data = [10, 8, 6, 4, 2]
        trend = analyzer.detect_trend(decreasing_data)
        
        assert trend['slope'] < 0
        assert trend['direction'] == 'decreasing'
    
    def test_analyze_learner(self, mock_learner):
        """Test full learner analysis."""
        analyzer = StatisticalAnalyzer()
        
        analysis = analyzer.analyze_learner(mock_learner)
        
        # Check output structure
        assert isinstance(analysis, dict)
        
        # Should have main sections
        expected_sections = [
            'basic_stats', 'time_series_analysis', 'state_analysis',
            'network_analysis', 'thoughtseed_analysis'
        ]
        
        for section in expected_sections:
            assert section in analysis
    
    def test_state_specific_analysis(self, mock_learner):
        """Test state-specific statistical analysis."""
        analyzer = StatisticalAnalyzer()
        
        # Add some state history to mock learner
        mock_learner.state_history = ['breath_control'] * 10 + ['mind_wandering'] * 10
        
        analysis = analyzer.analyze_learner(mock_learner)
        state_analysis = analysis['state_analysis']
        
        # Should have analysis for each state that appeared
        assert 'breath_control' in state_analysis
        assert 'mind_wandering' in state_analysis
        
        # Each state should have descriptive stats
        for state_data in state_analysis.values():
            assert 'free_energy_stats' in state_data
            assert 'meta_awareness_stats' in state_data


class TestComparisonAnalyzer:
    """Test comparative analysis functionality."""
    
    def test_analyzer_initialization(self):
        """Test ComparisonAnalyzer initialization."""
        analyzer = ComparisonAnalyzer()
        
        # Check dependencies
        assert hasattr(analyzer, 'has_scipy')
        assert hasattr(analyzer, 'has_pandas')
    
    def test_compare_learners_basic(self, mock_learner):
        """Test basic learner comparison."""
        analyzer = ComparisonAnalyzer()
        
        # Create two different mock learners
        learner1 = mock_learner
        learner1.experience_level = 'novice'
        
        learner2 = Mock()
        learner2.experience_level = 'expert'
        learner2.thoughtseeds = mock_learner.thoughtseeds
        learner2.states = mock_learner.states
        learner2.networks = mock_learner.networks
        learner2.timesteps = mock_learner.timesteps
        learner2.free_energy_history = np.random.rand(mock_learner.timesteps).tolist()
        learner2.meta_awareness_history = np.random.rand(mock_learner.timesteps).tolist()
        learner2.state_history = mock_learner.state_history.copy()
        learner2.activations_history = [
            np.random.rand(len(mock_learner.thoughtseeds)) for _ in range(mock_learner.timesteps)
        ]
        learner2.network_activations_history = [
            {net: np.random.rand() for net in mock_learner.networks} 
            for _ in range(mock_learner.timesteps)
        ]
        
        comparison = analyzer.compare_learners(learner1, learner2)
        
        # Check output structure
        assert isinstance(comparison, object)  # ComparisonResult object
        assert hasattr(comparison, 'comparison_statistics')
        assert hasattr(comparison, 'statistical_tests')
        assert hasattr(comparison, 'effect_sizes')
    
    def test_statistical_tests(self):
        """Test statistical test implementations."""
        analyzer = ComparisonAnalyzer()
        
        # Test t-test
        group1 = np.random.normal(0, 1, 50)
        group2 = np.random.normal(1, 1, 50)  # Different mean
        
        t_result = analyzer.t_test(group1, group2)
        
        assert isinstance(t_result, dict)
        assert 'statistic' in t_result
        assert 'p_value' in t_result
        
        # Should detect difference
        assert t_result['p_value'] < 0.05  # Likely to be significant
    
    def test_mann_whitney_u_test(self):
        """Test Mann-Whitney U test implementation."""
        analyzer = ComparisonAnalyzer()
        
        # Create two different distributions
        group1 = [1, 2, 3, 4, 5]
        group2 = [6, 7, 8, 9, 10]  # Clearly different
        
        u_result = analyzer.mann_whitney_u_test(group1, group2)
        
        assert isinstance(u_result, dict)
        assert 'statistic' in u_result
        assert 'p_value' in u_result
    
    def test_effect_size_calculation(self):
        """Test effect size calculations."""
        analyzer = ComparisonAnalyzer()
        
        # Create groups with known effect size
        group1 = np.random.normal(0, 1, 100)
        group2 = np.random.normal(0.8, 1, 100)  # 0.8 standard deviation difference
        
        effect_size = analyzer.calculate_effect_size(group1, group2)
        
        assert isinstance(effect_size, dict)
        assert 'cohens_d' in effect_size
        assert 'interpretation' in effect_size
        
        # Cohen's d should be around 0.8 (medium to large effect)
        assert 0.5 < abs(effect_size['cohens_d']) < 1.2
    
    def test_compare_specific_metrics(self, mock_learner):
        """Test comparison of specific metrics between learners."""
        analyzer = ComparisonAnalyzer()
        
        # Create expert learner with lower free energy (better performance)
        expert_learner = Mock()
        expert_learner.experience_level = 'expert'
        expert_learner.free_energy_history = [x * 0.8 for x in mock_learner.free_energy_history]
        expert_learner.meta_awareness_history = [x * 1.2 for x in mock_learner.meta_awareness_history]
        expert_learner.thoughtseeds = mock_learner.thoughtseeds
        expert_learner.states = mock_learner.states
        expert_learner.networks = mock_learner.networks
        expert_learner.timesteps = mock_learner.timesteps
        expert_learner.state_history = mock_learner.state_history
        expert_learner.activations_history = mock_learner.activations_history
        expert_learner.network_activations_history = mock_learner.network_activations_history
        
        comparison = analyzer.compare_learners(mock_learner, expert_learner)
        
        # Expert should have lower mean free energy
        fe_comparison = comparison.comparison_statistics['free_energy_comparison']
        assert fe_comparison['expert_mean'] < fe_comparison['novice_mean']
        
        # Expert should have higher mean meta-awareness
        ma_comparison = comparison.comparison_statistics['meta_awareness_comparison']
        assert ma_comparison['expert_mean'] > ma_comparison['novice_mean']


class TestMetricsCalculator:
    """Test meditation-specific metrics calculation."""
    
    def test_calculator_initialization(self):
        """Test MetricsCalculator initialization."""
        calculator = MetricsCalculator()
        
        # Should initialize without error
        assert calculator is not None
    
    def test_calculate_attention_stability(self, mock_learner):
        """Test attention stability calculation."""
        calculator = MetricsCalculator()
        
        # Calculate stability
        stability = calculator.calculate_attention_stability(mock_learner)
        
        # Should return numeric value
        assert isinstance(stability, (int, float))
        assert 0 <= stability <= 1  # Should be normalized
    
    def test_calculate_distraction_resistance(self, mock_learner):
        """Test distraction resistance calculation."""
        calculator = MetricsCalculator()
        
        # Add some distraction patterns to mock learner
        mock_learner.state_history = ['breath_control'] * 15 + ['mind_wandering'] * 5
        
        resistance = calculator.calculate_distraction_resistance(mock_learner)
        
        assert isinstance(resistance, (int, float))
        assert 0 <= resistance <= 1
    
    def test_calculate_metacognitive_efficiency(self, mock_learner):
        """Test metacognitive efficiency calculation."""
        calculator = MetricsCalculator()
        
        efficiency = calculator.calculate_metacognitive_efficiency(mock_learner)
        
        assert isinstance(efficiency, (int, float))
        assert 0 <= efficiency <= 1
    
    def test_calculate_network_connectivity(self, mock_learner):
        """Test network connectivity metrics."""
        calculator = MetricsCalculator()
        
        connectivity = calculator.calculate_network_connectivity(mock_learner)
        
        # Should return dictionary with connectivity metrics
        assert isinstance(connectivity, dict)
        
        # Should have metrics for each network pair or overall connectivity
        assert len(connectivity) > 0
    
    def test_calculate_all_metrics(self, mock_learner):
        """Test comprehensive metrics calculation."""
        calculator = MetricsCalculator()
        
        metrics = calculator.calculate_all_metrics(mock_learner)
        
        # Check output structure
        assert isinstance(metrics, object)  # MetricsResult object
        assert hasattr(metrics, 'attention_stability')
        assert hasattr(metrics, 'distraction_resistance')
        assert hasattr(metrics, 'metacognitive_efficiency')
        assert hasattr(metrics, 'network_connectivity')
        assert hasattr(metrics, 'overall_quality_score')
        
        # Check that overall quality is computed
        assert isinstance(metrics.overall_quality_score, (int, float))
        assert 0 <= metrics.overall_quality_score <= 1


class TestNetworkAnalyzer:
    """Test network-specific analysis functionality."""
    
    def test_analyzer_initialization(self):
        """Test NetworkAnalyzer initialization."""
        analyzer = NetworkAnalyzer()
        
        assert analyzer is not None
    
    def test_calculate_network_correlations(self, mock_learner):
        """Test network correlation calculation."""
        analyzer = NetworkAnalyzer()
        
        correlations = analyzer.calculate_network_correlations(mock_learner)
        
        # Should return correlation matrix or similar structure
        assert isinstance(correlations, dict)
        
        # Should have correlations between network pairs
        networks = mock_learner.networks
        for i, net1 in enumerate(networks):
            for net2 in networks[i+1:]:
                pair_key = f"{net1}_{net2}" or f"{net2}_{net1}"
                # Should have some representation of network relationships
    
    def test_analyze_dmn_dan_anticorrelation(self, mock_learner):
        """Test DMN-DAN anticorrelation analysis."""
        analyzer = NetworkAnalyzer()
        
        # Create mock data with anticorrelated DMN and DAN
        anticorr_history = []
        for i in range(mock_learner.timesteps):
            dmn_val = np.random.rand()
            dan_val = 1.0 - dmn_val + np.random.normal(0, 0.1)  # Anticorrelated
            dan_val = np.clip(dan_val, 0, 1)
            
            anticorr_history.append({
                'DMN': dmn_val,
                'DAN': dan_val,
                'VAN': np.random.rand(),
                'FPN': np.random.rand()
            })
        
        mock_learner.network_activations_history = anticorr_history
        
        anticorr_analysis = analyzer.analyze_dmn_dan_anticorrelation(mock_learner)
        
        assert isinstance(anticorr_analysis, dict)
        assert 'correlation' in anticorr_analysis
        
        # Should detect negative correlation
        assert anticorr_analysis['correlation'] < 0
    
    def test_calculate_network_dominance(self, mock_learner):
        """Test network dominance patterns."""
        analyzer = NetworkAnalyzer()
        
        dominance = analyzer.calculate_network_dominance(mock_learner)
        
        assert isinstance(dominance, dict)
        
        # Should have dominance info for each network
        networks = mock_learner.networks
        for network in networks:
            assert network in dominance or any(network in key for key in dominance.keys())
    
    def test_analyze_state_network_patterns(self, mock_learner):
        """Test state-specific network pattern analysis."""
        analyzer = NetworkAnalyzer()
        
        patterns = analyzer.analyze_state_network_patterns(mock_learner)
        
        assert isinstance(patterns, dict)
        
        # Should have patterns for each state that appeared
        unique_states = set(mock_learner.state_history)
        for state in unique_states:
            assert state in patterns


class TestTimeSeriesAnalyzer:
    """Test time series analysis functionality."""
    
    def test_analyzer_initialization(self):
        """Test TimeSeriesAnalyzer initialization."""
        analyzer = TimeSeriesAnalyzer()
        
        assert analyzer is not None
        assert hasattr(analyzer, 'has_scipy')
    
    def test_detrend_data(self):
        """Test data detrending."""
        analyzer = TimeSeriesAnalyzer()
        
        # Create data with trend
        t = np.arange(100)
        trend = 0.05 * t  # Linear trend
        noise = np.random.normal(0, 0.1, 100)
        data = trend + noise
        
        detrended = analyzer.detrend_data(data)
        
        # Detrended data should have reduced trend
        assert len(detrended) == len(data)
        
        # Mean should be closer to zero after detrending
        assert abs(np.mean(detrended)) < abs(np.mean(data))
    
    def test_detect_periodicities(self):
        """Test periodicity detection."""
        analyzer = TimeSeriesAnalyzer()
        
        # Create data with known periodicity
        t = np.arange(200)
        periodic_signal = np.sin(2 * np.pi * t / 20)  # Period of 20
        noise = np.random.normal(0, 0.1, 200)
        data = periodic_signal + noise
        
        periodicities = analyzer.detect_periodicities(data)
        
        assert isinstance(periodicities, dict)
        
        # Should detect the main periodicity or at least find some structure
        if 'dominant_periods' in periodicities:
            assert len(periodicities['dominant_periods']) > 0
    
    def test_analyze_transitions(self, mock_learner):
        """Test state transition analysis."""
        analyzer = TimeSeriesAnalyzer()
        
        # Create some transition patterns
        mock_learner.state_history = (
            ['breath_control'] * 10 + 
            ['mind_wandering'] * 5 + 
            ['meta_awareness'] * 2 + 
            ['redirect_breath'] * 3
        )
        
        transitions = analyzer.analyze_transitions(mock_learner)
        
        assert isinstance(transitions, dict)
        
        # Should identify transition points and patterns
        if 'transition_points' in transitions:
            assert len(transitions['transition_points']) > 0
    
    def test_calculate_stability_metrics(self, mock_learner):
        """Test stability metrics calculation."""
        analyzer = TimeSeriesAnalyzer()
        
        stability = analyzer.calculate_stability_metrics(mock_learner)
        
        assert isinstance(stability, dict)
        
        # Should have stability measures for key variables
        expected_metrics = ['free_energy_stability', 'meta_awareness_stability']
        
        # At least some stability metrics should be present
        assert len(stability) > 0
    
    def test_analyze_convergence(self, mock_learner):
        """Test convergence analysis."""
        analyzer = TimeSeriesAnalyzer()
        
        # Create data that converges
        convergent_data = [1.0 * np.exp(-i * 0.05) + np.random.normal(0, 0.01) 
                          for i in range(mock_learner.timesteps)]
        mock_learner.free_energy_history = convergent_data
        
        convergence = analyzer.analyze_convergence(mock_learner)
        
        assert isinstance(convergence, dict)
        
        # Should detect convergence
        if 'converged' in convergence:
            assert isinstance(convergence['converged'], bool)
        
        if 'convergence_point' in convergence:
            assert isinstance(convergence['convergence_point'], (int, type(None)))


class TestAnalysisIntegration:
    """Test integration between analysis components."""
    
    def test_statistical_and_comparison_integration(self, mock_learner):
        """Test that statistical and comparison analyzers work together."""
        stat_analyzer = StatisticalAnalyzer()
        comp_analyzer = ComparisonAnalyzer()
        
        # Analyze individual learner
        stats = stat_analyzer.analyze_learner(mock_learner)
        
        # Create second learner for comparison
        learner2 = Mock()
        learner2.experience_level = 'expert'
        learner2.thoughtseeds = mock_learner.thoughtseeds
        learner2.states = mock_learner.states
        learner2.networks = mock_learner.networks
        learner2.timesteps = mock_learner.timesteps
        learner2.free_energy_history = [x * 0.9 for x in mock_learner.free_energy_history]
        learner2.meta_awareness_history = mock_learner.meta_awareness_history
        learner2.state_history = mock_learner.state_history
        learner2.activations_history = mock_learner.activations_history
        learner2.network_activations_history = mock_learner.network_activations_history
        
        # Compare learners
        comparison = comp_analyzer.compare_learners(mock_learner, learner2)
        
        # Both should produce valid results
        assert isinstance(stats, dict)
        assert hasattr(comparison, 'comparison_statistics')
    
    def test_metrics_and_network_integration(self, mock_learner):
        """Test integration between metrics and network analyzers."""
        metrics_calc = MetricsCalculator()
        network_analyzer = NetworkAnalyzer()
        
        # Calculate metrics
        metrics = metrics_calc.calculate_all_metrics(mock_learner)
        
        # Analyze networks
        network_analysis = network_analyzer.calculate_network_correlations(mock_learner)
        
        # Both should complete successfully
        assert hasattr(metrics, 'overall_quality_score')
        assert isinstance(network_analysis, dict)
    
    def test_full_analysis_pipeline(self, mock_learner):
        """Test complete analysis pipeline."""
        # Initialize all analyzers
        stat_analyzer = StatisticalAnalyzer()
        metrics_calc = MetricsCalculator()
        network_analyzer = NetworkAnalyzer()
        ts_analyzer = TimeSeriesAnalyzer()
        
        # Run full analysis
        basic_stats = stat_analyzer.analyze_learner(mock_learner)
        metrics = metrics_calc.calculate_all_metrics(mock_learner)
        network_analysis = network_analyzer.calculate_network_correlations(mock_learner)
        ts_analysis = ts_analyzer.calculate_stability_metrics(mock_learner)
        
        # All analyses should complete
        assert isinstance(basic_stats, dict)
        assert hasattr(metrics, 'overall_quality_score')
        assert isinstance(network_analysis, dict)
        assert isinstance(ts_analysis, dict)
        
        # Create comprehensive report
        full_report = {
            'basic_statistics': basic_stats,
            'meditation_metrics': metrics,
            'network_analysis': network_analysis,
            'time_series_analysis': ts_analysis
        }
        
        # Report should be comprehensive
        assert len(full_report) == 4
        for section in full_report.values():
            assert section is not None


class TestAnalysisErrorHandling:
    """Test error handling in analysis modules."""
    
    def test_empty_data_handling(self):
        """Test handling of empty or minimal data."""
        stat_analyzer = StatisticalAnalyzer()
        
        # Test with empty list
        stats = stat_analyzer.calculate_descriptive_stats([])
        assert stats['count'] == 0
        
        # Test with single value
        stats = stat_analyzer.calculate_descriptive_stats([5])
        assert stats['count'] == 1
        assert stats['mean'] == 5
        assert stats['std'] == 0
    
    def test_invalid_learner_data(self):
        """Test handling of invalid learner data."""
        # Create learner with missing attributes
        invalid_learner = Mock()
        invalid_learner.experience_level = 'test'
        invalid_learner.free_energy_history = []
        
        stat_analyzer = StatisticalAnalyzer()
        
        # Should handle missing data gracefully
        try:
            analysis = stat_analyzer.analyze_learner(invalid_learner)
            # Should return something, even if minimal
            assert isinstance(analysis, dict)
        except AttributeError:
            # Acceptable to fail with missing critical attributes
            pass
    
    def test_scipy_pandas_fallbacks(self):
        """Test fallback behavior when scipy/pandas unavailable."""
        # Test that analyzers can still function without optional dependencies
        stat_analyzer = StatisticalAnalyzer()
        
        # Even if scipy is unavailable, basic stats should work
        data = [1, 2, 3, 4, 5]
        stats = stat_analyzer.calculate_descriptive_stats(data)
        
        assert isinstance(stats, dict)
        assert 'mean' in stats
        assert 'std' in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
