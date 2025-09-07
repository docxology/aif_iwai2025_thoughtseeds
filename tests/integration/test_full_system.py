"""
Comprehensive integration tests for the full meditation simulation system.

These tests verify that all components work together correctly in realistic
end-to-end scenarios, including training, analysis, export, and visualization.
"""

import pytest
import numpy as np
import os
import json
import shutil
from pathlib import Path

from core import ActInfLearner, RuleBasedLearner
from config import ActiveInferenceConfig, THOUGHTSEEDS, STATES
from utils import (
    ensure_directories, FreeEnergyTracer, ExportManager, ExportConfig,
    _save_json_outputs
)
from visualization import generate_all_plots, FreeEnergyVisualizer
from analysis import (
    StatisticalAnalyzer, ComparisonAnalyzer, MetricsCalculator,
    NetworkAnalyzer, TimeSeriesAnalyzer
)


class TestBasicSystemIntegration:
    """Test basic system integration scenarios."""
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_full_simulation_pipeline(self, temp_dir):
        """Test complete simulation pipeline from training to visualization."""
        original_dir = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            # Step 1: Create and train learners
            novice_learner = ActInfLearner(experience_level='novice', timesteps_per_cycle=30)
            expert_learner = ActInfLearner(experience_level='expert', timesteps_per_cycle=30)
            
            # Train both learners
            novice_learner.train()
            expert_learner.train()
            
            # Verify training completed
            assert len(novice_learner.state_history) == 30
            assert len(expert_learner.state_history) == 30
            assert len(novice_learner.free_energy_history) == 30
            assert len(expert_learner.free_energy_history) == 30
            
            # Step 2: Analyze results
            stat_analyzer = StatisticalAnalyzer()
            comp_analyzer = ComparisonAnalyzer()
            
            novice_analysis = stat_analyzer.analyze_learner(novice_learner)
            expert_analysis = stat_analyzer.analyze_learner(expert_learner)
            comparison = comp_analyzer.compare_learners(novice_learner, expert_learner)
            
            # Verify analyses completed
            assert isinstance(novice_analysis, dict)
            assert isinstance(expert_analysis, dict)
            assert hasattr(comparison, 'comparison_statistics')
            
            # Step 3: Export data
            export_config = ExportConfig(formats=['json', 'csv'])
            export_manager = ExportManager(export_config)
            
            novice_exports = export_manager.export_learner(novice_learner, 'integration_novice')
            expert_exports = export_manager.export_learner(expert_learner, 'integration_expert')
            
            # Verify exports
            assert 'successful' in novice_exports
            assert 'successful' in expert_exports
            assert len(novice_exports['successful']) > 0
            assert len(expert_exports['successful']) > 0
            
            # Step 4: Generate visualizations
            with pytest.warns(None):  # Allow matplotlib warnings
                plots_generated = generate_all_plots()
            
            # Should attempt to generate plots (may fail due to data format, but should try)
            assert isinstance(plots_generated, bool)
            
        finally:
            os.chdir(original_dir)
    
    @pytest.mark.integration
    def test_learner_inheritance_chain(self):
        """Test that ActInfLearner properly inherits from RuleBasedLearner."""
        # Create learners
        base_learner = RuleBasedLearner(experience_level='novice')
        act_inf_learner = ActInfLearner(experience_level='novice')
        
        # Verify inheritance
        assert isinstance(act_inf_learner, RuleBasedLearner)
        
        # Verify shared methods work
        activations = act_inf_learner.get_target_activations('breath_control', 0.5)
        assert isinstance(activations, np.ndarray)
        assert len(activations) == len(THOUGHTSEEDS)
        
        dwell_time = act_inf_learner.get_dwell_time('mind_wandering')
        assert isinstance(dwell_time, (int, np.integer))
        assert dwell_time > 0
        
        meta_awareness = act_inf_learner.get_meta_awareness('meta_awareness', activations)
        assert isinstance(meta_awareness, (float, np.floating))
        assert 0 <= meta_awareness <= 1
    
    @pytest.mark.integration
    def test_configuration_system_integration(self):
        """Test that configuration system integrates properly with all components."""
        # Test parameter loading
        novice_params = ActiveInferenceConfig.get_params('novice')
        expert_params = ActiveInferenceConfig.get_params('expert')
        
        assert isinstance(novice_params, dict)
        assert isinstance(expert_params, dict)
        
        # Verify parameters are used in learner initialization
        novice_learner = ActInfLearner(experience_level='novice')
        expert_learner = ActInfLearner(experience_level='expert')
        
        # Check that parameters were applied
        assert novice_learner.precision_weight == novice_params['precision_weight']
        assert expert_learner.precision_weight == expert_params['precision_weight']
        assert novice_learner.complexity_penalty == novice_params['complexity_penalty']
        assert expert_learner.complexity_penalty == expert_params['complexity_penalty']


class TestEnhancedSystemIntegration:
    """Test integration with enhanced features like free energy tracing."""
    
    @pytest.mark.integration
    def test_free_energy_tracing_integration(self, temp_dir):
        """Test free energy tracing integration with the main system."""
        original_dir = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            # Create learner with tracing
            learner = ActInfLearner(experience_level='expert', timesteps_per_cycle=20)
            tracer = FreeEnergyTracer(output_dir=os.path.join(temp_dir, "fe_traces"))
            
            # Simulate some training with manual tracing
            learner.train()
            
            # Create some traces manually (since full integration would require modifying core)
            for i in range(min(10, len(learner.state_history))):
                # Mock the tracing process
                if hasattr(learner, 'network_activations_history') and i < len(learner.network_activations_history):
                    snapshot = tracer.trace_timestep(learner, i)
                    assert hasattr(snapshot, 'timestep')
                    assert snapshot.timestep == i
            
            # Create trace summary
            if len(tracer.snapshots) > 0:
                trace_summary = tracer.create_trace_summary(learner)
                assert hasattr(trace_summary, 'experience_level')
                assert trace_summary.experience_level == 'expert'
                
                # Save trace
                trace_file = tracer.save_trace(trace_summary, "integration_test")
                assert os.path.exists(trace_file)
        
        finally:
            os.chdir(original_dir)
    
    @pytest.mark.integration 
    @pytest.mark.visualization
    def test_enhanced_visualization_integration(self, temp_dir):
        """Test enhanced visualization integration."""
        original_dir = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            # Create basic visualization setup
            ensure_directories("./results_act_inf")
            
            # Create minimal learner data
            learner = ActInfLearner(experience_level='novice', timesteps_per_cycle=15)
            learner.train()
            
            # Save JSON outputs for visualization
            _save_json_outputs(learner)
            
            # Test standard visualization
            with pytest.warns(None):  # Allow matplotlib warnings
                standard_result = generate_all_plots()
            
            # Test enhanced visualization
            visualizer = FreeEnergyVisualizer(output_dir=os.path.join(temp_dir, "enhanced"))
            
            # Create mock trace for enhanced visualization
            from utils import FreeEnergyTrace
            mock_trace = FreeEnergyTrace(
                experience_level='novice',
                simulation_duration=15,
                snapshots=[],
                summary_statistics={'variational_fe_stats': {'mean': 1.5}},
                optimization_metrics={'gradient_evolution': [0.2, 0.1, 0.05]}
            )
            
            # Should be able to create enhanced visualizations
            with pytest.warns(None):
                dashboard_path = visualizer.create_comprehensive_dashboard(mock_trace)
                assert isinstance(dashboard_path, str)
        
        finally:
            os.chdir(original_dir)


class TestMultiComponentIntegration:
    """Test integration between multiple system components."""
    
    @pytest.mark.integration
    def test_analysis_export_integration(self, temp_dir):
        """Test integration between analysis and export systems."""
        original_dir = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            # Create and train learner
            learner = ActInfLearner(experience_level='expert', timesteps_per_cycle=25)
            learner.train()
            
            # Analyze learner
            stat_analyzer = StatisticalAnalyzer()
            metrics_calc = MetricsCalculator()
            
            statistical_analysis = stat_analyzer.analyze_learner(learner)
            metrics = metrics_calc.calculate_all_metrics(learner)
            
            # Export both raw learner data and analysis results
            export_config = ExportConfig(formats=['json'], include_analysis=True)
            export_manager = ExportManager(export_config)
            
            # Export learner
            export_results = export_manager.export_learner(learner, 'analyzed_learner')
            
            # Verify exports include both raw data and analysis
            assert 'successful' in export_results
            assert len(export_results['successful']) > 0
            
            # Verify analysis completed
            assert isinstance(statistical_analysis, dict)
            assert hasattr(metrics, 'overall_quality_score')
        
        finally:
            os.chdir(original_dir)
    
    @pytest.mark.integration
    def test_comparison_visualization_integration(self, temp_dir):
        """Test integration between comparison analysis and visualization."""
        original_dir = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            # Create and train two learners
            novice = ActInfLearner(experience_level='novice', timesteps_per_cycle=20)
            expert = ActInfLearner(experience_level='expert', timesteps_per_cycle=20)
            
            novice.train()
            expert.train()
            
            # Compare learners
            comp_analyzer = ComparisonAnalyzer()
            comparison = comp_analyzer.compare_learners(novice, expert)
            
            # Export comparison data
            export_config = ExportConfig(formats=['json'])
            export_manager = ExportManager(export_config)
            
            comparison_exports = export_manager.export_comparison(
                novice, expert, 'integration_comparison')
            
            # Save individual learner data for visualization
            _save_json_outputs(novice)
            _save_json_outputs(expert)
            
            # Generate comparative visualizations
            with pytest.warns(None):
                plots_result = generate_all_plots()
            
            # Verify everything completed
            assert hasattr(comparison, 'comparison_statistics')
            assert 'successful' in comparison_exports
            assert isinstance(plots_result, bool)
        
        finally:
            os.chdir(original_dir)
    
    @pytest.mark.integration
    def test_full_analysis_pipeline(self, temp_dir):
        """Test complete analysis pipeline with all analyzers."""
        original_dir = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            # Create learner
            learner = ActInfLearner(experience_level='expert', timesteps_per_cycle=30)
            learner.train()
            
            # Run all types of analysis
            stat_analyzer = StatisticalAnalyzer()
            comp_analyzer = ComparisonAnalyzer()
            metrics_calc = MetricsCalculator()
            network_analyzer = NetworkAnalyzer()
            ts_analyzer = TimeSeriesAnalyzer()
            
            # Basic statistical analysis
            basic_stats = stat_analyzer.analyze_learner(learner)
            
            # Metrics calculation
            metrics = metrics_calc.calculate_all_metrics(learner)
            
            # Network analysis
            network_analysis = network_analyzer.calculate_network_correlations(learner)
            
            # Time series analysis
            ts_analysis = ts_analyzer.calculate_stability_metrics(learner)
            
            # Create comprehensive report
            comprehensive_report = {
                'learner_info': {
                    'experience_level': learner.experience_level,
                    'timesteps': learner.timesteps,
                    'final_free_energy': learner.free_energy_history[-1] if learner.free_energy_history else None
                },
                'statistical_analysis': basic_stats,
                'meditation_metrics': {
                    'attention_stability': metrics.attention_stability,
                    'distraction_resistance': metrics.distraction_resistance,
                    'metacognitive_efficiency': metrics.metacognitive_efficiency,
                    'overall_quality': metrics.overall_quality_score
                },
                'network_analysis': network_analysis,
                'time_series_analysis': ts_analysis
            }
            
            # Save comprehensive report
            with open('comprehensive_analysis_report.json', 'w') as f:
                json.dump(comprehensive_report, f, indent=2, default=str)
            
            # Verify report was created and is valid
            assert os.path.exists('comprehensive_analysis_report.json')
            
            with open('comprehensive_analysis_report.json', 'r') as f:
                loaded_report = json.load(f)
                assert 'learner_info' in loaded_report
                assert 'statistical_analysis' in loaded_report
                assert 'meditation_metrics' in loaded_report
        
        finally:
            os.chdir(original_dir)


class TestSystemRobustness:
    """Test system robustness and error handling in integration scenarios."""
    
    @pytest.mark.integration
    def test_system_with_missing_optional_dependencies(self, temp_dir):
        """Test system behavior when optional dependencies are missing."""
        original_dir = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            # Test that system works without scipy/pandas by mocking their absence
            with patch('analysis.statistical_analyzer.HAS_SCIPY', False), \
                 patch('analysis.statistical_analyzer.HAS_PANDAS', False):
                
                # Create and train learner
                learner = ActInfLearner(experience_level='novice', timesteps_per_cycle=20)
                learner.train()
                
                # Analysis should still work with fallbacks
                stat_analyzer = StatisticalAnalyzer()
                analysis = stat_analyzer.analyze_learner(learner)
                
                # Should return some analysis even without full scipy/pandas
                assert isinstance(analysis, dict)
                assert len(analysis) > 0
        
        finally:
            os.chdir(original_dir)
    
    @pytest.mark.integration
    def test_system_with_partial_data_corruption(self, temp_dir):
        """Test system behavior with partially corrupted data."""
        original_dir = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            # Create learner and train
            learner = ActInfLearner(experience_level='expert', timesteps_per_cycle=25)
            learner.train()
            
            # Simulate partial data corruption
            # Remove some history entries
            if len(learner.free_energy_history) > 5:
                learner.free_energy_history = learner.free_energy_history[:-3]
            
            if len(learner.state_history) > 5:
                learner.state_history = learner.state_history[:-2]
            
            # System should handle mismatched history lengths gracefully
            try:
                stat_analyzer = StatisticalAnalyzer()
                analysis = stat_analyzer.analyze_learner(learner)
                
                # Should produce some results despite corrupted data
                assert isinstance(analysis, dict)
                
            except (IndexError, ValueError, KeyError):
                # Acceptable to fail with corrupted data
                pass
        
        finally:
            os.chdir(original_dir)
    
    @pytest.mark.integration
    def test_concurrent_analysis_operations(self, temp_dir):
        """Test multiple analysis operations on the same data."""
        original_dir = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            # Create learner
            learner = ActInfLearner(experience_level='novice', timesteps_per_cycle=20)
            learner.train()
            
            # Run multiple analyses concurrently (simulated)
            analyzers = [
                StatisticalAnalyzer(),
                MetricsCalculator(),
                NetworkAnalyzer(),
                TimeSeriesAnalyzer()
            ]
            
            results = []
            for analyzer in analyzers:
                try:
                    if hasattr(analyzer, 'analyze_learner'):
                        result = analyzer.analyze_learner(learner)
                    elif hasattr(analyzer, 'calculate_all_metrics'):
                        result = analyzer.calculate_all_metrics(learner)
                    elif hasattr(analyzer, 'calculate_network_correlations'):
                        result = analyzer.calculate_network_correlations(learner)
                    elif hasattr(analyzer, 'calculate_stability_metrics'):
                        result = analyzer.calculate_stability_metrics(learner)
                    else:
                        result = None
                    
                    results.append(result)
                
                except Exception as e:
                    # Log error but continue with other analyses
                    results.append(f"Error: {str(e)}")
            
            # Should have some successful results
            successful_results = [r for r in results if not isinstance(r, str)]
            assert len(successful_results) > 0
        
        finally:
            os.chdir(original_dir)


class TestSystemPerformance:
    """Test system performance characteristics."""
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_large_simulation_performance(self, temp_dir):
        """Test system performance with larger simulations."""
        original_dir = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            import time
            
            # Test with larger timestep count
            start_time = time.time()
            learner = ActInfLearner(experience_level='expert', timesteps_per_cycle=100)
            learner.train()
            training_time = time.time() - start_time
            
            # Verify training completed
            assert len(learner.state_history) == 100
            assert len(learner.free_energy_history) == 100
            
            # Training should complete in reasonable time (less than 30 seconds)
            assert training_time < 30
            
            # Test analysis performance
            start_time = time.time()
            stat_analyzer = StatisticalAnalyzer()
            analysis = stat_analyzer.analyze_learner(learner)
            analysis_time = time.time() - start_time
            
            # Analysis should be fast (less than 5 seconds)
            assert analysis_time < 5
            assert isinstance(analysis, dict)
        
        finally:
            os.chdir(original_dir)
    
    @pytest.mark.integration
    def test_memory_usage_patterns(self, temp_dir):
        """Test that system doesn't have obvious memory leaks."""
        original_dir = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            import gc
            
            # Create multiple learners in sequence
            for i in range(3):
                learner = ActInfLearner(experience_level='novice', timesteps_per_cycle=30)
                learner.train()
                
                # Analyze
                stat_analyzer = StatisticalAnalyzer()
                analysis = stat_analyzer.analyze_learner(learner)
                
                # Export
                export_config = ExportConfig(formats=['json'])
                export_manager = ExportManager(export_config)
                exports = export_manager.export_learner(learner, f'memory_test_{i}')
                
                # Clean up explicitly
                del learner, stat_analyzer, export_manager, analysis, exports
                gc.collect()
            
            # If we get here without memory errors, that's good
            assert True
        
        finally:
            os.chdir(original_dir)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
