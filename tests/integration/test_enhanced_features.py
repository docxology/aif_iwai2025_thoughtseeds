"""
Integration tests for enhanced features.

This module tests the integration of enhanced features like free energy tracing,
advanced visualizations, and comprehensive export/analysis systems.
"""

import pytest
import numpy as np
import os
import json
from unittest.mock import Mock, patch

from core import ActInfLearner
from utils import FreeEnergyTracer, ExportManager, ExportConfig
from visualization import FreeEnergyVisualizer
from analysis import StatisticalAnalyzer, ComparisonAnalyzer, MetricsCalculator


class TestEnhancedSimulationPipeline:
    """Test enhanced simulation features end-to-end."""
    
    @pytest.mark.integration
    def test_enhanced_learner_with_tracing(self, temp_dir):
        """Test ActInfLearner with free energy tracing integration."""
        original_dir = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            # Create learner
            learner = ActInfLearner(experience_level='expert', timesteps_per_cycle=25)
            
            # Create tracer
            tracer = FreeEnergyTracer(output_dir="./fe_traces")
            
            # Train with manual tracing integration
            learner.train()
            
            # Simulate tracing integration (would be built into enhanced learner)
            for i in range(min(10, len(learner.state_history))):
                snapshot = tracer.trace_timestep(learner, i)
                
                # Verify snapshot creation
                assert hasattr(snapshot, 'timestep')
                assert hasattr(snapshot, 'variational_free_energy')
                assert hasattr(snapshot, 'expected_free_energy')
                assert snapshot.timestep == i
                assert snapshot.state in learner.states
            
            # Create comprehensive trace
            if len(tracer.snapshots) > 0:
                trace_summary = tracer.create_trace_summary(learner)
                
                # Verify trace summary
                assert trace_summary.experience_level == 'expert'
                assert len(trace_summary.snapshots) > 0
                assert isinstance(trace_summary.summary_statistics, dict)
                assert isinstance(trace_summary.optimization_metrics, dict)
                
                # Save trace
                trace_file = tracer.save_trace(trace_summary, "enhanced_test")
                assert os.path.exists(trace_file)
                
                # Verify trace file content
                with open(trace_file, 'r') as f:
                    trace_data = json.load(f)
                    assert 'experience_level' in trace_data
                    assert 'snapshots' in trace_data
                    assert 'summary_statistics' in trace_data
        
        finally:
            os.chdir(original_dir)
    
    @pytest.mark.integration
    @pytest.mark.visualization
    def test_enhanced_visualization_pipeline(self, temp_dir):
        """Test enhanced visualization with free energy traces."""
        original_dir = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            # Create mock trace data
            from utils import FreeEnergyTrace, FreeEnergySnapshot
            
            # Create realistic snapshots
            snapshots = []
            for i in range(20):
                snapshot = FreeEnergySnapshot(
                    timestep=i,
                    state=['breath_control', 'mind_wandering', 'meta_awareness'][i % 3],
                    variational_free_energy=2.0 - i * 0.05 + np.random.normal(0, 0.1),
                    expected_free_energy=1.8 - i * 0.04 + np.random.normal(0, 0.08),
                    prediction_error=0.3 - i * 0.01 + np.random.normal(0, 0.02),
                    precision_weight=0.5 + np.random.normal(0, 0.05),
                    complexity_penalty=0.3,
                    network_predictions={'DMN': 0.4, 'VAN': 0.5, 'DAN': 0.6, 'FPN': 0.5},
                    network_observations={'DMN': 0.45, 'VAN': 0.55, 'DAN': 0.65, 'FPN': 0.55},
                    network_prediction_errors={'DMN': 0.05, 'VAN': 0.05, 'DAN': 0.05, 'FPN': 0.05},
                    thoughtseed_activations={
                        'breath_focus': 0.7, 'pain_discomfort': 0.2, 'pending_tasks': 0.1,
                        'self_reflection': 0.4, 'equanimity': 0.5
                    },
                    thoughtseed_predictions={
                        'breath_focus': 0.65, 'pain_discomfort': 0.25, 'pending_tasks': 0.15,
                        'self_reflection': 0.35, 'equanimity': 0.45
                    },
                    meta_awareness=0.6 + np.random.normal(0, 0.1),
                    attention_precision=0.8 + np.random.normal(0, 0.05),
                    cognitive_load=0.3 + np.random.normal(0, 0.02),
                    transition_probability=0.1,
                    state_entropy=1.2,
                    gradient_magnitude=0.2 - i * 0.005,
                    learning_rate_effective=0.01,
                    timestamp=f'2024-01-01T{i:02d}:00:00'
                )
                snapshots.append(snapshot)
            
            # Create trace summary
            trace = FreeEnergyTrace(
                experience_level='expert',
                simulation_duration=20,
                snapshots=snapshots,
                summary_statistics={
                    'variational_fe_stats': {
                        'mean': 1.5, 'std': 0.2, 'min': 1.0, 'max': 2.0,
                        'final': 1.2, 'initial': 2.0
                    },
                    'expected_fe_stats': {
                        'mean': 1.3, 'std': 0.15, 'convergence_timestep': 15
                    },
                    'optimization_efficiency': {
                        'total_reduction': 0.8, 'percent_reduction': 40.0,
                        'convergence_rate': 0.02
                    }
                },
                optimization_metrics={
                    'gradient_evolution': [0.2 - i * 0.005 for i in range(20)],
                    'learning_adaptation': [0.01] * 20,
                    'precision_evolution': [0.7 + i * 0.005 for i in range(20)],
                    'complexity_cost': [0.3] * 20
                }
            )
            
            # Test enhanced visualizer
            visualizer = FreeEnergyVisualizer(output_dir="./enhanced_viz")
            
            with patch('matplotlib.pyplot.savefig') as mock_save, \
                 patch('matplotlib.pyplot.close'):
                
                # Test comprehensive dashboard
                dashboard_path = visualizer.create_comprehensive_dashboard(trace)
                assert isinstance(dashboard_path, str)
                mock_save.assert_called()
                
                # Test detailed component analysis
                mock_save.reset_mock()
                component_path = visualizer.create_detailed_component_analysis(trace)
                assert isinstance(component_path, str)
                mock_save.assert_called()
                
                # Test optimization landscape
                mock_save.reset_mock()
                landscape_path = visualizer.create_optimization_landscape(trace)
                assert isinstance(landscape_path, str)
                mock_save.assert_called()
        
        finally:
            os.chdir(original_dir)
    
    @pytest.mark.integration
    @pytest.mark.export
    def test_comprehensive_export_pipeline(self, temp_dir):
        """Test comprehensive data export with all formats."""
        original_dir = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            # Create and train learner
            learner = ActInfLearner(experience_level='novice', timesteps_per_cycle=20)
            learner.train()
            
            # Test comprehensive export
            export_config = ExportConfig.comprehensive()
            export_manager = ExportManager(export_config)
            
            # Export learner data
            export_results = export_manager.export_learner(learner, 'comprehensive_test')
            
            # Should have successful exports
            assert 'successful' in export_results
            assert 'failed' in export_results
            assert len(export_results['successful']) > 0
            
            # Check that different formats were attempted
            successful_formats = set()
            for format_name, files in export_results['successful'].items():
                successful_formats.add(format_name)
                assert len(files) > 0  # Should have exported some files
            
            # Should have attempted multiple formats
            assert len(successful_formats) >= 2
            
            # JSON should definitely work
            assert 'json' in successful_formats
            
            # Verify JSON exports contain expected data
            json_files = export_results['successful']['json']
            for json_file in json_files:
                if os.path.exists(json_file):
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                        assert isinstance(data, dict)
                        assert len(data) > 0
        
        finally:
            os.chdir(original_dir)


class TestAdvancedAnalysisIntegration:
    """Test advanced analysis feature integration."""
    
    @pytest.mark.integration
    def test_multi_analyzer_integration(self, temp_dir):
        """Test integration of multiple analysis components."""
        original_dir = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            # Create and train learners
            novice = ActInfLearner(experience_level='novice', timesteps_per_cycle=25)
            expert = ActInfLearner(experience_level='expert', timesteps_per_cycle=25)
            
            novice.train()
            expert.train()
            
            # Initialize all analyzers
            stat_analyzer = StatisticalAnalyzer()
            comp_analyzer = ComparisonAnalyzer()
            metrics_calc = MetricsCalculator()
            
            # Run comprehensive analysis
            novice_stats = stat_analyzer.analyze_learner(novice)
            expert_stats = stat_analyzer.analyze_learner(expert)
            
            novice_metrics = metrics_calc.calculate_all_metrics(novice)
            expert_metrics = metrics_calc.calculate_all_metrics(expert)
            
            comparison = comp_analyzer.compare_learners(novice, expert)
            
            # Create integrated analysis report
            integrated_report = {
                'novice_analysis': {
                    'basic_statistics': novice_stats,
                    'meditation_metrics': {
                        'attention_stability': novice_metrics.attention_stability,
                        'distraction_resistance': novice_metrics.distraction_resistance,
                        'metacognitive_efficiency': novice_metrics.metacognitive_efficiency,
                        'overall_quality': novice_metrics.overall_quality_score
                    }
                },
                'expert_analysis': {
                    'basic_statistics': expert_stats,
                    'meditation_metrics': {
                        'attention_stability': expert_metrics.attention_stability,
                        'distraction_resistance': expert_metrics.distraction_resistance,
                        'metacognitive_efficiency': expert_metrics.metacognitive_efficiency,
                        'overall_quality': expert_metrics.overall_quality_score
                    }
                },
                'comparative_analysis': {
                    'statistical_comparisons': comparison.comparison_statistics,
                    'effect_sizes': comparison.effect_sizes,
                    'expert_advantages': {
                        'quality_improvement': expert_metrics.overall_quality_score - novice_metrics.overall_quality_score,
                        'stability_improvement': expert_metrics.attention_stability - novice_metrics.attention_stability,
                        'resistance_improvement': expert_metrics.distraction_resistance - novice_metrics.distraction_resistance
                    }
                }
            }
            
            # Save integrated report
            with open('integrated_analysis_report.json', 'w') as f:
                json.dump(integrated_report, f, indent=2, default=str)
            
            # Verify report structure
            assert os.path.exists('integrated_analysis_report.json')
            
            with open('integrated_analysis_report.json', 'r') as f:
                loaded_report = json.load(f)
                
                assert 'novice_analysis' in loaded_report
                assert 'expert_analysis' in loaded_report
                assert 'comparative_analysis' in loaded_report
                
                # Verify expert advantages
                expert_advantages = loaded_report['comparative_analysis']['expert_advantages']
                assert 'quality_improvement' in expert_advantages
                assert 'stability_improvement' in expert_advantages
                assert 'resistance_improvement' in expert_advantages
        
        finally:
            os.chdir(original_dir)
    
    @pytest.mark.integration
    def test_analysis_export_integration(self, temp_dir):
        """Test integration between analysis results and export system."""
        original_dir = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            # Create learner
            learner = ActInfLearner(experience_level='expert', timesteps_per_cycle=20)
            learner.train()
            
            # Analyze learner
            stat_analyzer = StatisticalAnalyzer()
            metrics_calc = MetricsCalculator()
            
            analysis_results = stat_analyzer.analyze_learner(learner)
            metrics_results = metrics_calc.calculate_all_metrics(learner)
            
            # Create export config that includes analysis
            export_config = ExportConfig(
                formats=['json'],
                include_analysis=True,
                include_metadata=True,
                statistical_summaries=True
            )
            
            export_manager = ExportManager(export_config)
            
            # Export with analysis integration
            export_results = export_manager.export_learner(learner, 'analyzed_export')
            
            # Should have successful exports
            assert 'successful' in export_results
            assert len(export_results['successful']['json']) > 0
            
            # Check that exported files contain analysis data
            json_files = export_results['successful']['json']
            analysis_found = False
            
            for json_file in json_files:
                if os.path.exists(json_file):
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                        
                        # Look for analysis indicators
                        if any(key in str(data).lower() for key in ['analysis', 'statistics', 'metrics']):
                            analysis_found = True
                            break
            
            # Should find analysis data in exports
            assert analysis_found, "Analysis data should be included in exports"
        
        finally:
            os.chdir(original_dir)


class TestSystemResilience:
    """Test system resilience with enhanced features."""
    
    @pytest.mark.integration
    def test_enhanced_features_with_partial_failures(self, temp_dir):
        """Test system behavior when some enhanced features fail."""
        original_dir = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            # Create learner
            learner = ActInfLearner(experience_level='novice', timesteps_per_cycle=15)
            learner.train()
            
            # Test with simulated tracing failure
            tracer = FreeEnergyTracer(output_dir="/invalid/path")
            
            # Should handle invalid path gracefully
            try:
                snapshot = tracer.trace_timestep(learner, 0)
                # If it succeeds, that's fine
                assert hasattr(snapshot, 'timestep')
            except (OSError, PermissionError):
                # Expected to fail with invalid path
                pass
            
            # Test with simulated visualization failure
            visualizer = FreeEnergyVisualizer(output_dir=temp_dir)
            
            # Create minimal trace
            from utils import FreeEnergyTrace
            minimal_trace = FreeEnergyTrace(
                experience_level='test',
                simulation_duration=1,
                snapshots=[],
                summary_statistics={},
                optimization_metrics={}
            )
            
            # Should handle empty trace gracefully
            with patch('matplotlib.pyplot.savefig', side_effect=Exception("Mock failure")), \
                 patch('matplotlib.pyplot.close'):
                
                try:
                    dashboard_path = visualizer.create_comprehensive_dashboard(minimal_trace)
                    # If it handles the exception, that's good
                    assert isinstance(dashboard_path, str)
                except Exception:
                    # Acceptable to fail with mocked exception
                    pass
            
            # Test with simulated export failure
            export_config = ExportConfig(formats=['json', 'csv', 'hdf5'])  # HDF5 might not be available
            export_manager = ExportManager(export_config)
            
            export_results = export_manager.export_learner(learner, 'resilience_test')
            
            # Should have both successful and failed exports
            assert 'successful' in export_results
            assert 'failed' in export_results
            
            # Should have at least JSON success
            assert 'json' in export_results['successful']
            
        finally:
            os.chdir(original_dir)
    
    @pytest.mark.integration
    def test_enhanced_system_memory_management(self, temp_dir):
        """Test memory management in enhanced system components."""
        original_dir = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            import gc
            
            # Run multiple enhanced simulations
            for i in range(3):
                # Create components
                learner = ActInfLearner(experience_level=['novice', 'expert'][i % 2], 
                                     timesteps_per_cycle=20)
                tracer = FreeEnergyTracer(output_dir=f"./trace_{i}")
                visualizer = FreeEnergyVisualizer(output_dir=f"./viz_{i}")
                
                # Train
                learner.train()
                
                # Trace some timesteps
                for t in range(5):
                    tracer.trace_timestep(learner, t)
                
                # Create trace summary
                if len(tracer.snapshots) > 0:
                    trace_summary = tracer.create_trace_summary(learner)
                    
                    # Save trace
                    tracer.save_trace(trace_summary, f"memory_test_{i}")
                    
                    # Create visualization
                    with patch('matplotlib.pyplot.savefig'), \
                         patch('matplotlib.pyplot.close'):
                        visualizer.create_comprehensive_dashboard(trace_summary)
                
                # Clean up explicitly
                del learner, tracer, visualizer, trace_summary
                gc.collect()
            
            # If we get here without memory errors, that's good
            assert True
        
        finally:
            os.chdir(original_dir)


class TestEnhancedFeaturePerformance:
    """Test performance of enhanced features."""
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_free_energy_tracing_performance(self, temp_dir):
        """Test performance of free energy tracing."""
        original_dir = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            import time
            
            # Create learner and tracer
            learner = ActInfLearner(experience_level='expert', timesteps_per_cycle=50)
            tracer = FreeEnergyTracer(output_dir="./performance_test")
            
            # Train learner
            start_time = time.time()
            learner.train()
            training_time = time.time() - start_time
            
            # Trace all timesteps
            start_time = time.time()
            for i in range(min(30, len(learner.state_history))):
                tracer.trace_timestep(learner, i)
            tracing_time = time.time() - start_time
            
            # Create trace summary
            start_time = time.time()
            trace_summary = tracer.create_trace_summary(learner)
            summary_time = time.time() - start_time
            
            # Performance assertions
            assert training_time < 15, f"Training took too long: {training_time}s"
            assert tracing_time < 5, f"Tracing took too long: {tracing_time}s"
            assert summary_time < 2, f"Summary creation took too long: {summary_time}s"
            
            # Verify quality
            assert len(tracer.snapshots) > 0
            assert trace_summary.experience_level == 'expert'
            assert len(trace_summary.snapshots) == len(tracer.snapshots)
        
        finally:
            os.chdir(original_dir)
    
    @pytest.mark.integration
    @pytest.mark.visualization
    def test_enhanced_visualization_performance(self, temp_dir):
        """Test performance of enhanced visualization system."""
        original_dir = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            import time
            
            # Create realistic trace data
            from utils import FreeEnergyTrace, FreeEnergySnapshot
            
            snapshots = []
            for i in range(50):  # Larger dataset
                snapshot = FreeEnergySnapshot(
                    timestep=i, state='breath_control', variational_free_energy=1.5,
                    expected_free_energy=1.3, prediction_error=0.2, precision_weight=0.5,
                    complexity_penalty=0.3, network_predictions={}, network_observations={},
                    network_prediction_errors={}, thoughtseed_activations={}, thoughtseed_predictions={},
                    meta_awareness=0.6, attention_precision=0.8, cognitive_load=0.3,
                    transition_probability=0.1, state_entropy=1.2, gradient_magnitude=0.15,
                    learning_rate_effective=0.01, timestamp=f'2024-01-01T{i:02d}:00:00'
                )
                snapshots.append(snapshot)
            
            trace = FreeEnergyTrace(
                experience_level='test', simulation_duration=50, snapshots=snapshots,
                summary_statistics={'test': 'data'}, optimization_metrics={'test': 'data'}
            )
            
            # Test visualization performance
            visualizer = FreeEnergyVisualizer(output_dir="./viz_performance")
            
            with patch('matplotlib.pyplot.savefig'), \
                 patch('matplotlib.pyplot.close'):
                
                # Test dashboard creation performance
                start_time = time.time()
                dashboard_path = visualizer.create_comprehensive_dashboard(trace)
                dashboard_time = time.time() - start_time
                
                # Test detailed analysis performance
                start_time = time.time()
                analysis_path = visualizer.create_detailed_component_analysis(trace)
                analysis_time = time.time() - start_time
                
                # Performance assertions
                assert dashboard_time < 10, f"Dashboard creation took too long: {dashboard_time}s"
                assert analysis_time < 8, f"Analysis visualization took too long: {analysis_time}s"
                
                # Verify outputs
                assert isinstance(dashboard_path, str)
                assert isinstance(analysis_path, str)
        
        finally:
            os.chdir(original_dir)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
