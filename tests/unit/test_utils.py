"""
Comprehensive unit tests for the utils module.

This module tests all utility functions including data management,
file operations, free energy tracing, and export functionality.
"""

import pytest
import numpy as np
import json
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from utils import (
    ensure_directories, convert_numpy_to_lists, _save_json_outputs,
    create_output_structure, validate_data_integrity, get_latest_results,
    FreeEnergyTracer, FreeEnergyTrace, FreeEnergySnapshot,
    ExportManager, ExportConfig, ExportPipeline
)


class TestDataManagement:
    """Test data management utilities."""
    
    def test_ensure_directories(self, temp_dir):
        """Test directory creation utility."""
        base_dir = os.path.join(temp_dir, "test_output")
        
        # Test default behavior
        ensure_directories(base_dir)
        
        # Check that directories were created
        assert os.path.exists(os.path.join(base_dir, "data"))
        assert os.path.exists(os.path.join(base_dir, "plots"))
        
        # Test that it doesn't fail on existing directories
        ensure_directories(base_dir)  # Should not raise error
        
        assert os.path.exists(os.path.join(base_dir, "data"))
        assert os.path.exists(os.path.join(base_dir, "plots"))
    
    def test_convert_numpy_to_lists_basic(self):
        """Test numpy to list conversion with basic types."""
        # Test numpy array
        np_array = np.array([1, 2, 3])
        result = convert_numpy_to_lists(np_array)
        assert result == [1, 2, 3]
        assert isinstance(result, list)
        
        # Test regular list (should pass through)
        regular_list = [1, 2, 3]
        result = convert_numpy_to_lists(regular_list)
        assert result == [1, 2, 3]
        assert isinstance(result, list)
        
        # Test regular value (should pass through)
        value = 42
        result = convert_numpy_to_lists(value)
        assert result == 42
    
    def test_convert_numpy_to_lists_nested(self):
        """Test numpy to list conversion with nested structures."""
        # Test nested dictionary
        nested_dict = {
            'array_data': np.array([[1, 2], [3, 4]]),
            'list_data': [np.array([5, 6]), 7],
            'regular_data': {'nested': np.array([8, 9])}
        }
        
        result = convert_numpy_to_lists(nested_dict)
        
        # Check structure is preserved
        assert isinstance(result, dict)
        assert set(result.keys()) == {'array_data', 'list_data', 'regular_data'}
        
        # Check conversions
        assert result['array_data'] == [[1, 2], [3, 4]]
        assert result['list_data'] == [[5, 6], 7]
        assert result['regular_data']['nested'] == [8, 9]
    
    def test_convert_numpy_to_lists_dtypes(self):
        """Test numpy conversion with different data types."""
        # Float array
        float_array = np.array([1.1, 2.2, 3.3])
        result = convert_numpy_to_lists(float_array)
        assert result == [1.1, 2.2, 3.3]
        
        # Integer array
        int_array = np.array([1, 2, 3], dtype=np.int32)
        result = convert_numpy_to_lists(int_array)
        assert result == [1, 2, 3]
        
        # Boolean array
        bool_array = np.array([True, False, True])
        result = convert_numpy_to_lists(bool_array)
        assert result == [True, False, True]
    
    @pytest.mark.export
    def test_save_json_outputs(self, temp_dir, mock_learner):
        """Test JSON output saving."""
        # Change to temp directory
        original_dir = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            # Create expected directory structure
            os.makedirs("results_act_inf/data", exist_ok=True)
            
            # Test with mock learner
            _save_json_outputs(mock_learner)
            
            # Check that files were created
            expected_files = [
                f"results_act_inf/data/thoughtseed_params_{mock_learner.experience_level}.json",
                f"results_act_inf/data/active_inference_params_{mock_learner.experience_level}.json"
            ]
            
            for file_path in expected_files:
                assert os.path.exists(file_path)
                
                # Check that files are valid JSON
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    assert isinstance(data, dict)
                    assert len(data) > 0
        
        finally:
            os.chdir(original_dir)
    
    def test_save_json_outputs_content(self, temp_dir, mock_learner):
        """Test JSON output content structure."""
        original_dir = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            os.makedirs("results_act_inf/data", exist_ok=True)
            _save_json_outputs(mock_learner)
            
            # Check thoughtseed params content
            ts_file = f"results_act_inf/data/thoughtseed_params_{mock_learner.experience_level}.json"
            with open(ts_file, 'r') as f:
                ts_data = json.load(f)
            
            # Required sections
            assert 'agent_parameters' in ts_data
            assert 'time_series' in ts_data
            
            # Check time series data
            time_series = ts_data['time_series']
            required_ts_fields = [
                'activations_history', 'network_activations_history',
                'meta_awareness_history', 'free_energy_history',
                'state_history', 'dominant_ts_history'
            ]
            
            for field in required_ts_fields:
                assert field in time_series
                assert isinstance(time_series[field], list)
            
            # Check active inference params content
            ai_file = f"results_act_inf/data/active_inference_params_{mock_learner.experience_level}.json"
            with open(ai_file, 'r') as f:
                ai_data = json.load(f)
            
            # Required AI parameters
            required_ai_fields = [
                'precision_weight', 'complexity_penalty', 'learning_rate',
                'average_free_energy_by_state'
            ]
            
            for field in required_ai_fields:
                assert field in ai_data
        
        finally:
            os.chdir(original_dir)


class TestFileOperations:
    """Test file operation utilities."""
    
    def test_create_output_structure(self, temp_dir):
        """Test output structure creation."""
        original_dir = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            structure = create_output_structure()
            
            # Check return value
            assert isinstance(structure, dict)
            assert 'results_dir' in structure
            assert 'data_dir' in structure
            assert 'plots_dir' in structure
            
            # Check directories exist
            assert os.path.exists(structure['results_dir'])
            assert os.path.exists(structure['data_dir'])
            assert os.path.exists(structure['plots_dir'])
        
        finally:
            os.chdir(original_dir)
    
    def test_validate_data_integrity(self, mock_learner):
        """Test data integrity validation."""
        # Should pass with valid mock learner
        is_valid, issues = validate_data_integrity(mock_learner)
        assert isinstance(is_valid, bool)
        assert isinstance(issues, list)
        
        # Test with invalid learner (missing attribute)
        invalid_learner = Mock()
        invalid_learner.activations_history = []  # Missing other required attributes
        
        is_valid, issues = validate_data_integrity(invalid_learner)
        assert isinstance(is_valid, bool)
        assert isinstance(issues, list)
    
    def test_get_latest_results(self, temp_dir):
        """Test latest results retrieval."""
        # Create some test result files
        results_dir = os.path.join(temp_dir, "results_act_inf", "data")
        os.makedirs(results_dir, exist_ok=True)
        
        # Create test files
        test_files = [
            "thoughtseed_params_novice.json",
            "thoughtseed_params_expert.json",
            "active_inference_params_novice.json",
            "active_inference_params_expert.json"
        ]
        
        for filename in test_files:
            filepath = os.path.join(results_dir, filename)
            with open(filepath, 'w') as f:
                json.dump({"test": "data"}, f)
        
        # Test retrieval
        original_dir = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            latest_results = get_latest_results()
            
            # Check return structure
            assert isinstance(latest_results, dict)
            
            # Should contain both experience levels
            assert 'novice' in latest_results
            assert 'expert' in latest_results
            
            # Check file paths
            for level in ['novice', 'expert']:
                level_data = latest_results[level]
                assert 'thoughtseed_params' in level_data
                assert 'active_inference_params' in level_data
        
        finally:
            os.chdir(original_dir)


class TestFreeEnergyTracer:
    """Test free energy tracing functionality."""
    
    def test_tracer_initialization(self, temp_dir):
        """Test FreeEnergyTracer initialization."""
        tracer = FreeEnergyTracer(output_dir=os.path.join(temp_dir, "fe_traces"))
        
        # Check initialization
        assert hasattr(tracer, 'output_dir')
        assert hasattr(tracer, 'snapshots')
        assert hasattr(tracer, 'detailed_log')
        assert hasattr(tracer, 'component_histories')
        
        # Check initial state
        assert len(tracer.snapshots) == 0
        assert len(tracer.detailed_log) == 0
        assert isinstance(tracer.component_histories, dict)
        
        # Check output directory creation
        assert os.path.exists(tracer.output_dir)
    
    def test_trace_timestep(self, free_energy_tracer, mock_learner):
        """Test timestep tracing."""
        # Mock learner with required attributes for tracing
        mock_learner.state_history = ['breath_control', 'mind_wandering', 'meta_awareness']
        mock_learner.network_activations_history = [
            {'DMN': 0.4, 'VAN': 0.6, 'DAN': 0.7, 'FPN': 0.5}
        ]
        mock_learner.thoughtseeds = ['breath_focus', 'pain_discomfort', 'pending_tasks', 
                                   'self_reflection', 'equanimity']
        mock_learner.activations_history = [np.array([0.7, 0.2, 0.1, 0.5, 0.6])]
        mock_learner.meta_awareness_history = [0.6]
        
        # Add required learned profiles
        mock_learner.learned_network_profiles = {
            "thoughtseed_contributions": {
                ts: {'DMN': 0.5, 'VAN': 0.5, 'DAN': 0.5, 'FPN': 0.5}
                for ts in mock_learner.thoughtseeds
            }
        }
        
        # Test tracing
        snapshot = free_energy_tracer.trace_timestep(mock_learner, timestep=0)
        
        # Check snapshot
        assert isinstance(snapshot, FreeEnergySnapshot)
        assert snapshot.timestep == 0
        assert snapshot.state == 'breath_control'
        
        # Check that snapshot was recorded
        assert len(free_energy_tracer.snapshots) == 1
        assert len(free_energy_tracer.component_histories['variational_fe']) == 1
    
    def test_create_trace_summary(self, free_energy_tracer, mock_learner):
        """Test trace summary creation."""
        # Set up mock learner
        mock_learner.experience_level = 'expert'
        mock_learner.state_history = ['breath_control'] * 10
        mock_learner.thoughtseeds = ['breath_focus', 'pain_discomfort', 'pending_tasks', 
                                   'self_reflection', 'equanimity']
        mock_learner.learned_network_profiles = {
            "thoughtseed_contributions": {
                ts: {'DMN': 0.5, 'VAN': 0.5, 'DAN': 0.5, 'FPN': 0.5}
                for ts in mock_learner.thoughtseeds
            }
        }
        
        # Create some fake snapshots
        for i in range(5):
            free_energy_tracer.trace_timestep(mock_learner, i)
        
        # Create summary
        trace_summary = free_energy_tracer.create_trace_summary(mock_learner)
        
        # Check summary
        assert isinstance(trace_summary, FreeEnergyTrace)
        assert trace_summary.experience_level == 'expert'
        assert len(trace_summary.snapshots) == 5
        assert isinstance(trace_summary.summary_statistics, dict)
        assert isinstance(trace_summary.optimization_metrics, dict)
    
    def test_save_trace(self, free_energy_tracer, mock_learner):
        """Test trace saving functionality."""
        # Create minimal trace
        mock_learner.experience_level = 'novice'
        trace_summary = FreeEnergyTrace(
            experience_level='novice',
            simulation_duration=10,
            snapshots=[],
            summary_statistics={'test': 'data'},
            optimization_metrics={'test': 'data'}
        )
        
        # Test saving
        filepath = free_energy_tracer.save_trace(trace_summary, "test_trace")
        
        # Check file was created
        assert os.path.exists(filepath)
        
        # Check file content
        with open(filepath, 'r') as f:
            saved_data = json.load(f)
        
        assert saved_data['experience_level'] == 'novice'
        assert saved_data['simulation_duration'] == 10


class TestExportSystem:
    """Test the comprehensive export system."""
    
    def test_export_config_creation(self):
        """Test ExportConfig creation and presets."""
        # Test basic config
        config = ExportConfig(formats=['json', 'csv'])
        assert 'json' in config.formats
        assert 'csv' in config.formats
        
        # Test comprehensive preset
        comprehensive_config = ExportConfig.comprehensive()
        assert len(comprehensive_config.formats) > 2
        assert comprehensive_config.include_metadata
        assert comprehensive_config.include_time_series
        assert comprehensive_config.statistical_summaries
    
    def test_export_manager_initialization(self, export_config):
        """Test ExportManager initialization."""
        manager = ExportManager(export_config)
        
        assert hasattr(manager, 'config')
        assert hasattr(manager, 'exporters')
        assert len(manager.exporters) > 0
    
    @pytest.mark.export
    def test_export_learner(self, temp_dir, mock_learner):
        """Test learner export functionality."""
        original_dir = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            # Create export manager
            config = ExportConfig(formats=['json', 'csv'], include_metadata=True)
            manager = ExportManager(config)
            
            # Test export
            export_results = manager.export_learner(mock_learner, 'test_export')
            
            # Check results
            assert isinstance(export_results, dict)
            assert 'successful' in export_results
            assert 'failed' in export_results
            
            # Should have some successful exports
            assert len(export_results['successful']) > 0
            
        finally:
            os.chdir(original_dir)
    
    @pytest.mark.export 
    def test_export_comparison(self, temp_dir, mock_learner):
        """Test comparison export functionality."""
        original_dir = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            # Create two mock learners
            novice_learner = mock_learner
            novice_learner.experience_level = 'novice'
            
            expert_learner = Mock()
            expert_learner.experience_level = 'expert'
            expert_learner.thoughtseeds = mock_learner.thoughtseeds
            expert_learner.states = mock_learner.states
            expert_learner.networks = mock_learner.networks
            expert_learner.timesteps = mock_learner.timesteps
            expert_learner.activations_history = mock_learner.activations_history
            expert_learner.network_activations_history = mock_learner.network_activations_history
            expert_learner.state_history = mock_learner.state_history
            expert_learner.free_energy_history = mock_learner.free_energy_history
            expert_learner.precision_weight = 0.5
            expert_learner.complexity_penalty = 0.2
            expert_learner.learning_rate = 0.02
            
            # Create export manager
            config = ExportConfig(formats=['json'], include_metadata=True)
            manager = ExportManager(config)
            
            # Test comparison export
            export_results = manager.export_comparison(
                novice_learner, expert_learner, 'test_comparison')
            
            # Check results
            assert isinstance(export_results, dict)
            assert 'successful' in export_results
            
        finally:
            os.chdir(original_dir)


class TestUtilsIntegration:
    """Test integration between utility components."""
    
    def test_directory_and_export_integration(self, temp_dir, mock_learner):
        """Test that directory creation and export work together."""
        original_dir = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            # Create directories
            ensure_directories('./test_results')
            
            # Create export manager
            config = ExportConfig(formats=['json'])
            manager = ExportManager(config)
            
            # Export should work with created directories
            results = manager.export_learner(mock_learner, 'integration_test')
            
            assert isinstance(results, dict)
            assert len(results['successful']) > 0
            
        finally:
            os.chdir(original_dir)
    
    def test_tracer_and_export_integration(self, temp_dir, mock_learner):
        """Test integration between tracer and export systems."""
        original_dir = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            # Set up mock learner for tracing
            mock_learner.state_history = ['breath_control'] * 5
            mock_learner.network_activations_history = [
                {'DMN': 0.4, 'VAN': 0.6, 'DAN': 0.7, 'FPN': 0.5}
            ] * 5
            mock_learner.learned_network_profiles = {
                "thoughtseed_contributions": {
                    ts: {'DMN': 0.5, 'VAN': 0.5, 'DAN': 0.5, 'FPN': 0.5}
                    for ts in mock_learner.thoughtseeds
                }
            }
            
            # Create tracer and trace some timesteps
            tracer = FreeEnergyTracer(output_dir="./fe_traces")
            
            for i in range(3):
                tracer.trace_timestep(mock_learner, i)
            
            # Create trace summary
            trace_summary = tracer.create_trace_summary(mock_learner)
            
            # Save trace
            trace_file = tracer.save_trace(trace_summary, "integration_test")
            
            # Verify file exists and is valid JSON
            assert os.path.exists(trace_file)
            
            with open(trace_file, 'r') as f:
                trace_data = json.load(f)
                assert isinstance(trace_data, dict)
                assert 'experience_level' in trace_data
            
        finally:
            os.chdir(original_dir)
    
    def test_json_conversion_in_export(self, temp_dir):
        """Test that numpy conversion works properly in exports."""
        original_dir = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            # Create test data with numpy arrays
            test_data = {
                'numpy_array': np.array([1.1, 2.2, 3.3]),
                'nested_data': {
                    'another_array': np.array([[1, 2], [3, 4]]),
                    'regular_list': [1, 2, 3]
                }
            }
            
            # Convert using utility function
            converted_data = convert_numpy_to_lists(test_data)
            
            # Should be JSON serializable now
            json_str = json.dumps(converted_data)
            reconstructed = json.loads(json_str)
            
            # Check that data survived the round trip
            assert reconstructed['numpy_array'] == [1.1, 2.2, 3.3]
            assert reconstructed['nested_data']['another_array'] == [[1, 2], [3, 4]]
            assert reconstructed['nested_data']['regular_list'] == [1, 2, 3]
            
        finally:
            os.chdir(original_dir)


class TestUtilsErrorHandling:
    """Test error handling in utility functions."""
    
    def test_ensure_directories_permission_error(self):
        """Test ensure_directories with permission issues."""
        # Test with invalid path (should handle gracefully)
        try:
            # Try to create directory in non-existent parent
            ensure_directories('/nonexistent/parent/child')
        except (PermissionError, FileNotFoundError, OSError):
            # This is expected behavior for invalid paths
            pass
    
    def test_convert_numpy_invalid_input(self):
        """Test numpy conversion with edge cases."""
        # Test with None
        result = convert_numpy_to_lists(None)
        assert result is None
        
        # Test with empty structures
        result = convert_numpy_to_lists({})
        assert result == {}
        
        result = convert_numpy_to_lists([])
        assert result == []
    
    def test_export_with_missing_data(self, temp_dir):
        """Test export behavior with incomplete learner data."""
        original_dir = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            # Create minimal mock learner (missing some attributes)
            minimal_learner = Mock()
            minimal_learner.experience_level = 'test'
            minimal_learner.thoughtseeds = ['breath_focus']
            minimal_learner.activations_history = []
            
            # Export should handle missing attributes gracefully
            config = ExportConfig(formats=['json'])
            manager = ExportManager(config)
            
            # This should not crash, but may have failures
            results = manager.export_learner(minimal_learner, 'incomplete_test')
            
            # Should return results structure even if some exports failed
            assert isinstance(results, dict)
            assert 'successful' in results
            assert 'failed' in results
            
        finally:
            os.chdir(original_dir)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
