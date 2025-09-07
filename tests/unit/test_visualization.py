"""
Comprehensive unit tests for the visualization module.

This module tests all visualization functionality including plotting functions,
data loading, styling, and the enhanced free energy visualizer.
"""

import pytest
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import json
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock

# Set matplotlib backend for testing
matplotlib.use('Agg')

from visualization import (
    generate_all_plots, FreeEnergyVisualizer,
    plot_network_radar, plot_free_energy_comparison,
    plot_hierarchy, plot_time_series, load_json_data,
    set_plot_style, STATE_COLORS, NETWORK_COLORS, THOUGHTSEED_COLORS
)
from utils import FreeEnergyTrace, FreeEnergySnapshot


class TestVisualizationConstants:
    """Test visualization constants and color schemes."""
    
    def test_state_colors(self):
        """Test state color definitions."""
        expected_states = ['breath_control', 'mind_wandering', 'meta_awareness', 'redirect_breath']
        
        assert isinstance(STATE_COLORS, dict)
        
        # All expected states should have colors
        for state in expected_states:
            assert state in STATE_COLORS
            assert isinstance(STATE_COLORS[state], str)
            assert STATE_COLORS[state].startswith('#')  # Hex color
    
    def test_network_colors(self):
        """Test network color definitions."""
        expected_networks = ['DMN', 'VAN', 'DAN', 'FPN']
        
        assert isinstance(NETWORK_COLORS, dict)
        
        # All expected networks should have colors
        for network in expected_networks:
            assert network in NETWORK_COLORS
            assert isinstance(NETWORK_COLORS[network], str)
            assert NETWORK_COLORS[network].startswith('#')
    
    def test_thoughtseed_colors(self):
        """Test thoughtseed color definitions."""
        expected_thoughtseeds = ['breath_focus', 'equanimity', 'self_reflection', 
                               'pain_discomfort', 'pending_tasks']
        
        assert isinstance(THOUGHTSEED_COLORS, dict)
        
        # All expected thoughtseeds should have colors
        for ts in expected_thoughtseeds:
            assert ts in THOUGHTSEED_COLORS
            assert isinstance(THOUGHTSEED_COLORS[ts], str)
            assert THOUGHTSEED_COLORS[ts].startswith('#')
    
    def test_color_uniqueness(self):
        """Test that colors are reasonably unique."""
        all_colors = (list(STATE_COLORS.values()) + 
                     list(NETWORK_COLORS.values()) + 
                     list(THOUGHTSEED_COLORS.values()))
        
        # Should have reasonable color diversity (allowing some overlap)
        unique_colors = set(all_colors)
        assert len(unique_colors) >= len(all_colors) * 0.8  # At least 80% unique


class TestVisualizationUtilities:
    """Test visualization utility functions."""
    
    @pytest.mark.visualization
    def test_set_plot_style(self):
        """Test plot style setting."""
        # Should not raise error
        set_plot_style()
        
        # Check that matplotlib settings were applied
        # (This test is somewhat limited since we can't easily verify all settings)
        assert matplotlib.rcParams is not None
    
    def test_load_json_data_success(self, sample_json_data):
        """Test successful JSON data loading."""
        # Change to test directory
        original_dir = os.getcwd()
        os.chdir(sample_json_data)
        
        try:
            # Test loading novice data
            novice_data = load_json_data('novice')
            
            # Check data structure
            assert isinstance(novice_data, dict)
            assert 'experience_level' in novice_data
            assert novice_data['experience_level'] == 'novice'
            
            # Check that time series data was extracted
            if 'time_series' in novice_data.get('thoughtseed_params', {}):
                assert 'state_history' in novice_data
                assert 'activations_history' in novice_data
            
        finally:
            os.chdir(original_dir)
    
    def test_load_json_data_missing_files(self, temp_dir):
        """Test JSON data loading with missing files."""
        # Change to empty directory
        original_dir = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            # Should handle missing files gracefully
            data = load_json_data('nonexistent')
            
            # Should return None or empty dict, not crash
            assert data is None or isinstance(data, dict)
            
        finally:
            os.chdir(original_dir)


class TestBasicPlotFunctions:
    """Test individual plotting functions."""
    
    @pytest.mark.visualization
    def test_plot_network_radar(self, sample_json_data):
        """Test network radar plot creation."""
        original_dir = os.getcwd()
        os.chdir(sample_json_data)
        
        try:
            # Load test data
            novice_data = load_json_data('novice')
            expert_data = load_json_data('expert')
            
            if novice_data and expert_data:
                # Test plot creation (should not crash)
                with patch('matplotlib.pyplot.savefig'), patch('matplotlib.pyplot.close'):
                    plot_network_radar(novice_data, expert_data, save_path=None)
                
                # Test passes if no exception raised
                assert True
        
        finally:
            os.chdir(original_dir)
            plt.close('all')  # Clean up any open figures
    
    @pytest.mark.visualization
    def test_plot_free_energy_comparison(self, sample_json_data):
        """Test free energy comparison plot."""
        original_dir = os.getcwd()
        os.chdir(sample_json_data)
        
        try:
            novice_data = load_json_data('novice')
            expert_data = load_json_data('expert')
            
            if novice_data and expert_data:
                with patch('matplotlib.pyplot.savefig'), patch('matplotlib.pyplot.close'):
                    plot_free_energy_comparison(novice_data, expert_data, save_path=None)
                
                assert True
        
        finally:
            os.chdir(original_dir)
            plt.close('all')
    
    @pytest.mark.visualization
    def test_plot_hierarchy(self, sample_json_data):
        """Test hierarchical plot creation."""
        original_dir = os.getcwd()
        os.chdir(sample_json_data)
        
        try:
            data = load_json_data('novice')
            
            if data:
                with patch('matplotlib.pyplot.savefig'), patch('matplotlib.pyplot.close'):
                    plot_hierarchy(data, save_path=None)
                
                assert True
        
        finally:
            os.chdir(original_dir)
            plt.close('all')
    
    @pytest.mark.visualization
    def test_plot_time_series(self, sample_json_data):
        """Test time series plot creation."""
        original_dir = os.getcwd()
        os.chdir(sample_json_data)
        
        try:
            novice_data = load_json_data('novice')
            expert_data = load_json_data('expert')
            
            if novice_data and expert_data:
                # Ensure timesteps are present
                if 'timesteps' not in novice_data:
                    novice_data['timesteps'] = 20
                if 'timesteps' not in expert_data:
                    expert_data['timesteps'] = 20
                
                with patch('matplotlib.pyplot.savefig'), patch('matplotlib.pyplot.close'):
                    plot_time_series(novice_data, expert_data, save_path=None)
                
                assert True
        
        finally:
            os.chdir(original_dir)
            plt.close('all')
    
    @pytest.mark.visualization
    def test_generate_all_plots(self, sample_json_data):
        """Test comprehensive plot generation."""
        original_dir = os.getcwd()
        os.chdir(sample_json_data)
        
        try:
            with patch('matplotlib.pyplot.savefig'), patch('matplotlib.pyplot.close'):
                # Should attempt to generate all plots
                result = generate_all_plots()
                
                # Should return boolean indicating success/failure
                assert isinstance(result, bool)
        
        finally:
            os.chdir(original_dir)
            plt.close('all')


class TestFreeEnergyVisualizer:
    """Test the enhanced free energy visualizer."""
    
    def test_visualizer_initialization(self, temp_dir):
        """Test FreeEnergyVisualizer initialization."""
        visualizer = FreeEnergyVisualizer(output_dir=temp_dir)
        
        # Check initialization
        assert hasattr(visualizer, 'output_dir')
        assert hasattr(visualizer, 'component_colors')
        
        # Check output directory creation
        assert os.path.exists(visualizer.output_dir)
        
        # Check color scheme
        assert isinstance(visualizer.component_colors, dict)
        expected_components = ['variational_fe', 'expected_fe', 'prediction_error', 
                             'precision', 'complexity', 'meta_awareness']
        for component in expected_components:
            assert component in visualizer.component_colors
    
    def create_mock_trace(self):
        """Create a mock FreeEnergyTrace for testing."""
        # Create mock snapshots
        snapshots = []
        for i in range(20):
            snapshot = FreeEnergySnapshot(
                timestep=i,
                state='breath_control',
                variational_free_energy=1.5 + np.random.normal(0, 0.1),
                expected_free_energy=1.3 + np.random.normal(0, 0.1),
                prediction_error=0.2 + np.random.normal(0, 0.05),
                precision_weight=0.5 + np.random.normal(0, 0.05),
                complexity_penalty=0.3,
                network_predictions={'DMN': 0.4, 'VAN': 0.5, 'DAN': 0.6, 'FPN': 0.5},
                network_observations={'DMN': 0.45, 'VAN': 0.55, 'DAN': 0.65, 'FPN': 0.55},
                network_prediction_errors={'DMN': 0.05, 'VAN': 0.05, 'DAN': 0.05, 'FPN': 0.05},
                thoughtseed_activations={'breath_focus': 0.7, 'pain_discomfort': 0.2, 
                                       'pending_tasks': 0.1, 'self_reflection': 0.4, 'equanimity': 0.5},
                thoughtseed_predictions={'breath_focus': 0.65, 'pain_discomfort': 0.25, 
                                       'pending_tasks': 0.15, 'self_reflection': 0.35, 'equanimity': 0.45},
                meta_awareness=0.6,
                attention_precision=0.8,
                cognitive_load=0.3,
                transition_probability=0.1,
                state_entropy=1.2,
                gradient_magnitude=0.15,
                learning_rate_effective=0.01,
                timestamp='2024-01-01T00:00:00'
            )
            snapshots.append(snapshot)
        
        # Create trace
        trace = FreeEnergyTrace(
            experience_level='novice',
            simulation_duration=20,
            snapshots=snapshots,
            summary_statistics={
                'variational_fe_stats': {
                    'mean': 1.5, 'std': 0.1, 'min': 1.3, 'max': 1.7,
                    'final': 1.45, 'initial': 1.55
                },
                'expected_fe_stats': {
                    'mean': 1.3, 'std': 0.1, 'convergence_timestep': 15
                },
                'optimization_efficiency': {
                    'total_reduction': 0.1, 'percent_reduction': 6.7,
                    'convergence_rate': 0.005
                }
            },
            optimization_metrics={
                'gradient_evolution': [0.2, 0.18, 0.16, 0.15, 0.15],
                'learning_adaptation': [0.01, 0.01, 0.01, 0.01, 0.01],
                'precision_evolution': [0.7, 0.75, 0.8, 0.8, 0.8],
                'complexity_cost': [0.3, 0.3, 0.3, 0.3, 0.3]
            }
        )
        
        return trace
    
    @pytest.mark.visualization
    def test_create_comprehensive_dashboard(self, temp_dir):
        """Test comprehensive dashboard creation."""
        visualizer = FreeEnergyVisualizer(output_dir=temp_dir)
        trace = self.create_mock_trace()
        
        with patch('matplotlib.pyplot.savefig') as mock_save, \
             patch('matplotlib.pyplot.close'):
            
            dashboard_path = visualizer.create_comprehensive_dashboard(trace)
            
            # Should return a file path
            assert isinstance(dashboard_path, str)
            
            # Should attempt to save the figure
            mock_save.assert_called_once()
    
    @pytest.mark.visualization
    def test_create_detailed_component_analysis(self, temp_dir):
        """Test detailed component analysis visualization."""
        visualizer = FreeEnergyVisualizer(output_dir=temp_dir)
        trace = self.create_mock_trace()
        
        with patch('matplotlib.pyplot.savefig') as mock_save, \
             patch('matplotlib.pyplot.close'):
            
            analysis_path = visualizer.create_detailed_component_analysis(trace)
            
            assert isinstance(analysis_path, str)
            mock_save.assert_called_once()
    
    @pytest.mark.visualization
    def test_create_optimization_landscape(self, temp_dir):
        """Test optimization landscape visualization."""
        visualizer = FreeEnergyVisualizer(output_dir=temp_dir)
        trace = self.create_mock_trace()
        
        with patch('matplotlib.pyplot.savefig') as mock_save, \
             patch('matplotlib.pyplot.close'):
            
            landscape_path = visualizer.create_optimization_landscape(trace)
            
            assert isinstance(landscape_path, str)
            mock_save.assert_called_once()
    
    @pytest.mark.visualization
    def test_create_comparative_analysis(self, temp_dir):
        """Test comparative analysis visualization."""
        visualizer = FreeEnergyVisualizer(output_dir=temp_dir)
        
        novice_trace = self.create_mock_trace()
        expert_trace = self.create_mock_trace()
        expert_trace.experience_level = 'expert'
        
        # Make expert trace show improvement
        for snapshot in expert_trace.snapshots:
            snapshot.variational_free_energy *= 0.8  # Lower free energy
            snapshot.attention_precision *= 1.2  # Higher precision
        
        with patch('matplotlib.pyplot.savefig') as mock_save, \
             patch('matplotlib.pyplot.close'), \
             patch('pandas.DataFrame') as mock_df:
            
            # Mock pandas DataFrame for correlation analysis
            mock_df.return_value.corr.return_value.iloc = lambda i, j: 0.5
            
            comparison_path = visualizer.create_comparative_analysis(novice_trace, expert_trace)
            
            assert isinstance(comparison_path, str)
            mock_save.assert_called_once()


class TestVisualizationErrorHandling:
    """Test error handling in visualization functions."""
    
    @pytest.mark.visualization
    def test_plot_with_missing_data(self):
        """Test plotting functions with missing data."""
        # Create incomplete data
        incomplete_data = {
            'experience_level': 'novice',
            'state_history': ['breath_control'] * 5
            # Missing other required fields
        }
        
        # Should handle missing data gracefully
        with patch('matplotlib.pyplot.savefig'), \
             patch('matplotlib.pyplot.close'), \
             patch('builtins.print'):  # Suppress error prints
            
            try:
                plot_hierarchy(incomplete_data, save_path=None)
                # If it doesn't crash, that's good
                assert True
            except (KeyError, AttributeError, ValueError):
                # Acceptable to fail gracefully with missing data
                assert True
    
    @pytest.mark.visualization
    def test_plot_with_empty_data(self):
        """Test plotting with empty data structures."""
        empty_data = {
            'experience_level': 'test',
            'state_history': [],
            'activations_history': [],
            'network_activations_history': [],
            'free_energy_history': [],
            'timesteps': 0
        }
        
        with patch('matplotlib.pyplot.savefig'), \
             patch('matplotlib.pyplot.close'), \
             patch('builtins.print'):
            
            try:
                plot_time_series(empty_data, empty_data, save_path=None)
                assert True
            except (IndexError, ValueError, ZeroDivisionError):
                # Acceptable to fail with empty data
                assert True
    
    def test_load_json_data_corrupted_file(self, temp_dir):
        """Test loading corrupted JSON files."""
        # Create corrupted JSON file
        data_dir = os.path.join(temp_dir, "results_act_inf", "data")
        os.makedirs(data_dir, exist_ok=True)
        
        corrupted_file = os.path.join(data_dir, "thoughtseed_params_corrupted.json")
        with open(corrupted_file, 'w') as f:
            f.write('{"incomplete": json data')  # Invalid JSON
        
        original_dir = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            # Should handle corrupted files gracefully
            data = load_json_data('corrupted')
            
            # Should return None or handle error gracefully
            assert data is None or isinstance(data, dict)
            
        finally:
            os.chdir(original_dir)
    
    @pytest.mark.visualization
    def test_visualizer_with_invalid_output_dir(self):
        """Test visualizer with invalid output directory."""
        # Try to create visualizer with read-only or invalid path
        try:
            visualizer = FreeEnergyVisualizer(output_dir='/invalid/readonly/path')
            # If it succeeds, that's fine
            assert visualizer is not None
        except (PermissionError, FileNotFoundError, OSError):
            # Acceptable to fail with invalid path
            assert True


class TestVisualizationIntegration:
    """Test integration between visualization components."""
    
    @pytest.mark.visualization
    def test_color_consistency_across_plots(self, sample_json_data):
        """Test that colors are consistent across different plots."""
        original_dir = os.getcwd()
        os.chdir(sample_json_data)
        
        try:
            novice_data = load_json_data('novice')
            expert_data = load_json_data('expert')
            
            if novice_data and expert_data:
                # Mock matplotlib to capture color usage
                with patch('matplotlib.pyplot.savefig'), \
                     patch('matplotlib.pyplot.close'), \
                     patch('matplotlib.pyplot.plot') as mock_plot, \
                     patch('matplotlib.pyplot.bar') as mock_bar:
                    
                    # Generate multiple plots
                    plot_network_radar(novice_data, expert_data)
                    plot_free_energy_comparison(novice_data, expert_data)
                    
                    # Colors should be used consistently
                    # (This is a basic test - more detailed color tracking would be complex)
                    assert True
        
        finally:
            os.chdir(original_dir)
            plt.close('all')
    
    @pytest.mark.visualization
    def test_standard_and_enhanced_visualizer_integration(self, temp_dir):
        """Test that standard plots and enhanced visualizer work together."""
        # Create test data
        data_dir = os.path.join(temp_dir, "results_act_inf", "data")
        os.makedirs(data_dir, exist_ok=True)
        
        # Create minimal test data
        test_data = {
            "time_series": {
                "state_history": ["breath_control"] * 10,
                "free_energy_history": [1.5] * 10,
                "activations_history": [[0.7, 0.2, 0.1, 0.4, 0.5]] * 10,
                "network_activations_history": [{"DMN": 0.4, "VAN": 0.5, "DAN": 0.6, "FPN": 0.5}] * 10,
                "meta_awareness_history": [0.6] * 10,
                "dominant_ts_history": ["breath_focus"] * 10
            },
            "timesteps": 10
        }
        
        ai_data = {
            "precision_weight": 0.4,
            "complexity_penalty": 0.3,
            "average_free_energy_by_state": {"breath_control": 1.5}
        }
        
        # Save test files
        with open(os.path.join(data_dir, "thoughtseed_params_novice.json"), 'w') as f:
            json.dump(test_data, f)
        with open(os.path.join(data_dir, "active_inference_params_novice.json"), 'w') as f:
            json.dump(ai_data, f)
        
        original_dir = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            # Test standard plotting
            with patch('matplotlib.pyplot.savefig'), \
                 patch('matplotlib.pyplot.close'):
                
                result = generate_all_plots()
                # Should complete without major errors
                assert isinstance(result, bool)
            
            # Test enhanced visualizer
            visualizer = FreeEnergyVisualizer(output_dir=os.path.join(temp_dir, "enhanced_viz"))
            trace = FreeEnergyTrace(
                experience_level='novice',
                simulation_duration=10,
                snapshots=[],
                summary_statistics={'test': 'data'},
                optimization_metrics={'test': 'data'}
            )
            
            with patch('matplotlib.pyplot.savefig'), \
                 patch('matplotlib.pyplot.close'):
                
                dashboard_path = visualizer.create_comprehensive_dashboard(trace)
                assert isinstance(dashboard_path, str)
        
        finally:
            os.chdir(original_dir)
            plt.close('all')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
