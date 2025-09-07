"""
MATLAB export functionality for scientific computing integration.

This module provides MATLAB (.mat) export capabilities for seamless
integration with MATLAB-based analysis workflows and scientific computing
environments. Requires scipy package.

Note: Falls back gracefully if scipy is not available.
"""

from typing import Dict, Any, Optional
import numpy as np
from .base_exporter import BaseExporter

try:
    from scipy.io import savemat
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class MATLABExporter(BaseExporter):
    """
    MATLAB .mat file exporter for scientific computing integration.
    
    Exports simulation data in MATLAB-compatible format suitable for
    analysis in MATLAB, Octave, and other scientific computing environments.
    Requires scipy package.
    """
    
    def __init__(self, output_dir: str = "./exports",
                 format: str = '5',
                 long_field_names: bool = True,
                 do_compression: bool = True,
                 **kwargs):
        """
        Initialize MATLAB exporter.
        
        Args:
            output_dir: Directory for exported files
            format: MATLAB file format ('4', '5', or '7.3')
            long_field_names: Allow long field names (format 5+ only)
            do_compression: Enable compression
            **kwargs: Additional arguments for base exporter
        """
        super().__init__(output_dir, **kwargs)
        
        if not SCIPY_AVAILABLE:
            raise ImportError(
                "scipy is required for MATLAB export. Install with: pip install scipy"
            )
        
        self.format = format
        self.long_field_names = long_field_names
        self.do_compression = do_compression
    
    def export(self, learner: Any, filename: Optional[str] = None) -> Dict[str, str]:
        """Export learner data to MATLAB .mat format."""
        base_name = filename or "meditation_data"
        filepath = self._generate_filename(base_name, "mat", learner.experience_level)
        
        # Prepare MATLAB-compatible data structure
        matlab_data = self._prepare_matlab_data(learner)
        
        # Save to .mat file
        savemat(
            filepath,
            matlab_data,
            format=self.format,
            long_field_names=self.long_field_names,
            do_compression=self.do_compression
        )
        
        return {'matlab_data': filepath}
    
    def _prepare_matlab_data(self, learner: Any) -> Dict[str, Any]:
        """Prepare data in MATLAB-compatible format."""
        # Convert lists to numpy arrays for MATLAB compatibility
        activations_array = np.array(learner.activations_history)
        meta_awareness_array = np.array(learner.meta_awareness_history)
        
        # Create main data structure
        matlab_data = {
            # Metadata
            'metadata': {
                'experience_level': learner.experience_level,
                'timesteps': learner.timesteps,
                'thoughtseeds': np.array(learner.thoughtseeds, dtype='U'),
                'states': np.array(learner.states, dtype='U'),
                'natural_transitions': getattr(learner, 'natural_transition_count', 0),
                'forced_transitions': getattr(learner, 'forced_transition_count', 0),
                'export_timestamp': self.metadata['export_timestamp']
            },
            
            # Time series data
            'time_series': {
                'thoughtseed_activations': activations_array,
                'state_history': np.array(learner.state_history, dtype='U'),
                'meta_awareness_history': meta_awareness_array,
                'dominant_thoughtseed_history': np.array(learner.dominant_ts_history, dtype='U'),
                'timesteps': np.arange(len(learner.activations_history))
            },
            
            # Parameters
            'parameters': {
                'noise_level': getattr(learner, 'noise_level', np.nan),
                'precision_weight': getattr(learner, 'precision_weight', np.nan),
                'complexity_penalty': getattr(learner, 'complexity_penalty', np.nan),
                'learning_rate': getattr(learner, 'learning_rate', np.nan),
                'memory_factor': getattr(learner, 'memory_factor', np.nan)
            },
            
            # Statistics
            'statistics': self._calculate_matlab_statistics(learner, activations_array)
        }
        
        # Add network data if available
        if hasattr(learner, 'network_activations_history'):
            network_data = self._prepare_network_data(learner)
            matlab_data['networks'] = network_data
        
        return matlab_data
    
    def _calculate_matlab_statistics(self, learner: Any, activations_array: np.ndarray) -> Dict[str, Any]:
        """Calculate statistics in MATLAB-compatible format."""
        stats = {
            # Thoughtseed statistics
            'thoughtseed_means': np.mean(activations_array, axis=0),
            'thoughtseed_stds': np.std(activations_array, axis=0),
            'thoughtseed_maxs': np.max(activations_array, axis=0),
            'thoughtseed_mins': np.min(activations_array, axis=0),
            
            # State frequencies
            'state_frequencies': np.array([
                learner.state_history.count(state) / len(learner.state_history)
                for state in learner.states
            ]),
            
            # Meta-awareness statistics
            'meta_awareness_mean': np.mean(learner.meta_awareness_history),
            'meta_awareness_std': np.std(learner.meta_awareness_history),
            'meta_awareness_trend': self._calculate_trend_slope(learner.meta_awareness_history),
            
            # Dominance matrix (thoughtseed × timestep)
            'dominance_matrix': self._create_dominance_matrix(learner),
            
            # State transition matrix
            'transition_matrix': self._create_transition_matrix(learner)
        }
        
        return stats
    
    def _prepare_network_data(self, learner: Any) -> Dict[str, Any]:
        """Prepare network data in MATLAB format."""
        # Convert network activations to matrix (timestep × network)
        network_matrix = np.array([
            [step[net] for net in learner.networks]
            for step in learner.network_activations_history
        ])
        
        network_data = {
            'network_names': np.array(learner.networks, dtype='U'),
            'activations': network_matrix,
            'free_energy_history': np.array(learner.free_energy_history),
            'prediction_error_history': np.array(learner.prediction_error_history),
            'precision_history': np.array(learner.precision_history),
            
            # Network statistics
            'network_means': np.mean(network_matrix, axis=0),
            'network_stds': np.std(network_matrix, axis=0),
            'network_correlations': np.corrcoef(network_matrix.T),
            
            # Network-state relationships
            'network_by_state': self._calculate_network_by_state(learner, network_matrix)
        }
        
        return network_data
    
    def _create_dominance_matrix(self, learner: Any) -> np.ndarray:
        """Create binary dominance matrix (thoughtseed × timestep)."""
        dominance_matrix = np.zeros((len(learner.thoughtseeds), len(learner.dominant_ts_history)))
        
        for t, dominant_ts in enumerate(learner.dominant_ts_history):
            ts_index = learner.thoughtseeds.index(dominant_ts)
            dominance_matrix[ts_index, t] = 1
        
        return dominance_matrix
    
    def _create_transition_matrix(self, learner: Any) -> np.ndarray:
        """Create state transition count matrix."""
        n_states = len(learner.states)
        transition_matrix = np.zeros((n_states, n_states))
        
        # Count transitions
        for i in range(len(learner.state_history) - 1):
            from_state = learner.state_history[i]
            to_state = learner.state_history[i + 1]
            
            from_idx = learner.states.index(from_state)
            to_idx = learner.states.index(to_state)
            
            transition_matrix[from_idx, to_idx] += 1
        
        return transition_matrix
    
    def _calculate_network_by_state(self, learner: Any, network_matrix: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate network activation profiles by meditation state."""
        network_by_state = {}
        
        for state in learner.states:
            # Find indices for this state
            state_indices = [i for i, s in enumerate(learner.state_history) if s == state]
            
            if state_indices:
                # Calculate mean network activations for this state
                state_networks = np.mean(network_matrix[state_indices, :], axis=0)
                network_by_state[f'{state}_means'] = state_networks
                
                # Calculate standard deviations
                state_stds = np.std(network_matrix[state_indices, :], axis=0)
                network_by_state[f'{state}_stds'] = state_stds
        
        return network_by_state
    
    def _calculate_trend_slope(self, values: list) -> float:
        """Calculate linear trend slope for time series."""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        return float(slope)


class MATLABScriptGenerator:
    """
    Utility class for generating MATLAB analysis scripts.
    
    Provides templates and script generation for common analysis tasks
    with the exported MATLAB data.
    """
    
    @staticmethod
    def generate_analysis_script(mat_filepath: str, output_dir: str = "./matlab_scripts") -> str:
        """Generate basic MATLAB analysis script."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        script_content = f"""
%% Meditation Simulation Analysis Script
%% Auto-generated for data: {mat_filepath}

clear; clc;

%% Load data
fprintf('Loading meditation simulation data...\\n');
data = load('{mat_filepath}');

%% Display basic information
fprintf('Experience Level: %s\\n', data.metadata.experience_level);
fprintf('Timesteps: %d\\n', data.metadata.timesteps);
fprintf('Thoughtseeds: %s\\n', strjoin(cellstr(data.metadata.thoughtseeds), ', '));
fprintf('States: %s\\n', strjoin(cellstr(data.metadata.states), ', '));

%% Plot thoughtseed activations
figure('Name', 'Thoughtseed Activations Over Time');
plot(data.time_series.timesteps, data.time_series.thoughtseed_activations);
title('Thoughtseed Activations Over Time');
xlabel('Timestep');
ylabel('Activation Level');
legend(cellstr(data.metadata.thoughtseeds), 'Location', 'best');
grid on;

%% Plot meta-awareness
figure('Name', 'Meta-Awareness Over Time');
plot(data.time_series.timesteps, data.time_series.meta_awareness_history, 'LineWidth', 2);
title('Meta-Awareness Over Time');
xlabel('Timestep');
ylabel('Meta-Awareness Level');
grid on;

%% Network analysis (if available)
if isfield(data, 'networks')
    figure('Name', 'Network Activations');
    plot(data.time_series.timesteps, data.networks.activations);
    title('Network Activations Over Time');
    xlabel('Timestep');
    ylabel('Network Activation');
    legend(cellstr(data.networks.network_names), 'Location', 'best');
    grid on;
    
    figure('Name', 'Free Energy');
    plot(data.time_series.timesteps, data.networks.free_energy_history, 'LineWidth', 2);
    title('Free Energy Over Time');
    xlabel('Timestep');
    ylabel('Free Energy');
    grid on;
    
    %% Network correlation heatmap
    figure('Name', 'Network Correlations');
    imagesc(data.networks.network_correlations);
    colorbar;
    title('Network Activation Correlations');
    set(gca, 'XTick', 1:length(data.networks.network_names), ...
             'XTickLabel', cellstr(data.networks.network_names), ...
             'YTick', 1:length(data.networks.network_names), ...
             'YTickLabel', cellstr(data.networks.network_names));
    axis equal tight;
end

%% State transition analysis
figure('Name', 'State Transitions');
imagesc(data.statistics.transition_matrix);
colorbar;
title('State Transition Matrix');
set(gca, 'XTick', 1:length(data.metadata.states), ...
         'XTickLabel', cellstr(data.metadata.states), ...
         'YTick', 1:length(data.metadata.states), ...
         'YTickLabel', cellstr(data.metadata.states));
xlabel('To State');
ylabel('From State');

%% Summary statistics
fprintf('\\n=== Summary Statistics ===\\n');
fprintf('Thoughtseed means: %s\\n', mat2str(data.statistics.thoughtseed_means, 3));
fprintf('Meta-awareness mean: %.3f (std: %.3f)\\n', ...
        data.statistics.meta_awareness_mean, data.statistics.meta_awareness_std);

if isfield(data, 'networks')
    fprintf('Free energy mean: %.3f\\n', mean(data.networks.free_energy_history));
end

fprintf('\\nAnalysis complete!\\n');
"""
        
        script_path = os.path.join(output_dir, "analyze_meditation_data.m")
        with open(script_path, 'w') as f:
            f.write(script_content.strip())
        
        return script_path
