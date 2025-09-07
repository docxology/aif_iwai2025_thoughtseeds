"""
Enhanced Free Energy visualization system.

This module provides comprehensive visualization capabilities for all aspects
of Free Energy calculations including variational free energy, expected free energy,
component breakdowns, optimization trajectories, and comparative analyses.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
import json
from pathlib import Path

from utils.free_energy_tracer import FreeEnergyTrace, FreeEnergySnapshot


class FreeEnergyVisualizer:
    """
    Comprehensive visualization system for Free Energy calculations.
    
    Provides detailed visual analysis of all free energy components,
    optimization trajectories, and comparative studies.
    """
    
    def __init__(self, output_dir: str = "./free_energy_visualizations"):
        """Initialize visualizer with output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Define colors for components
        self.component_colors = {
            'variational_fe': '#e74c3c',      # Red
            'expected_fe': '#3498db',         # Blue  
            'prediction_error': '#f39c12',    # Orange
            'precision': '#27ae60',           # Green
            'complexity': '#9b59b6',          # Purple
            'meta_awareness': '#1abc9c',      # Teal
            'state_entropy': '#34495e'        # Dark gray
        }
    
    def create_comprehensive_dashboard(self, trace: FreeEnergyTrace, 
                                     comparison_trace: Optional[FreeEnergyTrace] = None) -> str:
        """Create comprehensive free energy analysis dashboard."""
        
        fig = plt.figure(figsize=(20, 16))
        gs = gridspec.GridSpec(4, 4, hspace=0.3, wspace=0.3)
        
        # 1. Main free energy evolution
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_free_energy_evolution(ax1, trace, comparison_trace)
        
        # 2. Component breakdown
        ax2 = fig.add_subplot(gs[0, 2:])
        self._plot_component_breakdown(ax2, trace)
        
        # 3. Precision and attention dynamics
        ax3 = fig.add_subplot(gs[1, :2])
        self._plot_precision_dynamics(ax3, trace)
        
        # 4. State-dependent free energy
        ax4 = fig.add_subplot(gs[1, 2:])
        self._plot_state_dependent_fe(ax4, trace)
        
        # 5. Optimization trajectory
        ax5 = fig.add_subplot(gs[2, :2])
        self._plot_optimization_trajectory(ax5, trace)
        
        # 6. Network prediction errors
        ax6 = fig.add_subplot(gs[2, 2:])
        self._plot_network_prediction_errors(ax6, trace)
        
        # 7. Convergence analysis
        ax7 = fig.add_subplot(gs[3, :2])
        self._plot_convergence_analysis(ax7, trace)
        
        # 8. Summary statistics
        ax8 = fig.add_subplot(gs[3, 2:])
        self._plot_summary_statistics(ax8, trace, comparison_trace)
        
        # Main title
        experience = trace.experience_level.capitalize()
        comparison_text = f" vs {comparison_trace.experience_level.capitalize()}" if comparison_trace else ""
        fig.suptitle(f'Free Energy Analysis Dashboard: {experience}{comparison_text}', 
                    fontsize=16, fontweight='bold')
        
        # Save
        filename = f"fe_dashboard_{trace.experience_level}"
        if comparison_trace:
            filename += f"_vs_{comparison_trace.experience_level}"
        
        filepath = self.output_dir / f"{filename}.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Free Energy Dashboard saved: {filepath}")
        return str(filepath)
    
    def _plot_free_energy_evolution(self, ax, trace: FreeEnergyTrace, 
                                   comparison_trace: Optional[FreeEnergyTrace] = None):
        """Plot evolution of variational and expected free energy."""
        timesteps = range(len(trace.snapshots))
        vfe_values = [s.variational_free_energy for s in trace.snapshots]
        efe_values = [s.expected_free_energy for s in trace.snapshots]
        
        # Plot main trace
        ax.plot(timesteps, vfe_values, label=f'Variational FE ({trace.experience_level})', 
                color=self.component_colors['variational_fe'], linewidth=2)
        ax.plot(timesteps, efe_values, label=f'Expected FE ({trace.experience_level})', 
                color=self.component_colors['expected_fe'], linewidth=2, linestyle='--')
        
        # Plot comparison if provided
        if comparison_trace:
            comp_timesteps = range(len(comparison_trace.snapshots))
            comp_vfe = [s.variational_free_energy for s in comparison_trace.snapshots]
            comp_efe = [s.expected_free_energy for s in comparison_trace.snapshots]
            
            ax.plot(comp_timesteps, comp_vfe, label=f'Variational FE ({comparison_trace.experience_level})', 
                    color=self.component_colors['variational_fe'], linewidth=2, alpha=0.7, linestyle=':')
            ax.plot(comp_timesteps, comp_efe, label=f'Expected FE ({comparison_trace.experience_level})', 
                    color=self.component_colors['expected_fe'], linewidth=2, alpha=0.7, linestyle='-.')
        
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Free Energy')
        ax.set_title('Free Energy Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_component_breakdown(self, ax, trace: FreeEnergyTrace):
        """Plot breakdown of free energy components."""
        timesteps = range(len(trace.snapshots))
        
        components = {
            'Prediction Error': [s.prediction_error for s in trace.snapshots],
            'Precision Weight': [s.precision_weight for s in trace.snapshots],
            'Complexity Penalty': [s.complexity_penalty for s in trace.snapshots],
            'Meta-awareness': [s.meta_awareness for s in trace.snapshots]
        }
        
        colors = ['#f39c12', '#27ae60', '#9b59b6', '#1abc9c']
        
        for i, (component, values) in enumerate(components.items()):
            ax.plot(timesteps, values, label=component, color=colors[i], linewidth=1.5)
        
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Component Value')
        ax.set_title('Free Energy Component Breakdown')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_precision_dynamics(self, ax, trace: FreeEnergyTrace):
        """Plot precision weighting and attention dynamics."""
        timesteps = range(len(trace.snapshots))
        
        precision_values = [s.precision_weight for s in trace.snapshots]
        attention_precision = [s.attention_precision for s in trace.snapshots]
        meta_awareness = [s.meta_awareness for s in trace.snapshots]
        
        ax.plot(timesteps, precision_values, label='Base Precision', 
                color=self.component_colors['precision'], linewidth=2)
        ax.plot(timesteps, attention_precision, label='Attention Precision', 
                color='darkgreen', linewidth=2, linestyle='--')
        ax.plot(timesteps, meta_awareness, label='Meta-awareness', 
                color=self.component_colors['meta_awareness'], linewidth=1.5, alpha=0.8)
        
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Precision / Awareness')
        ax.set_title('Precision & Attention Dynamics')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_state_dependent_fe(self, ax, trace: FreeEnergyTrace):
        """Plot free energy by meditation state."""
        state_fe_data = {}
        
        for snapshot in trace.snapshots:
            state = snapshot.state
            if state not in state_fe_data:
                state_fe_data[state] = []
            state_fe_data[state].append(snapshot.variational_free_energy)
        
        states = list(state_fe_data.keys())
        fe_means = [np.mean(state_fe_data[state]) for state in states]
        fe_stds = [np.std(state_fe_data[state]) for state in states]
        
        colors = ['#2ca02c', '#1f77b4', '#d62728', '#ff7f0e'][:len(states)]
        
        bars = ax.bar(states, fe_means, yerr=fe_stds, capsize=5, 
                     color=colors, alpha=0.7, edgecolor='black')
        
        ax.set_xlabel('Meditation State')
        ax.set_ylabel('Mean Variational Free Energy')
        ax.set_title('Free Energy by Meditation State')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Rotate x-axis labels if needed
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    def _plot_optimization_trajectory(self, ax, trace: FreeEnergyTrace):
        """Plot optimization trajectory and learning dynamics."""
        timesteps = range(len(trace.snapshots))
        
        gradient_magnitudes = [s.gradient_magnitude for s in trace.snapshots]
        effective_lr = [s.learning_rate_effective for s in trace.snapshots]
        cognitive_load = [s.cognitive_load for s in trace.snapshots]
        
        # Normalize values for comparison
        grad_norm = np.array(gradient_magnitudes) / (np.max(gradient_magnitudes) + 1e-8)
        lr_norm = np.array(effective_lr) / (np.max(effective_lr) + 1e-8)
        load_norm = np.array(cognitive_load) / (np.max(cognitive_load) + 1e-8)
        
        ax.plot(timesteps, grad_norm, label='Gradient Magnitude (norm)', 
                color='red', linewidth=2)
        ax.plot(timesteps, lr_norm, label='Effective Learning Rate (norm)', 
                color='blue', linewidth=2)
        ax.plot(timesteps, load_norm, label='Cognitive Load (norm)', 
                color='purple', linewidth=1.5, alpha=0.8)
        
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Normalized Value')
        ax.set_title('Optimization Trajectory')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_network_prediction_errors(self, ax, trace: FreeEnergyTrace):
        """Plot network-specific prediction errors."""
        # Collect network prediction errors
        network_errors = {}
        
        for snapshot in trace.snapshots:
            for net, error in snapshot.network_prediction_errors.items():
                if net not in network_errors:
                    network_errors[net] = []
                network_errors[net].append(error)
        
        if not network_errors:
            ax.text(0.5, 0.5, 'No network prediction error data available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Network Prediction Errors')
            return
        
        timesteps = range(len(next(iter(network_errors.values()))))
        
        colors = ['#CA3542', '#B77FB4', '#2C8B4B', '#E58429']  # DMN, VAN, DAN, FPN colors
        
        for i, (network, errors) in enumerate(network_errors.items()):
            color = colors[i] if i < len(colors) else f'C{i}'
            ax.plot(timesteps, errors, label=f'{network} PE', 
                   color=color, linewidth=2)
        
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Prediction Error')
        ax.set_title('Network Prediction Errors')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_convergence_analysis(self, ax, trace: FreeEnergyTrace):
        """Plot convergence analysis and stability metrics."""
        timesteps = range(len(trace.snapshots))
        vfe_values = [s.variational_free_energy for s in trace.snapshots]
        
        # Calculate moving averages for stability
        window_size = max(5, len(vfe_values) // 20)
        moving_avg = []
        moving_std = []
        
        for i in range(len(vfe_values)):
            start_idx = max(0, i - window_size + 1)
            window = vfe_values[start_idx:i+1]
            moving_avg.append(np.mean(window))
            moving_std.append(np.std(window))
        
        # Plot original and smoothed curves
        ax.plot(timesteps, vfe_values, label='Variational FE', 
                color=self.component_colors['variational_fe'], alpha=0.6, linewidth=1)
        ax.plot(timesteps, moving_avg, label='Moving Average', 
                color='darkred', linewidth=2)
        
        # Add stability bands
        moving_avg = np.array(moving_avg)
        moving_std = np.array(moving_std)
        ax.fill_between(timesteps, moving_avg - moving_std, moving_avg + moving_std,
                       alpha=0.2, color='red', label='±1 SD')
        
        # Mark convergence point if detected
        convergence_point = trace.summary_statistics['expected_fe_stats'].get('convergence_timestep', -1)
        if convergence_point > 0:
            ax.axvline(x=convergence_point, color='green', linestyle='--', linewidth=2,
                      label=f'Convergence at t={convergence_point}')
        
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Free Energy')
        ax.set_title('Convergence Analysis')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_summary_statistics(self, ax, trace: FreeEnergyTrace, 
                                comparison_trace: Optional[FreeEnergyTrace] = None):
        """Plot summary statistics comparison."""
        ax.axis('off')  # Turn off axis for text-based summary
        
        # Create summary text
        vfe_stats = trace.summary_statistics['variational_fe_stats']
        opt_metrics = trace.summary_statistics['optimization_efficiency']
        
        summary_text = f"""
FREE ENERGY SUMMARY ({trace.experience_level.upper()})

Variational Free Energy:
  • Initial: {vfe_stats['initial']:.3f}
  • Final: {vfe_stats['final']:.3f}
  • Mean: {vfe_stats['mean']:.3f} ± {vfe_stats['std']:.3f}
  • Range: [{vfe_stats['min']:.3f}, {vfe_stats['max']:.3f}]

Optimization Performance:
  • Total Reduction: {opt_metrics['total_reduction']:.3f}
  • Percent Reduction: {opt_metrics['percent_reduction']:.1f}%
  • Convergence Rate: {opt_metrics['convergence_rate']:.4f}

Simulation Details:
  • Duration: {trace.simulation_duration} timesteps
  • Experience Level: {trace.experience_level}
        """.strip()
        
        # Add comparison if available
        if comparison_trace:
            comp_vfe_stats = comparison_trace.summary_statistics['variational_fe_stats']
            comp_opt_metrics = comparison_trace.summary_statistics['optimization_efficiency']
            
            improvement_text = f"""

COMPARISON ({comparison_trace.experience_level.upper()}):
  • Final FE Improvement: {vfe_stats['final'] - comp_vfe_stats['final']:.3f}
  • Mean FE Improvement: {vfe_stats['mean'] - comp_vfe_stats['mean']:.3f}
  • Reduction Difference: {comp_opt_metrics['percent_reduction'] - opt_metrics['percent_reduction']:.1f}%
            """.strip()
            
            summary_text += improvement_text
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, 
                fontsize=10, fontfamily='monospace', verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        
        ax.set_title('Summary Statistics')
    
    def create_detailed_component_analysis(self, trace: FreeEnergyTrace) -> str:
        """Create detailed analysis of all free energy components."""
        
        fig, axes = plt.subplots(3, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        timesteps = range(len(trace.snapshots))
        
        # 1. Variational Free Energy decomposition
        vfe_values = [s.variational_free_energy for s in trace.snapshots]
        axes[0].plot(timesteps, vfe_values, color=self.component_colors['variational_fe'], linewidth=2)
        axes[0].set_title('Variational Free Energy')
        axes[0].grid(True, alpha=0.3)
        
        # 2. Expected Free Energy
        efe_values = [s.expected_free_energy for s in trace.snapshots]
        axes[1].plot(timesteps, efe_values, color=self.component_colors['expected_fe'], linewidth=2)
        axes[1].set_title('Expected Free Energy')
        axes[1].grid(True, alpha=0.3)
        
        # 3. Prediction Error Evolution
        pe_values = [s.prediction_error for s in trace.snapshots]
        axes[2].plot(timesteps, pe_values, color=self.component_colors['prediction_error'], linewidth=2)
        axes[2].set_title('Prediction Error')
        axes[2].grid(True, alpha=0.3)
        
        # 4. Precision Weighting
        precision_values = [s.precision_weight for s in trace.snapshots]
        axes[3].plot(timesteps, precision_values, color=self.component_colors['precision'], linewidth=2)
        axes[3].set_title('Precision Weighting')
        axes[3].grid(True, alpha=0.3)
        
        # 5. Complexity Penalty
        complexity_values = [s.complexity_penalty for s in trace.snapshots]
        axes[4].plot(timesteps, complexity_values, color=self.component_colors['complexity'], linewidth=2)
        axes[4].set_title('Complexity Penalty')
        axes[4].grid(True, alpha=0.3)
        
        # 6. State Entropy
        entropy_values = [s.state_entropy for s in trace.snapshots]
        axes[5].plot(timesteps, entropy_values, color=self.component_colors['state_entropy'], linewidth=2)
        axes[5].set_title('State Entropy')
        axes[5].grid(True, alpha=0.3)
        
        # 7. Transition Probabilities
        transition_probs = [s.transition_probability for s in trace.snapshots]
        axes[6].plot(timesteps, transition_probs, color='orange', linewidth=2)
        axes[6].set_title('Transition Probabilities')
        axes[6].grid(True, alpha=0.3)
        
        # 8. Cognitive Load
        cognitive_load = [s.cognitive_load for s in trace.snapshots]
        axes[7].plot(timesteps, cognitive_load, color='purple', linewidth=2)
        axes[7].set_title('Cognitive Load')
        axes[7].grid(True, alpha=0.3)
        
        # 9. Attention Precision
        att_precision = [s.attention_precision for s in trace.snapshots]
        axes[8].plot(timesteps, att_precision, color='teal', linewidth=2)
        axes[8].set_title('Attention Precision')
        axes[8].grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig.suptitle(f'Detailed Free Energy Component Analysis - {trace.experience_level.capitalize()}', 
                    fontsize=16, fontweight='bold', y=1.02)
        
        filepath = self.output_dir / f"fe_components_detailed_{trace.experience_level}.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Detailed Component Analysis saved: {filepath}")
        return str(filepath)
    
    def create_optimization_landscape(self, trace: FreeEnergyTrace) -> str:
        """Create 3D visualization of optimization landscape."""
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(15, 10))
        
        # Main 3D plot
        ax1 = fig.add_subplot(221, projection='3d')
        
        timesteps = range(len(trace.snapshots))
        vfe_values = [s.variational_free_energy for s in trace.snapshots]
        efe_values = [s.expected_free_energy for s in trace.snapshots]
        precision_values = [s.precision_weight for s in trace.snapshots]
        
        # Create 3D trajectory
        ax1.plot(vfe_values, efe_values, precision_values, 
                color='red', linewidth=2, alpha=0.8)
        ax1.scatter(vfe_values[0], efe_values[0], precision_values[0], 
                   color='green', s=100, label='Start')
        ax1.scatter(vfe_values[-1], efe_values[-1], precision_values[-1], 
                   color='red', s=100, label='End')
        
        ax1.set_xlabel('Variational FE')
        ax1.set_ylabel('Expected FE')  
        ax1.set_zlabel('Precision')
        ax1.set_title('Free Energy Optimization Trajectory')
        ax1.legend()
        
        # 2D projections
        ax2 = fig.add_subplot(222)
        ax2.plot(vfe_values, efe_values, color='blue', linewidth=2)
        ax2.scatter(vfe_values[0], efe_values[0], color='green', s=50)
        ax2.scatter(vfe_values[-1], efe_values[-1], color='red', s=50)
        ax2.set_xlabel('Variational FE')
        ax2.set_ylabel('Expected FE')
        ax2.set_title('VFE vs EFE Trajectory')
        ax2.grid(True, alpha=0.3)
        
        ax3 = fig.add_subplot(223)
        ax3.plot(timesteps, vfe_values, color=self.component_colors['variational_fe'], 
                linewidth=2, label='Variational FE')
        ax3.plot(timesteps, efe_values, color=self.component_colors['expected_fe'], 
                linewidth=2, label='Expected FE')
        ax3.set_xlabel('Timestep')
        ax3.set_ylabel('Free Energy')
        ax3.set_title('Free Energy Evolution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Optimization metrics
        ax4 = fig.add_subplot(224)
        gradient_mags = [s.gradient_magnitude for s in trace.snapshots]
        ax4.plot(timesteps, gradient_mags, color='purple', linewidth=2)
        ax4.set_xlabel('Timestep')
        ax4.set_ylabel('Gradient Magnitude')
        ax4.set_title('Optimization Gradient')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig.suptitle(f'Free Energy Optimization Landscape - {trace.experience_level.capitalize()}', 
                    fontsize=14, fontweight='bold', y=1.02)
        
        filepath = self.output_dir / f"fe_optimization_landscape_{trace.experience_level}.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Optimization Landscape saved: {filepath}")
        return str(filepath)
    
    def create_comparative_analysis(self, novice_trace: FreeEnergyTrace, 
                                   expert_trace: FreeEnergyTrace) -> str:
        """Create comprehensive comparative analysis visualization."""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # 1. Free Energy Comparison
        timesteps_n = range(len(novice_trace.snapshots))
        timesteps_e = range(len(expert_trace.snapshots))
        
        vfe_n = [s.variational_free_energy for s in novice_trace.snapshots]
        vfe_e = [s.variational_free_energy for s in expert_trace.snapshots]
        
        axes[0,0].plot(timesteps_n, vfe_n, label='Novice', color='blue', linewidth=2)
        axes[0,0].plot(timesteps_e, vfe_e, label='Expert', color='red', linewidth=2)
        axes[0,0].set_xlabel('Timestep')
        axes[0,0].set_ylabel('Variational Free Energy')
        axes[0,0].set_title('Free Energy Comparison')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Optimization Efficiency
        opt_n = novice_trace.summary_statistics['optimization_efficiency']
        opt_e = expert_trace.summary_statistics['optimization_efficiency']
        
        metrics = ['total_reduction', 'percent_reduction', 'convergence_rate']
        novice_vals = [opt_n[m] for m in metrics]
        expert_vals = [opt_e[m] for m in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        axes[0,1].bar(x - width/2, novice_vals, width, label='Novice', color='blue', alpha=0.7)
        axes[0,1].bar(x + width/2, expert_vals, width, label='Expert', color='red', alpha=0.7)
        axes[0,1].set_xlabel('Optimization Metric')
        axes[0,1].set_ylabel('Value')
        axes[0,1].set_title('Optimization Efficiency')
        axes[0,1].set_xticks(x)
        axes[0,1].set_xticklabels(['Total Red.', 'Percent Red.', 'Conv. Rate'])
        axes[0,1].legend()
        
        # 3. Precision Dynamics Comparison
        precision_n = [s.attention_precision for s in novice_trace.snapshots]
        precision_e = [s.attention_precision for s in expert_trace.snapshots]
        
        axes[0,2].plot(timesteps_n, precision_n, label='Novice', color='blue', linewidth=2)
        axes[0,2].plot(timesteps_e, precision_e, label='Expert', color='red', linewidth=2)
        axes[0,2].set_xlabel('Timestep')
        axes[0,2].set_ylabel('Attention Precision')
        axes[0,2].set_title('Precision Dynamics')
        axes[0,2].legend()
        axes[0,2].grid(True, alpha=0.3)
        
        # 4. State-dependent FE comparison
        self._plot_state_fe_comparison(axes[1,0], novice_trace, expert_trace)
        
        # 5. Component correlation heatmap
        self._plot_component_correlations(axes[1,1], expert_trace)
        
        # 6. Performance summary
        self._plot_performance_summary(axes[1,2], novice_trace, expert_trace)
        
        plt.tight_layout()
        fig.suptitle('Expert vs Novice Free Energy Analysis', 
                    fontsize=16, fontweight='bold', y=1.02)
        
        filepath = self.output_dir / "fe_comparative_analysis.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Comparative Analysis saved: {filepath}")
        return str(filepath)
    
    def _plot_state_fe_comparison(self, ax, novice_trace, expert_trace):
        """Plot state-dependent free energy comparison."""
        # Collect state-wise FE data
        states = set()
        novice_state_fe = {}
        expert_state_fe = {}
        
        for snapshot in novice_trace.snapshots:
            state = snapshot.state
            states.add(state)
            if state not in novice_state_fe:
                novice_state_fe[state] = []
            novice_state_fe[state].append(snapshot.variational_free_energy)
        
        for snapshot in expert_trace.snapshots:
            state = snapshot.state
            states.add(state)
            if state not in expert_state_fe:
                expert_state_fe[state] = []
            expert_state_fe[state].append(snapshot.variational_free_energy)
        
        states = sorted(list(states))
        novice_means = [np.mean(novice_state_fe.get(state, [0])) for state in states]
        expert_means = [np.mean(expert_state_fe.get(state, [0])) for state in states]
        
        x = np.arange(len(states))
        width = 0.35
        
        ax.bar(x - width/2, novice_means, width, label='Novice', color='blue', alpha=0.7)
        ax.bar(x + width/2, expert_means, width, label='Expert', color='red', alpha=0.7)
        ax.set_xlabel('Meditation State')
        ax.set_ylabel('Mean Free Energy')
        ax.set_title('State-dependent Free Energy')
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace('_', '\n') for s in states])
        ax.legend()
    
    def _plot_component_correlations(self, ax, trace):
        """Plot correlation matrix of free energy components."""
        # Extract component values
        components_data = {
            'VFE': [s.variational_free_energy for s in trace.snapshots],
            'EFE': [s.expected_free_energy for s in trace.snapshots],
            'PE': [s.prediction_error for s in trace.snapshots],
            'Precision': [s.precision_weight for s in trace.snapshots],
            'Complexity': [s.complexity_penalty for s in trace.snapshots],
            'Meta-awareness': [s.meta_awareness for s in trace.snapshots]
        }
        
        # Calculate correlation matrix
        import pandas as pd
        df = pd.DataFrame(components_data)
        corr_matrix = df.corr()
        
        # Plot heatmap
        im = ax.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        
        # Add labels
        ax.set_xticks(range(len(corr_matrix.columns)))
        ax.set_yticks(range(len(corr_matrix.columns)))
        ax.set_xticklabels(corr_matrix.columns, rotation=45)
        ax.set_yticklabels(corr_matrix.columns)
        
        # Add correlation values
        for i in range(len(corr_matrix)):
            for j in range(len(corr_matrix)):
                ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', 
                       ha='center', va='center', color='white' if abs(corr_matrix.iloc[i, j]) > 0.5 else 'black')
        
        ax.set_title('Component Correlations')
        plt.colorbar(im, ax=ax, shrink=0.8)
    
    def _plot_performance_summary(self, ax, novice_trace, expert_trace):
        """Plot performance summary comparison."""
        ax.axis('off')
        
        # Calculate key metrics
        novice_final_fe = novice_trace.snapshots[-1].variational_free_energy
        expert_final_fe = expert_trace.snapshots[-1].variational_free_energy
        fe_improvement = (novice_final_fe - expert_final_fe) / novice_final_fe * 100
        
        novice_opt = novice_trace.summary_statistics['optimization_efficiency']
        expert_opt = expert_trace.summary_statistics['optimization_efficiency']
        
        summary_text = f"""
PERFORMANCE SUMMARY

Free Energy Improvement:
  Expert Final FE: {expert_final_fe:.3f}
  Novice Final FE: {novice_final_fe:.3f}
  Improvement: {fe_improvement:.1f}%

Optimization Efficiency:
  Expert Reduction: {expert_opt['percent_reduction']:.1f}%
  Novice Reduction: {novice_opt['percent_reduction']:.1f}%
  
Convergence:
  Expert Rate: {expert_opt['convergence_rate']:.4f}
  Novice Rate: {novice_opt['convergence_rate']:.4f}
        """.strip()
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
                fontsize=11, fontfamily='monospace', verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))
        
        ax.set_title('Performance Summary')
