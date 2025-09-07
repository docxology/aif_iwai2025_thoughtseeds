"""
Advanced Free Energy Visualization System.

This module provides sophisticated visualizations for free energy calculations,
decompositions, optimization landscapes, and energy flow dynamics in meditation.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
import os

from .plotting import set_plot_style


class AdvancedFreeEnergyVisualizer:
    """Advanced visualizer for free energy dynamics and decomposition."""
    
    def __init__(self):
        """Initialize advanced free energy visualizer."""
        self.plot_dir = "results_act_inf/plots"
        os.makedirs(self.plot_dir, exist_ok=True)
        
    def create_energy_decomposition_plot(self, learner_data: Dict[str, Any], save_path: Optional[str] = None):
        """Create detailed free energy decomposition visualization."""
        set_plot_style()
        
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle(f'Free Energy Decomposition Analysis - {learner_data.get("experience_level", "").title()}',
                     fontsize=16, fontweight='bold')
        
        # Extract energy components
        free_energy = learner_data.get('free_energy_history', [])
        prediction_error = learner_data.get('prediction_error_history', [])
        precision_values = learner_data.get('precision_history', [])
        
        if not free_energy:
            print("Warning: No free energy data available for decomposition")
            return
            
        time_steps = np.arange(len(free_energy))
        
        # Calculate energy components
        complexity_penalty = learner_data.get('complexity_penalty', 0.1)
        learning_rate = learner_data.get('learning_rate', 0.01)
        
        # Estimated complexity term (simplified)
        complexity_term = [complexity_penalty * (i * learning_rate) for i in range(len(free_energy))]
        
        # Variational free energy components
        accuracy_term = prediction_error if prediction_error else [fe * 0.7 for fe in free_energy]
        
        # 1. Main free energy evolution with components
        ax1 = axes[0, 0]
        ax1.plot(time_steps, free_energy, 'k-', linewidth=3, label='Total Free Energy', alpha=0.9)
        ax1.plot(time_steps, accuracy_term, 'r--', linewidth=2, label='Accuracy Term', alpha=0.7)
        ax1.plot(time_steps, complexity_term, 'b--', linewidth=2, label='Complexity Term', alpha=0.7)
        
        ax1.fill_between(time_steps, accuracy_term, alpha=0.2, color='red')
        ax1.fill_between(time_steps, complexity_term, alpha=0.2, color='blue')
        
        ax1.set_title('Free Energy Decomposition')
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Energy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Energy gradient and curvature analysis
        ax2 = axes[0, 1]
        if len(free_energy) > 2:
            fe_gradient = np.gradient(free_energy)
            fe_curvature = np.gradient(fe_gradient)
            
            ax2_twin = ax2.twinx()
            
            line1 = ax2.plot(time_steps, fe_gradient, 'g-', linewidth=2, label='Gradient (1st derivative)')
            line2 = ax2_twin.plot(time_steps, fe_curvature, 'orange', linewidth=2, label='Curvature (2nd derivative)')
            
            ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax2_twin.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            
            ax2.set_xlabel('Time Step')
            ax2.set_ylabel('Gradient', color='g')
            ax2_twin.set_ylabel('Curvature', color='orange')
            ax2.tick_params(axis='y', labelcolor='g')
            ax2_twin.tick_params(axis='y', labelcolor='orange')
            
            # Combined legend
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax2.legend(lines, labels, loc='upper right')
        
        ax2.set_title('Energy Dynamics (Gradient & Curvature)')
        ax2.grid(True, alpha=0.3)
        
        # 3. Precision weighting effects
        ax3 = axes[1, 0]
        if precision_values:
            # Show how precision affects energy
            weighted_energy = [fe * (1 + p) for fe, p in zip(free_energy, precision_values)]
            
            ax3.plot(time_steps, free_energy, 'b-', linewidth=2, label='Unweighted Energy')
            ax3.plot(time_steps, weighted_energy, 'r-', linewidth=2, label='Precision-Weighted Energy')
            ax3.plot(time_steps, precision_values, 'g--', linewidth=2, label='Precision Values', alpha=0.7)
            
            ax3.fill_between(time_steps, free_energy, weighted_energy, 
                           alpha=0.3, color='red', label='Precision Effect')
        else:
            # Estimated precision effect
            estimated_precision = [0.5 + 0.3 * np.sin(i * 0.1) for i in range(len(free_energy))]
            weighted_energy = [fe * (1 + p) for fe, p in zip(free_energy, estimated_precision)]
            
            ax3.plot(time_steps, free_energy, 'b-', linewidth=2, label='Base Energy')
            ax3.plot(time_steps, weighted_energy, 'r-', linewidth=2, label='Estimated Weighted Energy')
            ax3.plot(time_steps, estimated_precision, 'g--', linewidth=1, label='Est. Precision', alpha=0.7)
        
        ax3.set_title('Precision Weighting Effects')
        ax3.set_xlabel('Time Step')
        ax3.set_ylabel('Energy')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Energy landscape topology
        ax4 = axes[1, 1]
        # Create energy landscape approximation
        window_size = min(20, len(free_energy) // 5)
        landscape_points = []
        landscape_energies = []
        
        for i in range(0, len(free_energy) - window_size, window_size):
            window_energy = free_energy[i:i+window_size]
            landscape_points.append(i + window_size // 2)
            landscape_energies.append(np.mean(window_energy))
        
        if len(landscape_points) > 2:
            # Interpolate for smooth landscape
            interp_points = np.linspace(landscape_points[0], landscape_points[-1], 100)
            interp_energies = np.interp(interp_points, landscape_points, landscape_energies)
            
            ax4.plot(interp_points, interp_energies, 'purple', linewidth=3, alpha=0.8)
            ax4.scatter(landscape_points, landscape_energies, c='red', s=50, zorder=5, alpha=0.8)
            ax4.fill_between(interp_points, interp_energies, alpha=0.3, color='purple')
            
            # Mark local minima
            for i in range(1, len(landscape_energies) - 1):
                if (landscape_energies[i] < landscape_energies[i-1] and 
                    landscape_energies[i] < landscape_energies[i+1]):
                    ax4.scatter(landscape_points[i], landscape_energies[i], 
                              c='green', s=100, marker='*', zorder=10, 
                              label='Local Minimum' if i == 1 else "")
        
        ax4.set_title('Energy Landscape Topology')
        ax4.set_xlabel('Time Window')
        ax4.set_ylabel('Average Energy')
        ax4.grid(True, alpha=0.3)
        if 'Local Minimum' in ax4.get_legend_handles_labels()[1]:
            ax4.legend()
        
        # 5. Expected vs Variational Free Energy
        ax5 = axes[2, 0]
        # Calculate Expected Free Energy approximation
        meta_awareness = learner_data.get('meta_awareness_history', [])
        if meta_awareness:
            # EFE approximation: higher meta-awareness -> lower expected future energy
            expected_fe = [fe * (1 - ma * 0.5) for fe, ma in zip(free_energy, meta_awareness)]
        else:
            # Estimated EFE based on trend
            expected_fe = [fe * (1 - i * 0.001) for i, fe in enumerate(free_energy)]
        
        ax5.plot(time_steps, free_energy, 'b-', linewidth=2, label='Variational Free Energy (VFE)')
        ax5.plot(time_steps, expected_fe, 'r-', linewidth=2, label='Expected Free Energy (EFE)')
        
        # Highlight divergence regions
        divergence = [abs(vfe - efe) for vfe, efe in zip(free_energy, expected_fe)]
        high_div_threshold = np.percentile(divergence, 75)
        high_div_indices = [i for i, d in enumerate(divergence) if d > high_div_threshold]
        
        if high_div_indices:
            ax5.scatter([time_steps[i] for i in high_div_indices], 
                       [free_energy[i] for i in high_div_indices],
                       c='orange', s=30, alpha=0.7, label='High Divergence')
        
        ax5.set_title('VFE vs EFE Comparison')
        ax5.set_xlabel('Time Step')
        ax5.set_ylabel('Free Energy')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Energy optimization trajectory
        ax6 = axes[2, 1]
        if len(free_energy) > 10:
            # Create optimization trajectory in energy-gradient space
            fe_gradient = np.gradient(free_energy)
            
            # Color points by time (trajectory evolution)
            colors = np.arange(len(free_energy))
            scatter = ax6.scatter(free_energy, fe_gradient, c=colors, cmap='viridis', 
                                alpha=0.7, s=20)
            
            # Add trajectory arrows
            for i in range(0, len(free_energy)-5, max(1, len(free_energy)//20)):
                ax6.annotate('', xy=(free_energy[i+5], fe_gradient[i+5]),
                           xytext=(free_energy[i], fe_gradient[i]),
                           arrowprops=dict(arrowstyle='->', alpha=0.5, lw=1))
            
            # Mark start and end points
            ax6.scatter(free_energy[0], fe_gradient[0], c='red', s=100, marker='o', 
                       label='Start', zorder=10)
            ax6.scatter(free_energy[-1], fe_gradient[-1], c='green', s=100, marker='s', 
                       label='End', zorder=10)
            
            plt.colorbar(scatter, ax=ax6, label='Time Step')
        
        ax6.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax6.axvline(x=np.mean(free_energy), color='gray', linestyle='--', alpha=0.5)
        ax6.set_title('Optimization Trajectory')
        ax6.set_xlabel('Free Energy')
        ax6.set_ylabel('Energy Gradient')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(f"{self.plot_dir}/energy_decomposition_{learner_data.get('experience_level', 'unknown')}.png",
                       dpi=300, bbox_inches='tight')
        plt.close()
        
    def create_energy_surface_plot(self, novice_data: Dict[str, Any], expert_data: Dict[str, Any], 
                                 save_path: Optional[str] = None):
        """Create 3D energy surface comparison plot."""
        set_plot_style()
        
        fig = plt.figure(figsize=(16, 6))
        
        for i, (data, title) in enumerate([(novice_data, 'Novice'), (expert_data, 'Expert')]):
            ax = fig.add_subplot(1, 2, i+1, projection='3d')
            
            free_energy = data.get('free_energy_history', [])
            meta_awareness = data.get('meta_awareness_history', [])
            
            if not free_energy or not meta_awareness:
                continue
            
            # Create meshgrid for surface
            time_range = np.arange(len(free_energy))
            fe_range = np.linspace(min(free_energy), max(free_energy), 20)
            ma_range = np.linspace(min(meta_awareness), max(meta_awareness), 20)
            
            FE, MA = np.meshgrid(fe_range, ma_range)
            
            # Create energy surface (simplified model)
            Z = FE * (1 - MA * 0.5) + MA * FE * 0.3
            
            # Plot surface
            surf = ax.plot_surface(FE, MA, Z, cmap='viridis', alpha=0.8)
            
            # Plot actual trajectory
            ax.scatter(free_energy, meta_awareness, 
                      [free_energy[j] * (1 - meta_awareness[j] * 0.5) for j in range(len(free_energy))],
                      c='red', s=20, alpha=0.8)
            
            # Connect trajectory points
            for j in range(len(free_energy)-1):
                z1 = free_energy[j] * (1 - meta_awareness[j] * 0.5)
                z2 = free_energy[j+1] * (1 - meta_awareness[j+1] * 0.5)
                ax.plot([free_energy[j], free_energy[j+1]], 
                       [meta_awareness[j], meta_awareness[j+1]],
                       [z1, z2], 'r-', alpha=0.5, linewidth=1)
            
            ax.set_xlabel('Free Energy')
            ax.set_ylabel('Meta-awareness')
            ax.set_zlabel('Energy Landscape')
            ax.set_title(f'{title} Energy Surface')
            
            # Add colorbar
            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        
        fig.suptitle('3D Energy Landscape Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(f"{self.plot_dir}/energy_surface_3d.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    def create_energy_flow_diagram(self, learner_data: Dict[str, Any], save_path: Optional[str] = None):
        """Create energy flow and transformation diagram."""
        set_plot_style()
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Free Energy Flow Analysis - {learner_data.get("experience_level", "").title()}',
                     fontsize=16, fontweight='bold')
        
        free_energy = learner_data.get('free_energy_history', [])
        if not free_energy:
            return
            
        time_steps = np.arange(len(free_energy))
        
        # 1. Energy dissipation rate
        ax1 = axes[0, 0]
        if len(free_energy) > 1:
            energy_change = np.diff(free_energy)
            dissipation_rate = [-change if change < 0 else 0 for change in energy_change]
            accumulation_rate = [change if change > 0 else 0 for change in energy_change]
            
            ax1.bar(time_steps[1:], dissipation_rate, color='green', alpha=0.7, 
                   label='Energy Dissipation', width=0.8)
            ax1.bar(time_steps[1:], accumulation_rate, color='red', alpha=0.7,
                   label='Energy Accumulation', width=0.8)
            
            ax1.set_title('Energy Flow Rate')
            ax1.set_xlabel('Time Step')
            ax1.set_ylabel('Energy Change Rate')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 2. Cumulative energy balance
        ax2 = axes[0, 1]
        cumulative_dissipation = np.cumsum([max(0, -np.diff(free_energy)[i-1]) 
                                          for i in range(1, len(free_energy))])
        cumulative_accumulation = np.cumsum([max(0, np.diff(free_energy)[i-1]) 
                                           for i in range(1, len(free_energy))])
        
        ax2.plot(time_steps[1:], cumulative_dissipation, 'g-', linewidth=3, 
                label='Cumulative Dissipation')
        ax2.plot(time_steps[1:], cumulative_accumulation, 'r-', linewidth=3,
                label='Cumulative Accumulation')
        
        ax2.fill_between(time_steps[1:], cumulative_dissipation, alpha=0.3, color='green')
        ax2.fill_between(time_steps[1:], cumulative_accumulation, alpha=0.3, color='red')
        
        ax2.set_title('Cumulative Energy Balance')
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Cumulative Energy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Energy efficiency over time
        ax3 = axes[1, 0]
        window_size = max(5, len(free_energy) // 10)
        efficiency_values = []
        efficiency_time = []
        
        for i in range(window_size, len(free_energy)):
            window_energy = free_energy[i-window_size:i]
            initial_energy = window_energy[0]
            final_energy = window_energy[-1]
            
            # Efficiency as energy reduction per unit time
            if initial_energy > 0:
                efficiency = max(0, (initial_energy - final_energy) / (initial_energy * window_size))
                efficiency_values.append(efficiency)
                efficiency_time.append(i)
        
        if efficiency_values:
            ax3.plot(efficiency_time, efficiency_values, 'purple', linewidth=2)
            ax3.fill_between(efficiency_time, efficiency_values, alpha=0.3, color='purple')
            
            # Add trend line
            if len(efficiency_values) > 2:
                z = np.polyfit(efficiency_time, efficiency_values, 1)
                trend_line = np.poly1d(z)
                ax3.plot(efficiency_time, trend_line(efficiency_time), 'r--', 
                        linewidth=2, alpha=0.8, label=f'Trend (slope: {z[0]:.4f})')
                ax3.legend()
        
        ax3.set_title('Energy Optimization Efficiency')
        ax3.set_xlabel('Time Step')
        ax3.set_ylabel('Efficiency Index')
        ax3.grid(True, alpha=0.3)
        
        # 4. Phase transitions in energy dynamics
        ax4 = axes[1, 1]
        if len(free_energy) > 20:
            # Detect phase transitions using change point detection
            window_size = len(free_energy) // 10
            transitions = []
            
            for i in range(window_size, len(free_energy) - window_size):
                before_mean = np.mean(free_energy[i-window_size:i])
                after_mean = np.mean(free_energy[i:i+window_size])
                
                # Significant change in mean indicates phase transition
                if abs(before_mean - after_mean) > 0.1 * np.std(free_energy):
                    transitions.append((i, before_mean, after_mean))
            
            # Plot energy with phase transitions marked
            ax4.plot(time_steps, free_energy, 'b-', linewidth=2, alpha=0.8)
            
            # Mark transitions
            for i, (trans_point, before, after) in enumerate(transitions):
                ax4.axvline(x=trans_point, color='red', linestyle='--', alpha=0.7)
                ax4.scatter(trans_point, free_energy[trans_point], c='red', s=100, 
                           zorder=10, label='Phase Transition' if i == 0 else "")
            
            # Color different phases
            phase_starts = [0] + [t[0] for t in transitions] + [len(free_energy)-1]
            colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral', 'lightpink']
            
            for i in range(len(phase_starts) - 1):
                start = phase_starts[i]
                end = phase_starts[i + 1]
                ax4.axvspan(start, end, alpha=0.2, 
                           color=colors[i % len(colors)])
            
            if transitions:
                ax4.legend()
        
        ax4.set_title('Energy Phase Transitions')
        ax4.set_xlabel('Time Step')
        ax4.set_ylabel('Free Energy')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(f"{self.plot_dir}/energy_flow_{learner_data.get('experience_level', 'unknown')}.png",
                       dpi=300, bbox_inches='tight')
        plt.close()
        
    def create_variational_landscape(self, learner_data: Dict[str, Any], save_path: Optional[str] = None):
        """Create variational free energy landscape visualization."""
        set_plot_style()
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'Variational Free Energy Landscape - {learner_data.get("experience_level", "").title()}',
                     fontsize=16, fontweight='bold')
        
        free_energy = learner_data.get('free_energy_history', [])
        meta_awareness = learner_data.get('meta_awareness_history', [])
        
        if not free_energy:
            return
            
        # Create parameter spaces for visualization
        precision_range = np.linspace(0.1, 1.0, 50)
        complexity_range = np.linspace(0.01, 0.5, 50)
        P, C = np.meshgrid(precision_range, complexity_range)
        
        # 1. Precision-Complexity Energy Landscape
        ax1 = axes[0, 0]
        # Simplified energy function: higher precision reduces energy, higher complexity increases it
        energy_surface = np.exp(-P) + C * 2
        
        contour = ax1.contour(P, C, energy_surface, levels=20, colors='black', alpha=0.5)
        contourf = ax1.contourf(P, C, energy_surface, levels=20, cmap='viridis', alpha=0.8)
        
        # Mark actual learner parameters
        learner_precision = learner_data.get('precision_weight', 0.5)
        learner_complexity = learner_data.get('complexity_penalty', 0.1)
        ax1.scatter(learner_precision, learner_complexity, c='red', s=200, 
                   marker='*', zorder=10, label='Learner Parameters')
        
        ax1.set_xlabel('Precision Weight')
        ax1.set_ylabel('Complexity Penalty')
        ax1.set_title('Energy Landscape (Precision-Complexity)')
        ax1.legend()
        plt.colorbar(contourf, ax=ax1)
        
        # 2. Temporal energy evolution with confidence bands
        ax2 = axes[0, 1]
        time_steps = np.arange(len(free_energy))
        
        # Calculate confidence bands using running statistics
        window_size = max(5, len(free_energy) // 20)
        energy_mean = []
        energy_std = []
        energy_time = []
        
        for i in range(window_size, len(free_energy), window_size//2):
            window_data = free_energy[max(0, i-window_size):i+window_size]
            energy_mean.append(np.mean(window_data))
            energy_std.append(np.std(window_data))
            energy_time.append(i)
        
        if energy_mean:
            energy_mean = np.array(energy_mean)
            energy_std = np.array(energy_std)
            
            ax2.plot(time_steps, free_energy, 'b-', linewidth=1, alpha=0.6, label='Raw Energy')
            ax2.plot(energy_time, energy_mean, 'r-', linewidth=3, label='Mean Energy')
            ax2.fill_between(energy_time, energy_mean - energy_std, energy_mean + energy_std,
                           alpha=0.3, color='red', label='±1 Std')
            ax2.fill_between(energy_time, energy_mean - 2*energy_std, energy_mean + 2*energy_std,
                           alpha=0.2, color='red', label='±2 Std')
        
        ax2.set_title('Energy Evolution with Uncertainty')
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Free Energy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Energy density distribution
        ax3 = axes[0, 2]
        ax3.hist(free_energy, bins=30, density=True, alpha=0.7, color='skyblue', edgecolor='black')
        
        # Fit and plot distribution
        mu, sigma = np.mean(free_energy), np.std(free_energy)
        x_dist = np.linspace(min(free_energy), max(free_energy), 100)
        y_dist = (1/(sigma * np.sqrt(2*np.pi))) * np.exp(-0.5*((x_dist - mu)/sigma)**2)
        ax3.plot(x_dist, y_dist, 'r-', linewidth=2, label=f'Normal (μ={mu:.3f}, σ={sigma:.3f})')
        
        ax3.axvline(mu, color='red', linestyle='--', alpha=0.8, label='Mean')
        ax3.axvline(mu + sigma, color='orange', linestyle='--', alpha=0.8, label='±1σ')
        ax3.axvline(mu - sigma, color='orange', linestyle='--', alpha=0.8)
        
        ax3.set_title('Energy Distribution')
        ax3.set_xlabel('Free Energy')
        ax3.set_ylabel('Density')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Variational bound tightness
        ax4 = axes[1, 0]
        if meta_awareness:
            # Estimate bound tightness: higher meta-awareness indicates tighter bound
            bound_tightness = [ma * (1 - fe/max(free_energy)) for ma, fe in zip(meta_awareness, free_energy)]
            
            ax4.plot(time_steps, bound_tightness, 'g-', linewidth=2, label='Bound Tightness')
            ax4.fill_between(time_steps, bound_tightness, alpha=0.3, color='green')
            
            # Add trend analysis
            if len(bound_tightness) > 2:
                z = np.polyfit(time_steps, bound_tightness, 1)
                trend = np.poly1d(z)
                ax4.plot(time_steps, trend(time_steps), 'r--', linewidth=2, 
                        label=f'Trend (slope: {z[0]:.4f})')
        
        ax4.set_title('Variational Bound Tightness')
        ax4.set_xlabel('Time Step')
        ax4.set_ylabel('Tightness Index')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Energy minimization trajectory
        ax5 = axes[1, 1]
        if len(free_energy) > 10:
            # Phase space: energy vs energy change
            energy_changes = np.diff(free_energy)
            
            # Color by time progression
            colors = np.arange(len(energy_changes))
            scatter = ax5.scatter(free_energy[1:], energy_changes, c=colors, cmap='plasma',
                                alpha=0.7, s=30)
            
            # Add nullcline (where energy change = 0)
            ax5.axhline(y=0, color='black', linestyle='-', alpha=0.8, linewidth=2, 
                       label='Energy Equilibrium')
            
            # Mark convergence regions
            low_change_indices = [i for i, change in enumerate(energy_changes) 
                                if abs(change) < 0.1 * np.std(energy_changes)]
            if low_change_indices:
                ax5.scatter([free_energy[i+1] for i in low_change_indices],
                          [energy_changes[i] for i in low_change_indices],
                          c='red', s=80, marker='s', alpha=0.8, label='Low Change Regions')
            
            plt.colorbar(scatter, ax=ax5, label='Time Progression')
        
        ax5.set_title('Energy Minimization Phase Space')
        ax5.set_xlabel('Free Energy')
        ax5.set_ylabel('Energy Change')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Multi-scale energy analysis
        ax6 = axes[1, 2]
        # Wavelet-like analysis using different window sizes
        scales = [5, 10, 20, 40]
        colors = ['red', 'orange', 'green', 'blue']
        
        for scale, color in zip(scales, colors):
            if scale < len(free_energy):
                smoothed_energy = []
                smooth_time = []
                
                for i in range(scale, len(free_energy), scale//2):
                    window_data = free_energy[i-scale:i]
                    smoothed_energy.append(np.mean(window_data))
                    smooth_time.append(i - scale//2)
                
                ax6.plot(smooth_time, smoothed_energy, color=color, linewidth=2,
                        label=f'Scale {scale}', alpha=0.8)
        
        ax6.set_title('Multi-scale Energy Analysis')
        ax6.set_xlabel('Time Step')
        ax6.set_ylabel('Smoothed Energy')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(f"{self.plot_dir}/variational_landscape_{learner_data.get('experience_level', 'unknown')}.png",
                       dpi=300, bbox_inches='tight')
        plt.close()
