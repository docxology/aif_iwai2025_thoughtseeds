"""
Advanced dynamics visualization for meditation simulation data.

This module provides sophisticated visualizations focusing on temporal dynamics,
phase space analysis, and dynamic system representations of the meditation process.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
import os

from .plotting import STATE_COLORS, NETWORK_COLORS, THOUGHTSEED_COLORS, set_plot_style


class DynamicsVisualizer:
    """Advanced visualizer for meditation dynamics and phase space analysis."""
    
    def __init__(self, style_theme: str = 'seaborn-v0_8-whitegrid'):
        """Initialize dynamics visualizer."""
        self.style_theme = style_theme
        self.plot_dir = "results_act_inf/plots"
        os.makedirs(self.plot_dir, exist_ok=True)
        
    def create_phase_portrait(self, learner_data: Dict[str, Any], save_path: Optional[str] = None):
        """Create phase portrait of free energy vs meta-awareness dynamics."""
        set_plot_style()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Extract data
        free_energy = learner_data.get('free_energy_history', [])
        meta_awareness = learner_data.get('meta_awareness_history', [])
        state_history = learner_data.get('state_history', [])
        
        if not free_energy or not meta_awareness:
            print("Warning: Missing free energy or meta-awareness data for phase portrait")
            return
            
        # Create phase portrait
        for i, state in enumerate(set(state_history)):
            # Get indices for this state
            state_indices = [j for j, s in enumerate(state_history) if s == state]
            if not state_indices:
                continue
                
            state_fe = [free_energy[j] for j in state_indices]
            state_ma = [meta_awareness[j] for j in state_indices]
            
            ax.scatter(state_fe, state_ma, 
                      c=STATE_COLORS.get(state, '#333333'),
                      label=state.replace('_', ' ').title(),
                      alpha=0.6, s=30)
        
        # Add trajectory arrows
        if len(free_energy) > 1:
            for i in range(0, len(free_energy)-1, max(1, len(free_energy)//20)):
                if i+1 < len(free_energy):
                    ax.annotate('', xy=(free_energy[i+1], meta_awareness[i+1]),
                              xytext=(free_energy[i], meta_awareness[i]),
                              arrowprops=dict(arrowstyle='->', alpha=0.4, lw=0.8))
        
        ax.set_xlabel('Free Energy', fontsize=12)
        ax.set_ylabel('Meta-awareness', fontsize=12)
        ax.set_title(f'Phase Portrait: Free Energy vs Meta-awareness\n{learner_data.get("experience_level", "").title()} Meditator', 
                     fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(f"{self.plot_dir}/phase_portrait_{learner_data.get('experience_level', 'unknown')}.png",
                       dpi=300, bbox_inches='tight')
        plt.close()
        
    def create_network_flow_diagram(self, learner_data: Dict[str, Any], save_path: Optional[str] = None):
        """Create dynamic network flow visualization showing connectivity changes."""
        set_plot_style()
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle(f'Network Activation Flow - {learner_data.get("experience_level", "").title()}', 
                     fontsize=16, fontweight='bold')
        
        network_history = learner_data.get('network_activations_history', [])
        if not network_history:
            print("Warning: No network activation history available")
            return
            
        networks = ['DMN', 'VAN', 'DAN', 'FPN']
        
        # Extract network time series
        network_series = {net: [step.get(net, 0) for step in network_history] for net in networks}
        
        # 1. Network activation over time
        ax1 = axes[0, 0]
        time_steps = np.arange(len(network_history))
        for net in networks:
            ax1.plot(time_steps, network_series[net], 
                    color=NETWORK_COLORS[net], linewidth=2, label=net)
        ax1.set_title('Network Activation Time Series')
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Activation Level')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Cross-correlation matrix
        ax2 = axes[0, 1]
        corr_matrix = np.corrcoef([network_series[net] for net in networks])
        im = ax2.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        ax2.set_xticks(range(len(networks)))
        ax2.set_yticks(range(len(networks)))
        ax2.set_xticklabels(networks)
        ax2.set_yticklabels(networks)
        ax2.set_title('Network Cross-Correlation')
        
        # Add correlation values
        for i in range(len(networks)):
            for j in range(len(networks)):
                ax2.text(j, i, f'{corr_matrix[i, j]:.2f}',
                        ha='center', va='center', fontweight='bold')
        
        plt.colorbar(im, ax=ax2)
        
        # 3. DMN-DAN anticorrelation over time
        ax3 = axes[1, 0]
        window_size = min(20, len(network_history) // 4)
        anticorr_values = []
        
        for i in range(window_size, len(network_history) - window_size):
            window_dmn = network_series['DMN'][i-window_size:i+window_size]
            window_dan = network_series['DAN'][i-window_size:i+window_size]
            corr = np.corrcoef(window_dmn, window_dan)[0, 1]
            anticorr_values.append(-corr if not np.isnan(corr) else 0)
        
        anticorr_time = time_steps[window_size:len(network_history)-window_size]
        ax3.plot(anticorr_time, anticorr_values, color='purple', linewidth=2)
        ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax3.set_title('DMN-DAN Anticorrelation Over Time')
        ax3.set_xlabel('Time Step')
        ax3.set_ylabel('Anticorrelation Strength')
        ax3.grid(True, alpha=0.3)
        
        # 4. Network dominance pie chart (final state)
        ax4 = axes[1, 1]
        final_activations = [network_series[net][-1] for net in networks]
        colors = [NETWORK_COLORS[net] for net in networks]
        wedges, texts, autotexts = ax4.pie(final_activations, labels=networks, colors=colors, 
                                          autopct='%1.1f%%', startangle=90)
        ax4.set_title('Final Network Dominance')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(f"{self.plot_dir}/network_flow_{learner_data.get('experience_level', 'unknown')}.png",
                       dpi=300, bbox_inches='tight')
        plt.close()
        
    def create_thoughtseed_competition_plot(self, learner_data: Dict[str, Any], save_path: Optional[str] = None):
        """Visualize thoughtseed competition dynamics."""
        set_plot_style()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Thoughtseed Competition Dynamics - {learner_data.get("experience_level", "").title()}',
                     fontsize=16, fontweight='bold')
        
        # Get thoughtseed data
        activations_history = learner_data.get('activations_history', [])
        dominant_ts_history = learner_data.get('dominant_ts_history', [])
        thoughtseeds = ['breath_focus', 'equanimity', 'self_reflection', 'pain_discomfort', 'pending_tasks']
        
        if not activations_history:
            print("Warning: No thoughtseed activation history available")
            return
            
        # Convert to time series
        time_steps = np.arange(len(activations_history))
        ts_series = {}
        for i, ts in enumerate(thoughtseeds):
            if i < len(activations_history[0]):
                ts_series[ts] = [step[i] if i < len(step) else 0 for step in activations_history]
            else:
                ts_series[ts] = [0] * len(activations_history)
        
        # 1. Stacked area plot of thoughtseed activations
        ax1 = axes[0, 0]
        bottom = np.zeros(len(time_steps))
        for ts in thoughtseeds:
            if ts in ts_series:
                ax1.fill_between(time_steps, bottom, bottom + ts_series[ts],
                               color=THOUGHTSEED_COLORS.get(ts, '#888888'),
                               alpha=0.7, label=ts.replace('_', ' ').title())
                bottom += ts_series[ts]
        
        ax1.set_title('Thoughtseed Activation Stack')
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Activation Level')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 2. Dominance switching plot
        ax2 = axes[0, 1]
        dominant_mapping = {ts: i for i, ts in enumerate(thoughtseeds)}
        dominant_indices = [dominant_mapping.get(ts, 0) for ts in dominant_ts_history]
        
        ax2.scatter(time_steps, dominant_indices, 
                   c=[THOUGHTSEED_COLORS.get(ts, '#888888') for ts in dominant_ts_history],
                   s=20, alpha=0.8)
        
        # Add transition lines
        for i in range(len(dominant_indices)-1):
            if dominant_indices[i] != dominant_indices[i+1]:
                ax2.plot([time_steps[i], time_steps[i+1]], 
                        [dominant_indices[i], dominant_indices[i+1]], 
                        'k-', alpha=0.3, linewidth=0.5)
        
        ax2.set_yticks(range(len(thoughtseeds)))
        ax2.set_yticklabels([ts.replace('_', ' ').title() for ts in thoughtseeds])
        ax2.set_title('Dominant Thoughtseed Switching')
        ax2.set_xlabel('Time Step')
        ax2.grid(True, alpha=0.3)
        
        # 3. Competition intensity heatmap
        ax3 = axes[1, 0]
        window_size = 10
        competition_matrix = np.zeros((len(thoughtseeds), len(time_steps) // window_size))
        
        for i in range(0, len(time_steps) - window_size, window_size):
            window_data = activations_history[i:i+window_size]
            if window_data:
                avg_activations = np.mean(window_data, axis=0)
                for j, ts in enumerate(thoughtseeds):
                    if j < len(avg_activations):
                        competition_matrix[j, i // window_size] = avg_activations[j]
        
        im = ax3.imshow(competition_matrix, aspect='auto', cmap='viridis', origin='lower')
        ax3.set_yticks(range(len(thoughtseeds)))
        ax3.set_yticklabels([ts.replace('_', ' ').title() for ts in thoughtseeds])
        ax3.set_title('Competition Intensity Heatmap')
        ax3.set_xlabel('Time Window')
        plt.colorbar(im, ax=ax3, label='Activation Level')
        
        # 4. Final dominance distribution
        ax4 = axes[1, 1]
        dominance_counts = {}
        for ts in dominant_ts_history:
            dominance_counts[ts] = dominance_counts.get(ts, 0) + 1
        
        ts_names = list(dominance_counts.keys())
        counts = list(dominance_counts.values())
        colors = [THOUGHTSEED_COLORS.get(ts, '#888888') for ts in ts_names]
        
        bars = ax4.bar(range(len(ts_names)), counts, color=colors, alpha=0.8)
        ax4.set_xticks(range(len(ts_names)))
        ax4.set_xticklabels([ts.replace('_', ' ').title() for ts in ts_names], rotation=45)
        ax4.set_title('Dominance Frequency')
        ax4.set_ylabel('Time Steps Dominant')
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(f"{self.plot_dir}/thoughtseed_competition_{learner_data.get('experience_level', 'unknown')}.png",
                       dpi=300, bbox_inches='tight')
        plt.close()
        
    def create_state_transition_analysis(self, learner_data: Dict[str, Any], save_path: Optional[str] = None):
        """Create detailed state transition analysis visualization."""
        set_plot_style()
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'State Transition Analysis - {learner_data.get("experience_level", "").title()}',
                     fontsize=16, fontweight='bold')
        
        state_history = learner_data.get('state_history', [])
        states = ['breath_control', 'mind_wandering', 'meta_awareness', 'redirect_breath']
        
        if not state_history:
            print("Warning: No state history available")
            return
        
        # 1. State duration distribution
        ax1 = axes[0, 0]
        state_durations = {}
        current_state = state_history[0]
        current_duration = 1
        
        for i in range(1, len(state_history)):
            if state_history[i] == current_state:
                current_duration += 1
            else:
                if current_state not in state_durations:
                    state_durations[current_state] = []
                state_durations[current_state].append(current_duration)
                current_state = state_history[i]
                current_duration = 1
        
        # Add final duration
        if current_state not in state_durations:
            state_durations[current_state] = []
        state_durations[current_state].append(current_duration)
        
        # Box plot of durations
        duration_data = []
        duration_labels = []
        for state in states:
            if state in state_durations:
                duration_data.append(state_durations[state])
                duration_labels.append(state.replace('_', ' ').title())
        
        bp = ax1.boxplot(duration_data, patch_artist=True)
        for patch, state in zip(bp['boxes'], [s for s in states if s in state_durations]):
            patch.set_facecolor(STATE_COLORS.get(state, '#888888'))
            patch.set_alpha(0.7)
        
        ax1.set_xticklabels(duration_labels, rotation=45)
        ax1.set_title('State Duration Distribution')
        ax1.set_ylabel('Duration (timesteps)')
        ax1.grid(True, alpha=0.3)
        
        # 2. Transition probability matrix
        ax2 = axes[0, 1]
        transition_matrix = np.zeros((len(states), len(states)))
        state_to_idx = {state: i for i, state in enumerate(states)}
        
        for i in range(len(state_history) - 1):
            from_state = state_history[i]
            to_state = state_history[i + 1]
            if from_state in state_to_idx and to_state in state_to_idx:
                transition_matrix[state_to_idx[from_state], state_to_idx[to_state]] += 1
        
        # Normalize to probabilities
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        transition_matrix = np.divide(transition_matrix, row_sums, 
                                    out=np.zeros_like(transition_matrix), where=row_sums!=0)
        
        im = ax2.imshow(transition_matrix, cmap='Blues', vmin=0, vmax=1)
        ax2.set_xticks(range(len(states)))
        ax2.set_yticks(range(len(states)))
        ax2.set_xticklabels([s.replace('_', ' ').title() for s in states], rotation=45)
        ax2.set_yticklabels([s.replace('_', ' ').title() for s in states])
        ax2.set_title('State Transition Probabilities')
        ax2.set_xlabel('To State')
        ax2.set_ylabel('From State')
        
        # Add probability values
        for i in range(len(states)):
            for j in range(len(states)):
                ax2.text(j, i, f'{transition_matrix[i, j]:.2f}',
                        ha='center', va='center', fontweight='bold')
        
        plt.colorbar(im, ax=ax2)
        
        # 3. State timeline with transitions
        ax3 = axes[1, 0]
        time_steps = np.arange(len(state_history))
        state_indices = [state_to_idx.get(state, 0) for state in state_history]
        
        # Plot state timeline
        for i, state in enumerate(states):
            state_mask = np.array(state_history) == state
            ax3.scatter(time_steps[state_mask], [i] * sum(state_mask),
                       c=STATE_COLORS.get(state, '#888888'), alpha=0.6, s=8)
        
        ax3.set_yticks(range(len(states)))
        ax3.set_yticklabels([s.replace('_', ' ').title() for s in states])
        ax3.set_xlabel('Time Step')
        ax3.set_title('State Timeline')
        ax3.grid(True, alpha=0.3)
        
        # 4. Transition frequency over time
        ax4 = axes[1, 1]
        window_size = max(10, len(state_history) // 20)
        transition_rates = []
        
        for i in range(window_size, len(state_history) - window_size):
            window_states = state_history[i-window_size:i+window_size]
            transitions = sum(1 for j in range(len(window_states)-1) 
                            if window_states[j] != window_states[j+1])
            transition_rates.append(transitions / (2 * window_size - 1))
        
        transition_time = time_steps[window_size:len(state_history)-window_size]
        ax4.plot(transition_time, transition_rates, 'b-', linewidth=2)
        ax4.set_xlabel('Time Step')
        ax4.set_ylabel('Transition Rate')
        ax4.set_title('Transition Rate Over Time')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(f"{self.plot_dir}/state_transitions_{learner_data.get('experience_level', 'unknown')}.png",
                       dpi=300, bbox_inches='tight')
        plt.close()
        
    def create_comprehensive_dashboard(self, novice_data: Dict[str, Any], expert_data: Dict[str, Any], 
                                     save_path: Optional[str] = None):
        """Create comprehensive comparative dashboard."""
        set_plot_style()
        
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 5, hspace=0.3, wspace=0.3)
        fig.suptitle('Comprehensive Meditation Dynamics Dashboard', fontsize=20, fontweight='bold')
        
        # Helper function to safely get data
        def safe_get(data, key, default=[]):
            return data.get(key, default) if data else default
        
        # 1. Free energy comparison (top row, spans 2 columns)
        ax1 = fig.add_subplot(gs[0, :2])
        nov_fe = safe_get(novice_data, 'free_energy_history')
        exp_fe = safe_get(expert_data, 'free_energy_history')
        
        if nov_fe and exp_fe:
            time_steps = np.arange(min(len(nov_fe), len(exp_fe)))
            ax1.plot(time_steps, nov_fe[:len(time_steps)], 'b-', linewidth=2, label='Novice', alpha=0.8)
            ax1.plot(time_steps, exp_fe[:len(time_steps)], 'r-', linewidth=2, label='Expert', alpha=0.8)
            ax1.fill_between(time_steps, nov_fe[:len(time_steps)], alpha=0.2, color='blue')
            ax1.fill_between(time_steps, exp_fe[:len(time_steps)], alpha=0.2, color='red')
        
        ax1.set_title('Free Energy Evolution Comparison', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Free Energy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Meta-awareness comparison (top row)
        ax2 = fig.add_subplot(gs[0, 2])
        nov_ma = safe_get(novice_data, 'meta_awareness_history')
        exp_ma = safe_get(expert_data, 'meta_awareness_history')
        
        if nov_ma and exp_ma:
            ax2.hist(nov_ma, bins=20, alpha=0.6, label='Novice', color='blue', density=True)
            ax2.hist(exp_ma, bins=20, alpha=0.6, label='Expert', color='red', density=True)
        
        ax2.set_title('Meta-awareness Distribution')
        ax2.set_xlabel('Meta-awareness Level')
        ax2.set_ylabel('Density')
        ax2.legend()
        
        # 3. Network correlation heatmaps (top row)
        for i, (data, title, pos) in enumerate([(novice_data, 'Novice Networks', gs[0, 3]), 
                                               (expert_data, 'Expert Networks', gs[0, 4])]):
            ax = fig.add_subplot(pos)
            net_history = safe_get(data, 'network_activations_history')
            
            if net_history:
                networks = ['DMN', 'VAN', 'DAN', 'FPN']
                net_series = {net: [step.get(net, 0) for step in net_history] for net in networks}
                corr_matrix = np.corrcoef([net_series[net] for net in networks])
                
                im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
                ax.set_xticks(range(len(networks)))
                ax.set_yticks(range(len(networks)))
                ax.set_xticklabels(networks, fontsize=8)
                ax.set_yticklabels(networks, fontsize=8)
                
                # Add correlation values
                for ii in range(len(networks)):
                    for jj in range(len(networks)):
                        ax.text(jj, ii, f'{corr_matrix[ii, jj]:.2f}',
                               ha='center', va='center', fontsize=8)
            
            ax.set_title(title, fontsize=10)
        
        # Second row: Thoughtseed dynamics
        ax3 = fig.add_subplot(gs[1, :3])
        thoughtseeds = ['breath_focus', 'equanimity', 'self_reflection', 'pain_discomfort', 'pending_tasks']
        
        # Combined thoughtseed activation comparison
        for data, label, style in [(novice_data, 'Novice', '-'), (expert_data, 'Expert', '--')]:
            activations = safe_get(data, 'activations_history')
            if activations:
                time_steps = np.arange(len(activations))
                for i, ts in enumerate(thoughtseeds):
                    if i < len(activations[0]):
                        ts_values = [step[i] if i < len(step) else 0 for step in activations]
                        alpha = 0.8 if style == '-' else 0.6
                        ax3.plot(time_steps, ts_values, style, 
                               color=THOUGHTSEED_COLORS.get(ts, '#888888'),
                               alpha=alpha, linewidth=1.5,
                               label=f'{ts} ({label})' if label == 'Novice' else None)
        
        ax3.set_title('Thoughtseed Competition Dynamics')
        ax3.set_xlabel('Time Step')
        ax3.set_ylabel('Activation Level')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax3.grid(True, alpha=0.3)
        
        # State transition comparison (second row)
        for i, (data, title, pos) in enumerate([(novice_data, 'Novice States', gs[1, 3]), 
                                               (expert_data, 'Expert States', gs[1, 4])]):
            ax = fig.add_subplot(pos)
            state_history = safe_get(data, 'state_history')
            
            if state_history:
                states = ['breath_control', 'mind_wandering', 'meta_awareness', 'redirect_breath']
                state_counts = {state: state_history.count(state) for state in states}
                
                colors = [STATE_COLORS.get(state, '#888888') for state in states]
                wedges, texts = ax.pie(state_counts.values(), labels=[s.replace('_', ' ').title() 
                                     for s in state_counts.keys()], 
                                     colors=colors, startangle=90)
                
                for text in texts:
                    text.set_fontsize(8)
            
            ax.set_title(title, fontsize=10)
        
        # Third row: Advanced dynamics
        # Free energy gradient analysis
        ax4 = fig.add_subplot(gs[2, :2])
        for data, label, color in [(novice_data, 'Novice', 'blue'), (expert_data, 'Expert', 'red')]:
            fe_data = safe_get(data, 'free_energy_history')
            if fe_data and len(fe_data) > 1:
                gradients = np.gradient(fe_data)
                time_steps = np.arange(len(gradients))
                ax4.plot(time_steps, gradients, color=color, linewidth=2, label=label, alpha=0.8)
        
        ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax4.set_title('Free Energy Gradient (Learning Rate)')
        ax4.set_xlabel('Time Step')
        ax4.set_ylabel('Energy Gradient')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Network synchronization analysis
        ax5 = fig.add_subplot(gs[2, 2])
        sync_data = []
        labels = []
        
        for data, label in [(novice_data, 'Novice'), (expert_data, 'Expert')]:
            net_history = safe_get(data, 'network_activations_history')
            if net_history:
                networks = ['DMN', 'VAN', 'DAN', 'FPN']
                net_series = np.array([[step.get(net, 0) for net in networks] for step in net_history])
                
                # Calculate synchronization as variance of network correlations
                if net_series.shape[0] > 1:
                    corrs = np.corrcoef(net_series.T)
                    sync_index = 1 / (np.var(corrs[np.triu_indices_from(corrs, k=1)]) + 1e-6)
                    sync_data.append(sync_index)
                    labels.append(label)
        
        if sync_data:
            bars = ax5.bar(labels, sync_data, color=['blue', 'red'], alpha=0.7)
            ax5.set_title('Network Synchronization')
            ax5.set_ylabel('Synchronization Index')
            
            # Add value labels
            for bar, value in zip(bars, sync_data):
                ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(sync_data)*0.02,
                        f'{value:.2f}', ha='center', va='bottom')
        
        # Attention stability comparison
        ax6 = fig.add_subplot(gs[2, 3:])
        stability_metrics = {}
        
        for data, label in [(novice_data, 'Novice'), (expert_data, 'Expert')]:
            activations = safe_get(data, 'activations_history')
            if activations and len(activations) > 0 and len(activations[0]) > 0:
                # Breath focus stability (assuming first thoughtseed is breath_focus)
                breath_series = [step[0] if len(step) > 0 else 0 for step in activations]
                stability = 1 / (np.std(breath_series) + 1e-6)
                stability_metrics[label] = min(stability / 10, 1.0)  # Normalize
        
        if stability_metrics:
            bars = ax6.bar(stability_metrics.keys(), stability_metrics.values(), 
                          color=['blue', 'red'], alpha=0.7)
            ax6.set_title('Attention Stability (Breath Focus)')
            ax6.set_ylabel('Stability Index')
            ax6.set_ylim(0, 1.1)
            
            # Add value labels
            for bar, (label, value) in zip(bars, stability_metrics.items()):
                ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{value:.3f}', ha='center', va='bottom')
        
        # Fourth row: Performance metrics summary
        ax7 = fig.add_subplot(gs[3, :])
        
        # Calculate comprehensive metrics
        metrics = {}
        for data, label in [(novice_data, 'Novice'), (expert_data, 'Expert')]:
            metrics[label] = {}
            
            # Average free energy
            fe_data = safe_get(data, 'free_energy_history')
            metrics[label]['Free Energy'] = np.mean(fe_data) if fe_data else 0
            
            # Average meta-awareness
            ma_data = safe_get(data, 'meta_awareness_history')
            metrics[label]['Meta-awareness'] = np.mean(ma_data) if ma_data else 0
            
            # Transition efficiency
            state_history = safe_get(data, 'state_history')
            if state_history:
                transitions = sum(1 for i in range(len(state_history)-1) 
                                if state_history[i] != state_history[i+1])
                metrics[label]['Transition Rate'] = transitions / len(state_history)
            else:
                metrics[label]['Transition Rate'] = 0
        
        # Create performance comparison radar chart
        if metrics:
            categories = list(metrics['Novice'].keys())
            nov_values = list(metrics['Novice'].values())
            exp_values = list(metrics['Expert'].values()) if 'Expert' in metrics else [0] * len(categories)
            
            # Normalize values for radar chart
            max_vals = [max(nov_values[i], exp_values[i]) for i in range(len(categories))]
            nov_norm = [nov_values[i] / (max_vals[i] + 1e-6) for i in range(len(categories))]
            exp_norm = [exp_values[i] / (max_vals[i] + 1e-6) for i in range(len(categories))]
            
            angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
            nov_norm += [nov_norm[0]]  # Close the polygon
            exp_norm += [exp_norm[0]]
            angles += [angles[0]]
            
            ax7 = plt.subplot(gs[3, :], projection='polar')
            ax7.plot(angles, nov_norm, 'b-', linewidth=2, label='Novice')
            ax7.fill(angles, nov_norm, 'blue', alpha=0.2)
            ax7.plot(angles, exp_norm, 'r-', linewidth=2, label='Expert')
            ax7.fill(angles, exp_norm, 'red', alpha=0.2)
            
            ax7.set_xticks(angles[:-1])
            ax7.set_xticklabels(categories)
            ax7.set_ylim(0, 1)
            ax7.set_title('Performance Metrics Comparison (Normalized)', pad=20)
            ax7.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(f"{self.plot_dir}/comprehensive_dashboard.png", dpi=300, bbox_inches='tight')
        plt.close()
