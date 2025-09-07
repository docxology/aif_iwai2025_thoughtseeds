import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import json
import os, sys
import pickle
from typing import Dict, List, Any
from matplotlib.gridspec import GridSpec
import seaborn as sns
import matplotlib.cm as cm

# Define consistent color schemes
STATE_COLORS = {
    "breath_control": "#2ca02c",    # Green (focused attention)
    "mind_wandering": "#1f77b4",    # Blue (default mode)
    "meta_awareness": "#d62728",    # Red (salience detection)
    "redirect_breath": "#ff7f0e",   # Orange (equanimous attention)
}

NETWORK_COLORS = {
    'DMN': '#CA3542',   # Red
    'VAN': '#B77FB4',   # Pink
    'DAN': '#2C8B4B',   # Green
    'FPN': '#E58429',   # Orange
}

THOUGHTSEED_COLORS = {
    'breath_focus': '#f58231',      # Orange
    'equanimity': '#3cb44b',        # Green
    'self_reflection': '#4363d8',   # Blue
    'pain_discomfort': '#e6194B',   # Red
    'pending_tasks': '#911eb4'      # Purple
}

# State abbreviation mapping
state_abbr = {
    "breath_control": "BF",
    "mind_wandering": "MW",
    "meta_awareness": "MA",
    "redirect_breath": "RA"
}

state_full = {
    "breath_control": "Breath Focus",
    "mind_wandering": "Mind Wandering",
    "meta_awareness": "Meta Awareness",
    "redirect_breath": "Redirect Attention"
}

def set_plot_style():
    """Set consistent style for all plots"""
    plt.style.use('seaborn-v0_8-whitegrid')
    mpl.rcParams['font.family'] = 'DejaVu Sans'
    mpl.rcParams['axes.linewidth'] = 0.5
    mpl.rcParams['grid.linewidth'] = 0.5
    mpl.rcParams['grid.alpha'] = 0.3

def load_json_data(experience_level):
    """Load data from JSON files with proper time series extraction"""
    data = {}
    data_dir = os.path.join("results_act_inf", "data")
    
    # Load thoughtseed parameters (contains time series data)
    ts_path = os.path.join(data_dir, f"thoughtseed_params_{experience_level}.json")
    if not os.path.exists(ts_path):
        print(f"ERROR: Required file not found: {ts_path}")
        return None
        
    with open(ts_path, 'r') as f:
        ts_data = json.load(f)
        # Keep the nested structure instead of flattening
        data['thoughtseed_params'] = ts_data
        
        # Extract time series data if available
        if "time_series" in ts_data:
            for key, value in ts_data["time_series"].items():
                data[key] = value
            
            
            # Add timesteps info
            if "state_history" in data:
                data["timesteps"] = len(data["state_history"])
    
    # Load active inference parameters
    ai_path = os.path.join(data_dir, f"active_inference_params_{experience_level}.json")
    if not os.path.exists(ai_path):
        print(f"ERROR: Required file not found: {ai_path}")
        return None
        
    with open(ai_path, 'r') as f:
        ai_data = json.load(f)
        # Keep the nested structure
        data['active_inference_params'] = ai_data
    
    # Add experience level to data
    data["experience_level"] = experience_level
    
    return data

def plot_network_radar(novice_data, expert_data, save_path=None):
    """
    Create radar plots showing network activation patterns for each state
    with expert vs novice comparison in a 2×2 grid.
    """
    # Get data
    nov_networks = novice_data['thoughtseed_params']['network_activations_by_state']
    exp_networks = expert_data['thoughtseed_params']['network_activations_by_state']
    
    # States to plot
    states = ["breath_control", "mind_wandering", "meta_awareness", "redirect_breath"]
    networks = ['DMN', 'VAN', 'DAN', 'FPN']
    
    # Create figure with subplots
    fig = plt.figure(figsize=(14, 12))
    fig.suptitle('Network Activation Profiles in Vipassana Meditation States', fontsize=16, fontweight='bold', y=0.98)
    
    # Set angles for radar plot
    angles = np.linspace(0, 2*np.pi, len(networks), endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon
    
    # Create subplots in a 2×2 grid
    for i, state in enumerate(states):
        # Create subplot
        ax = fig.add_subplot(2, 2, i+1, polar=True)
        
        # Extract values and close the polygon
        nov_values = [nov_networks[state][net] for net in networks]
        nov_values += [nov_values[0]]
        
        exp_values = [exp_networks[state][net] for net in networks]
        exp_values += [exp_values[0]]
        
        # Plot both datasets
        ax.plot(angles, nov_values, color=STATE_COLORS[state], linewidth=2, linestyle='--', label="Novice")
        ax.fill(angles, nov_values, color=STATE_COLORS[state], alpha=0.2)

        ax.plot(angles, exp_values, color=STATE_COLORS[state], linewidth=2, label="Expert")
        ax.fill(angles, exp_values, color=STATE_COLORS[state], alpha=0.1)
        
        # Set labels and ticks
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(networks, fontsize=10, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.set_title(f"{state.replace('_', ' ').title()}", fontsize=14, pad=15)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add legend for the first subplot only
        if i == 0:
            pass
    # Add a comprehensive legend in the center
    labels = ["Expert", "Novice"]
    handles = [
        plt.Line2D([0], [0], color='black', linewidth=2, label=labels[0]),          # Solid for Expert
        plt.Line2D([0], [0], color='black', linewidth=2, linestyle='--', label=labels[1])  # Dashed for Novice
    ]
    fig.legend(handles=handles, labels=labels, loc='upper center', 
              bbox_to_anchor=(0.5, 0.1), ncol=2, fontsize=12)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.90, bottom=0.15)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plot_dir = os.path.join("results_act_inf", "plots")
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, "network_radar_comparison.png"), dpi=300, bbox_inches='tight')
    
    plt.close()

def plot_free_energy_comparison(novice_data, expert_data, save_path=None):
    """
    Create bar chart comparing free energy across states
    between novice and expert meditators.
    """
    # Extract free energy data
    nov_fe = novice_data['active_inference_params']['average_free_energy_by_state']
    exp_fe = expert_data['active_inference_params']['average_free_energy_by_state']
    
    # States to plot
    states = ["breath_control", "mind_wandering", "meta_awareness", "redirect_breath"]
    
    # Create figure
    fig, ax = plt.figure(figsize=(10, 6)), plt.gca()
    
    # Bar positions
    x = np.arange(len(states))
    width = 0.35
    
    # Create bars
    nov_bars = ax.bar(x - width/2, [nov_fe[state] for state in states], width, 
                     label='Novice', color=[STATE_COLORS[state] for state in states],
                     alpha=0.7, edgecolor='black', linewidth=1)
    
    exp_bars = ax.bar(x + width/2, [exp_fe[state] for state in states], width,
                     label='Expert', color=[STATE_COLORS[state] for state in states],
                     alpha=0.4, edgecolor='black', linewidth=1, hatch='//')
    
    # Add labels, title and legend
    ax.set_ylabel('Free Energy', fontsize=12, fontweight='bold')
    ax.set_title('Free Energy Comparison Across Meditation States', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([state.replace('_', ' ').title() for state in states], fontsize=11)
    ax.legend(fontsize=11)
    
    # Add value labels on the bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(rect.get_x() + rect.get_width()/2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
    
    autolabel(nov_bars)
    autolabel(exp_bars)
    
    # Add grid lines for better readability
    ax.grid(True, axis='y', linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plot_dir = os.path.join("results_act_inf", "plots")
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, "free_energy_comparison.png"), dpi=300, bbox_inches='tight')
    
    plt.close()

def plot_hierarchy(data, save_path=None):
    """
    Create 3-level hierarchical visualization showing:
    1. Meta-awareness level
    2. Dominant thoughtseed
    3. Network activations
    
    """
    # Check for required data
    required_fields = ['state_history', 'meta_awareness_history', 'network_activations_history', 'dominant_ts_history']
    for field in required_fields:
        if field not in data:
            print(f"ERROR: Required data '{field}' missing for hierarchy plot")
            return
    
    time_steps = np.arange(data['timesteps'])
    
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(3, 1, height_ratios=[1, 1, 1.5], figure=fig)
    
    # 1. Level 3: Meta-awareness
    ax1 = fig.add_subplot(gs[0])
    meta_awareness = data['meta_awareness_history']
    
    # Smooth the data for better visualization
    smoothed_meta = np.zeros_like(meta_awareness)
    alpha = 0.3
    smoothed_meta[0] = meta_awareness[0]
    for j in range(1, len(meta_awareness)):
        smoothed_meta[j] = (1 - alpha) * smoothed_meta[j-1] + alpha * meta_awareness[j]
    
    ax1.plot(time_steps, smoothed_meta, color='#4363d8', linewidth=2)
    ax1.fill_between(time_steps, smoothed_meta, alpha=0.2, color='#4363d8')
    ax1.set_ylabel('Meta-Awareness', fontsize=12)
    ax1.set_title('Level 3: Metacognition', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, axis='y', linestyle='--', alpha=0.5)
    
    # 2. Level 2: Dominant Thoughtseed
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    thoughtseeds = ['breath_focus', 'equanimity', 'self_reflection', 'pain_discomfort', 'pending_tasks']
    ts_mapping = {ts: i for i, ts in enumerate(thoughtseeds)}
    
    # Create categorical scatter plot
    for i, ts in enumerate(data['dominant_ts_history']):
        ax2.scatter(i, ts_mapping[ts], color=THOUGHTSEED_COLORS[ts], s=25, 
                   edgecolors='white', linewidth=0.5, alpha=0.8)
    
    # Connect dots with thin lines
    prev_ts = data['dominant_ts_history'][0]
    prev_y = ts_mapping[prev_ts]
    for i in range(1, len(data['dominant_ts_history'])):
        curr_ts = data['dominant_ts_history'][i]
        curr_y = ts_mapping[curr_ts]
        if curr_ts != prev_ts:
            ax2.plot([i-1, i], [prev_y, curr_y], color='#aaaaaa', 
                    linestyle='-', linewidth=0.5, alpha=0.4)
        prev_ts = curr_ts
        prev_y = curr_y
    
    ax2.set_yticks(range(len(thoughtseeds)))
    ax2.set_yticklabels(thoughtseeds)
    ax2.invert_yaxis()
    ax2.set_ylabel('Dominant Thoughtseed', fontsize=12)
    ax2.set_title('Level 2: Dominant Thoughtseed', fontsize=14, fontweight='bold', pad=-15)
    ax2.grid(True, axis='y', linestyle='--', alpha=0.5)
    
    # 3. Level 1: Network Activations
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    
    for net in ['DMN', 'VAN', 'DAN', 'FPN']:
        # Extract data for this network
        net_acts = [n[net] for n in data['network_activations_history']]
        
        # Smooth the data
        smoothed_acts = np.zeros_like(net_acts)
        alpha = 0.3
        smoothed_acts[0] = net_acts[0]
        for j in range(1, len(net_acts)):
            smoothed_acts[j] = (1 - alpha) * smoothed_acts[j-1] + alpha * net_acts[j]
        
        ax3.plot(time_steps, smoothed_acts, label=net, color=NETWORK_COLORS[net], linewidth=2)
    
    # Highlight state transitions across all plots
    prev_state = None
    state_boundaries = []
    
    for i, state in enumerate(data['state_history']):
        if state != prev_state:
            state_boundaries.append(i)
            ax1.axvline(x=i, color='#bbbbbb', linestyle='--', alpha=0.5, zorder=0)
            ax2.axvline(x=i, color='#bbbbbb', linestyle='--', alpha=0.5, zorder=0)
            ax3.axvline(x=i, color='#bbbbbb', linestyle='--', alpha=0.5, zorder=0)
            
            # Add state label to top plot (ax1) instead of bottom plot
            ax1.text(i, -0.05, state_abbr[state], 
                rotation=90, fontsize=9, color=STATE_COLORS[state],
                transform=ax1.get_xaxis_transform(), ha='center', va='top')
            
            prev_state = state
            
        # Add state legend to explain abbreviations
        state_legend_elements = [
            plt.Line2D([0], [0], color=STATE_COLORS[state], lw=4, label=f"{abbr}: {state_full[state]}")
            for state, abbr in state_abbr.items()
        ]
        
        # Create a separate legend for state abbreviations below the plot
        state_legend = fig.legend(handles=state_legend_elements, loc='lower center', 
                                fontsize=10, frameon=True, ncol=4, bbox_to_anchor=(0.5, 0.01))
    
    ax3.set_xlabel('Timestep', fontsize=12)
    ax3.set_ylabel('Network Activation', fontsize=12)
    ax3.set_title('Level 1: Network Dynamics', fontsize=14, fontweight='bold', pad =-25)
    ax3.set_ylim(0, 1.05)
    ax3.grid(True, linestyle='--', alpha=0.5)
    
    # Create a more elegant legend
    ax3.legend(loc='upper right', framealpha=0.9, fancybox=True, fontsize=10)
    
    # Add overall title
    fig.suptitle(f'Hierarchical Dynamics of {data["experience_level"].title()} Meditation', 
               fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.25, bottom=0.12) 
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plot_dir = os.path.join("results_act_inf", "plots")
        os.makedirs(plot_dir, exist_ok=True)
        experience = data.get('experience_level', 'default')
        plt.savefig(os.path.join(plot_dir, f"hierarchy_{experience}.png"), 
                dpi=300, bbox_inches='tight')
    
    plt.close()

def plot_time_series(novice_data, expert_data, save_path=None):
    """
    Create time series visualization showing network dynamics and free energy
    for both novice and expert meditators side-by-side.
    """
    # Check for required data
    required_fields = ['state_history', 'network_activations_history', 'free_energy_history']
    for field in required_fields:
        if field not in novice_data or field not in expert_data:
            print(f"ERROR: Required data '{field}' missing for time series plot")
            return
    
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.5, 1])
    
    # Process both experience levels
    axes = []
        
    for col, (level, data) in enumerate([('novice', novice_data), ('expert', expert_data)]):
        time_steps = np.arange(data['timesteps'])
        
        # 1. Network activations (top row)
        ax1 = fig.add_subplot(gs[0, col])
        
        for net in ['DMN', 'VAN', 'DAN', 'FPN']:
            # Extract data for this network
            net_acts = [n[net] for n in data['network_activations_history']]
            
            ax1.plot(time_steps, net_acts, label=f"{net}", color=NETWORK_COLORS[net], linewidth=1.5)
        
        # Add state boundaries
        prev_state = None
        for i, state in enumerate(data['state_history']):
            if state != prev_state:
                ax1.axvline(x=i, color='gray', linestyle='--', alpha=0.5)
                # Add abbreviated state label
                ax1.text(i, 1.15, state_abbr[state], 
                        rotation=90, fontsize=9, color=STATE_COLORS[state],
                        transform=ax1.get_xaxis_transform(), ha='center')
                prev_state = state
        
        ax1.set_ylim(0, 1.1)
        ax1.set_title(f"Network Activations ({level.title()})", fontsize=14, fontweight='bold')
        ax1.set_xlabel('Timestep', fontsize=12)
        ax1.set_ylabel('Activation', fontsize=12)
        ax1.legend(loc='upper right', framealpha=0.9, fancybox=True, fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # 2. Free energy (bottom row)
        ax2 = fig.add_subplot(gs[1, col], sharex=ax1)
        ax2.plot(time_steps, data['free_energy_history'], color='#E74C3C', label="Free Energy", linewidth=2)
        
        # Add state boundaries (matching top plot)
        prev_state = None
        for i, state in enumerate(data['state_history']):
            if state != prev_state:
                ax2.axvline(x=i, color='gray', linestyle='--', alpha=0.5)
                prev_state = state
        
        # Shade background by state for better visualization
        prev_state = data['state_history'][0]
        start_idx = 0
        
        for i, state in enumerate(data['state_history']):
            if state != prev_state or i == len(data['state_history'])-1:
                # Shade the region
                ax2.axvspan(start_idx, i, alpha=0.1, color=STATE_COLORS[prev_state])
                start_idx = i
                prev_state = state
        
        ax2.set_title(f"Free Energy ({level.title()})", fontsize=14, fontweight='bold')
        ax2.set_xlabel('Timestep', fontsize=12)
        ax2.set_ylabel('Free Energy', fontsize=12)
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        # Store axes for later adjustment
        axes.append(ax2)
    
    # Ensure the same y-axis range for free energy plots
    # Find the global min and max across both plots
    y_min = min(min(novice_data['free_energy_history']), min(expert_data['free_energy_history']))
    y_max = max(max(novice_data['free_energy_history']), max(expert_data['free_energy_history']))
    
    # Add 10% padding
    y_range = y_max - y_min
    y_min -= y_range * 0.1
    y_max += y_range * 0.1
    
    # Set the same limits for both free energy plots
    for ax in axes:
        ax.set_ylim(y_min, y_max)
    
    # Add state legend to explain abbreviations
    state_legend_elements = [
        plt.Line2D([0], [0], color=STATE_COLORS[state], lw=4, label=f"{abbr}: {state_full[state]}")
        for state, abbr in state_abbr.items()
    ]
    
    fig.suptitle('Temporal Dynamics: Network Activation and Free Energy', 
               fontsize=16, fontweight='bold', y=0.98)
        
    # Create a separate legend for state abbreviations below the plot
    state_legend = fig.legend(handles=state_legend_elements, loc='upper center', 
                            fontsize=10, frameon=True, ncol=4, 
                            bbox_to_anchor=(0.5, 0.93))
    
    plt.tight_layout()
    
    plt.subplots_adjust(top=0.92, wspace=0.15, bottom=0.12)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plot_dir = os.path.join("results_act_inf", "plots")
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, "time_series_comparison.png"), dpi=300, bbox_inches='tight')
    
    plt.close()
    
def generate_all_plots():
    """
    Generate all plots from saved JSON files without requiring
    the original model instance.
    """
    # Set consistent plot style
    set_plot_style()
    
    try:
        # Load data from JSON files
        novice_json = load_json_data('novice')
        expert_json = load_json_data('expert')
        
        if novice_json and expert_json:
            # Add experience level to the JSON data
            novice_json['experience_level'] = 'novice'
            expert_json['experience_level'] = 'expert'
            
            # Create network radar plot
            plot_network_radar(novice_json, expert_json)
            
            # Create free energy comparison
            plot_free_energy_comparison(novice_json, expert_json)
            
            # Create hierarchy plots for each experience level
            plot_hierarchy(novice_json)
            plot_hierarchy(expert_json)
            
            # Create time series comparison
            plot_time_series(novice_json, expert_json)
            
            print("Generated all plots from JSON data.")
            return True
        else:
            print("Error: Could not load JSON data files.")
            return False
            
    except Exception as e:
        print(f"Error generating plots: {str(e)}")
        return False

if __name__ == "__main__":
    # When run directly, generate plots from saved data
    generate_all_plots()