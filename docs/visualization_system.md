# Visualization System Documentation

## Overview

This document provides comprehensive technical details on the visualization and plotting system in `act_inf_plots.py`. The system generates publication-quality visualizations for analyzing meditation simulation data, supporting comparative analysis between novice and expert practitioners across multiple representation formats.

---

## Core Visualization Architecture

### Module Structure
- **File**: `act_inf_plots.py`
- **Primary Approach**: JSON-based plotting from saved simulation data
- **Key Features**: Comparative analysis, multiple plot types, publication-quality output
- **Dependencies**: matplotlib, seaborn, numpy, json

### Color Scheme System

```python
# Consistent color schemes for different data types
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
```

**Design Principles**:
- **Semantic Colors**: Colors reflect functional meanings (green for focus, blue for wandering)
- **Accessibility**: High contrast ratios for readability
- **Consistency**: Same entities always use the same colors across all plots
- **Publication Ready**: Colors suitable for both screen and print

---

## Data Loading System

### JSON Data Integration

```python
def load_json_data(experience_level):
    """Load data from JSON files with proper time series extraction"""
    data = {}
    data_dir = os.path.join("results_act_inf", "data")
    
    # Load thoughtseed parameters (contains time series data)
    ts_path = os.path.join(data_dir, f"thoughtseed_params_{experience_level}.json")
    with open(ts_path, 'r') as f:
        ts_data = json.load(f)
        data['thoughtseed_params'] = ts_data
        
        # Extract time series data if available
        if "time_series" in ts_data:
            for key, value in ts_data["time_series"].items():
                data[key] = value
            
            if "state_history" in data:
                data["timesteps"] = len(data["state_history"])
    
    # Load active inference parameters
    ai_path = os.path.join(data_dir, f"active_inference_params_{experience_level}.json")
    with open(ai_path, 'r') as f:
        ai_data = json.load(f)
        data['active_inference_params'] = ai_data
    
    data["experience_level"] = experience_level
    return data
```

**Key Features**:
- **Decoupled Analysis**: Plots generated independently from simulation runs
- **Time Series Extraction**: Automatic extraction of temporal data
- **Multi-File Integration**: Combines thoughtseed and active inference data
- **Error Handling**: Graceful handling of missing files

### Data Structure Validation

The system validates required data fields before plotting:

```python
required_fields = ['state_history', 'meta_awareness_history', 'network_activations_history']
for field in required_fields:
    if field not in data:
        print(f"ERROR: Required data '{field}' missing for hierarchy plot")
        return
```

---

## Plot Types and Implementation

### 1. Network Radar Plots

#### Purpose and Design
Comparative radar charts showing network activation profiles across meditation states for expert vs novice comparison.

```python
def plot_network_radar(novice_data, expert_data, save_path=None):
    """
    Create radar plots showing network activation patterns for each state
    with expert vs novice comparison in a 2×2 grid.
    """
```

#### Implementation Details

```python
# Create figure with subplots
fig = plt.figure(figsize=(14, 12))
fig.suptitle('Network Activation Profiles in Vipassana Meditation States', 
             fontsize=16, fontweight='bold', y=0.98)

# Set angles for radar plot
angles = np.linspace(0, 2*np.pi, len(networks), endpoint=False).tolist()
angles += angles[:1]  # Close the polygon

# Create subplots in a 2×2 grid
for i, state in enumerate(states):
    ax = fig.add_subplot(2, 2, i+1, polar=True)
    
    # Extract values and close the polygon
    nov_values = [nov_networks[state][net] for net in networks]
    nov_values += [nov_values[0]]
    
    exp_values = [exp_networks[state][net] for net in networks]
    exp_values += [exp_values[0]]
    
    # Plot both datasets
    ax.plot(angles, nov_values, color=STATE_COLORS[state], 
            linewidth=2, linestyle='--', label="Novice")
    ax.fill(angles, nov_values, color=STATE_COLORS[state], alpha=0.2)

    ax.plot(angles, exp_values, color=STATE_COLORS[state], 
            linewidth=2, label="Expert")
    ax.fill(angles, exp_values, color=STATE_COLORS[state], alpha=0.1)
```

**Key Features**:
- **2×2 Grid Layout**: All four meditation states displayed simultaneously
- **Dual Visualization**: Novice (dashed) vs Expert (solid) lines
- **Polar Coordinates**: Natural representation for multi-dimensional network data
- **Semantic Coloring**: Each state uses its designated color
- **Fill Areas**: Semi-transparent fills show activation ranges

#### Typical Results Patterns

| State | DMN Pattern | DAN Pattern | VAN Pattern | FPN Pattern |
|-------|-------------|-------------|-------------|-------------|
| **Breath Control** | Expert < Novice | Expert > Novice | Similar | Expert > Novice |
| **Mind Wandering** | Expert < Novice | Expert > Novice | Similar | Expert > Novice |
| **Meta Awareness** | Expert ≤ Novice | Expert > Novice | Expert > Novice | Expert > Novice |
| **Redirect Breath** | Expert < Novice | Expert > Novice | Similar | Expert > Novice |

### 2. Free Energy Comparison

#### Bar Chart Implementation

```python
def plot_free_energy_comparison(novice_data, expert_data, save_path=None):
    """
    Create bar chart comparing free energy across states
    between novice and expert meditators.
    """
    # Extract free energy data
    nov_fe = novice_data['active_inference_params']['average_free_energy_by_state']
    exp_fe = expert_data['active_inference_params']['average_free_energy_by_state']
    
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
```

**Visualization Features**:
- **Side-by-Side Bars**: Direct comparison between experience levels
- **Value Labels**: Automatic labeling of exact free energy values
- **State Colors**: Consistent color mapping across states
- **Pattern Differentiation**: Hatching distinguishes expert bars
- **Grid Lines**: Background grid for easier value reading

#### Auto-Labeling Function

```python
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                   xy=(rect.get_x() + rect.get_width()/2, height),
                   xytext=(0, 3),  # 3 points vertical offset
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=9)
```

### 3. Hierarchical Dynamics Visualization

#### Three-Level Architecture Display

```python
def plot_hierarchy(data, save_path=None):
    """
    Create 3-level hierarchical visualization showing:
    1. Meta-awareness level
    2. Dominant thoughtseed
    3. Network activations
    """
```

#### Implementation Structure

```python
fig = plt.figure(figsize=(14, 10))
gs = GridSpec(3, 1, height_ratios=[1, 1, 1.5], figure=fig)

# Level 3: Meta-awareness
ax1 = fig.add_subplot(gs[0])
smoothed_meta = np.zeros_like(meta_awareness)
alpha = 0.3  # Smoothing factor
smoothed_meta[0] = meta_awareness[0]
for j in range(1, len(meta_awareness)):
    smoothed_meta[j] = (1 - alpha) * smoothed_meta[j-1] + alpha * meta_awareness[j]

ax1.plot(time_steps, smoothed_meta, color='#4363d8', linewidth=2)
ax1.fill_between(time_steps, smoothed_meta, alpha=0.2, color='#4363d8')

# Level 2: Dominant Thoughtseed
ax2 = fig.add_subplot(gs[1], sharex=ax1)
thoughtseeds = ['breath_focus', 'equanimity', 'self_reflection', 'pain_discomfort', 'pending_tasks']
ts_mapping = {ts: i for i, ts in enumerate(thoughtseeds)}

for i, ts in enumerate(data['dominant_ts_history']):
    ax2.scatter(i, ts_mapping[ts], color=THOUGHTSEED_COLORS[ts], s=25, 
               edgecolors='white', linewidth=0.5, alpha=0.8)

# Level 1: Network Activations
ax3 = fig.add_subplot(gs[2], sharex=ax1)
for net in ['DMN', 'VAN', 'DAN', 'FPN']:
    net_acts = [n[net] for n in data['network_activations_history']]
    # Apply smoothing
    smoothed_acts = np.zeros_like(net_acts)
    for j in range(1, len(net_acts)):
        smoothed_acts[j] = (1 - alpha) * smoothed_acts[j-1] + alpha * net_acts[j]
    
    ax3.plot(time_steps, smoothed_acts, label=net, color=NETWORK_COLORS[net], linewidth=2)
```

**Hierarchical Features**:
- **Level 3 (Top)**: Metacognitive control (meta-awareness)
- **Level 2 (Middle)**: Thoughtseed competition (dominant patterns)
- **Level 1 (Bottom)**: Network substrate (neural dynamics)
- **Temporal Smoothing**: Applied to reduce noise in visualization
- **State Transitions**: Vertical lines mark state boundaries
- **Shared X-Axis**: Time alignment across all levels

#### State Boundary Marking

```python
# Highlight state transitions across all plots
for i, state in enumerate(data['state_history']):
    if state != prev_state:
        ax1.axvline(x=i, color='#bbbbbb', linestyle='--', alpha=0.5, zorder=0)
        ax2.axvline(x=i, color='#bbbbbb', linestyle='--', alpha=0.5, zorder=0)
        ax3.axvline(x=i, color='#bbbbbb', linestyle='--', alpha=0.5, zorder=0)
        
        # Add state label abbreviation
        ax1.text(i, -0.05, state_abbr[state], 
            rotation=90, fontsize=9, color=STATE_COLORS[state],
            transform=ax1.get_xaxis_transform(), ha='center', va='top')
```

### 4. Time Series Comparison

#### Side-by-Side Temporal Analysis

```python
def plot_time_series(novice_data, expert_data, save_path=None):
    """
    Create time series visualization showing network dynamics and free energy
    for both novice and expert meditators side-by-side.
    """
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.5, 1])
```

#### Dual-Panel Layout

```python
for col, (level, data) in enumerate([('novice', novice_data), ('expert', expert_data)]):
    time_steps = np.arange(data['timesteps'])
    
    # Network activations (top row)
    ax1 = fig.add_subplot(gs[0, col])
    for net in ['DMN', 'VAN', 'DAN', 'FPN']:
        net_acts = [n[net] for n in data['network_activations_history']]
        ax1.plot(time_steps, net_acts, label=f"{net}", 
                color=NETWORK_COLORS[net], linewidth=1.5)
    
    # Free energy (bottom row)
    ax2 = fig.add_subplot(gs[1, col], sharex=ax1)
    ax2.plot(time_steps, data['free_energy_history'], 
            color='#E74C3C', label="Free Energy", linewidth=2)
```

**Comparison Features**:
- **Side-by-Side Layout**: Direct visual comparison
- **Shared Y-Axis Scaling**: Normalized ranges for fair comparison
- **State Background Shading**: Color-coded state regions
- **Synchronized Time Axes**: Aligned temporal progression

---

## Style and Formatting System

### Consistent Plot Styling

```python
def set_plot_style():
    """Set consistent style for all plots"""
    plt.style.use('seaborn-v0_8-whitegrid')
    mpl.rcParams['font.family'] = 'DejaVu Sans'
    mpl.rcParams['axes.linewidth'] = 0.5
    mpl.rcParams['grid.linewidth'] = 0.5
    mpl.rcParams['grid.alpha'] = 0.3
```

### State Abbreviation System

```python
state_abbr = {
    "breath_control": "BF",      # Breath Focus
    "mind_wandering": "MW",      # Mind Wandering
    "meta_awareness": "MA",      # Meta Awareness
    "redirect_breath": "RA"      # Redirect Attention
}

state_full = {
    "breath_control": "Breath Focus",
    "mind_wandering": "Mind Wandering",
    "meta_awareness": "Meta Awareness",
    "redirect_breath": "Redirect Attention"
}
```

### Legend and Annotation System

```python
# Comprehensive state legend
state_legend_elements = [
    plt.Line2D([0], [0], color=STATE_COLORS[state], lw=4, 
              label=f"{abbr}: {state_full[state]}")
    for state, abbr in state_abbr.items()
]

fig.legend(handles=state_legend_elements, loc='upper center', 
          fontsize=10, frameon=True, ncol=4, bbox_to_anchor=(0.5, 0.93))
```

---

## Data Processing and Smoothing

### Temporal Smoothing Algorithm

```python
# Exponential smoothing for cleaner visualization
smoothed_acts = np.zeros_like(net_acts)
alpha = 0.3  # Smoothing parameter
smoothed_acts[0] = net_acts[0]
for j in range(1, len(net_acts)):
    smoothed_acts[j] = (1 - alpha) * smoothed_acts[j-1] + alpha * net_acts[j]
```

**Smoothing Benefits**:
- **Noise Reduction**: Removes high-frequency noise while preserving trends
- **Visual Clarity**: Cleaner lines for publication-quality figures
- **Pattern Enhancement**: Makes underlying patterns more visible
- **Configurable**: Alpha parameter allows adjustment of smoothing strength

### State-Based Background Shading

```python
# Shade background by state for better visualization
prev_state = data['state_history'][0]
start_idx = 0

for i, state in enumerate(data['state_history']):
    if state != prev_state or i == len(data['state_history'])-1:
        # Shade the region
        ax2.axvspan(start_idx, i, alpha=0.1, color=STATE_COLORS[prev_state])
        start_idx = i
        prev_state = state
```

---

## Output Management System

### File Saving and Organization

```python
if save_path:
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
else:
    plot_dir = os.path.join("results_act_inf", "plots")
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, "network_radar_comparison.png"), 
            dpi=300, bbox_inches='tight')
```

**Output Features**:
- **High Resolution**: 300 DPI for publication quality
- **Tight Bounding**: Automatic cropping of whitespace
- **Organized Storage**: Systematic file organization in plots directory
- **Flexible Paths**: Support for custom save locations

### Generated Plot Files

| Plot File | Content | Comparison Type |
|-----------|---------|-----------------|
| `network_radar_comparison.png` | Network activation patterns | Expert vs Novice |
| `free_energy_comparison.png` | Free energy by state | Expert vs Novice |
| `hierarchy_novice.png` | Hierarchical dynamics | Novice only |
| `hierarchy_expert.png` | Hierarchical dynamics | Expert only |
| `time_series_comparison.png` | Temporal dynamics | Side-by-side |

---

## Master Plot Generation

### Orchestrated Plotting System

```python
def generate_all_plots():
    """
    Generate all plots from saved JSON files without requiring
    the original model instance.
    """
    set_plot_style()
    
    try:
        # Load data from JSON files
        novice_json = load_json_data('novice')
        expert_json = load_json_data('expert')
        
        if novice_json and expert_json:
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
```

**Orchestration Features**:
- **Single Command**: Generate all plots with one function call
- **Error Handling**: Graceful failure handling with informative messages
- **Independence**: Operates entirely from saved JSON data
- **Validation**: Checks data integrity before plotting

---

## Advanced Visualization Features

### Responsive Y-Axis Scaling

```python
# Ensure the same y-axis range for free energy plots
y_min = min(min(novice_data['free_energy_history']), min(expert_data['free_energy_history']))
y_max = max(max(novice_data['free_energy_history']), max(expert_data['free_energy_history']))

# Add 10% padding
y_range = y_max - y_min
y_min -= y_range * 0.1
y_max += y_range * 0.1

# Set the same limits for both plots
for ax in axes:
    ax.set_ylim(y_min, y_max)
```

### Thoughtseed Scatter Visualization

```python
# Create categorical scatter plot for dominant thoughtseeds
for i, ts in enumerate(data['dominant_ts_history']):
    ax2.scatter(i, ts_mapping[ts], color=THOUGHTSEED_COLORS[ts], s=25, 
               edgecolors='white', linewidth=0.5, alpha=0.8)

# Connect dots with thin lines for transition visualization
for i in range(1, len(data['dominant_ts_history'])):
    curr_ts = data['dominant_ts_history'][i]
    curr_y = ts_mapping[curr_ts]
    if curr_ts != prev_ts:
        ax2.plot([i-1, i], [prev_y, curr_y], color='#aaaaaa', 
                linestyle='-', linewidth=0.5, alpha=0.4)
```

### Interactive Elements Preparation

The visualization system is designed to support future interactive features:

```python
# Hover information preparation
hover_data = {
    'timestep': time_steps,
    'free_energy': data['free_energy_history'],
    'dominant_thoughtseed': data['dominant_ts_history'],
    'meta_awareness': data['meta_awareness_history']
}
```

---

## Quality Assurance and Validation

### Plot Data Validation

```python
# Validate required data fields
required_fields = ['state_history', 'network_activations_history', 'free_energy_history']
for field in required_fields:
    if field not in novice_data or field not in expert_data:
        print(f"ERROR: Required data '{field}' missing for time series plot")
        return
```

### Visual Quality Checks

```python
# Ensure reasonable value ranges
assert np.all(np.array(data['free_energy_history']) >= 0), "Negative free energy detected"
assert len(data['state_history']) == data['timesteps'], "State history length mismatch"
```

### Accessibility Features

- **High Contrast**: Color schemes tested for accessibility
- **Alternative Patterns**: Hatching and line styles for colorblind users
- **Clear Labels**: Large, readable fonts and labels
- **Legend Clarity**: Comprehensive legends explaining all visual elements

---

## Performance Optimization

### Efficient Data Loading

```python
# Load data only once and reuse across plots
def generate_all_plots():
    novice_json = load_json_data('novice')  # Single load
    expert_json = load_json_data('expert')  # Single load
    
    # Pass data to all plotting functions
    plot_network_radar(novice_json, expert_json)
    plot_free_energy_comparison(novice_json, expert_json)
```

### Memory Management

```python
# Close figures to prevent memory leaks
plt.close()  # Called after each plot function
```

### Vectorized Operations

```python
# Use numpy operations for efficiency
smoothed_acts = np.zeros_like(net_acts)
# Vectorized smoothing where possible
```

---

*This documentation covers the comprehensive visualization system that transforms meditation simulation data into publication-quality analytical plots. The system supports comparative analysis, temporal visualization, and hierarchical representation of the complex meditation dynamics modeled by the Active Inference framework.*
